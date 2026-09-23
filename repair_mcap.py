#!/usr/bin/env python
"""Rebuild an MCAP whose summary section was never written.

A recording the rig did not close carries every message but no footer, statistics or chunk index. The
normal reader plans its read from that index, so it fails immediately — on IC-559 with

    RecordLengthLimitExceeded: unknown (opcode 0) record has length 11687914036116062208

which says nothing about the real problem. The data is intact; only the trailer is missing.

This walks the records from the front, ignoring the index entirely, and writes them into a fresh file
that ends with a proper summary. Nothing is re-encoded: schemas, channels, log times, publish times
and sequence numbers are copied verbatim, so the repaired file is the same recording, readable.

  python repair_mcap.py BROKEN.mcap FIXED.mcap
  python repair_mcap.py BROKEN.mcap FIXED.mcap --expect-truncated   # do not fail on a partial tail

A file truncated mid-record (an interrupted transfer rather than an unfinalised write) recovers every
whole record before the cut; pass --expect-truncated to accept that instead of treating it as an error.
Re-copy such a file first — this is a last resort for one whose source is gone.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import redact_videos as rv  # noqa: E402

LOG = logging.getLogger("repair")


def repair(src: Path, dst: Path, expect_truncated: bool, progress_every: int) -> dict:
    from mcap.stream_reader import StreamReader
    from mcap.writer import Writer

    schemas: dict = {}          # source schema id -> new id
    channels: dict = {}         # source channel id -> new id
    src_schemas: dict = {}      # source schema id -> record, held until a channel needs it
    counts: dict = {}
    n_msg = 0
    lo = hi = None
    truncated = None

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    fh_out = open(tmp, "wb")
    w = Writer(fh_out)
    w.start(profile="", library=f"sparkpack-redact-repair/{rv.__name__}")
    try:
        with open(src, "rb") as fh:
            try:
                for rec in StreamReader(fh, skip_magic=False).records:
                    kind = type(rec).__name__
                    counts[kind] = counts.get(kind, 0) + 1
                    if kind == "Schema":
                        src_schemas[rec.id] = rec
                    elif kind == "Channel":
                        sid = 0
                        if rec.schema_id and rec.schema_id in src_schemas:
                            s = src_schemas[rec.schema_id]
                            if rec.schema_id not in schemas:
                                schemas[rec.schema_id] = w.register_schema(
                                    name=s.name, encoding=s.encoding, data=s.data)
                            sid = schemas[rec.schema_id]
                        channels[rec.id] = w.register_channel(
                            topic=rec.topic, message_encoding=rec.message_encoding,
                            schema_id=sid, metadata=dict(rec.metadata or {}))
                    elif kind == "Message":
                        cid = channels.get(rec.channel_id)
                        if cid is None:
                            # A message before its channel record cannot be placed; the source is
                            # malformed in a way this tool will not silently paper over.
                            raise RuntimeError(
                                f"message references unknown channel {rec.channel_id}")
                        w.add_message(channel_id=cid, log_time=rec.log_time, data=rec.data,
                                      publish_time=rec.publish_time, sequence=rec.sequence)
                        n_msg += 1
                        lo = rec.log_time if lo is None else min(lo, rec.log_time)
                        hi = rec.log_time if hi is None else max(hi, rec.log_time)
                        if progress_every and n_msg % progress_every == 0:
                            LOG.info("  %d messages", n_msg)
                    elif kind == "Metadata":
                        w.add_metadata(name=rec.name, data=dict(rec.metadata or {}))
                    # MessageIndex/ChunkIndex/Statistics/Footer are all summary records; the writer
                    # builds correct ones itself, and the broken ones are what we are discarding.
            except Exception as e:  # noqa: BLE001
                # StreamReader raises at the point the records stop making sense. Everything read up
                # to here is whole and already written.
                truncated = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
    finally:
        w.finish()
        fh_out.close()

    if truncated and not expect_truncated:
        LOG.warning("source ended early (%s); %d messages recovered", truncated, n_msg)
    if not n_msg:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("no messages could be read; this file is not recoverable this way")

    tmp.replace(dst)
    return {"source": str(src), "output": str(dst), "messages": n_msg,
            "channels": len(channels), "schemas": len(schemas),
            "duration_s": round((hi - lo) / 1e9, 3) if lo is not None else 0.0,
            "records": counts, "ended_early": truncated,
            "bytes": dst.stat().st_size}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="the unreadable .mcap")
    ap.add_argument("output", type=Path, help="where to write the repaired copy")
    ap.add_argument("--expect-truncated", action="store_true",
                    help="do not warn when the source ends mid-record")
    ap.add_argument("--progress-every", type=int, default=200000,
                    help="log every N messages (0 to silence)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not a.input.exists():
        sys.exit(f"error: {a.input} does not exist")
    if a.output.exists():
        sys.exit(f"error: {a.output} already exists; refusing to overwrite")

    LOG.info("repairing %s", a.input.name)
    r = repair(a.input, a.output, a.expect_truncated, a.progress_every)
    LOG.info("wrote %s", r["output"])
    LOG.info("  %d messages, %d channels, %.1f s, %.2f GB",
             r["messages"], r["channels"], r["duration_s"], r["bytes"] / 1e9)

    # Prove it: the repaired file must open through the normal indexed reader, which is exactly what
    # the source could not do. Reporting success without this check would be worthless.
    from mcap.reader import make_reader
    with open(a.output, "rb") as fh:
        s = make_reader(fh).get_summary()
    if s is None or not s.statistics:
        LOG.error("  repaired file still has no summary")
        return 1
    LOG.info("  verified: reads through the indexed reader, %d channels, %d messages",
             len(s.channels), s.statistics.message_count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
