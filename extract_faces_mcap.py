#!/usr/bin/env python
"""Cut the face-containing spans out of MCAP recordings and concatenate them into one new MCAP.

Reads a report from detect_faces_mcap.py, turns each detection into a padded time interval, and copies
the messages inside those intervals into a single output MCAP. Every channel of the source is carried
over (all five camera feeds plus IMU/log/thermal), so a segment is a faithful excerpt of the whole rig
at that instant, not just the camera that saw the face.

h264 constraint: a segment must begin on an IDR frame or it will not decode, so each video channel's
segment start is snapped BACKWARDS to the last keyframe at or before the interval. That means a little
extra footage at the head of each clip; it is never less than the detected span.

Message log times are preserved by default, so every frame keeps its provenance and can be traced back
to the source recording. Pass --contiguous to restamp segments back-to-back instead.

  python extract_faces_mcap.py FACES.json faces_only.mcap
  python extract_faces_mcap.py FACES.json out.mcap --pad-s 1.0 --video-only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import redact_videos as rv  # noqa: E402
from detect_faces_mcap import VIDEO_SCHEMA, parse_protobuf_fields  # noqa: E402

LOG = logging.getLogger("extract")
SCRIPT_VERSION = "1.0.0"


def is_keyframe(pkt: bytes) -> bool:
    """True if the access unit contains an IDR slice (NAL type 5)."""
    i, n = 0, len(pkt)
    while i < n - 3:
        if pkt[i] == 0 and pkt[i + 1] == 0:
            if pkt[i + 2] == 1:
                if (pkt[i + 3] & 0x1F) == 5:
                    return True
                i += 3
                continue
            if pkt[i + 2] == 0 and i + 4 < n and pkt[i + 3] == 1:
                if (pkt[i + 4] & 0x1F) == 5:
                    return True
                i += 4
                continue
        i += 1
    return False


def merge(intervals: list, gap_s: float) -> list:
    out: list = []
    for a, b in sorted(intervals):
        if out and a - out[-1][1] <= gap_s:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("faces_json", type=Path)
    ap.add_argument("output", type=Path, help="output .mcap")
    ap.add_argument("--pad-s", type=float, default=0.5, help="seconds kept either side of each detection")
    ap.add_argument("--merge-s", type=float, default=2.0, help="intervals closer than this become one segment")
    ap.add_argument("--video-only", action="store_true", help="carry only CompressedVideo channels")
    ap.add_argument("--topics", nargs="+", default=None, help="restrict to these topics")
    ap.add_argument("--preserve-timestamps", dest="contiguous", action="store_false", default=True,
                    help="keep each message's original log time. Off by default: the source recordings are "
                         "hours apart, so preserving them scatters a few seconds of footage across a day-long "
                         "timeline. Segment provenance is recorded in the .json sidecar either way.")
    ap.add_argument("--min-segment-s", type=float, default=0.0, help="drop segments shorter than this")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from mcap.reader import make_reader
    from mcap.writer import Writer

    report = json.loads(args.faces_json.read_text())
    files = [f for f in report["files"] if f.get("status") == "ok" and f.get("frames_with_faces_total")]
    if not files:
        LOG.error("no files with detections in %s", args.faces_json)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_fh = open(args.output, "wb")
    writer = Writer(out_fh)
    writer.start(profile="", library=f"sparkpack-extract-faces/{SCRIPT_VERSION}")
    schema_ids: dict[tuple, int] = {}
    channel_ids: dict[tuple, int] = {}
    plan = []
    written = 0
    time_cursor = None

    for f in files:
        src = Path(f["file"])
        t0 = f["log_start_time"]
        # union the per-channel intervals onto the shared log-time axis, then pad and merge
        raw = []
        for c in f["channels"]:
            off = c["start_log_time"] - t0
            raw += [[off + a, off + b] for a, b in c["intervals_rel_s"]]
        padded = [[max(0.0, a - args.pad_s), b + args.pad_s] for a, b in raw]
        segments = [s for s in merge(padded, args.merge_s) if s[1] - s[0] >= args.min_segment_s]
        if not segments:
            continue
        LOG.info("%s: %d detection span(s) -> %d segment(s): %s", src.name, len(raw), len(segments),
                 [[round(a, 2), round(b, 2)] for a, b in segments])

        with open(src, "rb") as fh:
            reader = make_reader(fh)
            summary = reader.get_summary()
            wanted = {}
            for cid, ch in summary.channels.items():
                sname = summary.schemas[ch.schema_id].name if ch.schema_id else ""
                if args.video_only and sname != VIDEO_SCHEMA:
                    continue
                if args.topics and ch.topic not in args.topics:
                    continue
                wanted[cid] = (ch, sname)

            # Pass 1: for each video channel, find the keyframe times that segment starts must snap back to.
            kf_times: dict[str, list] = defaultdict(list)
            video_topics = {ch.topic for ch, sname in wanted.values() if sname == VIDEO_SCHEMA}
            if video_topics:
                for schema, channel, msg in reader.iter_messages(topics=list(video_topics)):
                    data = parse_protobuf_fields(msg.data).get(3)
                    if data and is_keyframe(data):
                        kf_times[channel.topic].append(msg.log_time / 1e9 - t0)

            # snap each segment start back per channel; use the earliest so all feeds stay aligned
            snapped = []
            for a, b in segments:
                start = a
                for topic in video_topics:
                    prior = [k for k in kf_times[topic] if k <= a]
                    start = min(start, prior[-1] if prior else 0.0)
                snapped.append((start, b, a))
            LOG.info("   after snapping to keyframes: %s",
                     [[round(s, 2), round(e, 2)] for s, e, _ in snapped])

        # Pass 2: copy every message that falls inside a snapped segment
        with open(src, "rb") as fh:
            reader = make_reader(fh)
            summary = reader.get_summary()
            seg_counts = [0] * len(snapped)
            seg_first_last: list = [[None, None] for _ in snapped]
            # where each segment starts on the output timeline when restamping
            seg_offsets = []
            cur = time_cursor or 0
            for s_, e_, _ in snapped:
                seg_offsets.append(cur)
                cur += int((e_ - s_) * 1e9)
            for schema, channel, msg in reader.iter_messages(topics=[c.topic for c, _ in wanted.values()]):
                rel = msg.log_time / 1e9 - t0
                for i, (s, e, _) in enumerate(snapped):
                    if s <= rel <= e:
                        key = (channel.topic, channel.message_encoding, schema.name if schema else "")
                        if key not in channel_ids:
                            skey = (schema.name, schema.encoding, bytes(schema.data)) if schema else None
                            sid = 0
                            if skey is not None:
                                if skey not in schema_ids:
                                    schema_ids[skey] = writer.register_schema(
                                        name=schema.name, encoding=schema.encoding, data=schema.data)
                                sid = schema_ids[skey]
                            meta = dict(channel.metadata or {})
                            meta["excerpt_of"] = "multiple recordings; see the .json sidecar for segment provenance"
                            channel_ids[key] = writer.register_channel(
                                topic=channel.topic, message_encoding=channel.message_encoding,
                                schema_id=sid, metadata=meta)
                        log_time = msg.log_time
                        if args.contiguous:
                            log_time = int(seg_offsets[i] + (rel - s) * 1e9)
                        writer.add_message(channel_id=channel_ids[key], log_time=log_time,
                                           data=msg.data, publish_time=log_time, sequence=msg.sequence)
                        written += 1
                        seg_counts[i] += 1
                        fl = seg_first_last[i]
                        fl[0] = msg.log_time if fl[0] is None else min(fl[0], msg.log_time)
                        fl[1] = msg.log_time if fl[1] is None else max(fl[1], msg.log_time)
                        break
            if args.contiguous:
                time_cursor = cur
            for i, (s, e, orig) in enumerate(snapped):
                plan.append({
                    "source": str(src), "segment": i, "detected_start_s": round(orig, 3),
                    "output_start_s": round(seg_offsets[i] / 1e9, 3) if args.contiguous else None,
                    "kept_start_s": round(s, 3), "kept_end_s": round(e, 3),
                    "duration_s": round(e - s, 3), "messages": seg_counts[i],
                    "source_log_time_range": [round((seg_first_last[i][0] or 0) / 1e9, 6),
                                              round((seg_first_last[i][1] or 0) / 1e9, 6)],
                })

    writer.finish()
    out_fh.close()
    if written == 0:
        LOG.error("no messages copied; output is empty")
        return 1

    sidecar = args.output.with_suffix(args.output.suffix + ".json")
    rv.atomic_write_json(sidecar, {
        "schema": "sparkpack-face-excerpt/1", "created_utc": rv.utc_now(),
        "script": Path(__file__).name, "script_version": SCRIPT_VERSION,
        "source_report": str(args.faces_json),
        "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "output": {"path": str(args.output), "bytes": args.output.stat().st_size,
                   "sha256": rv.sha256_file(args.output), "messages": written,
                   "channels": len(channel_ids)},
        "segments": plan,
    })
    total = sum(p["duration_s"] for p in plan)
    LOG.info("wrote %s: %d messages, %d channels, %d segments, %.2fs of footage",
             args.output, written, len(channel_ids), len(plan), total)
    LOG.info("sidecar: %s", sidecar)
    return 0


if __name__ == "__main__":
    sys.exit(main())
