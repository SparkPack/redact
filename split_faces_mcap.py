#!/usr/bin/env python
"""Split a folder of MCAP recordings into face-free and face-only copies.

Builds a self-contained deliverable:

    <out_root>/source/     the originals, untouched (hardlinked by default, --copy-source to duplicate)
    <out_root>/cleaned/    same filenames, with every face-containing span removed
    <out_root>/face_cuts/  same filenames, containing ONLY those spans (absent if a file has no faces)
    <out_root>/split_manifest.json

Every channel travels together: cutting a span removes it from all five cameras plus the IMU, log and
thermal channels, so the cleaned recording stays internally consistent and the face_cuts file is a
faithful excerpt of the whole rig at that instant.

h264 constrains where the boundaries can fall, and the two outputs share them so nothing is lost:

    cut_start ....... the padded start of the face span; cleaned keeps everything before it
    cut_end ......... the next keyframe at or after the padded end; cleaned resumes here so it decodes
    excerpt_start ... the previous keyframe at or before cut_start; face_cuts starts here so IT decodes

face_cuts therefore holds [excerpt_start, cut_end) and cleaned holds everything outside
[cut_start, cut_end). They overlap by a fraction of a second of face-free lead-in, and together they
still cover the whole recording. Log timestamps are preserved on both sides, so a cleaned file simply
has a gap where the faces were.

  python split_faces_mcap.py ~/cocoapack/"card 1" ~/cocoapack/card1_split
  python split_faces_mcap.py IN OUT --faces-json existing_report.json   # reuse a detection pass
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import redact_videos as rv  # noqa: E402
from detect_faces_mcap import VIDEO_SCHEMA, parse_protobuf_fields  # noqa: E402
from extract_faces_mcap import is_keyframe, merge  # noqa: E402

LOG = logging.getLogger("split")
SCRIPT_VERSION = "1.0.0"


class McapCopy:
    """Incremental MCAP writer that lazily mirrors schemas/channels from a source file."""

    def __init__(self, path: Path, library: str):
        from mcap.writer import Writer

        self.path = path
        self.fh = open(path, "wb")
        self.w = Writer(self.fh)
        self.w.start(profile="", library=library)
        self._schemas: dict = {}
        self._channels: dict = {}
        self.count = 0

    def add(self, schema, channel, msg) -> None:
        key = (channel.topic, channel.message_encoding, schema.name if schema else "")
        if key not in self._channels:
            sid = 0
            if schema is not None:
                skey = (schema.name, schema.encoding, bytes(schema.data))
                if skey not in self._schemas:
                    self._schemas[skey] = self.w.register_schema(
                        name=schema.name, encoding=schema.encoding, data=schema.data)
                sid = self._schemas[skey]
            self._channels[key] = self.w.register_channel(
                topic=channel.topic, message_encoding=channel.message_encoding,
                schema_id=sid, metadata=dict(channel.metadata or {}))
        self.w.add_message(channel_id=self._channels[key], log_time=msg.log_time,
                           data=msg.data, publish_time=msg.publish_time, sequence=msg.sequence)
        self.count += 1

    def close(self) -> int:
        self.w.finish()
        self.fh.close()
        return self.count


def keyframe_times(reader, topics: list, t0: float) -> dict:
    kf = defaultdict(list)
    if not topics:
        return kf
    for schema, channel, msg in reader.iter_messages(topics=topics):
        data = parse_protobuf_fields(msg.data).get(3)
        if data and is_keyframe(data):
            kf[channel.topic].append(msg.log_time / 1e9 - t0)
    return kf


def plan_cuts(entry: dict, kf: dict, video_topics: set, pad_s: float, merge_s: float, duration: float) -> list:
    """Face spans -> [(excerpt_start, cut_start, cut_end)] on the file's relative time axis."""
    t0 = entry["log_start_time"]
    raw = []
    for c in entry["channels"]:
        off = c["start_log_time"] - t0
        raw += [[off + a, off + b] for a, b in c["intervals_rel_s"]]
    padded = [[max(0.0, a - pad_s), b + pad_s] for a, b in raw]
    out = []
    for a, b in merge(padded, merge_s):
        # cleaned resumes at the LATEST next-keyframe across cameras, so every feed decodes
        end = b
        for t in video_topics:
            later = [k for k in kf[t] if k >= b]
            end = max(end, later[0] if later else duration + 1.0)
        # face_cuts begins at the EARLIEST prior keyframe, for the same reason
        start = a
        for t in video_topics:
            prior = [k for k in kf[t] if k <= a]
            start = min(start, prior[-1] if prior else 0.0)
        out.append((start, a, end))
    return merge_overlapping(out)


def merge_overlapping(cuts: list) -> list:
    """Collapse cuts whose keyframe-expanded ranges now touch."""
    out: list = []
    for es, cs, ce in sorted(cuts, key=lambda x: x[1]):
        if out and cs <= out[-1][2]:
            out[-1] = (min(out[-1][0], es), out[-1][1], max(out[-1][2], ce))
        else:
            out.append((es, cs, ce))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", type=Path, help="folder of source .mcap files")
    ap.add_argument("out_root", type=Path, help="deliverable root; source/ cleaned/ face_cuts/ are created here")
    ap.add_argument("--faces-json", type=Path, default=None,
                    help="reuse a detect_faces_mcap report instead of running detection")
    ap.add_argument("--pad-s", type=float, default=0.5, help="seconds removed either side of each detection")
    ap.add_argument("--merge-s", type=float, default=2.0, help="spans closer than this become one cut")
    ap.add_argument("--copy-source", action="store_true",
                    help="copy originals into source/ instead of hardlinking them")
    ap.add_argument("--detect-args", default="", help="extra args passed to detect_faces_mcap.py")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from mcap.reader import make_reader

    srcs = sorted(p for p in args.input_dir.glob("*.mcap"))
    if not srcs:
        LOG.error("no .mcap files in %s", args.input_dir)
        return 2
    src_dir, clean_dir, cut_dir = (args.out_root / d for d in ("source", "cleaned", "face_cuts"))
    for d in (src_dir, clean_dir, cut_dir):
        d.mkdir(parents=True, exist_ok=True)

    faces_json = args.faces_json
    if faces_json is None:
        faces_json = args.out_root / "faces.json"
        cmd = [sys.executable, str(Path(__file__).parent / "detect_faces_mcap.py"), str(args.input_dir),
               "-o", str(faces_json)] + args.detect_args.split()
        LOG.info("running detection: %s", " ".join(cmd))
        if subprocess.run(cmd).returncode > 1:
            LOG.error("detection failed")
            return 1
    report = json.loads(Path(faces_json).read_text())
    by_file = {Path(f["file"]).name: f for f in report["files"]}

    lib = f"sparkpack-split-faces/{SCRIPT_VERSION}"
    records = []
    for src in srcs:
        rec = {"file": src.name, "bytes": src.stat().st_size, "sha256": rv.sha256_file(src)}
        # source/ mirror
        dst = src_dir / src.name
        if not dst.exists():
            if args.copy_source:
                shutil.copy2(src, dst)
            else:
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)

        entry = by_file.get(src.name)
        if entry is None:
            # Never certify a recording the detector has not examined. This catches stray files that
            # appeared in the source folder after the detection pass (including this tool's own output).
            LOG.error("%s is NOT in the detection report - skipping. Re-run detection over the source "
                      "folder, or remove the stray file.", src.name)
            (src_dir / src.name).unlink(missing_ok=True)
            rec.update(status="unscanned", cleaned=None, face_cuts=None, cuts=[])
            records.append(rec)
            continue
        has_faces = bool(entry.get("status") == "ok" and entry.get("frames_with_faces_total"))
        if not has_faces:
            # nothing to remove: cleaned/ is a byte-identical copy, no face_cuts/ entry
            shutil.copy2(src, clean_dir / src.name)
            rec.update(status="no_faces" if entry and entry.get("status") == "ok" else
                       (entry or {}).get("status", "unreadable"),
                       cleaned={"path": str(clean_dir / src.name), "sha256": rec["sha256"], "identical": True},
                       face_cuts=None, cuts=[])
            records.append(rec)
            continue

        with open(src, "rb") as fh:
            reader = make_reader(fh)
            summary = reader.get_summary()
            video_topics = {ch.topic for ch in summary.channels.values()
                            if ch.schema_id and summary.schemas[ch.schema_id].name == VIDEO_SCHEMA}
            t0 = entry["log_start_time"]
            kf = keyframe_times(reader, sorted(video_topics), t0)
        cuts = plan_cuts(entry, kf, video_topics, args.pad_s, args.merge_s, entry["duration_s"])
        LOG.info("%s: %d cut(s) %s", src.name,
                 len(cuts), [[round(cs, 2), round(ce, 2)] for _, cs, ce in cuts])

        cleaned = McapCopy(clean_dir / src.name, lib)
        cutfile = McapCopy(cut_dir / src.name, lib)
        dropped = 0
        with open(src, "rb") as fh:
            for schema, channel, msg in make_reader(fh).iter_messages():
                rel = msg.log_time / 1e9 - t0
                in_cut = in_excerpt = False
                for es, cs, ce in cuts:
                    if cs <= rel < ce:
                        in_cut = in_excerpt = True
                        break
                    if es <= rel < cs:
                        in_excerpt = True
                        break
                if in_excerpt:
                    cutfile.add(schema, channel, msg)
                if in_cut:
                    dropped += 1
                else:
                    cleaned.add(schema, channel, msg)
        n_clean, n_cut = cleaned.close(), cutfile.close()
        cp, xp = clean_dir / src.name, cut_dir / src.name
        rec.update(
            status="cut",
            cuts=[{"excerpt_start_s": round(es, 3), "cut_start_s": round(cs, 3), "cut_end_s": round(ce, 3),
                   "removed_s": round(ce - cs, 3)} for es, cs, ce in cuts],
            removed_s=round(sum(ce - cs for _, cs, ce in cuts), 3),
            cleaned={"path": str(cp), "bytes": cp.stat().st_size, "sha256": rv.sha256_file(cp),
                     "messages": n_clean, "messages_removed": dropped},
            face_cuts={"path": str(xp), "bytes": xp.stat().st_size, "sha256": rv.sha256_file(xp),
                       "messages": n_cut},
        )
        LOG.info("   cleaned %d msgs (-%d), face_cuts %d msgs, %.2fs removed",
                 n_clean, dropped, n_cut, rec["removed_s"])
        records.append(rec)

    manifest = {
        "schema": "sparkpack-face-split/1", "created_utc": rv.utc_now(),
        "script": Path(__file__).name, "script_version": SCRIPT_VERSION,
        "source_dir": str(args.input_dir), "faces_report": str(faces_json),
        "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "files": records,
        "summary": {
            "files": len(records),
            "with_cuts": sum(1 for r in records if r["status"] == "cut"),
            "untouched": sum(1 for r in records if r["status"] not in ("cut", "unscanned")),
            "unscanned_skipped": sum(1 for r in records if r["status"] == "unscanned"),
            "removed_s": round(sum(r.get("removed_s", 0) for r in records), 3),
        },
    }
    out = args.out_root / "split_manifest.json"
    rv.atomic_write_json(out, manifest)
    LOG.info("done: %s", json.dumps(manifest["summary"]))
    LOG.info("  source/    %d files", len(list(src_dir.glob('*.mcap'))))
    LOG.info("  cleaned/   %d files", len(list(clean_dir.glob('*.mcap'))))
    LOG.info("  face_cuts/ %d files", len(list(cut_dir.glob('*.mcap'))))
    LOG.info("  manifest:  %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
