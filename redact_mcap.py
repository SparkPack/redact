#!/usr/bin/env python
"""One entry point for redacting rig MCAPs, in either of two modes.

    blur  - keep every frame, blur the faces.       Parameters: --conf
    cut   - keep no faces, drop the footage instead. Parameters: --conf, --min-keep-s

Blur preserves the whole recording and every message, so nothing is lost but the face pixels. Cut
preserves image fidelity and throws away time: the recording is split into the face-free stretches,
and any stretch shorter than --min-keep-s is discarded rather than kept as an unusable fragment. That
minimum is what collapses a scatter of small cuts into a few long, clean clips.

  redact_mcap.py SRC OUT --mode blur --conf 0.95
  redact_mcap.py SRC OUT --mode cut  --conf 0.95 --min-keep-s 30

Choosing --conf: run it against footage you know is face-free and raise the threshold until nothing is
detected. On IC-559 rig footage that landed at 0.95 (0.90 still let 6 false positives through in four
minutes). Calibrate per rig; do not assume this number transfers.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import redact_videos as rv  # noqa: E402
from detect_faces_mcap import VIDEO_SCHEMA, parse_protobuf_fields  # noqa: E402
from extract_faces_mcap import is_keyframe  # noqa: E402
from plan_face_cuts import complement, merge  # noqa: E402
from split_faces_mcap import McapCopy  # noqa: E402

LOG = logging.getLogger("redact")
SCRIPT_VERSION = "1.0.0"
HERE = Path(__file__).resolve().parent


def run_detection(src: Path, out_json: Path, args) -> dict:
    cmd = [sys.executable, str(HERE / "detect_faces_mcap.py"), str(src), "-o", str(out_json),
           "--face-detector", "egoblur", "--egoblur-weights", str(args.egoblur_weights),
           "--scale", "1.0", "--conf", str(args.conf), "--conf-topic", args.conf_topic,
           "--stride", str(args.stride), "--device", args.device, "--no-sha256"]
    if args.half:
        cmd.append("--half")
    LOG.info("detecting: %s", " ".join(cmd[-10:]))
    if subprocess.run(cmd).returncode > 1:
        raise RuntimeError("detection failed")
    return json.loads(out_json.read_text())


def cut_one(src: Path, out_dir: Path, entry: dict, args) -> list:
    """Write each face-free stretch of one recording as its own MCAP."""
    from mcap.reader import make_reader

    t0 = entry["log_start_time"]
    duration = entry["duration_s"]
    raw = []
    for c in entry["channels"]:
        off = c["start_log_time"] - t0
        raw += [[off + a, off + b] for a, b in c["intervals_rel_s"]]
    drop = merge([[max(0.0, a - args.pad_s), b + args.pad_s] for a, b in raw], args.merge_s)
    keep = complement(drop, duration, args.min_keep_s)
    if not keep:
        LOG.warning("   %s: nothing survives a %.0fs minimum", src.name, args.min_keep_s)
        return []

    # Each kept clip must begin on a keyframe or it will not decode. Snap FORWARD to the next one so
    # the clip never reaches back into the removed span.
    with open(src, "rb") as fh:
        reader = make_reader(fh)
        summary = reader.get_summary()
        vt = {ch.topic for ch in summary.channels.values()
              if ch.schema_id and summary.schemas[ch.schema_id].name == VIDEO_SCHEMA}
        kf: dict = {t: [] for t in vt}
        for schema, channel, msg in reader.iter_messages(topics=sorted(vt)):
            d = parse_protobuf_fields(msg.data).get(3)
            if d and is_keyframe(d):
                kf[channel.topic].append(msg.log_time / 1e9 - t0)
    snapped = []
    for a, b in keep:
        start = a
        for t in vt:                       # latest next-keyframe, so every camera decodes
            later = [k for k in kf[t] if k >= a]
            start = max(start, later[0] if later else b)
        if b - start >= args.min_keep_s:
            snapped.append((start, b))
    LOG.info("   %s: %d clip(s) %s", src.name, len(snapped),
             [[round(a, 1), round(b, 1)] for a, b in snapped])

    # The removed spans go to their own folder so they can be reviewed: this is the footage the tool
    # decided contains faces, and it is the only way to check the decision was right.
    removed_dir = out_dir / "removed"
    removed = [] if args.no_removed else [(x, y) for x, y in _gaps(snapped, duration)
                                          if y - x >= args.min_removed_s]
    if removed:
        removed_dir.mkdir(parents=True, exist_ok=True)

    targets = [(out_dir / f"{src.stem}_clip{i:02d}.mcap", a, b, "clip")
               for i, (a, b) in enumerate(snapped)]
    targets += [(removed_dir / f"{src.stem}_removed{i:02d}.mcap", a, b, "removed")
                for i, (a, b) in enumerate(removed)]

    # One pass over the source per output would re-read 14 GB per clip; write them all together.
    writers = {}
    for dst, a, b, kind in targets:
        writers[dst] = (McapCopy(dst, f"sparkpack-redact-cut/{SCRIPT_VERSION}"), a, b, kind)
    with open(src, "rb") as fh:
        for schema, channel, msg in make_reader(fh).iter_messages():
            rel = msg.log_time / 1e9 - t0
            for w, a, b, _ in writers.values():
                if a <= rel <= b:
                    w.add(schema, channel, msg)
    written, cut_out = [], []
    for dst, (w, a, b, kind) in writers.items():
        n = w.close()
        rec = {"path": str(dst), "start_s": round(a, 3), "end_s": round(b, 3),
               "duration_s": round(b - a, 3), "messages": n,
               "bytes": dst.stat().st_size, "sha256": rv.sha256_file(dst)}
        (written if kind == "clip" else cut_out).append(rec)
        LOG.info("      %-8s %7.1f-%7.1fs (%6.1fs, %7d msgs)  %s", kind, a, b, b - a, n, dst.name)
    return written, cut_out


def _gaps(keep: list, duration: float) -> list:
    """The spans NOT covered by the kept clips - i.e. what was removed."""
    out, cur = [], 0.0
    for a, b in keep:
        if a > cur:
            out.append((cur, a))
        cur = max(cur, b)
    if duration > cur:
        out.append((cur, duration))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, help=".mcap file or folder")
    p.add_argument("output", type=Path, help="output folder")
    p.add_argument("--mode", choices=["blur", "cut"], required=True)
    p.add_argument("--conf", type=float, default=0.95,
                   help="face score threshold; calibrate on known face-free footage (default 0.95)")
    p.add_argument("--conf-topic", default="wrist:0.99",
                   help="per-camera overrides; wrist cameras point at the work surface and almost never "
                        "see a face, so they carry a stricter bar")
    p.add_argument("--min-keep-s", type=float, default=30.0,
                   help="CUT MODE: discard any face-free stretch shorter than this instead of keeping a "
                        "fragment. This is the knob that turns many small cuts into a few long clips.")
    p.add_argument("--pad-s", type=float, default=1.0, help="cut mode: margin around each face")
    p.add_argument("--merge-s", type=float, default=0.0, help="cut mode: merge faces closer than this")
    p.add_argument("--stride", type=int, default=3, help="detect every Nth frame")
    p.add_argument("--egoblur-weights", type=Path, default=HERE / "weights/ego_blur_face_gen2.jit")
    p.add_argument("--device", default="1")
    p.add_argument("--no-half", dest="half", action="store_false", default=True)
    p.add_argument("--faces-json", type=Path, default=None, help="reuse an existing detection report")
    p.add_argument("--no-removed", action="store_true",
                   help="cut mode: do not write the removed (face-containing) spans to OUT/removed/")
    p.add_argument("--min-removed-s", type=float, default=0.5,
                   help="cut mode: skip writing removed spans shorter than this")
    # blur-mode pass-throughs
    p.add_argument("--dilate-px", type=int, default=30)
    p.add_argument("--smooth-window", type=int, default=15)
    p.add_argument("--blur-block", type=int, default=28)
    p.add_argument("--channel-workers", type=int, default=5)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a.output.mkdir(parents=True, exist_ok=True)

    if a.mode == "blur":
        cmd = [sys.executable, str(HERE / "blur_faces_mcap.py"), str(a.input), str(a.output),
               "--conf", str(a.conf), "--conf-topic", a.conf_topic, "--stride", str(a.stride),
               "--dilate-px", str(a.dilate_px), "--smooth-window", str(a.smooth_window),
               "--blur-block", str(a.blur_block), "--channel-workers", str(a.channel_workers),
               "--egoblur-weights", str(a.egoblur_weights), "--device", a.device]
        if not a.half:
            cmd.append("--no-half")
        LOG.info("blur mode, conf %.2f", a.conf)
        return subprocess.run(cmd).returncode

    # cut mode
    srcs = sorted(a.input.glob("*.mcap")) if a.input.is_dir() else [a.input]
    report = (json.loads(a.faces_json.read_text()) if a.faces_json
              else run_detection(a.input, a.output / "faces.json", a))
    by = {Path(f["file"]).name: f for f in report["files"]}
    LOG.info("cut mode, conf %.2f, minimum clip %.0fs", a.conf, a.min_keep_s)
    records = []
    for src in srcs:
        e = by.get(src.name)
        if e is None or e.get("status") != "ok":
            LOG.warning("   %s: not in the detection report, skipped", src.name)
            continue
        clips, removed = cut_one(src, a.output, e, a)
        kept = sum(c["duration_s"] for c in clips)
        records.append({"file": src.name, "duration_s": e["duration_s"], "clips": clips,
                        "removed": removed, "kept_s": round(kept, 1),
                        "removed_s": round(sum(c["duration_s"] for c in removed), 1),
                        "kept_pct": round(100 * kept / max(e["duration_s"], 1e-9), 1)})
    man = a.output / "cut_manifest.json"
    rv.atomic_write_json(man, {
        "schema": "sparkpack-redact-cut/1", "created_utc": rv.utc_now(),
        "script_version": SCRIPT_VERSION,
        "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
        "files": records,
        "summary": {"files": len(records), "clips": sum(len(r["clips"]) for r in records),
                    "kept_s": round(sum(r["kept_s"] for r in records), 1),
                    "source_s": round(sum(r["duration_s"] for r in records), 1)},
    })
    for r in records:
        LOG.info("%s: %d clip(s) kept %.0fs of %.0fs (%.0f%%), %d removed span(s) totalling %.0fs",
                 r["file"], len(r["clips"]), r["kept_s"], r["duration_s"], r["kept_pct"],
                 len(r["removed"]), r["removed_s"])
    LOG.info("manifest: %s", man)
    return 0


if __name__ == "__main__":
    sys.exit(main())
