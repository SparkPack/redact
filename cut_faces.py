#!/usr/bin/env python
"""Cut face intervals out of synchronized camera feeds using a faces.json from detect_faces.py.

All videos in one group (default: every video in the manifest = the feeds of one episode) share a
timeline. Face intervals from every feed are unioned, padded, merged, snapped outward to keyframes of
each feed, and the face-free remainder is written as numbered segments per feed with `ffmpeg -c copy`
(no re-encode, no GPU). Segment k of every feed covers the same time span, so the feeds stay aligned.

  python cut_faces.py OUT_DIR/faces.json CUT_DIR
  python cut_faces.py faces.json CUT_DIR --group-by parent      # one group per source folder
  python cut_faces.py faces.json CUT_DIR --reencode              # frame-accurate cuts via NVENC instead
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import redact_videos as rv  # noqa: E402

LOG = logging.getLogger("cut")


def pad_merge(intervals: list[list[float]], pad_s: float, merge_s: float, duration: float) -> list[list[float]]:
    iv = sorted([max(0.0, a - pad_s), min(duration, b + pad_s)] for a, b in intervals)
    out: list[list[float]] = []
    for a, b in iv:
        if out and a - out[-1][1] <= merge_s:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def complement(drop: list[list[float]], duration: float, min_len: float) -> list[list[float]]:
    keep, cur = [], 0.0
    for a, b in drop:
        if a - cur >= min_len:
            keep.append([cur, a])
        cur = max(cur, b)
    if duration - cur >= min_len:
        keep.append([cur, duration])
    return keep


def keyframes(tools: rv.FFTools, path: Path) -> list[float]:
    out = subprocess.run([tools.ffprobe, "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey",
                          "-show_entries", "frame=pts_time", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return sorted(float(x) for x in out.replace(",", "\n").split() if x.replace(".", "", 1).isdigit())


def snap_outward(drop: list[list[float]], kf: list[float], duration: float) -> list[list[float]]:
    """Grow each drop interval to the keyframe at or before its start and the keyframe at or after its end."""
    import bisect

    out = []
    for a, b in drop:
        i = bisect.bisect_right(kf, a) - 1
        j = bisect.bisect_left(kf, b)
        a2 = kf[i] if i >= 0 else 0.0
        b2 = kf[j] if j < len(kf) else duration
        out.append([a2, b2])
    return out


def cut_segment(tools: rv.FFTools, src: Path, dst: Path, start: float, end: float, reencode: bool,
                strip_audio: bool, gpu_index: int | None) -> None:
    cmd = [tools.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if reencode:
        cmd += (["-hwaccel", "cuda"] if tools.hwaccel_cuda else []) + ["-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src)]
        cmd += ["-c:v", "h264_nvenc" if "h264_nvenc" in tools.encoders else "libx264", "-preset", "p5", "-cq", "20", "-pix_fmt", "yuv420p"]
        if gpu_index is not None and "h264_nvenc" in tools.encoders:
            cmd += ["-gpu", str(gpu_index)]
    else:
        cmd += ["-ss", f"{start:.3f}", "-i", str(src), "-t", f"{end - start:.3f}", "-c", "copy", "-avoid_negative_ts", "make_zero"]
    cmd += ["-an"] if strip_audio else ["-c:a", "copy"] if not reencode else ["-c:a", "aac"]
    cmd += ["-map_metadata", "0", "-movflags", "+faststart", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg cut failed for {dst.name}: {r.stderr[-400:]}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("faces_json", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--group-by", choices=["all", "parent"], default="all",
                   help="all = every video shares one timeline; parent = videos in the same folder do")
    p.add_argument("--pad-s", type=float, default=0.5, help="extra time removed before/after each face interval")
    p.add_argument("--merge-s", type=float, default=2.0, help="face intervals closer than this become one cut")
    p.add_argument("--min-segment-s", type=float, default=1.0, help="drop kept segments shorter than this")
    p.add_argument("--reencode", action="store_true", help="frame-accurate cuts via NVENC instead of keyframe-aligned stream copy")
    p.add_argument("--keep-audio", action="store_true")
    p.add_argument("--device", default="0")
    p.add_argument("--dry-run", action="store_true", help="print the cut plan, write nothing")
    args = p.parse_args(argv)
    _, gpu_index = rv.parse_device(args.device)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tools = rv.FFTools.discover(None, None)
    m = json.loads(args.faces_json.read_text())
    vids = [v for v in m["videos"] if v["status"] == "ok"]
    if not vids:
        LOG.error("no successful videos in %s", args.faces_json)
        return 2
    groups: dict[str, list[dict]] = defaultdict(list)
    for v in vids:
        key = "all" if args.group_by == "all" else str(Path(v["input"]["path"]).parent)
        groups[key].append(v)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = {"schema": "sparkpack-cuts/1", "created_utc": rv.utc_now(), "source_manifest": str(args.faces_json),
            "params": {k: (str(x) if isinstance(x, Path) else x) for k, x in vars(args).items()}, "groups": []}
    t0 = time.time()
    n_seg = 0
    for key, members in groups.items():
        duration = min(v["input"]["duration_s"] for v in members)  # shared timeline ends with the shortest feed
        raw = [iv for v in members for iv in v["detection"]["intervals"]]
        drop = pad_merge(raw, args.pad_s, args.merge_s, duration)
        if not args.reencode:
            kf_union = sorted(set(t for v in members for t in keyframes(tools, Path(v["input"]["path"]))))
            drop = pad_merge(snap_outward(drop, kf_union, duration), 0.0, 0.0, duration)  # re-merge after snapping
        keep = complement(drop, duration, args.min_segment_s)
        g = {"group": key, "feeds": [v["input"]["path"] for v in members], "timeline_s": round(duration, 3),
             "face_intervals_raw": raw, "dropped": [[round(a, 3), round(b, 3)] for a, b in drop],
             "kept": [[round(a, 3), round(b, 3)] for a, b in keep],
             "kept_s": round(sum(b - a for a, b in keep), 2), "dropped_s": round(sum(b - a for a, b in drop), 2), "segments": []}
        LOG.info("group %s: %d feeds, %.1fs timeline, %d face cuts removing %.1fs, %d segments kept (%.1fs)",
                 key, len(members), duration, len(drop), g["dropped_s"], len(keep), g["kept_s"])
        for k, (a, b) in enumerate(keep):
            for v in members:
                src = Path(v["input"]["path"])
                dst = args.output_dir / f"{src.stem}_seg{k:02d}.mp4"
                g["segments"].append({"index": k, "start_s": round(a, 3), "end_s": round(b, 3), "feed": str(src), "path": str(dst)})
                if args.dry_run:
                    continue
                cut_segment(tools, src, dst, a, b, args.reencode, not args.keep_audio, gpu_index)
                g["segments"][-1]["sha256"] = rv.sha256_file(dst)
                n_seg += 1
        plan["groups"].append(g)
    plan["summary"] = {"groups": len(plan["groups"]), "segments_written": n_seg, "wall_s": round(time.time() - t0, 1),
                       "mode": "dry-run" if args.dry_run else ("reencode" if args.reencode else "stream-copy")}
    out = args.output_dir / "cuts.json"
    if args.dry_run:
        print(json.dumps(plan, indent=2))
    else:
        rv.atomic_write_json(out, plan)
        with open(args.output_dir / "SHA256SUMS", "w") as f:
            for g in plan["groups"]:
                for s in g["segments"]:
                    f.write(f"{s['sha256']}  {Path(s['path']).name}\n")
        LOG.info("done: %d segments -> %s", n_seg, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
