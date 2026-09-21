#!/usr/bin/env python
"""Detect-only pass: find faces in a folder of videos, write intervals to a manifest. No blur, no re-encode.

Per video: ffmpeg decode (NVDEC when available) at a frame stride -> YuNet (CPU) or EgoBlur (GPU) face
detector -> per-frame boxes -> merged face intervals. One worker process per video (default 4, one per
rig camera). Feed the manifest to cut_faces.py to split the clips.

  python detect_faces.py IN OUT_DIR                      # YuNet, stride 3, half-res
  python detect_faces.py IN OUT_DIR --face-detector egoblur --egoblur-weights w.jit --device 0
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import redact_videos as rv  # noqa: E402

LOG = logging.getLogger("detect")
SCRIPT_VERSION = "1.0.0"


def merge_intervals(times: list[float], gap_s: float, hold_s: float) -> list[list[float]]:
    """Turn sorted detection timestamps into [start, end] intervals.

    Consecutive detections closer than gap_s belong to the same interval; each interval extends hold_s past
    its last detection (one stride period, so the un-checked frames after the last hit are covered).
    """
    out: list[list[float]] = []
    for t in times:
        if out and t - out[-1][1] <= gap_s:
            out[-1][1] = t
        else:
            out.append([t, t])
    return [[round(a, 3), round(b + hold_s, 3)] for a, b in out]


def detect_one(video: str, cfg: dict) -> dict:
    """Worker: detect faces in one video. Runs in its own process."""
    cv2.setNumThreads(cfg["cv_threads"])
    tools = rv.FFTools(**cfg["tools"])
    path = Path(video)
    t0 = time.time()
    info = rv.ffprobe_video(tools, path)
    W, H = info.display_width, info.display_height
    stride = cfg["stride"]
    if cfg["face_detector"] == "yunet":
        det = rv.YuNetFaceDetector(Path(cfg["yunet_weights"]), cfg["conf"], cfg["nms"], 1.0, cfg["scale"])
        # re-create with score access: YuNetFaceDetector.detect drops scores, so call the underlying model here
        yn = det.det
        scale = cfg["scale"]

        def run(frame):
            sw, sh = int(round(W * scale)), int(round(H * scale))
            small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA) if scale != 1.0 else frame
            yn.setInputSize((sw, sh))
            _, faces = yn.detect(small)
            if faces is None:
                return np.zeros((0, 5), np.float32)
            f = faces[:, :4] / scale
            return np.stack([f[:, 0], f[:, 1], f[:, 0] + f[:, 2], f[:, 1] + f[:, 3], faces[:, 14]], 1).astype(np.float32)
    else:
        det = rv.FaceDetector(Path(cfg["egoblur_weights"]), cfg["torch_dev"], cfg["egoblur_gen"], cfg["conf"], cfg["nms"], 1.0)

        def run(frame):
            return det.detect(frame)  # already Nx5 (xyxy + score)

    reader = rv.FrameReader(tools, path, W, H, cfg["gpu_decode"], cfg["gpu_index"], select_stride=stride)
    dets = []
    checked = 0
    min_px = cfg["min_face_px"]
    try:
        i = 0
        while True:
            if cfg["max_frames"] and i * stride >= cfg["max_frames"]:
                break
            f = reader.read()
            if f is None:
                break
            fidx = i * stride
            i += 1
            checked += 1
            boxes = run(f)
            if len(boxes):
                keep = np.minimum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]) >= min_px
                boxes = boxes[keep]
            if len(boxes):
                dets.append({"frame": fidx, "time_s": round(fidx / info.fps_float, 3),
                             "boxes": [[int(x1), int(y1), int(x2), int(y2), round(float(s), 3)] for x1, y1, x2, y2, s in boxes]})
    finally:
        err = reader.close()
    hold = stride / info.fps_float
    intervals = merge_intervals([d["time_s"] for d in dets], gap_s=cfg["gap_s"], hold_s=hold)
    face_time = sum(b - a for a, b in intervals)
    dt = time.time() - t0
    return {
        "input": {**asdict(info), "sha256": rv.sha256_file(path) if cfg["sha256"] else None},
        "status": "ok",
        "detection": {
            "detector": cfg["face_detector"], "stride": stride, "scale": cfg["scale"] if cfg["face_detector"] == "yunet" else None,
            "conf": cfg["conf"], "min_face_px": min_px, "frames_checked": checked, "frames_with_faces": len(dets),
            "max_faces_in_frame": max((len(d["boxes"]) for d in dets), default=0),
            "largest_face_px": max((min(b[2] - b[0], b[3] - b[1]) for d in dets for b in d["boxes"]), default=0),
            "face_time_s": round(face_time, 3), "face_fraction": round(face_time / max(info.duration_s, 1e-6), 4),
            "intervals": intervals, "detections": dets,
        },
        "timing": {"total_s": round(dt, 2), "source_fps": round(checked * stride / max(dt, 1e-6), 1),
                   "decoder_stderr": err[-300:] if err and "Broken pipe" not in err else ""},
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_dir", type=Path)
    p.add_argument("output_dir", type=Path, help="where faces.json is written")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--workers", type=int, default=4, help="parallel videos (one process each)")
    p.add_argument("--cv-threads", type=int, default=8, help="OpenCV threads per worker")
    p.add_argument("--stride", type=int, default=3, help="check every Nth frame (3 = 20 Hz at 60 fps)")
    p.add_argument("--max-frames", type=int, default=None, help="debug: stop after N source frames per video")
    p.add_argument("--no-sha256", action="store_true", help="skip input checksums (faster on big files)")
    g = p.add_argument_group("detector")
    g.add_argument("--face-detector", choices=["yunet", "egoblur"], default="yunet")
    g.add_argument("--yunet-weights", type=Path, default=Path(__file__).parent / "weights/face_detection_yunet_2023mar.onnx")
    g.add_argument("--scale", type=float, default=0.5, help="YuNet input scale; 0.5 finds faces >= ~40 px tall at 1080p")
    g.add_argument("--egoblur-weights", type=Path, default=None)
    g.add_argument("--egoblur-gen", choices=["auto", "1", "2"], default="auto")
    g.add_argument("--conf", type=float, default=0.6)
    g.add_argument("--nms", type=float, default=0.3)
    g.add_argument("--min-face-px", type=int, default=24, help="ignore boxes whose short side is smaller (full-res px)")
    g.add_argument("--gap-s", type=float, default=1.0, help="detections closer than this merge into one interval")
    g.add_argument("--device", default="0", help="GPU for NVDEC (and EgoBlur); cpu disables NVDEC")
    g.add_argument("--no-gpu-decode", action="store_true")
    g.add_argument("--ffmpeg", default=None)
    g.add_argument("--ffprobe", default=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    torch_dev, gpu_index = rv.parse_device(args.device)
    if not args.input_dir.is_dir():
        sys.exit(f"error: {args.input_dir} is not a directory")
    if args.face_detector == "egoblur" and (args.egoblur_weights is None or not args.egoblur_weights.is_file()):
        sys.exit("error: --egoblur-weights required for --face-detector egoblur")
    if args.face_detector == "yunet" and not args.yunet_weights.is_file():
        sys.exit(f"error: YuNet weights not found at {args.yunet_weights}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(args.output_dir / "detect.log")])
    tools = rv.FFTools.discover(args.ffmpeg, args.ffprobe)
    vids = rv.find_videos(args.input_dir, args.recursive)
    if not vids:
        LOG.error("no videos in %s", args.input_dir)
        return 2
    cfg = dict(
        tools={"ffmpeg": tools.ffmpeg, "ffprobe": tools.ffprobe, "version": tools.version,
               "hwaccel_cuda": tools.hwaccel_cuda, "encoders": set()},
        stride=args.stride, max_frames=args.max_frames, sha256=not args.no_sha256, cv_threads=args.cv_threads,
        face_detector=args.face_detector, yunet_weights=str(args.yunet_weights), scale=args.scale,
        egoblur_weights=str(args.egoblur_weights) if args.egoblur_weights else None, egoblur_gen=args.egoblur_gen,
        torch_dev=torch_dev, conf=args.conf, nms=args.nms, min_face_px=args.min_face_px, gap_s=args.gap_s,
        gpu_decode=tools.hwaccel_cuda and not args.no_gpu_decode and gpu_index is not None, gpu_index=gpu_index,
    )
    LOG.info("%d videos, %s stride %d, %d workers, NVDEC=%s", len(vids), args.face_detector, args.stride,
             args.workers, cfg["gpu_decode"])
    manifest = {
        "schema": "sparkpack-faces/1", "created_utc": rv.utc_now(), "script": Path(__file__).name,
        "script_version": SCRIPT_VERSION, "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "ffmpeg": tools.version, "videos": [],
    }
    t0 = time.time()
    results: dict[str, dict] = {}
    workers = max(1, min(args.workers, len(vids)))
    if args.face_detector == "egoblur":
        workers = 1  # one CUDA context; parallelism via NVDEC pipe already
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(detect_one, str(v), cfg): v for v in vids}
        for fut in as_completed(futs):
            v = futs[fut]
            try:
                r = fut.result()
                d = r["detection"]
                LOG.info("%s: %d/%d checked frames have faces, %d intervals, %.1fs of %.1fs (%.0f source fps)",
                         v.name, d["frames_with_faces"], d["frames_checked"], len(d["intervals"]),
                         d["face_time_s"], r["input"]["duration_s"], r["timing"]["source_fps"])
            except Exception as e:  # noqa: BLE001
                LOG.error("%s FAILED: %s", v.name, e)
                LOG.debug(traceback.format_exc())
                r = {"input": {"path": str(v)}, "status": "error", "error": f"{type(e).__name__}: {e}"}
            results[str(v)] = r
    manifest["videos"] = [results[k] for k in sorted(results)]
    ok = [r for r in manifest["videos"] if r["status"] == "ok"]
    manifest["summary"] = {
        "videos": len(manifest["videos"]), "ok": len(ok), "error": len(manifest["videos"]) - len(ok),
        "videos_with_faces": sum(1 for r in ok if r["detection"]["intervals"]),
        "total_face_time_s": round(sum(r["detection"]["face_time_s"] for r in ok), 2),
        "total_video_time_s": round(sum(r["input"]["duration_s"] for r in ok), 2),
        "wall_s": round(time.time() - t0, 1),
        "aggregate_source_fps": round(sum(r["input"]["nb_frames"] for r in ok) / max(time.time() - t0, 1e-6), 1),
    }
    out = args.output_dir / "faces.json"
    rv.atomic_write_json(out, manifest)
    (args.output_dir / "faces.json.sha256").write_text(f"{rv.sha256_file(out)}  faces.json\n")
    LOG.info("done: %s  (%s)", out, json.dumps(manifest["summary"]))
    return 1 if manifest["summary"]["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
