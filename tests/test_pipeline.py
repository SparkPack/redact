"""Pipeline tests that do not need the gated SAM 3 / EgoBlur weights.

Run from anywhere:  python redact/tests/test_pipeline.py [--sam2] [--clip PATH]
  --sam2  also drives the ultralytics frame-feeding shim with SAM 2.1 tiny (auto-downloaded).
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import redact_videos as rv  # noqa: E402


def make_clip(path: Path, tools: rv.FFTools, seconds: int = 3) -> None:
    subprocess.run([
        tools.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"testsrc2=size=1920x1080:rate=30:duration={seconds}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
        "-vf", "drawbox=x=1200:y=600:w=400:h=300:color=0x8B5A2B:t=fill",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
    ], check=True)


def test_smoother_max():
    sm = rv.TemporalSmoother(5, "max")
    masks = [np.zeros((4, 4), np.uint8) for _ in range(10)]
    masks[5][1, 1] = 1
    out = []
    for i, m in enumerate(masks):
        out += sm.push(i, np.full((4, 4, 3), i, np.uint8), m)
    out += sm.flush()
    assert [o[0] for o in out] == list(range(10))
    assert all(o[1][0, 0, 0] == o[0] for o in out), "frame/mask misaligned"
    assert [o[0] for o in out if o[2][1, 1]] == [3, 4, 5, 6, 7]
    print("smoother max OK")


def test_smoother_majority():
    sm = rv.TemporalSmoother(3, "majority")
    masks = [np.zeros((2, 2), np.uint8) for _ in range(6)]
    masks[2][0, 0] = 1  # one-frame blip -> suppressed
    for i in (3, 4, 5):
        masks[i][1, 1] = 1  # persistent -> kept
    out = []
    for i, m in enumerate(masks):
        out += sm.push(i, np.zeros((2, 2, 3), np.uint8), m)
    out += sm.flush()
    assert not any(o[2][0, 0] for o in out)
    assert [o[0] for o in out if o[2][1, 1]] == [3, 4, 5]
    print("smoother majority OK")


class _FakeTracker:
    """Duck-types Sam3Tracker: masks the static box region on every frame."""
    prompts = ["box"]
    version = "fake"

    def union(self, r, h, w):
        return rv.Sam3Tracker.union_mask(r, h, w)

    def run(self, source):
        import torch
        for _, imgs, _ in source:
            img = imgs[0]
            m = torch.zeros((1,) + img.shape[:2], dtype=torch.bool)
            m[0, 600:900, 1200:1600] = True
            yield types.SimpleNamespace(orig_img=img, masks=types.SimpleNamespace(data=m),
                                        boxes=types.SimpleNamespace(cls=torch.tensor([0])))


def test_fill_encode(tools: rv.FFTools, clip: Path, tmp: Path):
    args = rv.build_parser().parse_args([str(clip.parent), str(tmp), "--no-ocr", "--dilate-px", "10", "--smooth-window", "5"])
    info = rv.ffprobe_video(tools, clip)
    out = tmp / "filled.mp4"
    rep = rv.process_video(args, tools, info, out, _FakeTracker(), None, None, 0, "h264_nvenc")
    assert rep.status == "ok" and rep.redaction["frames"] == info.nb_frames == rep.output["nb_frames"], rep
    assert rep.redaction["frames_redacted"] == info.nb_frames
    assert rep.redaction["per_prompt_frames"] == {"box": info.nb_frames}
    cap = cv2.VideoCapture(str(out))
    n = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        n += 1
        inside = f[620:880, 1220:1580].astype(int)  # region minus dilation margin
        assert np.abs(inside - 128).max() <= 6, f"frame {n}: fill not grey, max dev {np.abs(inside - 128).max()}"
        ring = f[560:580, 1200:1600].astype(int)  # 20px above the dilated box (600-10=590)
        assert np.abs(ring - 128).mean() > 10, f"frame {n}: outside region looks grey too"
    assert n == info.nb_frames, (n, info.nb_frames)
    # audio must be gone
    streams = subprocess.run([tools.ffprobe, "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(out)],
                             capture_output=True, text=True).stdout.split()
    assert streams == ["video"], streams
    print(f"fill/encode OK: {n} frames, sha256 {rep.output['sha256'][:12]}..., {rep.timing['redact_fps']} fps")


def test_sam2_shim(tools: rv.FFTools, clip: Path):
    """Drive ultralytics' SAM2VideoPredictor through PipeFrameSource (same dataset contract SAM 3 uses)."""
    from ultralytics.models.sam import SAM2VideoPredictor

    rv._install_ultralytics_source_shim()
    info = rv.ffprobe_video(tools, clip)
    reader = rv.FrameReader(tools, clip, info.display_width, info.display_height, tools.hwaccel_cuda, 0)
    src = rv.PipeFrameSource(reader, clip, info.nb_frames, info.fps_float, max_frames=20)
    pred = SAM2VideoPredictor(overrides=dict(model="sam2.1_t.pt", task="segment", mode="predict", imgsz=1024,
                                             conf=0.25, verbose=False, save=False, show=False, batch=1, device="0"))
    n, covers = 0, []
    for r in pred(source=src, bboxes=[[1200, 600, 1600, 900]], stream=True):
        union, cls_ids = rv.Sam3Tracker.union_mask(r, info.height, info.width)
        assert union.shape == (info.height, info.width) and union.dtype == bool
        assert r.orig_img.shape == (info.height, info.width, 3)
        covers.append(union[600:900, 1200:1600].mean())
        n += 1
    reader.close()
    assert n == 20, n
    assert src.frame == 20
    assert min(covers) > 0.8, f"box coverage per frame: {[round(c, 2) for c in covers]}"
    print(f"SAM2 shim OK: {n} frames, min box coverage {min(covers):.2f}")


def test_rotated_clip(tools: rv.FFTools, clip: Path, tmp: Path):
    """Portrait phone footage: rotation tag must swap the frame dimensions or rows scramble."""
    rot = tmp / "rot" / "rotated90.mp4"
    rot.parent.mkdir(parents=True)
    subprocess.run([tools.ffmpeg, "-v", "error", "-y", "-display_rotation", "90", "-i", str(clip), "-c", "copy",
                    "-map", "0:v:0", str(rot)], check=True)
    info = rv.ffprobe_video(tools, rot)
    assert info.rotation == 90 and (info.display_width, info.display_height) == (1080, 1920), info
    args = rv.build_parser().parse_args([str(rot.parent), str(tmp / "rot_out"), "--no-ocr"])
    out = tmp / "rot_out" / "rotated90.mp4"
    rep = rv.process_video(args, tools, info, out, None, None, None, 0, "h264_nvenc")
    assert (rep.output["width"], rep.output["height"]) == (1080, 1920), rep.output
    # output must match ffmpeg's own autorotated decode of the source: PSNR well above scramble level
    r = subprocess.run([tools.ffmpeg, "-hide_banner", "-i", str(rot), "-i", str(out), "-lavfi", "[0:v][1:v]psnr",
                        "-frames:v", "30", "-f", "null", "-"], capture_output=True, text=True).stderr
    m = re.search(r"average:([\d.]+)", r)
    assert m and float(m.group(1)) > 35, f"PSNR vs autorotated source too low: {r[-400:]}"
    print(f"rotated clip OK: {rep.output['width']}x{rep.output['height']}, PSNR {m.group(1)} dB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam2", action="store_true")
    ap.add_argument("--clip", type=Path, default=None)
    a = ap.parse_args()
    tools = rv.FFTools.discover(None, None)
    test_smoother_max()
    test_smoother_majority()
    tmp = Path(tempfile.mkdtemp(prefix="redact_test_"))
    try:
        clip = a.clip or tmp / "src" / "clip.mp4"
        if a.clip is None:
            clip.parent.mkdir()
            make_clip(clip, tools)
        test_fill_encode(tools, clip, tmp / "out")
        test_rotated_clip(tools, clip, tmp)
        if a.sam2:
            test_sam2_shim(tools, clip)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ALL OK")


if __name__ == "__main__":
    main()
