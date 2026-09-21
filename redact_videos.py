"""Shared library for the MCAP face-redaction tools.

ffmpeg process wrappers, the EgoBlur TorchScript detector, mask growth and fill, temporal smoothing,
and small MCAP helpers. There is no command line here; see redact_mcap.py.

"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

SCRIPT_VERSION = "1.0.0"
DEFAULT_PROMPTS = ["chocolate bar", "wrapper", "cardboard box", "label"]
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv"}
LOG = logging.getLogger("redact")


# --------------------------------------------------------------------------- utils
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_device(dev: str) -> tuple[str, int | None]:
    """Return (torch_device_string, gpu_index_or_None)."""
    d = dev.strip().lower()
    if d == "cpu":
        return "cpu", None
    if d.startswith("cuda:"):
        d = d[5:]
    if d.isdigit():
        return f"cuda:{d}", int(d)
    raise argparse.ArgumentTypeError(f"bad --device {dev!r}; use cpu, N or cuda:N")


def atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- ffmpeg
@dataclass
class FFTools:
    ffmpeg: str
    ffprobe: str
    version: str = ""
    hwaccel_cuda: bool = False
    encoders: set = field(default_factory=set)

    @classmethod
    def discover(cls, ffmpeg: str | None, ffprobe: str | None) -> "FFTools":
        # prefer the static NVENC build installed next to the interpreter (do NOT resolve() the symlinked
        # venv python, that lands in uv's python store); fall back to $PATH
        cands = [Path(sys.executable).parent, Path(sys.prefix) / "bin"]
        ff = ffmpeg or next((str(d / "ffmpeg") for d in cands if (d / "ffmpeg").exists()), None) or shutil.which("ffmpeg")
        fp = ffprobe or next((str(d / "ffprobe") for d in cands if (d / "ffprobe").exists()), None) or shutil.which("ffprobe")
        if not ff or not fp:
            sys.exit("ffmpeg/ffprobe not found; install the NVENC-capable static build into the venv bin")
        t = cls(ffmpeg=ff, ffprobe=fp)
        t.version = subprocess.run([ff, "-version"], capture_output=True, text=True).stdout.splitlines()[0]
        hw = subprocess.run([ff, "-hide_banner", "-hwaccels"], capture_output=True, text=True).stdout
        t.hwaccel_cuda = "cuda" in hw.split()
        enc = subprocess.run([ff, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        t.encoders = {ln.split()[1] for ln in enc.splitlines() if ln.startswith(" V") and len(ln.split()) > 1}
        return t


@dataclass
class VideoInfo:
    path: str
    width: int   # coded size as stored in the file
    height: int
    display_width: int   # size after ffmpeg's autorotate (what the decoder pipe delivers)
    display_height: int
    rotation: int  # display-matrix rotation normalised to 0/90/180/270
    fps: str  # rational string, e.g. "30000/1001"
    fps_float: float
    nb_frames: int
    duration_s: float
    codec: str
    pix_fmt: str
    bytes: int


def ffprobe_video(tools: FFTools, path: Path) -> VideoInfo:
    cmd = [
        tools.ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,pix_fmt"
                         ":stream_side_data=rotation:stream_tags=rotate:format=duration",
        "-of", "json", str(path),
    ]
    info = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    if not info.get("streams"):
        raise RuntimeError(f"no video stream in {path}")
    s = info["streams"][0]
    # Phone footage carries a display-matrix rotation; ffmpeg autorotates on decode, so the pipe delivers
    # display-oriented frames (portrait clips come out H x W). Honour it or every row is misaligned.
    rot = 0.0
    for sd in s.get("side_data_list", []) or []:
        if sd.get("rotation") is not None:
            rot = float(sd["rotation"])
    if not rot and (s.get("tags") or {}).get("rotate"):
        rot = float(s["tags"]["rotate"])
    rotation = int(round(rot / 90.0)) * 90 % 360
    cw, ch = int(s["width"]), int(s["height"])
    dw, dh = (ch, cw) if rotation in (90, 270) else (cw, ch)
    fps_str = s.get("avg_frame_rate") or s.get("r_frame_rate") or "30/1"
    if fps_str in ("0/0", "0"):
        fps_str = s.get("r_frame_rate") or "30/1"
    fps = float(Fraction(fps_str))
    dur = s.get("duration") or info.get("format", {}).get("duration")
    dur = float(dur) if dur not in (None, "N/A") else 0.0
    nb = s.get("nb_frames")
    if nb in (None, "N/A"):
        cnt = subprocess.run(
            [tools.ffprobe, "-v", "error", "-select_streams", "v:0", "-count_packets",
             "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True,
        ).stdout.strip()
        nb = int(cnt) if cnt.isdigit() else int(round(dur * fps))
    return VideoInfo(
        path=str(path), width=cw, height=ch, display_width=dw, display_height=dh, rotation=rotation,
        fps=fps_str, fps_float=fps,
        nb_frames=int(nb), duration_s=dur, codec=s.get("codec_name", ""), pix_fmt=s.get("pix_fmt", ""),
        bytes=path.stat().st_size,
    )


class FrameReader:
    """ffmpeg -> raw bgr24 frames over a pipe (NVDEC via -hwaccel cuda when enabled)."""

    def __init__(self, tools: FFTools, path: Path, width: int, height: int, use_cuda: bool,
                 gpu_index: int | None, select_stride: int = 1):
        self.w, self.h = width, height
        self.nbytes = width * height * 3
        cmd = [tools.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
        if use_cuda:
            cmd += ["-hwaccel", "cuda"]
            if gpu_index is not None:
                cmd += ["-hwaccel_device", str(gpu_index)]
        cmd += ["-i", str(path), "-an", "-sn", "-dn", "-map", "0:v:0"]
        if select_stride > 1:
            cmd += ["-vf", f"select=not(mod(n\\,{select_stride}))", "-fps_mode", "passthrough"]
        cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
        self.cmd = cmd
        self.err = tempfile.NamedTemporaryFile(prefix="ffdec_", suffix=".log", delete=False)
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=self.err, bufsize=self.nbytes * 4)
        self.frames_read = 0

    def read(self) -> np.ndarray | None:
        buf = bytearray(self.nbytes)
        view = memoryview(buf)
        got = 0
        while got < self.nbytes:
            n = self.proc.stdout.readinto(view[got:])
            if not n:
                break
            got += n
        if got == 0:
            return None
        if got < self.nbytes:
            LOG.warning("decoder: truncated final frame (%d/%d bytes) discarded", got, self.nbytes)
            return None
        self.frames_read += 1
        return np.frombuffer(buf, dtype=np.uint8).reshape(self.h, self.w, 3)

    def close(self) -> str:
        try:
            self.proc.stdout.close()
        except Exception:
            pass
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
        self.err.close()
        msg = Path(self.err.name).read_text(errors="replace").strip()
        os.unlink(self.err.name)
        return msg


class FrameWriter:
    """raw bgr24 frames -> ffmpeg encoder (NVENC) in a background thread; no audio."""

    def __init__(self, tools: FFTools, out_path: Path, width: int, height: int, fps: str, codec: str,
                 preset: str, cq: int, gpu_index: int | None, queue_size: int = 24):
        cmd = [tools.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", fps, "-i", "pipe:0",
               "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", codec]
        if codec.endswith("_nvenc"):
            cmd += ["-preset", preset, "-tune", "hq", "-rc", "vbr", "-cq", str(cq), "-b:v", "0", "-pix_fmt", "yuv420p"]
            if gpu_index is not None:
                cmd += ["-gpu", str(gpu_index)]
        elif codec in ("libx264", "libx265"):
            cmd += ["-preset", "medium", "-crf", str(cq), "-pix_fmt", "yuv420p"]
        if codec.startswith("hevc") or codec == "libx265":
            cmd += ["-tag:v", "hvc1"]
        cmd += ["-movflags", "+faststart", str(out_path)]
        self.cmd = cmd
        self.err = tempfile.NamedTemporaryFile(prefix="ffenc_", suffix=".log", delete=False)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self.err, bufsize=width * height * 3 * 4)
        self.q: queue.Queue = queue.Queue(maxsize=queue_size)
        self.exc: BaseException | None = None
        self.frames_written = 0
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        try:
            while True:
                item = self.q.get()
                if item is None:
                    break
                self.proc.stdin.write(item)
                self.frames_written += 1
        except BaseException as e:  # noqa: BLE001
            self.exc = e
        finally:
            try:
                self.proc.stdin.close()
            except Exception:
                pass

    def _stderr(self) -> str:
        try:
            self.err.flush()
            return Path(self.err.name).read_text(errors="replace").strip()
        except Exception:
            return ""

    def write(self, frame: np.ndarray) -> None:
        if self.exc:
            rc = self.proc.poll()
            raise RuntimeError(f"ffmpeg encoder failed (exit {rc}): {self._stderr() or self.exc!r}")
        self.q.put(np.ascontiguousarray(frame).tobytes())

    def abort(self) -> None:
        """Kill the encoder without waiting (error path)."""
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
        self.t.join(timeout=5)
        self.err.close()
        try:
            os.unlink(self.err.name)
        except OSError:
            pass

    def close(self) -> str:
        self.q.put(None)
        self.t.join()
        rc = self.proc.wait()
        msg = self._stderr()
        self.err.close()
        os.unlink(self.err.name)
        if rc != 0 or self.exc:
            raise RuntimeError(f"ffmpeg encoder exited {rc}: {msg or self.exc!r}")
        return msg


# --------------------------------------------------------------------------- SAM 3
def _build_gen2_unwrapper(model):
    """Wrap a detectron2/d2go scripted detector so only plain tensors cross into Python.

    `.inference()` returns detectron2's scripted `Instances` class, which pybind cannot convert to a
    Python object ("ScriptedInstances1 is not found"). Scripting this wrapper keeps the field access
    inside TorchScript and hands back (boxes, scores). The class is defined here, in a real module
    file, because torch.jit.script needs to read its source.
    """
    import torch
    from typing import Tuple

    class _Gen2Unwrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            out = self.m.inference([{"image": image}], None, False)
            inst = out[0]
            # The public `pred_boxes` / `scores` names exist only as Python properties, which do not
            # survive the export; the scripted class stores Optional fields under these names instead.
            boxes = inst._pred_boxes
            scores = inst._scores
            if boxes is None or scores is None:
                empty = torch.zeros((0, 4), dtype=torch.float32, device=image.device)
                return empty, torch.zeros((0,), dtype=torch.float32, device=image.device)
            return boxes.tensor, scores

    return torch.jit.script(_Gen2Unwrapper(model))


def _build_gen2_batch_unwrapper(model):
    """Batched sibling of `_build_gen2_unwrapper`: N images in, N (boxes, scores) pairs out.

    detectron2's GeneralizedRCNN pads a batch to a common size internally, so the images may differ in
    shape; ours do not. One call for all channels keeps the GPU far better fed than one call per frame.
    """
    import torch
    from typing import Dict, List, Tuple

    class _Gen2BatchUnwrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, images: List[torch.Tensor]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
            inputs: List[Dict[str, torch.Tensor]] = []
            for im in images:
                inputs.append({"image": im})
            out = self.m.inference(inputs, None, False)
            res: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for inst in out:
                boxes = inst._pred_boxes
                scores = inst._scores
                if boxes is None or scores is None:
                    dev = images[0].device
                    res.append((torch.zeros((0, 4), dtype=torch.float32, device=dev),
                                torch.zeros((0,), dtype=torch.float32, device=dev)))
                else:
                    res.append((boxes.tensor, scores))
            return res

    return torch.jit.script(_Gen2BatchUnwrapper(model))


class FaceDetector:
    """EgoBlur face detector (TorchScript). gen1: native-res BGR uint8 CHW -> (boxes, labels, scores, dims).
    gen2: short-edge-1200 resize; scripted (`.inference`) or traced module."""

    GEN2_SHORT, GEN2_MAX = 1200, 1200

    def __init__(self, weights: Path, device: str, gen: str, conf: float, nms_iou: float, box_scale: float,
                 half: bool = False):
        import torch
        import torchvision  # noqa: F401  (registers ops used by the scripted model)

        self.torch = torch
        self.device = torch.device(device)
        self.model = torch.jit.load(str(weights), map_location="cpu").to(self.device)
        self.half = bool(half) and self.device.type == "cuda"
        if self.half:
            self.model.half()
        try:
            self.model.eval()
        except RuntimeError as e:
            # d2go/detectron2 exports bake `training` in as a constant, so .eval() raises.
            # Such modules are already exported in inference mode.
            if "constant 'training'" not in str(e):
                raise
            LOG.debug("%s: .eval() not applicable to this scripted module (%s)", weights.name, e)
        if gen == "auto":
            gen = "2" if "gen2" in weights.name.lower() else "1"
        self.gen = int(gen)
        self.scripted = hasattr(self.model, "inference")
        self.unwrapped = _build_gen2_unwrapper(self.model) if self.scripted else None
        self.batched = _build_gen2_batch_unwrapper(self.model) if self.scripted else None
        self.conf, self.nms_iou, self.box_scale = conf, nms_iou, box_scale
        self.weights = weights

    def _gen2_size(self, h: int, w: int) -> tuple[int, int]:
        scale = self.GEN2_SHORT / min(h, w)
        nh, nw = h * scale, w * scale
        if max(nh, nw) > self.GEN2_MAX:
            s = self.GEN2_MAX / max(nh, nw)
            nh, nw = nh * s, nw * s
        return int(nh + 0.5), int(nw + 0.5)

    def _to_input(self, bgr: np.ndarray):
        """BGR HWC uint8 -> (CHW model input tensor, x-scale, y-scale back to original pixels)."""
        torch = self.torch
        h, w = bgr.shape[:2]
        t = torch.from_numpy(np.ascontiguousarray(bgr.transpose(2, 0, 1))).to(self.device)
        sx = sy = 1.0
        if self.gen == 2:
            nh, nw = self._gen2_size(h, w)
            if (nh, nw) != (h, w):
                t = torch.nn.functional.interpolate(t[None].float(), size=(nh, nw), mode="bilinear",
                                                    align_corners=False)[0].round().clamp(0, 255).to(torch.uint8)
                sx, sy = w / nw, h / nh
        return (t.half() if self.half else t), sx, sy

    def _finish(self, boxes, scores, sx: float, sy: float, w: int, h: int) -> np.ndarray:
        """NMS, threshold, rescale to original pixels, expand about centre. Returns Nx5."""
        tv = __import__("torchvision")
        boxes, scores = boxes.float(), scores.float()
        if boxes.numel() == 0:
            return np.zeros((0, 5), np.float32)
        keep = tv.ops.nms(boxes, scores, self.nms_iou)
        boxes, scores = boxes[keep], scores[keep]
        keep = scores > self.conf
        boxes, scores = boxes[keep].cpu().numpy(), scores[keep].cpu().numpy()
        if len(boxes) == 0:
            return np.zeros((0, 5), np.float32)
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy
        cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
        bw, bh = (boxes[:, 2] - boxes[:, 0]) * self.box_scale, (boxes[:, 3] - boxes[:, 1]) * self.box_scale
        out = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
        out[:, [0, 2]] = out[:, [0, 2]].clip(0, w)
        out[:, [1, 3]] = out[:, [1, 3]].clip(0, h)
        return np.concatenate([out, scores[:, None]], 1).astype(np.float32)

    def detect_batch(self, frames: list) -> list:
        """Run one inference call over several frames. Returns a list of Nx5 arrays, one per frame."""
        if not frames:
            return []
        if not self.scripted:
            return [self.detect(f) for f in frames]  # traced gen1 exports are single-image only
        torch = self.torch
        with torch.no_grad():
            prepared = [self._to_input(f) for f in frames]
            pairs = self.batched([t for t, _, _ in prepared])
        return [self._finish(b, sc, sx, sy, f.shape[1], f.shape[0])
                for (b, sc), (_, sx, sy), f in zip(pairs, prepared, frames)]

    def detect(self, bgr: np.ndarray) -> np.ndarray:
        """Return Nx5 float32 [x1,y1,x2,y2,score] in original pixel coords (expanded by box_scale)."""
        torch = self.torch
        h, w = bgr.shape[:2]
        with torch.no_grad():
            t, sx, sy = self._to_input(bgr)
            if self.scripted:
                boxes, scores = self.unwrapped(t)
            else:
                out = self.model(t)
                if len(out) == 4:
                    boxes, _, scores, _ = out
                elif len(out) == 5:
                    boxes, _, _, scores, _ = out
                else:
                    raise RuntimeError(f"unexpected EgoBlur output arity {len(out)}")
        return self._finish(boxes, scores, sx, sy, w, h)


class YuNetFaceDetector:
    """Fallback face detector: OpenCV's bundled YuNet (face_detection_yunet_2023mar.onnx, CPU).

    Stand-in while the EgoBlur weights are pending. Runs on a downscaled copy for speed; boxes are
    returned in full-resolution xyxy, expanded by box_scale like the EgoBlur path.
    """

    URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

    def __init__(self, weights: Path, conf: float, nms_iou: float, box_scale: float, scale: float = 0.5):
        if not weights.is_file():
            import urllib.request
            weights.parent.mkdir(parents=True, exist_ok=True)
            LOG.info("downloading YuNet model to %s", weights)
            urllib.request.urlretrieve(self.URL, weights)
        self.weights = weights
        self.conf, self.nms_iou, self.box_scale, self.scale = conf, nms_iou, box_scale, scale
        self.det = cv2.FaceDetectorYN.create(str(weights), "", (320, 320), score_threshold=conf,
                                             nms_threshold=nms_iou, top_k=200)
        self.size = None
        self.gen, self.scripted = 0, False

    def detect(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        sw, sh = int(round(w * self.scale)), int(round(h * self.scale))
        small = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA) if self.scale != 1.0 else bgr
        if self.size != (sw, sh):
            self.det.setInputSize((sw, sh))
            self.size = (sw, sh)
        _, faces = self.det.detect(small)
        if faces is None or len(faces) == 0:
            return np.zeros((0, 5), np.float32)
        scores = faces[:, 14].astype(np.float32)
        f = faces[:, :4].astype(np.float32) / self.scale  # x, y, w, h at full res
        boxes = np.stack([f[:, 0], f[:, 1], f[:, 0] + f[:, 2], f[:, 1] + f[:, 3]], 1)
        cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
        bw, bh = (boxes[:, 2] - boxes[:, 0]) * self.box_scale, (boxes[:, 3] - boxes[:, 1]) * self.box_scale
        out = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
        out[:, [0, 2]] = out[:, [0, 2]].clip(0, w)
        out[:, [1, 3]] = out[:, [1, 3]].clip(0, h)
        return np.concatenate([out, scores[:, None]], 1).astype(np.float32)


def reject_coloured_boxes(frame: np.ndarray, boxes: np.ndarray, max_blue_frac: float = 0.35) -> np.ndarray:
    """Drop detections whose interior is mostly blue: nitrile gloves, not faces.

    EgoBlur scores a blue-gloved hand held near a wrist camera as high as 0.92, overlapping real faces,
    so no score threshold separates them. Colour does. Measured on IC-559 footage over 39 glove boxes
    and 40 verified face boxes: gloves have median hue 109 (blue) and 98% blue pixels; faces have median
    hue 8 (skin) and 0%. Rejecting above 35% blue removes two thirds of glove detections and none of the
    faces. Saturated pixels only, so grey and washed-out areas do not vote.
    """
    if not len(boxes):
        return boxes
    keep = []
    h, w = frame.shape[:2]
    for b in boxes:
        x1, y1, x2, y2 = (max(0, int(b[0])), max(0, int(b[1])), min(w, int(b[2])), min(h, int(b[3])))
        crop = frame[y1:y2, x1:x2]
        if crop.size < 300:
            keep.append(True)
            continue
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hue, sat = hsv[..., 0].ravel(), hsv[..., 1].ravel()
        m = sat > 60
        if m.sum() < 50:                       # too washed out to judge: keep it
            keep.append(True)
            continue
        blue = float(((hue[m] >= 95) & (hue[m] <= 135)).mean())
        keep.append(blue <= max_blue_frac)
    return boxes[np.array(keep, dtype=bool)]


def paint_faces(mask: np.ndarray, boxes: np.ndarray, shape: str) -> None:
    """boxes: Nx4 or Nx5; a trailing score column is ignored."""
    for x1, y1, x2, y2 in boxes[:, :4].astype(int):
        if x2 <= x1 or y2 <= y1:
            continue
        if shape == "ellipse":
            cv2.ellipse(mask, ((x1 + x2) // 2, (y1 + y2) // 2), ((x2 - x1) // 2, (y2 - y1) // 2), 0, 0, 360, 1, -1)
        else:
            cv2.rectangle(mask, (x1, y1), (x2 - 1, y2 - 1), 1, -1)


# --------------------------------------------------------------------------- mask post
def dilate_fast(mask_u8: np.ndarray, radius: int) -> np.ndarray:
    """Elliptical dilation (radius > 0) or erosion (radius < 0) by |radius| px.

    Erosion keeps the object's outline sharp and redacts only its interior (labels, print).
    For |radius| >= 8 the morphology runs on a 1/4-resolution mask and is upsampled nearest
    (~1 ms instead of ~23 ms at 1080p). Downscaling is conservative in the direction of the
    operation: any coverage counts when dilating, full coverage is required when eroding.
    """
    if radius == 0 or not mask_u8.any():
        return mask_u8
    h, w = mask_u8.shape
    op = cv2.dilate if radius > 0 else cv2.erode
    r = abs(radius)
    if r < 8:
        k = 2 * r + 1
        return op(mask_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    ds = 4
    small = cv2.resize(mask_u8 * 255, (w // ds, h // ds), interpolation=cv2.INTER_AREA)
    small = ((small > 0) if radius > 0 else (small >= 255)).view(np.uint8)
    k = 2 * int(np.ceil(r / ds)) + 1
    small = op(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def tint_from_seed(seed: str) -> tuple[int, int, int]:
    """Deterministic random BGR tint (full saturation, mid-high value) from a string seed."""
    hue = int(hashlib.sha256(seed.encode()).hexdigest()[:4], 16) % 180  # OpenCV hue range 0-179
    hsv = np.array([[[hue, 200, 230]]], dtype=np.uint8)
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
    return int(b), int(g), int(r)


def apply_fill(frame: np.ndarray, mask_u8: np.ndarray, mode: str, grey: int, block: int,
               tint: tuple[int, int, int] | None = None) -> np.ndarray:
    """Redact `frame` where mask_u8 != 0. Returns a new frame.

    grey:     flat fill (irreversible; the only mode that destroys text with certainty)
    blur:     strong blur = downscale by `block`, small gaussian, upscale bilinear
    pixelate: downscale by `block`, upscale nearest (mosaic)
    anon:     blur, then discard the object's colours: low-frequency shading (folds, highlights) is kept
              as luminance and re-tinted with a per-video random colour. Geometry and shading survive for
              training; print, artwork layout and brand colours do not. Doubles as colour randomisation.
    cv2.copyTo with a mask is ~10x faster than boolean fancy indexing on 1080p frames.
    """
    h, w = frame.shape[:2]
    if mode == "grey":
        src = np.full_like(frame, grey)
    else:
        sw, sh = max(1, w // block), max(1, h // block)
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        if mode == "pixelate":
            src = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            small = cv2.GaussianBlur(small, (5, 5), 0)
            if mode == "anon":
                lum = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                lum = cv2.normalize(lum, None, 90, 235, cv2.NORM_MINMAX)  # keep shading, drop absolute brightness cue
                t = np.array(tint if tint is not None else (128, 128, 128), dtype=np.float32) / 255.0
                small = np.clip(lum[..., None].astype(np.float32) * t[None, None, :], 0, 255).astype(np.uint8)
            src = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    out = frame.copy()
    cv2.copyTo(src, mask_u8, out)
    return out


class TemporalSmoother:
    """Windowed temporal filter over per-frame masks.

    mode="max": a pixel is redacted if masked in ANY frame within +-k (hold; never shrinks coverage).
    mode="majority": redacted if masked in more than half of the available frames within +-k.
    Frames are emitted with a delay of k frames; call flush() at end of stream.
    """

    def __init__(self, window: int, mode: str):
        if window < 1 or window % 2 == 0:
            raise ValueError("--smooth-window must be an odd integer >= 1")
        self.k = window // 2
        self.mode = mode
        self.masks: deque[tuple[int, np.ndarray]] = deque()
        self.frames: deque[tuple[int, np.ndarray]] = deque()

    def push(self, idx: int, frame: np.ndarray, mask: np.ndarray) -> list[tuple[int, np.ndarray, np.ndarray]]:
        self.frames.append((idx, frame))
        self.masks.append((idx, mask))
        return self._drain(final=False)

    def flush(self) -> list[tuple[int, np.ndarray, np.ndarray]]:
        return self._drain(final=True)

    def _drain(self, final: bool):
        out = []
        while self.frames:
            i, frame = self.frames[0]
            if not final and self.masks[-1][0] < i + self.k:
                break
            sel = [m for (j, m) in self.masks if i - self.k <= j <= i + self.k]
            if self.k == 0 or len(sel) == 1:
                sm = sel[0].astype(bool)
            elif self.mode == "max":
                sm = np.logical_or.reduce(sel)
            else:
                sm = np.sum(sel, axis=0, dtype=np.int16) > (len(sel) // 2)
            out.append((i, frame, sm))
            self.frames.popleft()
            while self.masks and self.masks[0][0] < i + 1 - self.k:
                self.masks.popleft()
        return out


# --------------------------------------------------------------------------- OCR


# --------------------------------------------------------------------------- mcap helpers
def is_keyframe(pkt: bytes) -> bool:
    """True if this h264 access unit contains an IDR slice (NAL type 5).

    A clip must begin on one or it will not decode, so cut boundaries are snapped to these.
    """
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


class McapCopy:
    """Incremental MCAP writer that mirrors schemas and channels from a source file on demand."""

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
