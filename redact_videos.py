#!/usr/bin/env python
"""Batch visual redaction for MP4 folders.

Per video:
  1. Decode with ffmpeg (NVDEC when available) to raw BGR frames.
  2. SAM 3 text-prompted video tracking (SAM3VideoSemanticPredictor) for the
     configured prompts, plus EgoBlur face detection (TorchScript, gen1 or gen2).
  3. Union of all masks -> spatial dilation -> temporal smoothing.
  4. Flat-fill the smoothed mask with a grey value.
  5. Encode with NVENC (h264/hevc), no audio, to <output_dir>/<stem>.mp4.
  6. Run PaddleOCR on sampled frames of the *output* to flag remaining readable text.
  7. Append an entry (SHA-256 of input and output, stats, OCR flags) to manifest.json.

The SAM 3 weights (sam3.pt) and EgoBlur weights (*.jit) are gated downloads and
must be supplied with --sam3-weights / --egoblur-weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Iterator

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
class PipeFrameSource:
    """Duck-types ultralytics' LoadImagesAndVideos for one video fed from a FrameReader.

    SAM3VideoSemanticPredictor needs `.mode == "video"`, `.frames` (used to size per-frame
    prompt lists) and `.frame` (1-based current index) from the dataset object.
    """

    def __init__(self, reader: FrameReader, path: Path, est_frames: int, fps: float, max_frames: int | None):
        from ultralytics.data.loaders import SourceTypes

        self.reader = reader
        self.path = str(path)
        self.source_type = SourceTypes(stream=False, screenshot=False, from_img=False, tensor=False)
        self.mode = "video"
        self.bs = 1
        self.fps = fps
        self.video_flag = [True]
        # generous margin: ffprobe frame counts can undershoot the decoded count
        self.frames = est_frames + max(64, est_frames // 10)
        self.limit = min(self.frames, max_frames) if max_frames else self.frames
        self.frame = 0  # 1-based index of the frame most recently returned
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.frame >= self.limit:
            raise StopIteration
        f = self.reader.read()
        if f is None:
            raise StopIteration
        self.frame += 1
        self.count += 1
        return [self.path], [f], [""]

    def __len__(self):
        return 1


def _install_ultralytics_source_shim() -> None:
    """Let BasePredictor.setup_source accept a PipeFrameSource unchanged."""
    import ultralytics.engine.predictor as pred_mod

    if getattr(pred_mod, "_redact_shim_installed", False):
        return
    orig = pred_mod.load_inference_source

    def _load(source=None, **kw):
        if isinstance(source, PipeFrameSource):
            return source
        return orig(source=source, **kw)

    pred_mod.load_inference_source = _load
    pred_mod._redact_shim_installed = True


class Sam3Tracker:
    def __init__(self, weights: Path, prompts: list[str], conf: float, imgsz: int, half: bool, device: str):
        import torch  # noqa: F401
        from ultralytics.models.sam import SAM3VideoSemanticPredictor

        _install_ultralytics_source_shim()
        self.prompts = list(prompts)
        self.overrides = dict(
            model=str(weights), task="segment", mode="predict", imgsz=imgsz, conf=conf,
            quantize=16 if half else None, device=device.replace("cuda:", ""), verbose=False, save=False,
            show=False, batch=1,
        )
        self.predictor = SAM3VideoSemanticPredictor(overrides=self.overrides)
        self.predictor.setup_model(model=None)  # load now so weight problems fail fast
        self.version = __import__("ultralytics").__version__

    def reset(self) -> None:
        # SAM3VideoSemanticPredictor.init_state() is a no-op when inference_state is non-empty,
        # so clear it before every video to start fresh masklets.
        self.predictor.inference_state = {}
        if hasattr(self.predictor, "tracker"):
            self.predictor.tracker.inference_state = {}

    def run(self, source: PipeFrameSource) -> Iterator:
        self.reset()
        return self.predictor(source=source, text=self.prompts, stream=True)

    def union(self, result, height: int, width: int) -> tuple[np.ndarray, list[int]]:
        return self.union_mask(result, height, width)

    @staticmethod
    def union_mask(result, height: int, width: int) -> tuple[np.ndarray, list[int]]:
        """Return (bool HxW union of tracked masks, list of class ids present)."""
        masks = getattr(result, "masks", None)
        if masks is None or masks.data.shape[0] == 0:
            return np.zeros((height, width), dtype=bool), []
        m = masks.data
        union = m.any(dim=0) if m.dtype == __import__("torch").bool else (m > 0.5).any(dim=0)
        cls_ids = result.boxes.cls.int().tolist() if result.boxes is not None else []
        return union.cpu().numpy(), cls_ids


class YoloeTracker:
    """Fallback object masker: YOLOE open-vocabulary instance segmentation, per frame, same text prompts.

    Stand-in while sam3.pt access is pending. No cross-frame tracking; the temporal smoother
    handles flicker. Weights (yoloe-*-seg.pt) and the MobileCLIP text encoder auto-download.
    """

    def __init__(self, weights: Path, prompts: list[str], conf: float, imgsz: int, half: bool, device: str):
        import contextlib

        from ultralytics import YOLOE

        self.prompts = list(prompts)
        self.conf, self.imgsz, self.half = conf, imgsz, half
        self.device = device.replace("cuda:", "")
        weights = weights.resolve()  # absolute: the chdir below would otherwise re-root a relative path
        weights.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.chdir(weights.parent):  # ultralytics drops mobileclip_blt.ts into the cwd
            self.model = YOLOE(str(weights))
            self.model.set_classes(self.prompts, self.model.get_text_pe(self.prompts))
            # warm up once so the predictor exists with our settings
            self.model.predict(np.zeros((64, 64, 3), np.uint8), imgsz=64, verbose=False, device=self.device,
                               quantize=16 if half else None)
        self.weights = weights
        self.version = __import__("ultralytics").__version__

    def run(self, source: "PipeFrameSource") -> Iterator:
        for _, imgs, _ in source:
            # retina_masks=False: masks stay at network resolution; upsampling every instance to 1080p inside
            # ultralytics costs ~15 ms/frame, upsampling our single union costs ~1.5 ms (see union()).
            yield self.model.predict(imgs[0], conf=self.conf, imgsz=self.imgsz, quantize=16 if self.half else None,
                                     device=self.device, retina_masks=False, verbose=False)[0]

    @staticmethod
    def union(result, height: int, width: int) -> tuple[np.ndarray, list[int]]:
        masks = getattr(result, "masks", None)
        if masks is None or masks.data.shape[0] == 0:
            return np.zeros((height, width), dtype=bool), []
        from ultralytics.utils import ops

        u = masks.data.any(dim=0)[None, None].float()  # letterboxed network-res union
        u = ops.scale_masks(u, (height, width))[0, 0] > 0.5  # strips letterbox padding, resizes to frame
        return u.cpu().numpy(), result.boxes.cls.int().tolist()


# --------------------------------------------------------------------------- EgoBlur
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
class TextFlagger:
    def __init__(self, lang: str, device: str, min_conf: float, min_chars: int):
        import paddleocr

        self.min_conf, self.min_chars = min_conf, min_chars
        self.version = paddleocr.__version__
        major = int(self.version.split(".")[0])
        pd_dev = "cpu" if device == "cpu" else "gpu:" + device.replace("cuda:", "")
        if major >= 3:
            self.api = 3
            self.ocr = paddleocr.PaddleOCR(
                lang=lang, device=pd_dev, use_doc_orientation_classify=False,
                use_doc_unwarping=False, use_textline_orientation=False,
            )
        else:
            self.api = 2
            self.ocr = paddleocr.PaddleOCR(lang=lang, use_angle_cls=False, use_gpu=device != "cpu", show_log=False)

    @staticmethod
    def _readable(text: str, min_chars: int) -> bool:
        t = text.strip()
        return len(t) >= min_chars and any(c.isalnum() for c in t)

    def read(self, bgr: np.ndarray) -> list[dict]:
        hits = []
        if self.api == 3:
            for res in self.ocr.predict(bgr):
                d = res if isinstance(res, dict) else getattr(res, "json", {}).get("res", {})
                texts = d.get("rec_texts", []) or []
                scores = d.get("rec_scores", []) or []
                polys = d.get("rec_polys", None)
                if polys is None:
                    polys = d.get("dt_polys", [None] * len(texts))
                for text, score, poly in zip(texts, scores, polys):
                    if float(score) >= self.min_conf and self._readable(text, self.min_chars):
                        box = np.asarray(poly).reshape(-1, 2) if poly is not None else None
                        hits.append({
                            "text": text, "score": round(float(score), 4),
                            "box_xyxy": [int(box[:, 0].min()), int(box[:, 1].min()), int(box[:, 0].max()), int(box[:, 1].max())] if box is not None else None,
                        })
        else:
            res = self.ocr.ocr(bgr, cls=False) or []
            for page in res:
                for item in page or []:
                    poly, (text, score) = item
                    if float(score) >= self.min_conf and self._readable(text, self.min_chars):
                        box = np.asarray(poly).reshape(-1, 2)
                        hits.append({"text": text, "score": round(float(score), 4),
                                     "box_xyxy": [int(box[:, 0].min()), int(box[:, 1].min()), int(box[:, 0].max()), int(box[:, 1].max())]})
        return hits


# --------------------------------------------------------------------------- pipeline
@dataclass
class VideoReport:
    input: dict
    output: dict | None = None
    redaction: dict | None = None
    ocr: dict | None = None
    timing: dict = field(default_factory=dict)
    status: str = "pending"
    error: str | None = None


def process_video(args, tools: FFTools, info: VideoInfo, out_path: Path, sam3: Sam3Tracker | None,
                  faces: FaceDetector | None, ocr: TextFlagger | None, gpu_index: int | None, codec: str) -> VideoReport:
    from tqdm import tqdm

    src = Path(info.path)
    report = VideoReport(input={**asdict(info), "sha256": None})
    t0 = time.time()
    report.input["sha256"] = sha256_file(src)
    report.timing["sha256_input_s"] = round(time.time() - t0, 2)

    W, H = info.display_width, info.display_height  # decoder pipe delivers display orientation
    if info.rotation:
        LOG.info("%s: rotation tag %d° -> processing upright at %dx%d", src.name, info.rotation, W, H)
    smoother = TemporalSmoother(args.smooth_window, args.smooth_mode)
    tint = tint_from_seed(report.input["sha256"] + str(args.anon_seed)) if args.fill == "anon" else None
    use_cuda_dec = tools.hwaccel_cuda and not args.no_gpu_decode

    stats = dict(
        frames=0, frames_with_objects=0, frames_with_faces=0, max_faces_in_frame=0, total_face_detections=0,
        frames_redacted=0, mean_masked_fraction=0.0, max_masked_fraction=0.0,
        per_prompt_frames={p: 0 for p in (sam3.prompts if sam3 else [])},
    )
    masked_frac_sum = 0.0
    partial = out_path.with_name(out_path.stem + ".partial.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reader = FrameReader(tools, src, W, H, use_cuda_dec, gpu_index)
    writer = FrameWriter(tools, partial, W, H, info.fps, codec, args.nvenc_preset, args.cq, gpu_index)
    t1 = time.time()
    dec_err = ""
    try:
        source = PipeFrameSource(reader, src, info.nb_frames, info.fps_float, args.max_frames)
        if sam3 is not None:
            results = sam3.run(source)

            def frame_iter():
                for r in results:
                    union, cls_ids = sam3.union(r, H, W)
                    yield r.orig_img, union, cls_ids
        else:
            def frame_iter():
                for _, imgs, _ in source:
                    yield imgs[0], np.zeros((H, W), dtype=bool), []

        # Stage 2 (worker thread): faces -> dilate -> temporal smoothing -> fill -> encoder.
        # Stage 1 (this thread): decode read -> object model -> union mask. Overlapping the two hides
        # the CPU post-processing behind GPU inference; cv2/numpy release the GIL.
        post_q: queue.Queue = queue.Queue(maxsize=48)
        post_exc: list[BaseException] = []
        masked = {"sum": 0.0}

        def emit(items):
            for _, frame, sm in items:
                frac = float(np.count_nonzero(sm)) / sm.size
                masked["sum"] += frac
                if frac > 0:
                    stats["frames_redacted"] += 1
                    stats["max_masked_fraction"] = max(stats["max_masked_fraction"], frac)
                    frame = apply_fill(frame, sm.view(np.uint8), args.fill, args.grey, args.blur_block, tint)
                writer.write(frame)

        def post_worker():
            try:
                for idx, frame, mask in iter(post_q.get, None):
                    if faces is not None:
                        boxes = faces.detect(frame)
                        if len(boxes):
                            stats["frames_with_faces"] += 1
                            stats["total_face_detections"] += int(len(boxes))
                            stats["max_faces_in_frame"] = max(stats["max_faces_in_frame"], int(len(boxes)))
                            paint_faces(mask, boxes, args.face_shape)
                    mask = dilate_fast(mask, args.dilate_px)
                    emit(smoother.push(idx, frame, mask))
                emit(smoother.flush())
            except BaseException as e:  # noqa: BLE001
                post_exc.append(e)
                while True:  # drain so the producer never blocks on a dead consumer
                    if post_q.get() is None:
                        break

        worker = threading.Thread(target=post_worker, daemon=True)
        worker.start()
        pbar = tqdm(total=source.limit if args.max_frames else info.nb_frames, unit="f", desc=src.name, leave=False, dynamic_ncols=True)
        for idx, (frame, union, cls_ids) in enumerate(frame_iter()):
            if post_exc:
                raise RuntimeError(f"post-processing failed: {post_exc[0]!r}") from post_exc[0]
            if cls_ids:
                stats["frames_with_objects"] += 1
                for c in set(cls_ids):
                    if 0 <= c < len(sam3.prompts):
                        stats["per_prompt_frames"][sam3.prompts[c]] += 1
            stats["frames"] += 1
            post_q.put((idx, frame, union.view(np.uint8).copy()))
            pbar.update(1)
        post_q.put(None)
        worker.join()
        pbar.close()
        if post_exc:
            raise RuntimeError(f"post-processing failed: {post_exc[0]!r}") from post_exc[0]
        masked_frac_sum = masked["sum"]
    except BaseException:
        writer.abort()
        partial.unlink(missing_ok=True)
        raise
    finally:
        dec_err = reader.close()
    enc_err = writer.close()
    if dec_err and not (args.max_frames and "Broken pipe" in dec_err):  # we close the pipe early on --max-frames
        LOG.warning("%s: decoder stderr: %s", src.name, dec_err[-500:])
    if enc_err:
        LOG.warning("%s: encoder stderr: %s", src.name, enc_err[-500:])
    if stats["frames"] == 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"no frames decoded from {src} ({dec_err[-300:]})")
    if writer.frames_written != stats["frames"]:
        raise RuntimeError(f"frame count mismatch: processed {stats['frames']} wrote {writer.frames_written}")
    report.timing["redact_s"] = round(time.time() - t1, 2)
    report.timing["redact_fps"] = round(stats["frames"] / max(1e-6, time.time() - t1), 2)
    stats["mean_masked_fraction"] = round(masked_frac_sum / stats["frames"], 6)
    stats["max_masked_fraction"] = round(stats["max_masked_fraction"], 6)
    os.replace(partial, out_path)

    out_info = ffprobe_video(tools, out_path)
    t2 = time.time()
    report.output = {**asdict(out_info), "sha256": sha256_file(out_path), "encoder": codec,
                     "audio": "stripped", "gpu_decode": use_cuda_dec}
    report.timing["sha256_output_s"] = round(time.time() - t2, 2)
    if out_info.nb_frames != stats["frames"]:
        LOG.warning("%s: output has %d frames, processed %d", out_path.name, out_info.nb_frames, stats["frames"])
    report.redaction = {
        "prompts": sam3.prompts if sam3 else [],
        "objects_enabled": sam3 is not None, "object_tracker": args.tracker if sam3 else None,
        "faces_enabled": faces is not None, "face_detector": args.face_detector if faces else None,
        "dilate_px": args.dilate_px, "smooth_window": args.smooth_window, "smooth_mode": args.smooth_mode,
        "fill": args.fill, "fill_bgr": [int(args.grey)] * 3 if args.fill == "grey" else None,
        "blur_block": args.blur_block if args.fill != "grey" else None,
        "anon_tint_bgr": list(tint) if tint else None, **stats,
    }

    # OCR pass on the encoded output
    if ocr is not None:
        t3 = time.time()
        flags = []
        checked = 0
        rd = FrameReader(tools, out_path, out_info.display_width, out_info.display_height, use_cuda_dec, gpu_index,
                         select_stride=args.ocr_stride)
        try:
            i = 0
            while True:
                f = rd.read()
                if f is None:
                    break
                fidx = i * args.ocr_stride
                i += 1
                checked += 1
                hits = ocr.read(f)
                if hits:
                    flags.append({"frame": fidx, "time_s": round(fidx / out_info.fps_float, 3), "texts": hits})
        finally:
            rd.close()
        report.ocr = {
            "engine": f"paddleocr {ocr.version}", "lang": args.ocr_lang, "stride": args.ocr_stride,
            "min_conf": args.ocr_min_conf, "min_chars": args.ocr_min_chars,
            "frames_checked": checked, "frames_flagged": len(flags),
            "distinct_texts": sorted({h["text"] for fl in flags for h in fl["texts"]}),
            "flags": flags,
        }
        report.timing["ocr_s"] = round(time.time() - t3, 2)
    report.timing["total_s"] = round(time.time() - t0, 2)
    report.status = "ok"
    return report


# --------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_dir", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--recursive", action="store_true", help="descend into sub-folders (outputs are flattened by stem)")
    p.add_argument("--overwrite", action="store_true", help="re-process videos whose output already exists")
    p.add_argument("--max-frames", type=int, default=None, help="debug: stop after N frames per video")
    p.add_argument("--shard", nargs=2, type=int, metavar=("I", "N"), default=None,
                   help="process only files with index %% N == I (run one shard per GPU)")
    p.add_argument("--list", action="store_true", help="probe and list the videos that would be processed, then exit")

    g = p.add_argument_group("models")
    g.add_argument("--tracker", choices=["sam3", "yoloe"], default="sam3",
                   help="object masker: sam3 (SAM3VideoSemanticPredictor, needs gated sam3.pt) or yoloe "
                        "(open-vocabulary YOLOE segmentation fallback, weights auto-download)")
    g.add_argument("--sam3-weights", type=Path, default=Path("sam3.pt"))
    g.add_argument("--yoloe-weights", type=Path, default=Path("weights/yoloe-11l-seg.pt"),
                   help="yoloe-{v8s,v8m,v8l,11s,11m,11l,26n..26x}-seg.pt; downloaded to this path if missing")
    g.add_argument("--yoloe-imgsz", type=int, default=1280, help="YOLOE inference size (long side)")
    g.add_argument("--yoloe-conf", type=float, default=0.15,
                   help="YOLOE score threshold (its open-vocabulary scores run lower than SAM 3's; "
                        "concrete nouns like 'box', 'soda can', 'person' score far better than 'label' or 'wrapper')")
    g.add_argument("--face-detector", choices=["egoblur", "yunet"], default="egoblur",
                   help="egoblur (gated TorchScript weights) or yunet (OpenCV fallback, auto-downloads 230 KB ONNX)")
    g.add_argument("--egoblur-weights", type=Path, default=None, help="EgoBlur face TorchScript (.jit)")
    g.add_argument("--yunet-weights", type=Path, default=Path("weights/face_detection_yunet_2023mar.onnx"))
    g.add_argument("--yunet-scale", type=float, default=0.5, help="downscale factor for YuNet input (speed)")
    g.add_argument("--egoblur-gen", choices=["auto", "1", "2"], default="auto")
    g.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS, help="SAM 3 text prompts")
    g.add_argument("--conf", type=float, default=0.3, help="SAM 3 mask confidence threshold")
    g.add_argument("--imgsz", type=int, default=1008, help="SAM 3 inference size (multiple of 14)")
    g.add_argument("--no-half", action="store_true", help="run SAM 3 in fp32")
    g.add_argument("--face-conf", type=float, default=0.5)
    g.add_argument("--face-nms", type=float, default=0.3)
    g.add_argument("--face-scale", type=float, default=1.15, help="enlarge face boxes about their centre")
    g.add_argument("--face-shape", choices=["rect", "ellipse"], default="rect")
    g.add_argument("--no-sam3", action="store_true")
    g.add_argument("--no-faces", action="store_true")
    g.add_argument("--device", default="0", help="cpu, N or cuda:N (torch, paddle, NVDEC and NVENC all use it)")

    g = p.add_argument_group("mask post-processing")
    g.add_argument("--dilate-px", type=int, default=15,
                   help="mask margin in px: >0 grows the mask outward (safer), <0 shrinks it inward so the object's "
                        "edges stay sharp and only its interior is redacted, 0 = as detected")
    g.add_argument("--smooth-window", type=int, default=5, help="odd temporal window (frames)")
    g.add_argument("--smooth-mode", choices=["max", "majority"], default="max",
                   help="max = hold (never shrinks coverage); majority = vote")
    g.add_argument("--fill", choices=["grey", "blur", "pixelate", "anon"], default="grey",
                   help="grey = flat fill (irreversible); blur = strong blur; pixelate = mosaic; anon = blur + "
                        "replace the object's colours with a per-video random tint while keeping its shading "
                        "(geometry-preserving anonymisation for training data). Only grey guarantees text is "
                        "unrecoverable.")
    g.add_argument("--anon-seed", default="", help="extra seed for the anon tint (tint is derived from input sha256 + seed)")
    g.add_argument("--grey", type=int, default=128, help="fill value 0-255 for all three channels (grey mode)")
    g.add_argument("--blur-block", type=int, default=24,
                   help="blur/pixelate coarseness in pixels at 1080p (frame is downscaled by this factor)")

    g = p.add_argument_group("encoding")
    g.add_argument("--codec", choices=["h264_nvenc", "hevc_nvenc", "libx264", "libx265"], default="h264_nvenc")
    g.add_argument("--nvenc-preset", default="p5", help="NVENC preset p1 (fast) .. p7 (quality)")
    g.add_argument("--cq", type=int, default=23, help="NVENC constant-quality level (or CRF for libx26x)")
    g.add_argument("--no-gpu-decode", action="store_true", help="disable NVDEC (-hwaccel cuda)")
    g.add_argument("--ffmpeg", default=None)
    g.add_argument("--ffprobe", default=None)

    g = p.add_argument_group("OCR check")
    g.add_argument("--no-ocr", action="store_true")
    g.add_argument("--ocr-stride", type=int, default=15, help="check every Nth output frame")
    g.add_argument("--ocr-lang", default="en")
    g.add_argument("--ocr-min-conf", type=float, default=0.6)
    g.add_argument("--ocr-min-chars", type=int, default=3)

    g = p.add_argument_group("manifest")
    g.add_argument("--manifest", default=None, help="manifest file name inside output_dir (default manifest.json)")
    return p


def find_videos(root: Path, recursive: bool) -> list[Path]:
    it = root.rglob("*") if recursive else root.iterdir()
    vids = sorted(p for p in it if p.is_file() and p.suffix.lower() in VIDEO_EXTS and not p.name.startswith("."))
    return vids


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    torch_dev, gpu_index = parse_device(args.device)
    if not args.input_dir.is_dir():
        sys.exit(f"error: input_dir {args.input_dir} is not a directory (pass the folder that holds your MP4s)")
    if not args.no_sam3 and args.tracker == "sam3" and not args.sam3_weights.is_file():
        sys.exit(f"error: SAM 3 weights not found at {args.sam3_weights}; pass --sam3-weights /real/path/sam3.pt "
                 "or --no-sam3 to run without it")
    if not args.no_faces and args.face_detector == "egoblur" and (args.egoblur_weights is None or not args.egoblur_weights.is_file()):
        sys.exit(f"error: EgoBlur weights not found at {args.egoblur_weights}; pass --egoblur-weights /real/path/*.jit "
                 "or --no-faces to run without it")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(args.output_dir / "redact.log")])
    tools = FFTools.discover(args.ffmpeg, args.ffprobe)
    LOG.info("ffmpeg: %s (cuda hwaccel=%s)", tools.version, tools.hwaccel_cuda)

    codec = args.codec
    if codec.endswith("_nvenc") and codec not in tools.encoders:
        LOG.warning("%s not available in this ffmpeg; falling back to libx264", codec)
        codec = "libx264"
    if codec.endswith("_nvenc") and gpu_index is None:
        LOG.warning("--device cpu with NVENC: encoder will use GPU 0")

    vids = find_videos(args.input_dir, args.recursive)
    if args.shard:
        i, n = args.shard
        vids = [v for k, v in enumerate(vids) if k % n == i]
    if not vids:
        LOG.error("no videos found in %s", args.input_dir)
        return 2
    if args.list:
        for v in vids:
            inf = ffprobe_video(tools, v)
            rot = f" rot{inf.rotation}->{inf.display_width}x{inf.display_height}" if inf.rotation else ""
            print(f"{v}  {inf.width}x{inf.height}{rot} {inf.fps_float:.3f}fps {inf.nb_frames}f {inf.codec}")
        return 0

    manifest_name = args.manifest or ("manifest.json" if not args.shard else f"manifest_shard{args.shard[0]}of{args.shard[1]}.json")
    manifest_path = args.output_dir / manifest_name
    manifest = {"schema": "sparkpack-redact/1", "created_utc": utc_now(), "videos": []}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            LOG.info("resuming manifest %s (%d entries)", manifest_path, len(manifest.get("videos", [])))
        except Exception as e:  # noqa: BLE001
            LOG.warning("could not parse existing manifest (%s); starting new", e)
            manifest = {"schema": "sparkpack-redact/1", "created_utc": utc_now(), "videos": []}
    by_input = {e["input"]["path"]: e for e in manifest["videos"]}

    # models
    sam3 = faces = ocr = None
    import torch

    gpu_name = torch.cuda.get_device_name(gpu_index) if gpu_index is not None and torch.cuda.is_available() else None
    if not args.no_sam3:
        t = time.time()
        if args.tracker == "yoloe":
            sam3 = YoloeTracker(args.yoloe_weights, args.prompts, args.yoloe_conf, args.yoloe_imgsz, not args.no_half, torch_dev)
            LOG.info("YOLOE fallback loaded (%.1fs) %s prompts=%s", time.time() - t, args.yoloe_weights.name, args.prompts)
        else:
            sam3 = Sam3Tracker(args.sam3_weights, args.prompts, args.conf, args.imgsz, not args.no_half, torch_dev)
            LOG.info("SAM 3 loaded (%.1fs) prompts=%s", time.time() - t, args.prompts)
    if not args.no_faces:
        if args.face_detector == "yunet":
            faces = YuNetFaceDetector(args.yunet_weights, args.face_conf, args.face_nms, args.face_scale, args.yunet_scale)
            LOG.info("YuNet face fallback loaded (%s, input scale %.2f)", args.yunet_weights.name, args.yunet_scale)
        else:
            faces = FaceDetector(args.egoblur_weights, torch_dev, args.egoblur_gen, args.face_conf, args.face_nms, args.face_scale)
            LOG.info("EgoBlur gen%d loaded (%s)", faces.gen, "scripted" if faces.scripted else "traced")
    if not args.no_ocr:
        t = time.time()
        ocr = TextFlagger(args.ocr_lang, torch_dev, args.ocr_min_conf, args.ocr_min_chars)
        LOG.info("PaddleOCR %s loaded (%.1fs)", ocr.version, time.time() - t)

    manifest["updated_utc"] = utc_now()
    manifest["tool"] = {
        "script": Path(__file__).name, "script_version": SCRIPT_VERSION, "script_sha256": sha256_file(Path(__file__)),
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "ultralytics": sam3.version if sam3 else None, "paddleocr": ocr.version if ocr else None,
        "opencv": cv2.__version__, "ffmpeg": tools.version, "gpu": gpu_name, "host": platform.node(),
    }
    manifest["params"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    manifest["params"]["codec_used"] = codec
    manifest["weights"] = {
        "objects": ({"tracker": args.tracker, "path": str(sam3.weights if args.tracker == "yoloe" else args.sam3_weights),
                     "sha256": sha256_file(sam3.weights if args.tracker == "yoloe" else args.sam3_weights)} if sam3 else None),
        "faces": {"detector": args.face_detector, "path": str(faces.weights), "sha256": sha256_file(faces.weights)} if faces else None,
    }

    n_ok = n_err = n_skip = 0
    for k, v in enumerate(vids, 1):
        out_path = args.output_dir / (v.stem + ".mp4")
        if out_path.resolve() == v.resolve():
            LOG.error("refusing to overwrite input %s (output_dir must differ from input_dir)", v)
            n_err += 1
            continue
        if out_path.exists() and not args.overwrite and str(v) in by_input and by_input[str(v)].get("status") == "ok":
            LOG.info("[%d/%d] skip %s (already in manifest)", k, len(vids), v.name)
            n_skip += 1
            continue
        LOG.info("[%d/%d] %s", k, len(vids), v)
        try:
            info = ffprobe_video(tools, v)
            rep = process_video(args, tools, info, out_path, sam3, faces, ocr, gpu_index, codec)
            n_ok += 1
            LOG.info("  ok: %d frames, %.1f fps, redacted %d frames (mean masked %.2f%%), faces in %d frames, OCR flags %s",
                     rep.redaction["frames"], rep.timing["redact_fps"], rep.redaction["frames_redacted"],
                     100 * rep.redaction["mean_masked_fraction"], rep.redaction["frames_with_faces"],
                     rep.ocr["frames_flagged"] if rep.ocr else "n/a")
        except Exception as e:  # noqa: BLE001
            n_err += 1
            LOG.error("  FAILED %s: %s", v.name, e)
            LOG.debug(traceback.format_exc())
            rep = VideoReport(input={"path": str(v)}, status="error", error=f"{type(e).__name__}: {e}")
            for junk in args.output_dir.glob(v.stem + ".partial.mp4"):
                junk.unlink(missing_ok=True)
        entry = asdict(rep)
        entry["processed_utc"] = utc_now()
        by_input[str(v)] = entry
        manifest["videos"] = [by_input[p] for p in sorted(by_input)]
        manifest["updated_utc"] = utc_now()
        manifest["summary"] = {
            "videos": len(manifest["videos"]),
            "ok": sum(e["status"] == "ok" for e in manifest["videos"]),
            "error": sum(e["status"] == "error" for e in manifest["videos"]),
            "ocr_flagged_videos": sum(1 for e in manifest["videos"] if e.get("ocr") and e["ocr"]["frames_flagged"] > 0),
        }
        atomic_write_json(manifest_path, manifest)
        # sha256sum-compatible list of outputs
        with open(args.output_dir / "SHA256SUMS", "w") as f:
            for e in manifest["videos"]:
                if e.get("output"):
                    f.write(f"{e['output']['sha256']}  {Path(e['output']['path']).name}\n")
    (args.output_dir / (manifest_name + ".sha256")).write_text(f"{sha256_file(manifest_path)}  {manifest_name}\n")
    LOG.info("done: %d ok, %d failed, %d skipped -> %s", n_ok, n_err, n_skip, manifest_path)
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
