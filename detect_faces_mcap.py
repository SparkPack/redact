#!/usr/bin/env python
"""Face detection straight out of an MCAP log: no extraction step, no re-encode.

Each `foxglove.CompressedVideo` channel in the file (wrist/chest/head feeds) is an Annex-B h264
elementary stream, one access unit per MCAP message. This script streams each channel's packets into
its own ffmpeg decoder, runs a face detector on the decoded frames, and reports every frame that
contains a face together with its MCAP log timestamp, so detections line up across feeds and with the
IMU/other channels in the same log.

  python detect_faces_mcap.py FILE.mcap                     # or a folder of .mcap files
  python detect_faces_mcap.py FILE.mcap --stride 2 --scale 0.5
  python detect_faces_mcap.py FILE.mcap --face-detector egoblur --egoblur-weights w.jit --device 0

Decoding and detection are CPU-only by default; pass --gpu-decode to use NVDEC.
"""
from __future__ import annotations

import argparse
import json
import logging
import queue
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import redact_videos as rv  # noqa: E402

LOG = logging.getLogger("mcapfaces")
SCRIPT_VERSION = "1.0.0"
VIDEO_SCHEMA = "foxglove.CompressedVideo"


# --------------------------------------------------------------------------- protobuf
def parse_protobuf_fields(buf: bytes) -> dict[int, bytes]:
    """Minimal protobuf reader for length-delimited fields (enough for foxglove.CompressedVideo).

    CompressedVideo: 1=timestamp(message) 2=frame_id(string) 3=data(bytes) 4=format(string)
    Varint and fixed-width fields are skipped rather than decoded.
    """
    out: dict[int, bytes] = {}
    i, n = 0, len(buf)
    while i < n:
        tag, shift = 0, 0
        while True:
            b = buf[i]; i += 1
            tag |= (b & 0x7F) << shift; shift += 7
            if not b & 0x80:
                break
        field, wire = tag >> 3, tag & 7
        if wire == 2:
            ln, shift = 0, 0
            while True:
                b = buf[i]; i += 1
                ln |= (b & 0x7F) << shift; shift += 7
                if not b & 0x80:
                    break
            out[field] = buf[i:i + ln]; i += ln
        elif wire == 0:
            while buf[i] & 0x80:
                i += 1
            i += 1
        elif wire == 1:
            i += 8
        elif wire == 5:
            i += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
    return out


def video_channels(reader) -> dict[int, str]:
    """{channel_id: topic} for every CompressedVideo channel in the summary."""
    s = reader.get_summary()
    if s is None:
        raise RuntimeError("MCAP has no summary section; cannot enumerate channels")
    return {cid: ch.topic for cid, ch in s.channels.items()
            if ch.schema_id and s.schemas[ch.schema_id].name == VIDEO_SCHEMA}


# --------------------------------------------------------------------------- decode
class PacketDecoder:
    """ffmpeg subprocess: Annex-B h264/h265 packets in on stdin, raw BGR frames out on stdout."""

    def __init__(self, tools: rv.FFTools, codec: str, width: int, height: int, scale: float,
                 gpu_decode: bool, gpu_index: int | None, queue_size: int = 8):
        self.ow = int(round(width * scale)) // 2 * 2
        self.oh = int(round(height * scale)) // 2 * 2
        self.nbytes = self.ow * self.oh * 3
        cmd = [tools.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
        if gpu_decode:
            cmd += ["-hwaccel", "cuda"] + (["-hwaccel_device", str(gpu_index)] if gpu_index is not None else [])
        cmd += ["-f", codec, "-i", "pipe:0", "-an", "-sn"]
        if (self.ow, self.oh) != (width, height):
            cmd += ["-vf", f"scale={self.ow}:{self.oh}:flags=area"]
        cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-fps_mode", "passthrough", "pipe:1"]
        self.cmd = cmd
        self.err = tempfile.NamedTemporaryFile(prefix="mcapdec_", suffix=".log", delete=False)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.err, bufsize=0)
        self.q: queue.Queue = queue.Queue(maxsize=queue_size)
        self.exc: BaseException | None = None
        self.feeder = threading.Thread(target=self._feed, daemon=True)
        self.feeder.start()

    def _feed(self):
        try:
            while True:
                pkt = self.q.get()
                if pkt is None:
                    break
                self.proc.stdin.write(pkt)
        except BrokenPipeError:
            pass  # decoder exited early; read side reports the real error
        except BaseException as e:  # noqa: BLE001
            self.exc = e
        finally:
            try:
                self.proc.stdin.close()
            except Exception:
                pass

    def write(self, pkt: bytes) -> None:
        self.q.put(pkt)

    def finish_input(self) -> None:
        self.q.put(None)

    def read(self) -> np.ndarray | None:
        buf = bytearray(self.nbytes)
        view = memoryview(buf)
        got = 0
        while got < self.nbytes:
            n = self.proc.stdout.readinto(view[got:])
            if not n:
                break
            got += n
        if got < self.nbytes:
            return None
        return np.frombuffer(buf, np.uint8).reshape(self.oh, self.ow, 3)

    def close(self) -> str:
        try:
            self.proc.stdout.close()
        except Exception:
            pass
        self.feeder.join(timeout=10)
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
        self.err.close()
        msg = Path(self.err.name).read_text(errors="replace").strip()
        Path(self.err.name).unlink(missing_ok=True)
        return msg


def probe_stream(tools: rv.FFTools, packets: list[bytes], codec: str) -> tuple[int, int]:
    """Width/height from the first few access units (they carry SPS/PPS)."""
    with tempfile.NamedTemporaryFile(suffix=f".{codec}", delete=False) as t:
        for p in packets:
            t.write(p)
        name = t.name
    try:
        out = subprocess.run([tools.ffprobe, "-v", "error", "-select_streams", "v:0",
                              "-show_entries", "stream=width,height", "-of", "csv=p=0", name],
                             capture_output=True, text=True, check=True).stdout.strip()
        w, h = (int(x) for x in out.split(",")[:2])
        return w, h
    finally:
        Path(name).unlink(missing_ok=True)


# --------------------------------------------------------------------------- detectors
def make_detector(args):
    """Return (name, callable(frame)->Nx5 xyxy+score in the frame's own pixel coords)."""
    if args.face_detector == "yunet":
        det = cv2.FaceDetectorYN.create(str(args.yunet_weights), "", (320, 320),
                                        score_threshold=args.conf, nms_threshold=args.nms, top_k=200)
        size = {"wh": None}

        def run(frame):
            h, w = frame.shape[:2]
            if size["wh"] != (w, h):
                det.setInputSize((w, h)); size["wh"] = (w, h)
            _, faces = det.detect(frame)
            if faces is None or len(faces) == 0:
                return np.zeros((0, 5), np.float32)
            f = faces[:, :4].astype(np.float32)
            return np.stack([f[:, 0], f[:, 1], f[:, 0] + f[:, 2], f[:, 1] + f[:, 3], faces[:, 14]], 1)
        return "yunet", run

    eb = rv.FaceDetector(args.egoblur_weights, args.torch_dev, args.egoblur_gen, args.conf, args.nms, 1.0,
                         half=args.half)

    def run(frame):
        return eb.detect(frame)  # already Nx5 (xyxy + score)
    run.batch = eb.detect_batch
    return f"egoblur-gen{eb.gen}", run


# --------------------------------------------------------------------------- batched multi-channel
def detect_channels_batched(topics: list, packets: dict, times: dict, codecs: dict,
                            tools: rv.FFTools, args, detect_batch) -> list:
    """Decode every channel in lockstep and run ONE inference call per frame index across all of them.

    The feeds are frame-synchronised, so index i of each channel is the same instant. Batching them
    keeps the GPU fed far better than one 1920x1200 frame at a time, and it costs nothing in accuracy
    (verified: batched and single-frame results agree to <0.1 px).
    """
    t0 = time.time()
    decs, scales, dims = {}, {}, {}
    for topic in topics:
        w, h = probe_stream(tools, packets[topic][:8], codecs[topic])
        dec = PacketDecoder(tools, codecs[topic], w, h, args.scale, args.gpu_decode, args.gpu_index)
        decs[topic] = dec
        dims[topic] = (w, h)
        scales[topic] = (w / dec.ow, h / dec.oh)
        threading.Thread(target=_feed, args=(dec, packets[topic]), daemon=True).start()

    dets = {t: [] for t in topics}
    checked = {t: 0 for t in topics}
    decoded = {t: 0 for t in topics}
    alive = list(topics)
    idx = 0
    min_px = args.min_face_px
    while alive:
        batch_topics, batch_frames = [], []
        for topic in list(alive):
            frame = decs[topic].read()
            if frame is None or idx >= len(times[topic]):
                alive.remove(topic)
                continue
            decoded[topic] += 1
            if args.stride <= 1 or idx % args.stride == 0:
                batch_topics.append(topic)
                batch_frames.append(frame)
        if batch_frames:
            if args.progress_every and idx and idx % args.progress_every == 0:
                el = time.time() - t0
                done = sum(decoded.values())
                total = sum(len(v) for v in times.values())
                LOG.info("   %6d/%d frames (%4.1f%%) %5.1f fps, eta %4.1f min, %d with faces so far",
                         done, total, 100.0 * done / max(1, total), done / max(el, 1e-6),
                         (total - done) / max(1e-6, done / el) / 60,
                         sum(len(v) for v in dets.values()))
            for topic, boxes in zip(batch_topics, detect_batch(batch_frames)):
                checked[topic] += 1
                sx, sy = scales[topic]
                if len(boxes):
                    boxes = boxes.copy()
                    boxes[:, [0, 2]] *= sx
                    boxes[:, [1, 3]] *= sy
                    keep = ((np.minimum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]) >= min_px)
                            & (boxes[:, 4] >= topic_conf(topic, args)))
                    boxes = boxes[keep]
                if len(boxes):
                    dets[topic].append({
                        "frame": idx, "log_time": round(times[topic][idx], 6),
                        "t_rel_s": round(times[topic][idx] - times[topic][0], 4),
                        "boxes": [[int(x1), int(y1), int(x2), int(y2), round(float(sc), 3)]
                                  for x1, y1, x2, y2, sc in boxes],
                    })
        idx += 1
    results = []
    total = sum(decoded.values())
    for topic in topics:
        err = decs[topic].close()
        if err and "Broken pipe" not in err:
            LOG.warning("%s: decoder stderr: %s", topic, err[-300:])
        results.append(_summarise(topic, codecs[topic], times[topic], dets[topic], decoded[topic],
                                  checked[topic], dims[topic], (decs[topic].ow, decs[topic].oh),
                                  args, time.time() - t0, total))
    return results


def _feed(dec: PacketDecoder, packets: list) -> None:
    try:
        for pkt in packets:
            dec.write(pkt)
    finally:
        dec.finish_input()


def _summarise(topic, codec, times, dets, decoded, checked, dims, det_dims, args, elapsed, total_frames):
    width, height = dims
    span = times[-1] - times[0]
    fps_src = (len(times) - 1) / span if span > 0 else 0.0
    hold = args.stride / fps_src if fps_src else 0.0
    intervals = _merge([d["t_rel_s"] for d in dets], args.gap_s, hold)
    return {
        "topic": topic, "codec": codec, "messages": len(times), "frames_decoded": decoded,
        "width": width, "height": height, "detect_width": det_dims[0], "detect_height": det_dims[1],
        "source_fps": round(fps_src, 3), "start_log_time": round(times[0], 6), "end_log_time": round(times[-1], 6),
        "frames_checked": checked, "frames_with_faces": len(dets),
        "max_faces_in_frame": max((len(d["boxes"]) for d in dets), default=0),
        "largest_face_px": int(max((min(b[2] - b[0], b[3] - b[1]) for d in dets for b in d["boxes"]), default=0)),
        "intervals_rel_s": intervals, "face_time_s": round(sum(b - a for a, b in intervals), 3),
        "detections": dets, "decode_detect_s": round(elapsed, 2),
        "fps": round(total_frames / max(elapsed, 1e-6), 1),
    }


# --------------------------------------------------------------------------- per-channel worker
def detect_channel(topic: str, packets: list[bytes], times: list[float], codec: str,
                   tools: rv.FFTools, args, detector_factory) -> dict:
    """Decode one channel's packets and detect faces. packets[i] <-> times[i] (message log time)."""
    t0 = time.time()
    _, detect = detector_factory()
    width, height = probe_stream(tools, packets[:8], codec)
    dec = PacketDecoder(tools, codec, width, height, args.scale, args.gpu_decode, args.gpu_index)

    def feed_all():
        # Every packet must be fed even when --stride skips detection: inter-frames depend on the
        # frames between them, so dropping any would desynchronise the decoder.
        try:
            for pkt in packets:
                dec.write(pkt)
        finally:
            dec.finish_input()

    threading.Thread(target=feed_all, daemon=True).start()

    sx, sy = width / dec.ow, height / dec.oh
    dets, checked, idx = [], 0, 0
    min_px = args.min_face_px
    while True:
        frame = dec.read()
        if frame is None:
            break
        if idx >= len(times):
            LOG.warning("%s: decoder produced more frames (%d) than messages (%d); ignoring extras",
                        topic, idx + 1, len(times))
            break
        if args.stride <= 1 or idx % args.stride == 0:
            checked += 1
            boxes = detect(frame)
            if len(boxes):
                boxes[:, [0, 2]] *= sx
                boxes[:, [1, 3]] *= sy
                keep = ((np.minimum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]) >= min_px)
                        & (boxes[:, 4] >= topic_conf(topic, args)))
                boxes = boxes[keep]
            if len(boxes):
                dets.append({
                    "frame": idx, "log_time": round(times[idx], 6), "t_rel_s": round(times[idx] - times[0], 4),
                    "boxes": [[int(x1), int(y1), int(x2), int(y2), round(float(s), 3)] for x1, y1, x2, y2, s in boxes],
                })
        idx += 1
    err = dec.close()
    if err and "Broken pipe" not in err:
        LOG.warning("%s: decoder stderr: %s", topic, err[-300:])
    if idx < len(times):
        LOG.warning("%s: decoded %d frames but channel has %d messages", topic, idx, len(times))
    fps_src = times[-1] - times[0]
    fps_src = (len(times) - 1) / fps_src if fps_src > 0 else 0.0
    hold = args.stride / fps_src if fps_src else 0.0
    intervals = _merge([d["t_rel_s"] for d in dets], args.gap_s, hold)
    return {
        "topic": topic, "codec": codec, "messages": len(times), "frames_decoded": idx,
        "width": width, "height": height, "detect_width": dec.ow, "detect_height": dec.oh,
        "source_fps": round(fps_src, 3), "start_log_time": round(times[0], 6), "end_log_time": round(times[-1], 6),
        "frames_checked": checked, "frames_with_faces": len(dets),
        "max_faces_in_frame": max((len(d["boxes"]) for d in dets), default=0),
        "largest_face_px": int(max((min(b[2] - b[0], b[3] - b[1]) for d in dets for b in d["boxes"]), default=0)),
        "intervals_rel_s": intervals, "face_time_s": round(sum(b - a for a, b in intervals), 3),
        "detections": dets, "decode_detect_s": round(time.time() - t0, 2),
        "fps": round(idx / max(time.time() - t0, 1e-6), 1),
    }


def topic_conf(topic: str, args) -> float:
    """Per-camera score threshold. Justified by rig geometry rather than this footage: the wrist
    cameras point at the work surface and almost never see a face, so they can be held to a much
    stricter bar than the head and chest cameras."""
    for pat, val in getattr(args, "conf_topic_map", []):
        if pat in topic:
            return val
    return args.conf


def _merge(times: list[float], gap_s: float, hold_s: float) -> list[list[float]]:
    out: list[list[float]] = []
    for t in times:
        if out and t - out[-1][1] <= gap_s:
            out[-1][1] = t
        else:
            out.append([t, t])
    return [[round(a, 3), round(b + hold_s, 3)] for a, b in out]


# --------------------------------------------------------------------------- file
def process_file(path: Path, tools: rv.FFTools, args, detector_factory) -> dict:
    from mcap.reader import make_reader

    t0 = time.time()
    with open(path, "rb") as fh:
        reader = make_reader(fh)
        chans = video_channels(reader)
        if args.topics:
            chans = {cid: t for cid, t in chans.items() if t in args.topics}
        if not chans:
            return {"file": str(path), "status": "skipped", "reason": f"no {VIDEO_SCHEMA} channels"}
        packets: dict[str, list[bytes]] = {t: [] for t in chans.values()}
        times: dict[str, list[float]] = {t: [] for t in chans.values()}
        formats: dict[str, Counter] = {t: Counter() for t in chans.values()}
        wanted = set(chans.values())
        for schema, channel, msg in reader.iter_messages(topics=list(wanted)):
            topic = channel.topic
            if topic not in wanted:
                continue
            f = parse_protobuf_fields(msg.data)
            data = f.get(3)
            if not data:
                continue
            packets[topic].append(data)
            times[topic].append(msg.log_time / 1e9)
            formats[topic][f.get(4, b"h264").decode()] += 1
            if args.max_messages and len(packets[topic]) >= args.max_messages:
                wanted.discard(topic)
                if not wanted:
                    break
    read_s = time.time() - t0
    LOG.info("%s: %d video channels, %d packets read in %.1fs", path.name, len(packets),
             sum(len(v) for v in packets.values()), read_s)

    results = []
    lock = threading.Lock()

    def work(topic):
        codec = formats[topic].most_common(1)[0][0]
        if codec not in ("h264", "h265", "hevc"):
            raise RuntimeError(f"{topic}: unsupported video format {codec!r}")
        r = detect_channel(topic, packets[topic], times[topic], "hevc" if codec == "h265" else codec,
                           tools, args, detector_factory)
        with lock:
            results.append(r)
        LOG.info("  %-22s %4d frames, %3d checked, %2d with faces, %d intervals (%.0f fps)",
                 topic, r["frames_decoded"], r["frames_checked"], r["frames_with_faces"],
                 len(r["intervals_rel_s"]), r["fps"])

    if args.batch_channels and args.face_detector == "egoblur":
        codecs = {t: ("hevc" if formats[t].most_common(1)[0][0] == "h265" else formats[t].most_common(1)[0][0])
                  for t in packets}
        bad = {t: c for t, c in codecs.items() if c not in ("h264", "hevc")}
        if bad:
            raise RuntimeError(f"unsupported video formats: {bad}")
        det = detector_factory()[1]  # one model on the GPU, shared by every channel
        results = detect_channels_batched(sorted(packets), packets, times, codecs, tools, args, det.batch)
        for r in results:
            LOG.info("  %-22s %4d frames, %3d checked, %2d with faces, %d intervals",
                     r["topic"], r["frames_decoded"], r["frames_checked"], r["frames_with_faces"],
                     len(r["intervals_rel_s"]))
    else:
        threads = []
        for topic in packets:
            if args.workers <= 1:
                work(topic)
            else:
                th = threading.Thread(target=work, args=(topic,)); th.start(); threads.append(th)
                while sum(t.is_alive() for t in threads) >= args.workers:
                    time.sleep(0.05)
        for th in threads:
            th.join()
    results.sort(key=lambda r: r["topic"])

    # union across feeds on the shared log-time axis
    t_start = min(r["start_log_time"] for r in results)
    union_raw = sorted([[r["start_log_time"] - t_start + a, r["start_log_time"] - t_start + b]
                        for r in results for a, b in r["intervals_rel_s"]])
    union: list[list[float]] = []
    for a, b in union_raw:
        if union and a - union[-1][1] <= args.gap_s:
            union[-1][1] = max(union[-1][1], b)
        else:
            union.append([a, b])
    return {
        "file": str(path), "sha256": rv.sha256_file(path) if not args.no_sha256 else None,
        "bytes": path.stat().st_size, "status": "ok",
        "log_start_time": round(t_start, 6),
        "duration_s": round(max(r["end_log_time"] for r in results) - t_start, 3),
        "channels": results,
        "union_intervals_rel_s": [[round(a, 3), round(b, 3)] for a, b in union],
        "union_face_time_s": round(sum(b - a for a, b in union), 3),
        "frames_with_faces_total": sum(r["frames_with_faces"] for r in results),
        "timing": {"read_s": round(read_s, 2), "total_s": round(time.time() - t0, 2)},
    }


# --------------------------------------------------------------------------- cli
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, help=".mcap file or a folder of them")
    p.add_argument("-o", "--output", type=Path, default=None, help="JSON report (default <input>_faces.json)")
    p.add_argument("--topics", nargs="+", default=None, help="only these video topics")
    p.add_argument("--stride", type=int, default=1, help="detect on every Nth frame (all frames are still decoded)")
    p.add_argument("--workers", type=int, default=5, help="channels decoded in parallel")
    p.add_argument("--cv-threads", type=int, default=4,
                   help="OpenCV threads per channel; the default avoids oversubscription across workers")
    p.add_argument("--max-messages", type=int, default=None, help="debug: stop after N messages per channel")
    p.add_argument("--no-sha256", action="store_true")
    p.add_argument("--progress-every", type=int, default=2000,
                   help="log progress every N frame indices (0 disables). Without it a 30-minute "
                        "recording prints nothing for half an hour.")
    p.add_argument("--quiet-detections", action="store_true", help="omit per-frame boxes from the JSON")
    g = p.add_argument_group("detector")
    g.add_argument("--face-detector", choices=["yunet", "egoblur"], default="yunet")
    g.add_argument("--yunet-weights", type=Path,
                   default=Path(__file__).resolve().parent / "weights/face_detection_yunet_2023mar.onnx")
    g.add_argument("--egoblur-weights", type=Path, default=None)
    g.add_argument("--egoblur-gen", choices=["auto", "1", "2"], default="auto")
    g.add_argument("--scale", type=float, default=0.5, help="decode/detect at this fraction of native size")
    g.add_argument("--conf", type=float, default=None,
                   help="Score threshold; defaults per detector (yunet 0.85, egoblur 0.90). Both were calibrated "
                        "on IC-559 rig footage 2026-09-18/19. Lower it to favour recall (a missed face is worse "
                        "than a spurious one if you are redacting; the reverse if you are cutting footage).")
    g.add_argument("--nms", type=float, default=0.3)
    g.add_argument("--min-face-px", type=int, default=24, help="ignore faces whose short side is smaller (native px)")
    g.add_argument("--conf-topic", default="wrist:0.9",
                   help="per-camera score thresholds, SUBSTRING:VALUE comma separated")
    g.add_argument("--gap-s", type=float, default=1.0, help="detections closer than this merge into one interval")
    g.add_argument("--half", action="store_true",
                   help="fp16 inference (egoblur only). Faster, but detections near the score threshold shift; "
                        "compare against fp32 before trusting it for a redaction guarantee.")
    g.add_argument("--batch-channels", action="store_true", default=None,
                   help="decode all channels in lockstep and run one batched inference per frame index. "
                        "Default: on with --half, off without. Measured on a 4090: batching is 1.3x FASTER in "
                        "fp16 but 1.6x SLOWER in fp32, so it is not a blanket win.")
    g.add_argument("--no-batch-channels", dest="batch_channels", action="store_false")
    g.add_argument("--gpu-decode", action="store_true", help="use NVDEC (default: CPU decode, GPUs untouched)")
    g.add_argument("--device", default="0", help="GPU index for NVDEC / EgoBlur")
    g.add_argument("--ffmpeg", default=None)
    g.add_argument("--ffprobe", default=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cv2.setNumThreads(args.cv_threads)
    args.torch_dev, args.gpu_index = rv.parse_device(args.device)
    if args.batch_channels is None:
        args.batch_channels = args.half
    args.conf_topic_map = []
    for item in (args.conf_topic or "").split(","):
        if item.strip():
            pat, _, val = item.partition(":")
            args.conf_topic_map.append((pat.strip(), float(val)))
    if args.conf is None:
        args.conf = 0.90 if args.face_detector == "egoblur" else 0.85
        LOG.info("using default conf %.2f for %s", args.conf, args.face_detector)
    if args.conf_topic_map:
        args.conf = min([args.conf] + [v for _, v in args.conf_topic_map])
    if args.face_detector == "yunet" and not args.yunet_weights.is_file():
        sys.exit(f"error: YuNet weights not found at {args.yunet_weights}")
    if args.face_detector == "egoblur" and (args.egoblur_weights is None or not args.egoblur_weights.is_file()):
        sys.exit("error: --egoblur-weights required for --face-detector egoblur")
    files = sorted(args.input.glob("*.mcap")) if args.input.is_dir() else [args.input]
    if not files:
        sys.exit(f"error: no .mcap files in {args.input}")
    tools = rv.FFTools.discover(args.ffmpeg, args.ffprobe)
    LOG.info("%d file(s), detector=%s scale=%.2f stride=%d gpu_decode=%s",
             len(files), args.face_detector, args.scale, args.stride, args.gpu_decode)

    report = {"schema": "sparkpack-mcap-faces/1", "created_utc": rv.utc_now(),
              "script": Path(__file__).name, "script_version": SCRIPT_VERSION,
              "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "ffmpeg": tools.version, "opencv": cv2.__version__, "files": []}
    factory = lambda: make_detector(args)  # noqa: E731  (one detector instance per channel thread)
    t0 = time.time()
    for f in files:
        try:
            r = process_file(f, tools, args, factory)
        except Exception as e:  # noqa: BLE001
            LOG.error("%s FAILED: %s", f.name, e)
            LOG.debug(traceback.format_exc())
            r = {"file": str(f), "status": "error", "error": f"{type(e).__name__}: {e}"}
        if r.get("status") == "skipped":
            LOG.info("%s: no video channels, skipped", f.name)
        if args.quiet_detections and r.get("status") == "ok":
            for c in r["channels"]:
                c.pop("detections", None)
        report["files"].append(r)
    ok = [f for f in report["files"] if f["status"] == "ok"]
    skipped = [f for f in report["files"] if f["status"] == "skipped"]
    report["summary"] = {
        "files": len(report["files"]), "ok": len(ok), "skipped_no_video": len(skipped),
        "error": len(report["files"]) - len(ok) - len(skipped),
        "files_with_faces": sum(1 for f in ok if f["frames_with_faces_total"]),
        "frames_with_faces": sum(f["frames_with_faces_total"] for f in ok),
        "face_time_s": round(sum(f["union_face_time_s"] for f in ok), 2),
        "video_time_s": round(sum(f["duration_s"] for f in ok), 2),
        "wall_s": round(time.time() - t0, 1),
    }
    out = args.output or (args.input.with_name(args.input.stem + "_faces.json") if args.input.is_file()
                          else args.input / "mcap_faces.json")
    rv.atomic_write_json(out, report)

    print(f"\n{'file':<48} {'topic':<20} {'frames':>7} {'faces':>6} {'largest':>8}  intervals (s, relative)")
    for f in ok:
        for c in f["channels"]:
            iv = ", ".join(f"{a:.2f}-{b:.2f}" for a, b in c["intervals_rel_s"][:4])
            if len(c["intervals_rel_s"]) > 4:
                iv += f", +{len(c['intervals_rel_s']) - 4} more"
            print(f"{Path(f['file']).name[:47]:<48} {c['topic']:<20} {c['frames_decoded']:>7} "
                  f"{c['frames_with_faces']:>6} {c['largest_face_px']:>7}px  {iv or '-'}")
    print(f"\n{json.dumps(report['summary'])}\nreport: {out}")
    return 1 if report["summary"]["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
