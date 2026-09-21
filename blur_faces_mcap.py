#!/usr/bin/env python
"""Blur faces inside MCAP recordings, keeping every frame.

For each `foxglove.CompressedVideo` channel: decode, detect faces with EgoBlur, grow and temporally
smooth the mask, blur those pixels, and re-encode one h264 access unit per frame so the message stream
maps 1:1 onto the original. Log times, publish times, sequence numbers, frame_id and format are all
preserved, and every non-video channel is copied through untouched.

Unlike cutting, a false positive costs you a blurred bottle rather than several seconds of footage, so
this runs at a deliberately low score threshold. Nothing is removed: the output has the same duration,
the same frame count and the same message count as the input.

  python blur_faces_mcap.py SRC_DIR OUT_DIR --egoblur-weights weights/ego_blur_face_gen2.jit
  python blur_faces_mcap.py in.mcap out.mcap --conf 0.3 --dilate-px 24 --blur-block 16
"""
from __future__ import annotations

import argparse
import json
import logging
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import redact_videos as rv  # noqa: E402
from detect_faces_mcap import PacketDecoder, VIDEO_SCHEMA, parse_protobuf_fields, probe_stream, _feed  # noqa: E402

LOG = logging.getLogger("blur")
SCRIPT_VERSION = "1.0.0"


def encode_protobuf_fields(fields: dict) -> bytes:
    """Re-serialise length-delimited protobuf fields in field-number order.

    Every field of foxglove.CompressedVideo is length-delimited (timestamp message, frame_id string,
    data bytes, format string), so rebuilding the message is just tag + varint length + payload.
    """
    out = bytearray()
    for num in sorted(fields):
        payload = fields[num]
        out.append((num << 3) | 2)
        n = len(payload)
        while True:
            b = n & 0x7F
            n >>= 7
            out.append(b | (0x80 if n else 0))
            if not n:
                break
        out += payload
    return bytes(out)


class PacketEncoder:
    """Raw BGR frames in, one Annex-B h264 access unit per frame out.

    `-aud 1` makes ffmpeg emit an access unit delimiter before every frame, which is what lets the
    output stream be split back into per-frame packets. `-bf 0` removes B-frames so encode order
    matches input order, and NVENC repeats SPS/PPS at each IDR by itself, so every keyframe stays a
    valid entry point.
    """

    def __init__(self, tools: rv.FFTools, width: int, height: int, fps: float, gop: int,
                 codec: str, preset: str, cq: int, gpu_index):
        cmd = [tools.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps:.6f}",
               "-i", "pipe:0", "-an", "-c:v", codec]
        if codec.endswith("_nvenc"):
            cmd += ["-preset", preset, "-rc", "vbr", "-cq", str(cq), "-b:v", "0",
                    "-bf", "0", "-g", str(gop), "-forced-idr", "1", "-aud", "1"]
            if gpu_index is not None:
                cmd += ["-gpu", str(gpu_index)]
        else:
            cmd += ["-preset", "veryfast", "-crf", str(cq), "-bf", "0", "-g", str(gop),
                    "-x264-params", "aud=1:repeat-headers=1"]
        cmd += ["-pix_fmt", "yuv420p", "-f", "h264", "pipe:1"]
        self.cmd = cmd
        self.err = tempfile.NamedTemporaryFile(prefix="blurenc_", suffix=".log", delete=False)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.err, bufsize=0)
        self.buf = bytearray()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        while True:
            chunk = self.proc.stdout.read(1 << 20)
            if not chunk:
                break
            self.buf += chunk

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def finish(self) -> list:
        self.proc.stdin.close()
        self.reader.join()
        rc = self.proc.wait()
        self.err.close()
        msg = Path(self.err.name).read_text(errors="replace").strip()
        Path(self.err.name).unlink(missing_ok=True)
        if rc != 0:
            raise RuntimeError(f"encoder exited {rc}: {msg[-400:]}")
        data = bytes(self.buf)
        aud = b"\x00\x00\x00\x01\x09"
        units, i = [], data.find(aud)
        if i < 0:
            raise RuntimeError("encoder produced no access unit delimiters")
        while True:
            j = data.find(aud, i + 1)
            if j < 0:
                units.append(data[i:])
                break
            units.append(data[i:j])
            i = j
        return units


def topic_conf(topic: str, args) -> float:
    """Score threshold for one camera, honouring --conf-topic overrides (first substring match wins)."""
    for pat, val in args.conf_topic_map:
        if pat in topic:
            return val
    return args.conf


def blur_channel(topic: str, packets: list, codec: str, tools: rv.FFTools, args, detect_batch) -> tuple:
    """Decode -> detect -> blur -> re-encode, with the four stages overlapped.

    Run serially these stages take 27 fps while the slowest alone allows 43, because each idles waiting
    for the others. Three threads joined by bounded queues fix that: ffmpeg decode and encode are
    separate processes, and both torch and OpenCV release the GIL, so they genuinely run at once.

    `--stride N` detects on every Nth frame only. Skipped frames contribute an empty mask and are
    covered by the temporal hold, so the window must span the gap (checked in main()). Detection is the
    only bottleneck, so this scales the whole channel almost linearly until decode becomes the limit.
    """
    w, h = probe_stream(tools, packets[:8], codec)
    dec = PacketDecoder(tools, codec, w, h, 1.0, False, None)
    threading.Thread(target=_feed, args=(dec, packets), daemon=True).start()
    enc = PacketEncoder(tools, w, h, args.fps, args.gop, args.codec if args.codec in tools.encoders
                        else "libx264", args.nvenc_preset, args.cq, args.gpu_index)
    smoother = rv.TemporalSmoother(args.smooth_window, "max")
    conf = topic_conf(topic, args)
    n_expected = len(packets)
    t_start = time.time()
    stats = {"frames": 0, "frames_detected": 0, "frames_with_faces": 0, "boxes": 0, "max_box_px": 0}
    errors: list = []
    q_frames: queue.Queue = queue.Queue(maxsize=48)   # decoded frames
    q_boxes: queue.Queue = queue.Queue(maxsize=48)    # (idx, frame, boxes|None), strictly in order

    def decode_worker():
        try:
            i = 0
            while True:
                f = dec.read()
                if f is None:
                    break
                q_frames.put((i, f))
                i += 1
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
        finally:
            q_frames.put(None)

    def detect_worker():
        """Buffers a window of frames so that batched detections and skipped frames stay in order."""
        buf: list = []          # [(idx, frame, wants_detect)]
        pending = 0

        def flush():
            nonlocal pending
            if not buf:
                return
            todo = [f for _, f, want in buf if want]
            got = detect_batch(todo) if todo else []
            stats["frames_detected"] += len(todo)
            it = iter(got)
            for idx, frame, want in buf:
                q_boxes.put((idx, frame, next(it) if want else None))
            buf.clear()
            pending = 0

        try:
            while True:
                item = q_frames.get()
                if item is None:
                    break
                idx, frame = item
                want = args.stride <= 1 or idx % args.stride == 0
                buf.append((idx, frame, want))
                pending += want
                if pending >= args.batch:
                    flush()
            flush()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
        finally:
            q_boxes.put(None)

    q_out: queue.Queue = queue.Queue(maxsize=48)      # finished frames awaiting the encoder

    def mask_worker():
        """Build and hold the mask, then apply the fill. Kept off the encoder thread: with a stride the
        detector gets cheap and these two together become the limit if they share one thread."""
        try:
            while True:
                item = q_boxes.get()
                if item is None:
                    break
                idx, frame, boxes = item
                mask = np.zeros((h, w), np.uint8)
                if boxes is not None and len(boxes):
                    # The detector runs at the lowest threshold in play; each camera then applies its
                    # own. Wrist cameras mostly see gloved hands, which EgoBlur scores like faces.
                    keep = (np.minimum(boxes[:, 2] - boxes[:, 0],
                                       boxes[:, 3] - boxes[:, 1]) >= args.min_face_px) & (boxes[:, 4] >= conf)
                    boxes = boxes[keep]
                    if args.max_blue_frac < 1.0 and len(boxes):
                        boxes = rv.reject_coloured_boxes(frame, boxes, args.max_blue_frac)
                    if len(boxes):
                        stats["frames_with_faces"] += 1
                        stats["boxes"] += len(boxes)
                        stats["max_box_px"] = max(stats["max_box_px"],
                                                  int(max(min(b[2] - b[0], b[3] - b[1]) for b in boxes)))
                        rv.paint_faces(mask, boxes, args.face_shape)
                        mask = rv.dilate_fast(mask, args.dilate_px)
                for _, f2, sm in smoother.push(stats["frames"], frame, mask):
                    q_out.put(rv.apply_fill(f2, sm.view(np.uint8), args.fill, args.grey, args.blur_block)
                              if sm.any() else f2)
                stats["frames"] += 1
                # Long recordings run for many minutes per channel; without this the tool looks hung.
                if args.progress_every and stats["frames"] % args.progress_every == 0:
                    el = time.time() - t_start
                    pct = 100.0 * stats["frames"] / max(1, n_expected)
                    eta = (n_expected - stats["frames"]) / max(1e-6, stats["frames"] / el)
                    LOG.info("   %-22s %6d/%d frames (%4.1f%%) %5.1f fps, eta %4.1f min",
                             topic, stats["frames"], n_expected, pct, stats["frames"] / el, eta / 60)
            for _, f2, sm in smoother.flush():
                q_out.put(rv.apply_fill(f2, sm.view(np.uint8), args.fill, args.grey, args.blur_block)
                          if sm.any() else f2)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
        finally:
            q_out.put(None)

    dt = threading.Thread(target=decode_worker, daemon=True)
    df = threading.Thread(target=detect_worker, daemon=True)
    mw = threading.Thread(target=mask_worker, daemon=True)
    for t in (dt, df, mw):
        t.start()
    while True:                                   # this thread only feeds the encoder pipe
        frame = q_out.get()
        if frame is None:
            break
        enc.write(frame)
    for t in (dt, df, mw):
        t.join(timeout=30)
    err = dec.close()
    if err and "Broken pipe" not in err:
        LOG.warning("%s: decoder stderr: %s", topic, err[-200:])
    if errors:
        raise RuntimeError(f"{topic}: {errors[0]!r}") from errors[0]
    units = enc.finish()
    return units, stats


def balance_shard(srcs: list, i: int, n: int) -> list:
    """Split files across shards by SIZE, not by index.

    Modulo sharding splits the count, not the work: on this card it handed one GPU three short
    recordings and the other the two long ones, so the first sat idle for half the run. Longest-
    processing-time-first assignment (each file to whichever shard is currently lightest) keeps the
    cards finishing together. File size is a good proxy for frame count at a fixed resolution.
    """
    loads = [0] * n
    buckets: list = [[] for _ in range(n)]
    for f in sorted(srcs, key=lambda x: -x.stat().st_size):
        k = loads.index(min(loads))
        buckets[k].append(f)
        loads[k] += f.stat().st_size
    return sorted(buckets[i])


def process_file(src: Path, dst: Path, tools: rv.FFTools, args, detectors: list) -> dict:
    from mcap.reader import make_reader
    from mcap.writer import Writer

    t0 = time.time()
    with open(src, "rb") as fh:
        reader = make_reader(fh)
        summary = reader.get_summary()
        if summary is None:
            raise RuntimeError("no summary")
        video_topics = sorted(ch.topic for ch in summary.channels.values()
                              if ch.schema_id and summary.schemas[ch.schema_id].name == VIDEO_SCHEMA)
        if not video_topics:
            raise RuntimeError("no video channels")
        packets: dict = {t: [] for t in video_topics}
        fields: dict = {t: [] for t in video_topics}
        fmts: dict = {t: set() for t in video_topics}
        cutoff = None
        for schema, channel, msg in reader.iter_messages(topics=video_topics):
            f = parse_protobuf_fields(msg.data)
            if not f.get(3):
                continue
            if args.max_messages and len(packets[channel.topic]) >= args.max_messages:
                continue
            packets[channel.topic].append(f[3])
            fields[channel.topic].append(f)
            fmts[channel.topic].add(f.get(4, b"h264").decode())
            cutoff = msg.log_time if cutoff is None else max(cutoff, msg.log_time)
        if args.max_messages:
            # Trim to a common length so the excerpt stays frame-aligned across cameras.
            n = min(len(v) for v in packets.values())
            for t in packets:
                packets[t], fields[t] = packets[t][:n], fields[t][:n]
            cutoff = max(fields[t][-1] and 0 for t in fields) or cutoff

    # Channels are independent, so run several at once: with a detection stride the GPU has idle time
    # and the CPU stages (decode, mask, fill, encode) are what remain, which parallelise across cores.
    new_packets, chan_stats, errs = {}, {}, []
    lock = threading.Lock()

    def work(topic, detect_batch):
        try:
            codec = "hevc" if "h265" in fmts[topic] else "h264"
            units, st = blur_channel(topic, packets[topic], codec, tools, args, detect_batch)
            if len(units) != len(packets[topic]):
                raise RuntimeError(f"{topic}: re-encoded {len(units)} access units for "
                                   f"{len(packets[topic])} source frames")
            with lock:
                new_packets[topic] = units
                chan_stats[topic] = st
            LOG.info("   %-22s %4d frames, %3d with faces, %d boxes (largest %dpx)",
                     topic, st["frames"], st["frames_with_faces"], st["boxes"], st["max_box_px"])
        except BaseException as e:  # noqa: BLE001
            errs.append(e)

    running: list = []
    for n, topic in enumerate(video_topics):
        while sum(t.is_alive() for t in running) >= max(1, args.channel_workers):
            time.sleep(0.05)
        if errs:
            break
        # Round-robin the channels over the available GPUs. Sharding alone only splits work when there
        # are several files; a single long recording would otherwise leave every card but one idle.
        t = threading.Thread(target=work, args=(topic, detectors[n % len(detectors)].detect_batch))
        t.start()
        running.append(t)
    for t in running:
        t.join()
    if errs:
        raise RuntimeError(f"channel failed: {errs[0]!r}") from errs[0]

    # rewrite the log, swapping only the video payloads
    cursor = {t: 0 for t in video_topics}
    out_fh = open(dst, "wb")
    writer = Writer(out_fh)
    writer.start(profile="", library=f"sparkpack-blur-faces/{SCRIPT_VERSION}")
    smap, cmap = {}, {}
    n_msg = 0
    with open(src, "rb") as fh:
        for schema, channel, msg in make_reader(fh).iter_messages():
            key = (channel.topic, channel.message_encoding, schema.name if schema else "")
            if key not in cmap:
                sid = 0
                if schema is not None:
                    skey = (schema.name, schema.encoding, bytes(schema.data))
                    if skey not in smap:
                        smap[skey] = writer.register_schema(name=schema.name, encoding=schema.encoding,
                                                            data=schema.data)
                    sid = smap[skey]
                cmap[key] = writer.register_channel(topic=channel.topic,
                                                    message_encoding=channel.message_encoding,
                                                    schema_id=sid, metadata=dict(channel.metadata or {}))
            if args.max_messages:
                if channel.topic in new_packets and cursor[channel.topic] >= len(new_packets[channel.topic]):
                    continue                      # this camera's excerpt is complete
                if channel.topic not in new_packets and cutoff and msg.log_time > cutoff:
                    continue                      # trim the other channels to the same span
            data = msg.data
            if channel.topic in new_packets:
                i = cursor[channel.topic]
                f = dict(fields[channel.topic][i])
                f[3] = new_packets[channel.topic][i]
                data = encode_protobuf_fields(f)
                cursor[channel.topic] += 1
            writer.add_message(channel_id=cmap[key], log_time=msg.log_time, data=data,
                               publish_time=msg.publish_time, sequence=msg.sequence)
            n_msg += 1
    writer.finish()
    out_fh.close()
    return {"file": src.name, "status": "blurred", "messages": n_msg,
            "channels": {t: chan_stats[t] for t in video_topics},
            "frames_with_faces": sum(s["frames_with_faces"] for s in chan_stats.values()),
            "frames": sum(s["frames"] for s in chan_stats.values()),
            "src_bytes": src.stat().st_size, "out_bytes": dst.stat().st_size,
            "sha256": rv.sha256_file(dst), "seconds": round(time.time() - t0, 1)}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, help=".mcap file or folder")
    p.add_argument("output", type=Path, help="output .mcap, or folder when the input is a folder")
    g = p.add_argument_group("detector")
    g.add_argument("--egoblur-weights", type=Path,
                   default=Path(__file__).resolve().parent / "weights/ego_blur_face_gen2.jit")
    g.add_argument("--egoblur-gen", choices=["auto", "1", "2"], default="auto")
    g.add_argument("--conf", type=float, default=0.5,
                   help="score threshold. Lower than the cutting default on purpose: a false positive "
                        "here blurs a bottle instead of deleting seconds of footage.")
    g.add_argument("--conf-topic", default="wrist:0.9",
                   help="per-camera thresholds as SUBSTRING:VALUE pairs, comma separated. Default "
                        "'wrist:0.9': measured on IC-559 footage, wrist cameras see gloved hands that "
                        "EgoBlur scores 0.3-0.87 like faces, while genuine faces on the head and chest "
                        "cameras sit at 0.75-1.0. Pass '' to use one threshold everywhere.")
    g.add_argument("--max-blue-frac", type=float, default=1.0,
                   help="OFF by default (1.0). Rejects detections more than this fraction blue. It cleanly "
                        "separated blue nitrile gloves from faces on IC-559 footage, but it is overfit to "
                        "that glove colour and would misfire on other gloves or a blue object near a face, "
                        "so it is opt-in only.")
    g.add_argument("--nms", type=float, default=0.3)
    g.add_argument("--min-face-px", type=int, default=16)
    g.add_argument("--no-half", dest="half", action="store_false", default=True)
    g.add_argument("--batch", type=int, default=5, help="frames per inference call")
    p.add_argument("--progress-every", type=int, default=2000,
                   help="log per-channel progress every N frames (0 disables). A 30-minute recording is "
                        "~59k frames per channel, so without this the tool prints nothing for many minutes.")
    g.add_argument("--channel-workers", type=int, default=3,
                   help="video channels processed concurrently. Each runs its own 4-stage pipeline, so "
                        "this is the lever once a stride has taken detection off the critical path.")
    g.add_argument("--stride", type=int, default=1,
                   help="detect on every Nth frame; skipped frames are covered by the temporal hold, so "
                        "--smooth-window must be larger than 2*stride. Detection is the only bottleneck, "
                        "so this is the main throughput lever.")
    g.add_argument("--device", default="0",
                   help="GPU index, or a comma-separated list like 0,1 to spread this file's camera "
                        "channels across several cards (one detector is loaded per card)")
    gg = p.add_argument_group("mask and fill")
    gg.add_argument("--dilate-px", type=int, default=20, help="grow each face mask by this many pixels")
    gg.add_argument("--smooth-window", type=int, default=9, help="odd temporal hold window in frames")
    gg.add_argument("--face-shape", choices=["rect", "ellipse"], default="ellipse")
    gg.add_argument("--fill", choices=["blur", "pixelate", "grey"], default="blur")
    gg.add_argument("--blur-block", type=int, default=20)
    gg.add_argument("--grey", type=int, default=128)
    ge = p.add_argument_group("encoding")
    ge.add_argument("--codec", default="libx264",
                   help="libx264 by default, not NVENC: GeForce cards cap concurrent NVENC sessions "
                        "(exceeding it kills encoders with a broken pipe once several channels run at "
                        "once), and at 1920x1200 libx264 measured 218 fps against NVENC's 206.")
    ge.add_argument("--nvenc-preset", default="p5")
    ge.add_argument("--cq", type=int, default=21)
    ge.add_argument("--gop", type=int, default=30, help="keyframe interval; match the source (30 here)")
    ge.add_argument("--fps", type=float, default=30.0)
    ge.add_argument("--ffmpeg", default=None)
    ge.add_argument("--ffprobe", default=None)
    p.add_argument("--max-messages", type=int, default=None,
                   help="only process the first N video frames per camera, writing a short excerpt. "
                        "3600 is 2 minutes at 30 fps - use it to try settings without a 25-minute run.")
    p.add_argument("--shard", nargs=2, type=int, metavar=("I", "N"), default=None,
                   help="process only files with index %% N == I, so one shard can run per GPU. Each shard "
                        "writes its own manifest.")
    p.add_argument("--no-copy-unreadable", dest="copy_unreadable", action="store_false", default=True,
                   help="fail on MCAPs that cannot be parsed instead of copying them through unscanned")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    devs = [d.strip() for d in str(args.device).split(",") if d.strip()]
    parsed = [rv.parse_device(d) for d in devs]
    torch_dev, args.gpu_index = parsed[0]
    if not args.egoblur_weights.is_file():
        sys.exit(f"error: EgoBlur weights not found at {args.egoblur_weights}")
    args.conf_topic_map = []
    for item in (args.conf_topic or "").split(","):
        if not item.strip():
            continue
        pat, _, val = item.partition(":")
        if not val:
            sys.exit(f"error: --conf-topic entry {item!r} must look like wrist:0.9")
        args.conf_topic_map.append((pat.strip(), float(val)))
    # run the detector at the loosest threshold in play; per-camera filtering happens downstream
    args.conf = min([args.conf] + [v for _, v in args.conf_topic_map]) if args.conf_topic_map else args.conf
    if args.stride > 1 and args.smooth_window <= 2 * args.stride:
        sys.exit(f"error: --smooth-window {args.smooth_window} cannot cover --stride {args.stride}; "
                 f"use at least {2 * args.stride + 1} so every skipped frame sits inside a held mask")
    tools = rv.FFTools.discover(args.ffmpeg, args.ffprobe)
    detectors = [rv.FaceDetector(args.egoblur_weights, td, args.egoblur_gen, args.conf, args.nms,
                                 1.0, half=args.half) for td, _ in parsed]
    LOG.info("EgoBlur gen%d on %s (conf %.2f%s, fp16=%s), fill=%s dilate=%dpx window=%d stride=%d",
             detectors[0].gen, ", ".join(d for d, _ in parsed), args.conf,
             "".join(f", {p}:{v}" for p, v in args.conf_topic_map), args.half, args.fill,
             args.dilate_px, args.smooth_window, args.stride)

    if args.input.is_dir():
        srcs = sorted(args.input.glob("*.mcap"))
        if args.shard:
            i, n = args.shard
            srcs = balance_shard(srcs, i, n)
            LOG.info("shard %d of %d: %d files, %.1f GB", i, n, len(srcs),
                     sum(f.stat().st_size for f in srcs) / 1e9)
        args.output.mkdir(parents=True, exist_ok=True)
        pairs = [(s, args.output / s.name) for s in srcs]
    else:
        pairs = [(args.input, args.output)]
        args.output.parent.mkdir(parents=True, exist_ok=True)
    if not pairs:
        LOG.error("no .mcap files in %s", args.input)
        return 2

    records, rc = [], 0
    for src, dst in pairs:
        try:
            LOG.info("%s", src.name)
            records.append(process_file(src, dst, tools, args, detectors))
            r = records[-1]
            LOG.info("   -> %s (%d msgs, %d/%d frames blurred, %.0fs)", dst.name, r["messages"],
                     r["frames_with_faces"], r["frames"], r["seconds"])
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "no video channels" in msg or "no summary" in msg:
                shutil.copy2(src, dst)          # nothing to blur: pass the file through untouched
                records.append({"file": src.name, "status": "copied", "reason": msg})
                LOG.info("   no video, copied through")
            elif args.copy_unreadable and "channel failed" not in msg and "access units" not in msg:
                # Unparseable logs (the card's non-final system_logs are truncated writes). They are
                # copied so the output folder stays complete, but they were NEVER SCANNED - the manifest
                # says so, and nothing here certifies them face-free.
                shutil.copy2(src, dst)
                records.append({"file": src.name, "status": "copied_unscanned", "reason": msg[:160]})
                LOG.warning("   UNPARSEABLE, copied through unscanned: %s", msg[:90])
            else:
                # A file that failed mid-blur must NOT be left in the output: an unblurred copy sitting
                # in the destination looks identical to a processed one.
                dst.unlink(missing_ok=True)
                LOG.error("   FAILED (no output written): %s", e)
                records.append({"file": src.name, "status": "error", "error": f"{type(e).__name__}: {e}"})
                rc = 1
    if args.output.is_dir():
        man = args.output / ("blur_manifest.json" if not args.shard
                             else f"blur_manifest_shard{args.shard[0]}of{args.shard[1]}.json")
        rv.atomic_write_json(man, {
            "schema": "sparkpack-face-blur/1", "created_utc": rv.utc_now(),
            "script": Path(__file__).name, "script_version": SCRIPT_VERSION,
            "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "files": records,
            "summary": {"files": len(records),
                        "blurred": sum(1 for r in records if r["status"] == "blurred"),
                        "copied": sum(1 for r in records if r["status"] == "copied"),
                        "copied_unscanned": sum(1 for r in records if r["status"] == "copied_unscanned"),
                        "error": sum(1 for r in records if r["status"] == "error"),
                        "frames_with_faces": sum(r.get("frames_with_faces", 0) for r in records)},
        })
        LOG.info("manifest: %s", man)
    return rc


if __name__ == "__main__":
    sys.exit(main())
