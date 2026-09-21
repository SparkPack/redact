#!/usr/bin/env python
"""Render every camera in an MCAP recording into one video, tiled side by side.

Each `foxglove.CompressedVideo` channel is decoded straight from the log, the feeds are placed on a grid
with their topic captioned, and the result is encoded with NVENC (falling back to libx264). Feeds are
aligned by MCAP log time rather than frame order, so a channel that drops a frame stays in sync instead
of drifting; a feed with no frame yet at a given instant holds its previous frame.

  python mcap_to_mosaic.py FILE.mcap out.mp4
  python mcap_to_mosaic.py FILE.mcap out.mp4 --cols 3 --tile-width 640 --fps 30
  python mcap_to_mosaic.py face_cuts/ out_dir/            # a folder of logs -> one mp4 each
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import redact_videos as rv  # noqa: E402
from detect_faces_mcap import PacketDecoder, VIDEO_SCHEMA, parse_protobuf_fields, probe_stream  # noqa: E402

LOG = logging.getLogger("mosaic")


def label(img: np.ndarray, text: str, sub: str = "") -> np.ndarray:
    """Caption a tile with a translucent band so the text stays readable over any footage."""
    h, w = img.shape[:2]
    band = max(24, h // 12)
    strip = img[:band].astype(np.float32) * 0.35
    img[:band] = strip.astype(np.uint8)
    scale = band / 34.0
    cv2.putText(img, text, (8, int(band * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255),
                max(1, int(scale * 2)), cv2.LINE_AA)
    if sub:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, int(scale * 2)))
        cv2.putText(img, sub, (16 + tw, int(band * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.8,
                    (170, 220, 255), max(1, int(scale * 1.5)), cv2.LINE_AA)
    return img


def render(path: Path, out: Path, args, tools: rv.FFTools) -> dict:
    from mcap.reader import make_reader

    with open(path, "rb") as fh:
        reader = make_reader(fh)
        summary = reader.get_summary()
        if summary is None:
            raise RuntimeError("no summary section")
        topics = sorted(ch.topic for ch in summary.channels.values()
                        if ch.schema_id and summary.schemas[ch.schema_id].name == VIDEO_SCHEMA)
        if args.topics:
            topics = [t for t in topics if t in args.topics]
        if not topics:
            raise RuntimeError(f"no {VIDEO_SCHEMA} channels")
        packets: dict = {t: [] for t in topics}
        times: dict = {t: [] for t in topics}
        fmts: dict = {t: set() for t in topics}
        for schema, channel, msg in reader.iter_messages(topics=topics):
            f = parse_protobuf_fields(msg.data)
            if not f.get(3):
                continue
            packets[channel.topic].append(f[3])
            times[channel.topic].append(msg.log_time / 1e9)
            fmts[channel.topic].add(f.get(4, b"h264").decode())

    t0 = min(v[0] for v in times.values() if v)
    t1 = max(v[-1] for v in times.values() if v)
    fps = args.fps or round(max((len(v) - 1) / (v[-1] - v[0]) for v in times.values() if len(v) > 1), 3)
    n_out = max(1, int(round((t1 - t0) * fps)) + 1)
    cols = args.cols or (3 if len(topics) > 4 else 2)
    rows = (len(topics) + cols - 1) // cols

    decs, dims = {}, {}
    for t in topics:
        codec = "hevc" if "h265" in fmts[t] else "h264"
        try:
            w, h = probe_stream(tools, packets[t][:8], codec)
        except Exception as e:  # noqa: BLE001
            # Usually means this channel's excerpt starts mid-GOP, so ffmpeg cannot open it.
            raise RuntimeError(f"{t}: cannot determine frame size - does this clip start on a "
                               f"keyframe? ({e})") from e
        if not w or not h:
            raise RuntimeError(f"{t}: ffmpeg reported a {w}x{h} frame; the clip likely starts mid-GOP")
        dims[t] = (w, h)
        decs[t] = PacketDecoder(tools, codec, w, h, 1.0, False, None)
        threading.Thread(target=_feed, args=(decs[t], packets[t]), daemon=True).start()

    tw = args.tile_width
    th = int(round(tw * dims[topics[0]][1] / dims[topics[0]][0])) // 2 * 2
    W, H = tw * cols, th * rows
    LOG.info("%s: %d feeds, %dx%d grid, %dx%d output, %.3f fps, %d frames",
             path.name, len(topics), cols, rows, W, H, fps, n_out)

    # Keep the SOURCE frame rate on the output so --every N yields an N-times-SHORTER video that plays
    # at speed, like a timelapse. Dividing the rate instead would keep the full duration and just make
    # playback choppy, which is the opposite of what a quick review needs.
    writer = rv.FrameWriter(tools, out, W, H, f"{fps:.6f}",
                            args.codec if args.codec in tools.encoders else "libx264",
                            args.nvenc_preset, args.cq, args.gpu_index)
    cur = {t: None for t in topics}       # most recent decoded frame per feed
    nxt_idx = {t: 0 for t in topics}      # next message index awaiting its turn
    blank = np.full((th, tw, 3), 24, np.uint8)
    written = 0
    try:
        for k in range(n_out):
            stamp = t0 + k / fps
            for t in topics:
                # pull every frame whose log time has arrived; the last one wins
                while nxt_idx[t] < len(times[t]) and times[t][nxt_idx[t]] <= stamp + 0.5 / fps:
                    fr = decs[t].read()
                    if fr is None:
                        nxt_idx[t] = len(times[t])
                        break
                    cur[t] = fr
                    nxt_idx[t] += 1
            if args.every > 1 and k % args.every:
                continue          # frame decoded and discarded: keeps the feeds in sync, shortens output
            tiles = []
            for t in topics:
                if cur[t] is None:
                    tile = blank.copy()
                else:
                    tile = cv2.resize(cur[t], (tw, th), interpolation=cv2.INTER_AREA)
                name = t.strip("/").replace("/video", "")
                tiles.append(label(tile, name, f"{stamp - t0:6.2f}s" if t == topics[0] else ""))
            while len(tiles) < rows * cols:
                tiles.append(blank.copy())
            grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
            writer.write(grid)
            written += 1
    finally:
        for t in topics:
            decs[t].close()
    writer.close()
    info = rv.ffprobe_video(tools, out)
    return {"source": str(path), "output": str(out), "feeds": topics, "grid": [cols, rows],
            "size": [W, H], "fps": fps, "frames": written, "duration_s": round(t1 - t0, 3),
            "bytes": out.stat().st_size, "sha256": rv.sha256_file(out), "codec": info.codec}


def _feed(dec: PacketDecoder, packets: list) -> None:
    try:
        for p in packets:
            dec.write(p)
    finally:
        dec.finish_input()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help=".mcap file or a folder of them")
    ap.add_argument("output", type=Path, help="output .mp4, or a folder when the input is a folder")
    ap.add_argument("--topics", nargs="+", default=None, help="only these video topics")
    ap.add_argument("--cols", type=int, default=None, help="grid columns (default 3 for 5+ feeds, else 2)")
    ap.add_argument("--tile-width", type=int, default=640, help="width of each tile in pixels")
    ap.add_argument("--fps", type=float, default=None, help="output frame rate (default: the source rate)")
    ap.add_argument("--every", type=int, default=1,
                    help="keep every Nth output frame: an N-times-shorter timelapse for quick review. "
                         "A 33-minute recording at --every 6 becomes a 5.5-minute video. Note this speeds "
                         "RENDERING only modestly (~35%%), because every source frame must still be decoded "
                         "for h264 inter-frame dependencies; it mainly saves your viewing time.")
    ap.add_argument("--codec", default="h264_nvenc", help="h264_nvenc, hevc_nvenc or libx264")
    ap.add_argument("--nvenc-preset", default="p5")
    ap.add_argument("--cq", type=int, default=23)
    ap.add_argument("--device", default="0", help="GPU index for NVENC")
    ap.add_argument("--ffmpeg", default=None)
    ap.add_argument("--ffprobe", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _, args.gpu_index = rv.parse_device(args.device)
    tools = rv.FFTools.discover(args.ffmpeg, args.ffprobe)

    if not args.input.exists():
        sys.exit(f"error: {args.input} does not exist. Pass a real .mcap file or a folder of them, "
                 f"e.g. ~/cocoapack/card_1/cut/removed")
    if args.input.is_dir():
        files = sorted(args.input.glob("*.mcap"))
        args.output.mkdir(parents=True, exist_ok=True)
        outs = [(f, args.output / (f.stem + ".mp4")) for f in files]
    else:
        files = [args.input]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        outs = [(args.input, args.output)]
    if not files:
        LOG.error("no .mcap files in %s", args.input)
        return 2

    rc = 0
    for src, dst in outs:
        try:
            r = render(src, dst, args, tools)
            LOG.info("  wrote %s (%d frames, %.1f MB)", dst, r["frames"], r["bytes"] / 1e6)
        except Exception as e:  # noqa: BLE001
            LOG.error("  %s FAILED: %s", src.name, e)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
