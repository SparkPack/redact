# redact

Face blurring or cutting for MCAP clips taken by the InstaWork Lens rig.

Removes faces from multi-camera robot-rig recordings, in whichever of two ways suits the job:

| mode | keeps | loses | use when |
|---|---|---|---|
| **blur** | every frame, every message | the face pixels | you need all the footage |
| **cut** | full image fidelity | the time around each face | you need untouched pixels |

Everything runs directly on `.mcap` logs. Video is never extracted to files and re-imported: frames are
decoded from the log, processed, and written back, so timestamps, sequence numbers and all non-video
channels (IMU, thermal, logs) survive intact.

```bash
source ~/sparkpack/.venv-redact/bin/activate
```

## The two commands

```bash
python redact_mcap.py SRC OUT --mode blur --conf 0.95
```

```bash
python redact_mcap.py SRC OUT --mode cut --conf 0.95 --min-keep-s 30
```

`SRC` is an `.mcap` file or a folder of them. In blur mode `OUT` gets one blurred copy per input, same
filename. In cut mode it gets `<recording>_clipNN.mcap` for each face-free stretch, plus a manifest.

## Setting `--conf`, the one number that matters

**Do not reuse a threshold from another rig.** Calibrate it, using footage you know contains no faces:

1. Run detection over a stretch you have watched and know is clean.
2. Raise `--conf` until it reports zero detections there.

Measured on IC-559 footage over four face-free minutes:

| `--conf` | real faces kept | false positives |
|---|---|---|
| 0.50 | 367 | 250 |
| 0.70 | 257 | 59 |
| 0.90 | 168 | 6 |
| **0.95** | **132** | **0** |

0.90 is not enough. Gloved hands score as high as 0.942.

```bash
python detect_faces_mcap.py CLEAN.mcap --face-detector egoblur \
  --egoblur-weights weights/ego_blur_face_gen2.jit --scale 1.0 --half \
  --conf 0.3 --conf-topic '' --stride 6 --device 1 -o calib.json
```

Then count what survives each threshold in `calib.json`.

### `--conf-topic`

Per-camera overrides, default `wrist:0.99`. Wrist cameras point at the work surface and almost never
see a face, so they carry a stricter bar. This is rig geometry, not a property of any one recording.

## Cut mode: `--min-keep-s`

The minimum length of a clip worth keeping. Anything shorter is discarded rather than kept as an
unusable fragment, and this is what turns a scatter of small cuts into a few long clips.

Measured over six minutes of IC-559 footage:

| `--min-keep-s` | clips | result |
|---|---|---|
| none | 9 | 0-9 s, 11-12 s, 16-21 s … slivers |
| 10 | 2 | a 12 s fragment plus the main run |
| **30** | **1** | one unbroken run from 86 s on |

At 30 s the merge distance stops mattering, because the minimum length decides everything. One knob
instead of two.

## Blur mode

`--dilate-px 30` grows each mask, `--smooth-window 15` holds it across frames so a face is covered
between detections, `--blur-block 28` sets coarseness. `--fill grey` if the blur must be irreversible.

**A face detector is not a valid check on blurred output.** After blurring, EgoBlur still fires at
0.85-0.92 on the blurred patch itself, because a smeared face keeps two dark eye-blobs on a lighter
ground. Lowering the threshold makes residual "detections" go *up*. Judge blurred output by eye, or
with a face *recognition* model. Detector-on-output only proves anything for cut mode, where the
pixels are genuinely gone.

## Reviewing the result

```bash
python mcap_to_mosaic.py OUT/recording.mcap preview.mp4 --device 1
vlc --avcodec-hw=none preview.mp4
```

All cameras tiled with labels and a timestamp. `--every 6` gives a 6x shorter timelapse.

Two traps. The mosaic scales each camera down by three, so a 50 px face becomes ~17 px and is easy to
miss — do not conclude a detection was spurious without checking the source at full resolution. And
play with `--avcodec-hw=none`: VLC's GPU decoding raised an Xid fault on GPU 0 that took the whole
machine down.

## Hardware notes

**GPU 0 drives the display.** Heavy CUDA there while video plays froze this machine. Default long jobs
to `--device 1`. `--device 0,1` spreads one recording's channels across both cards when nothing else
needs the display.

Throughput on two RTX 4090s, measured on a 32.7-minute five-camera recording:

| stage | rate |
|---|---|
| detection, stride 3 | ~230 fps (1.5x real time) |
| blur, stride 3, both GPUs | 189 fps (1.3x real time) |
| blur, one GPU | 154 fps (1.03x real time) |

A 32.7-minute recording blurs in ~24 minutes. Peak RAM ~34 GB, since all video packets are held in
memory — an 80-minute recording would need ~90 GB.

For several recordings, shard by file across the cards (balanced by size, not count):

```bash
python blur_faces_mcap.py SRC OUT --shard 0 2 --device 0 &
python blur_faces_mcap.py SRC OUT --shard 1 2 --device 1
```

## Weights

`weights/ego_blur_face_gen2.jit` (400 MB) is a gated download from
[projectaria.com/tools/egoblur](https://www.projectaria.com/tools/egoblur) behind an email form,
Apache 2.0. EgoBlur is trained on egocentric wearable-camera video, which is what a body-worn rig
produces, so it substantially beats general-purpose detectors here. YuNet
(`face_detection_yunet_2023mar.onnx`) is a CPU fallback that is markedly less accurate on this footage.

## Other tools

| file | purpose |
|---|---|
| `detect_faces_mcap.py` | detection only; writes per-frame boxes and intervals |
| `plan_face_cuts.py` | sweeps cut strategies, reports clip counts and lengths |
| `blur_faces_mcap.py` | the blur worker, with all its knobs |
| `mcap_to_mosaic.py` | multi-camera review video |
| `redact_videos.py` | shared library: detectors, masks, fills, ffmpeg |
| `tests/test_pipeline.py` | checks that need no gated weights |
| `README_mp4_pipeline.md` | the earlier MP4-based pipeline (SAM 3 objects, OCR) |

## Rebuilding the environment

```bash
./setup_venv.sh
```

Three traps it handles: PaddlePaddle downgrades torch's NVIDIA libraries and breaks it, uv gives
`--extra-index-url` priority over `--index-url`, and ultralytics pip-installs CLIP and timm at runtime
which fails in a uv venv. Also note libx264 is the default encoder rather than NVENC: GeForce cards cap
concurrent NVENC sessions, and at 1920x1200 libx264 measured slightly faster anyway.
