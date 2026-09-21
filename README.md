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

## Getting started on a fresh Ubuntu box

Tested on Ubuntu 22.04 with two RTX 4090s. You need an NVIDIA GPU with ~8 GB free, driver 560 or
newer (`nvidia-smi` should report a CUDA version of 12.6+), and about 10 GB of disk for the venv.

**1. Install uv**, the Python package manager the setup script uses:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Clone and build the environment.** This creates `.venv-redact` alongside the checkout, installs
CUDA PyTorch, ultralytics, the mcap libraries and a static ffmpeg with NVENC/NVDEC, and verifies them.
Takes about 10 minutes and downloads several GB:

```bash
git clone https://github.com/SparkPack/redact.git
cd redact && ./setup_venv.sh
```

**3. Get the EgoBlur face model.** It is a gated download, Apache 2.0 licensed, and the pipeline does
nothing without it:

  1. Go to [projectaria.com/tools/egoblur](https://www.projectaria.com/tools/egoblur) and find
     "Access the models"
  2. Enter your email and accept the licence. A download link arrives by email, usually in minutes
  3. The zip contains `ego_blur_face_gen2.jit` (400 MB)

```bash
mkdir -p weights && unzip -d weights ~/Downloads/ego_blur_face_gen2.zip
```

There are two model generations. Gen2 is the default here; both are Apache 2.0. The generation is
inferred from the filename, so keep the name as shipped.

**4. Activate and check it works:**

```bash
source ../.venv-redact/bin/activate
python redact_mcap.py --help
```

**5. Calibrate `--conf` on your own footage** before trusting any output. See below; do not reuse the
0.95 from this repo.

### Troubleshooting

| symptom | cause |
|---|---|
| `Unknown builtin op: torchvision::nms` | torchvision must be imported before `torch.jit.load`; handled in the code, but means the venv is broken if you see it |
| `undefined symbol: ncclCommGrow` | something reinstalled PaddlePaddle and downgraded torch's NVIDIA libraries. Re-run step 4 of `setup_venv.sh` |
| `no foxglove.CompressedVideo channels` | the log has no video, or the writer used a different schema |
| whole machine freezes | CUDA on the display GPU. Use `--device 1`; see Hardware notes |

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

## Other tools

| file | purpose |
|---|---|
| `detect_faces_mcap.py` | detection only; writes per-frame boxes and face intervals |
| `plan_face_cuts.py` | sweeps cut strategies, reports clip counts and lengths |
| `blur_faces_mcap.py` | the blur worker, with all its knobs |
| `mcap_to_mosaic.py` | multi-camera review video |
| `redact_videos.py` | shared library: ffmpeg wrappers, EgoBlur, masks, fills |
| `tests/test_pipeline.py` | unit checks that need no weights or footage |

## Notes on the environment

`setup_venv.sh` is idempotent; re-run it to rebuild. The dependency list is deliberately small: torch,
torchvision, opencv, numpy and the mcap libraries. Two things worth knowing:

- **torchvision must be imported before `torch.jit.load`**, or the EgoBlur export dies with
  "Unknown builtin op: torchvision::nms". The library does this; don't reorder it.
- **libx264 is the default encoder, not NVENC.** GeForce cards cap concurrent NVENC sessions, and
  exceeding it kills encoders mid-run with a broken pipe once several channels encode at once. At
  1920x1200 libx264 measured slightly faster anyway (218 vs 206 fps).
