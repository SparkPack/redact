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
| `mcap.exceptions.EndOfFile` | the source is truncated. Re-copy it; if its size is an exact multiple of 1 MiB the transfer was interrupted |
| `RecordLengthLimitExceeded` | the rig never finalised the file. Re-copying will not help |
| `N file(s) were NOT cut` + exit 1 | a source could not be read, so nothing was written for it. Never ignore this |
| `ffmpeg reported a 0x0 frame` | a clip whose feed starts before its own first keyframe. Handled in `mcap_to_mosaic.py`; if you see it, the file is genuinely undecodable |
| `bad --device '0,1'` | cut mode takes one GPU index. See Hardware notes for running two GPUs |

## The commands InstaWork should run

These are the exact settings used to redact the IC-559 cards, with every non-default spelled out.
Copy them as-is; the rest of this README explains why each number is what it is.

**Cut mode** — removes the time around each face, leaves every surviving pixel untouched:

```bash
python redact_mcap.py RECORDING.mcap OUT --mode cut \
  --conf 0.95 --conf-topic wrist:0.99 \
  --min-keep-s 30 --pad-s 1.0 --merge-s 20 --stride 3 --device 1
```

**Blur mode** — keeps every frame, blurs the face pixels:

```bash
python redact_mcap.py RECORDING.mcap OUT --mode blur \
  --conf 0.95 --conf-topic wrist:0.99 --stride 3 --device 1
```

### Defaults, and the two you must type anyway

| flag | default | used on IC-559 | what it does |
|---|---|---|---|
| `--conf` | 0.95 | 0.95 | face score threshold |
| `--conf-topic` | `wrist:0.99` | `wrist:0.99` | per-camera override |
| `--min-keep-s` | 30 | 30 | shortest clip, and shortest recording, worth keeping |
| `--pad-s` | 1.0 | 1.0 | margin cut either side of a face |
| `--merge-s` | 0.0 | 20 (no effect) | joins faces closer than this; redundant when `--min-keep-s` is larger |
| `--stride` | 3 | 3 | detect every Nth frame |
| `--device` | 1 | 1 | GPU index |
| `--min-removed-s` | 0.5 | 0.5 | shortest removed span written for review |

**`--merge-s` is redundant at these settings — you can leave it out.** It joins faces closer together
than its value into a single cut, but it can only absorb gaps shorter than itself, and any gap that
short has already been discarded by `--min-keep-s 30`. Measured on a 32.7-minute recording:

| `--min-keep-s` | `--merge-s` | clips | kept |
|---|---|---|---|
| 30 | 0 | 8 | 91.53% |
| 30 | 20 | 8 | 91.53% |
| 30 | 60 | 8 | 91.53% |
| 0 | 0 | 21 | 96.62% |
| 0 | 20 | 10 | 92.36% |

Identical wherever `--merge-s` is at or below `--min-keep-s`. It matters only if you lower the
minimum: at `--min-keep-s 0` it takes 21 clips down to 10. The IC-559 runs passed `--merge-s 20` and
it changed nothing.

`--device` defaults to 1 deliberately, not 0 — see Hardware notes.

### Running a folder of recordings

`SRC` may be a folder, but **folder mode globs every `.mcap` in it**, including the rig's
`system_logs_*.mcap`. Ten of the twenty-five on card 1 are unfinalised writes that cannot be parsed,
and one of them will stop the run. Loop over the video files instead:

```bash
for f in source/*.mcap; do
  case "$(basename "$f")" in system_logs_*) continue;; esac
  python redact_mcap.py "$f" OUT --mode cut \
    --conf 0.95 --conf-topic wrist:0.99 \
    --min-keep-s 30 --pad-s 1.0 --merge-s 20 --stride 3 --device 1 || echo "FAILED: $f"
done
```

**Check the exit code, not the log.** A recording that cannot be read is reported and the command
exits non-zero; a loop that ignores that will happily print nothing and leave you believing footage
was redacted when it was never opened.

### Check your sources before you start

A recording truncated in transfer, or never finalised by the rig, cannot be cut — and a 20 GB file
takes an hour to fail. Verify footers first:

```bash
python - <<'EOF'
import glob, sys
from mcap.reader import make_reader
for p in sorted(glob.glob('source/*.mcap')):
    if 'system_logs' in p: continue
    try:
        with open(p, 'rb') as fh:
            s = make_reader(fh).get_summary()
        print(f"ok      {p}  {(s.statistics.message_end_time - s.statistics.message_start_time)/6e10:.1f} min")
    except Exception as e:
        print(f"BROKEN  {p}  {type(e).__name__}")
EOF
```

Two failures look different and mean different things. `EndOfFile` means the file is truncated —
re-copy it, and check its size is not an exact multiple of 1 MiB, which is the signature of an
interrupted transfer. `RecordLengthLimitExceeded` means the rig never closed the file; re-copying
will not help and the footage is only recoverable from the rig itself.

### What comes out

```
OUT/
  <recording>_clip00.mcap        each face-free stretch, in order
  <recording>_clip01.mcap
  removed/
    <recording>_removed00.mcap   what was cut, for review
  cut_manifest.json              every clip with times, sizes and SHA-256
```

`cut_manifest.json` also carries `skipped_too_short`: recordings below `--min-keep-s` that produced
no output. They are listed rather than dropped silently, so a recording that leaves no files can be
told apart from one that was never processed.

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

**0.95 is a floor, not a guarantee, and there is no threshold that separates cleanly.** Measured over
204 detections on a 32.7-minute recording:

| detection | score | what it is |
|---|---|---|
| blue gloved hand on a dark bar | 0.951 | false |
| white plastic clip with a cable | 0.952 | false |
| bearded man, sharp, plainly identifiable | **0.957** | **real** |

Six thousandths separate the false positives from a face you could put a name to. Raising the bar to
0.96 to kill the gloves deletes that man as well, and going to 0.99 recovers only 2.4% more footage
while dropping most real faces. Accept the false positives — on IC-559 not one of them was the sole
reason for a cut, because every span they appeared in also held a genuine face.

**Do not try to filter by blur or face size either.** Both look reasonable and both fail. The false
positives are *sharp* (Laplacian variance 120-325) while the real faces on this footage are often
*blurry* (3.1, 3.8, 11.0), so a sharpness filter preferentially keeps gloves and deletes people.

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

At 30 s the merge distance stops mattering, because the minimum length decides everything — see the
`--merge-s` note above. One knob instead of two, and `--min-keep-s` is the one.

**The same number also decides whether a recording is processed at all.** A source shorter than
`--min-keep-s` cannot yield a clip worth keeping, so it produces no clips and no removed spans — it is
listed under `skipped_too_short` in the manifest instead. Seven of the twenty-three recordings across
the two IC-559 cards were 1.5-23 s and were skipped on this rule.

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

**Always pass `--faces` when reviewing a cut.** It draws a red box, labelled with its score, on every
detection, matched by source log time so a report made on the original recording lines up with a clip
taken out of it:

```bash
python mcap_to_mosaic.py OUT/removed/rec_removed00.mcap review.mp4 --faces faces.json --device 1
```

Without it the mosaic is actively misleading. Tiles scale 1920 down to 640, so a 63 px face renders at
about 21 px and reads as a smudge — a **correct** detection looks like a false positive, which would
push you to loosen the threshold and start leaking real faces. Never conclude "no face here" from an
unboxed mosaic; check the source at full resolution.

Play previews with `--avcodec-hw=none`: VLC's GPU decoding raised an Xid fault on GPU 0 that took the
whole machine down.

## Hardware notes

**GPU 0 drives the display.** Heavy CUDA there while a video plays froze this machine outright — VLC
raised an Xid 68 decoder fault on GPU 0 and a CUDA job on the same card took the whole session down.
Default long jobs to `--device 1`, and if you must use GPU 0, close your video players first.

**`--device` takes ONE index in cut mode.** `--device 0,1` works only in `blur_faces_mcap.py`, which
shards channels across cards; cut mode rejects it. To use two GPUs for cutting, run two processes on
different files:

```bash
python redact_mcap.py A.mcap OUT --mode cut ... --device 0 &
python redact_mcap.py B.mcap OUT --mode cut ... --device 1
```

That is close to a 2x speedup, since the files are independent. Both processes append to the same
`cut_manifest.json` safely — it is merged by filename, not rewritten.

Measured throughput, one RTX 4090, five 1920x1200 channels at 30 fps:

| stage | rate | per hour of footage |
|---|---|---|
| detection, stride 3 | ~110 fps | ~80 min |
| detection, stride 3, two GPUs on separate files | ~220 fps | ~40 min |
| blur, stride 3, one GPU | 154 fps | ~58 min |
| blur, stride 3, both GPUs | 189 fps | ~48 min |

Cutting 5.3 hours of five-camera footage took about 7 hours on one GPU, and the writing is disk-bound
rather than GPU-bound — expect the card to sit idle while a 20 GB recording is written out.

Peak RAM ~34 GB, since all video packets are held in memory — an 80-minute recording would need ~90 GB.

Budget roughly the size of your sources again for the output: cut mode keeps ~90% of the footage, and
the clips together are slightly larger than the source because each one repeats a keyframe.

For several recordings, shard by file across the cards (balanced by size, not count):

```bash
python blur_faces_mcap.py SRC OUT --shard 0 2 --device 0 &
python blur_faces_mcap.py SRC OUT --shard 1 2 --device 1
```

## What this produced on the IC-559 cards

Both cards, cut mode, the settings at the top of this README:

| | recordings | footage | kept | clips |
|---|---|---|---|---|
| card 1 | 10 | 5.26 h | 4.74 h (90.1%) | 96 |
| card 2 | 7+ | 1.54 h | 1.47 h (95.3%) | 23 |

Per-recording the spread is wide — 61% to 100% on card 1 — and it is driven by how often someone
walks through, not by recording length. Two recordings were unusable: one truncated in transfer
(`EndOfFile`) and one never finalised by the rig (`RecordLengthLimitExceeded`).

Expect to lose around 10% of footage, concentrated in a few spans rather than spread evenly. On the
recording examined in detail, two events — the session start and one cluster at minute 32 — accounted
for 68% of everything removed.

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
