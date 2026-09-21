# redact — batch visual redaction for 1080p MP4s (RTX 4090)

Pipeline per video: **ffmpeg NVDEC decode → SAM 3 text-prompted video tracking + EgoBlur face
detection → union mask → spatial dilation → temporal smoothing → flat grey fill → NVENC encode
(no audio) → PaddleOCR check of the output → manifest.json with SHA-256 checksums.**

## Environment

```bash
source /home/bizon/sparkpack/.venv-redact/bin/activate   # Python 3.12
```

| component | version | notes |
|---|---|---|
| torch / torchvision | 2.14.0+cu126 / 0.29.0+cu126 | PyTorch cu126 wheel index |
| ultralytics | 8.4.146 | provides `SAM3VideoSemanticPredictor`; `clip` (ultralytics fork) and `timm` pre-installed because SAM3 imports them at build time |
| supervision | 0.30.2 | |
| paddlepaddle-gpu / paddleocr | 3.3.1 (cu126) / 3.7.0 | OCR models auto-download to `~/.paddlex/official_models` on first run |
| opencv-python + opencv-contrib-python | 4.10.0.84 (both) | paddlex pins contrib; both dists must match |
| ffmpeg / ffprobe | n8.1 static GPL (BtbN) in `.venv-redact/bin` | `cuda` hwaccel, `h264_nvenc`/`hevc_nvenc`; the system ffmpeg 7.1.1 has neither |

Rebuild from scratch with `./setup_venv.sh` (see the header of that script for the two traps:
Paddle's nvidia-* pins break torch unless re-pinned, and uv gives `--extra-index-url` priority).
`requirements.lock.txt` is a full `uv pip freeze` of the working environment.

## Weights (gated, not included)

* `sam3.pt` — Meta SAM 3 checkpoint (HF gated). Pass with `--sam3-weights`.
* EgoBlur face detector TorchScript, gen1 (`ego_blur_face.jit`) or gen2 (`ego_blur_face_gen2.jit`)
  from projectaria.com. Pass with `--egoblur-weights`; generation is inferred from the file name
  (`--egoblur-gen 1|2` to force).

The manifest records the SHA-256 of both weight files.

## Usage

```bash
source .venv-redact/bin/activate
python redact/redact_videos.py /data/raw_mp4s /data/redacted \
    --sam3-weights /models/sam3.pt \
    --egoblur-weights /models/ego_blur_face.jit \
    --prompts "chocolate bar" "wrapper" "cardboard box" "label"
```

Verified with the real `sam3.pt` (2026-09-10) on a hand-held chocolate-bar clip. **SAM 3 cost scales with the
number of tracked objects**, and broad prompts ("label", "cardboard box") turn every envelope on a desk into
an object:

| configuration (1080p, RTX 4090) | tracked objects | end-to-end |
|---|---|---|
| SAM 3, prompts bar+wrapper+box+label | 8-13 | 2.3 fps, 40% of frame grey |
| SAM 3, prompt "chocolate bar" only | 1-3 | 7.9 fps, 24% grey (model-bound: ~126 ms/frame) |
| YOLOE fallback, 2-3 prompts | n/a | 36 fps |
| plumbing only (decode -> encode -> OCR) | – | 90-125 fps |

Prompt only for what must be hidden. Peak VRAM ~6-9 GB for SAM 3, ~12 GB with PaddleOCR loaded.
Add `--max-frames 300` on one clip to check mask quality before committing to a whole folder.
The loop is two-stage: decode + model on the main thread, faces/dilate/smooth/fill/encode on a worker, so
post-processing is hidden behind inference (dilation runs at 1/4 resolution, fill uses a masked copy).

Two GPUs: run one shard per card, each gets its own manifest file.

```bash
python redact/redact_videos.py IN OUT --device 0 --shard 0 2 ... &
python redact/redact_videos.py IN OUT --device 1 --shard 1 2 ...
```

Useful flags (defaults in parentheses):

* Masks: `--conf 0.3` SAM 3 score threshold, `--face-conf 0.5 --face-nms 0.3 --face-scale 1.15 --face-shape rect`
* Fill: `--fill grey|blur|pixelate|anon` (`grey` is the only irreversible option; `--blur-block 24` sets blur/mosaic
  coarseness). `--grey 128` value for grey mode. `anon` blurs, then throws away the object's colours and re-tints
  its shading with a per-video random colour (derived from the input sha256 + `--anon-seed`, recorded in the
  manifest): geometry and folds survive for training, print / artwork / brand colours do not, and the random
  colour doubles as appearance randomisation. Pair with an inward margin (`--dilate-px -3`) to keep the outline.
* Post: `--dilate-px 15` elliptical margin: positive grows the mask outward (safer), **negative shrinks it inward**
  (e.g. `-8`) so the object's outline stays sharp and only its interior is redacted; `--smooth-window 5 --smooth-mode max` — `max` holds a pixel
  redacted if it is masked in any frame within ±2 (never shrinks coverage); `majority` votes instead
  and removes one-frame flicker. `--grey 128` fill value.
* Rotation: phone footage with a display-matrix rotation tag (portrait MOVs) is decoded upright, so a
  1920x1080 portrait recording is processed and written as 1080x1920 with no rotation tag.
* Encode: `--codec h264_nvenc|hevc_nvenc|libx264 --nvenc-preset p5 --cq 23`. Output is CFR at the
  input's average frame rate; audio is dropped. Falls back to libx264 if NVENC is missing.
* OCR: `--ocr-stride 15` (every 15th output frame), `--ocr-min-conf 0.6 --ocr-min-chars 3`, `--no-ocr`.
* `--no-sam3` / `--no-faces` to run without one of the gated models; `--list` to probe only;
  `--overwrite` to redo; re-running without it resumes (finished entries in the manifest are skipped).

## YOLOE fallback (while SAM 3 access is pending)

`--tracker yoloe` swaps SAM 3 for YOLOE open-vocabulary instance segmentation (`yoloe-11l-seg.pt` and the
MobileCLIP text encoder auto-download into `weights/`). Same prompts flag, same mask post-processing and
encode path, but two differences matter:

* **Vocabulary.** YOLOE only knows concrete nouns from its training vocabulary. On a real clip,
  "chocolate bar", "wrapper" and "label" scored ~0.02 (i.e. never fire) while "packaged snack" (0.61),
  "box" (0.69), "food packaging" (0.36), "person" (0.9) worked. Phrase prompts for YOLOE accordingly;
  SAM 3 is the model that understands the literal packaging terms.
* **Scores run lower.** `--yoloe-conf` (default 0.15) is separate from SAM 3's `--conf`.

Working example on a hand-held chocolate bar:

```bash
python redact/redact_videos.py IN OUT --tracker yoloe --no-faces --prompts "packaged snack" box "food packaging" wrapper label
```

No cross-frame tracking, so a confidence dip on the target becomes a visible gap. Tuning on the
chocolate-bar clip (486 frames), judged by OCR flags on the output:

| `--yoloe-conf` | `--smooth-window` | `--dilate-px` | prompts | grey area | OCR flags |
|---|---|---|---|---|---|
| 0.15 | 5 | 15 | snack, box, packaging, wrapper, label | 34% | 3 (bar text visible at 10.5 s) |
| 0.08 | 31 | 15 | same | 53% | 0, but smears over hand and cloth |
| 0.15 | 15 | 15 | snack, box, packaging | 41% | 0 — good balance |
| 0.25 | 15 | 5 | snack, packaging | 27% | 9, bar edge exposed during a dip |
| 0.15 | 15 | 10 | snack, packaging (blur) | 32% | 4, all a cereal box behind the bar |
| 0.15 | 9 | 5 | snack, packaging (blur, `--blur-block 16`) | 23% | none from the bar on two clips — **recommended YOLOE setting** |
| 0.15 | 9 | -8 (inward) | snack, packaging (blur 16) | 20% | none from the bar; bar outline stays sharp, interior blurred |
| 0.15 | 9 | -3 (inward) | snack, packaging (blur 24) | 21% | none from the bar — **Eric's chosen setting 2026-09-10** |

Lower threshold / longer hold / bigger dilation = more coverage, fewer misses, uglier. Over-masking is the
safe direction; the OCR flags are the objective check.

## YuNet face fallback (while EgoBlur access is pending)

`--face-detector yunet` uses OpenCV's bundled YuNet detector (230 KB ONNX, auto-downloaded to `weights/`,
CPU, run at `--yunet-scale 0.5`). On a hand-held selfie clip it found the face in every one of 3245
frames, including 3/4-profile views. Swap back to `--face-detector egoblur --egoblur-weights ...` once
the EgoBlur file arrives; the box handling (`--face-scale`, `--face-shape`) is shared.

```bash
python redact/redact_videos.py IN OUT --no-sam3 --face-detector yunet
```

## Outputs

```
OUT/
  <stem>.mp4            redacted video (written as .partial.mp4, renamed on success)
  manifest.json         per-video: input {path, sha256, size, w×h, fps, frames, codec},
                        output {same + encoder}, redaction stats (frames with SAM 3 masks per prompt,
                        face counts, masked-area fraction), ocr {frames checked/flagged, distinct texts,
                        per-frame flags with text, score, box}, timing, status/error;
                        plus tool versions, all CLI params, weight checksums, GPU name
  manifest.json.sha256  checksum of the manifest
  SHA256SUMS            sha256sum -c compatible list of the output videos
  redact.log
```

Frames flagged by OCR mean readable text survived redaction; review those timestamps.

## Tests

```bash
python redact/tests/test_pipeline.py          # smoother + fake-tracker fill/encode/audio-strip checks
python redact/tests/test_pipeline.py --sam2   # also drives the ultralytics shim with SAM 2.1 tiny (downloads 75 MB)
```
