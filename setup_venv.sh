#!/usr/bin/env bash
# Recreate the redaction venv (.venv-redact) on an RTX 4090 box with driver >= 560.
# Usage: ./setup_venv.sh [venv_dir]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="${1:-$HERE/../.venv-redact}"
FF_ASSET="ffmpeg-n8.1-latest-linux64-gpl-8.1.tar.xz"
FF_URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/$FF_ASSET"
PADDLE_IDX="https://www.paddlepaddle.org.cn/packages/stable/cu126/"

command -v uv >/dev/null || { echo "uv not found (https://docs.astral.sh/uv/)"; exit 1; }

echo "== venv: $VENV"
uv venv --python 3.12 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "== 1/5 torch + torchvision (CUDA 12.6 wheels)"
uv pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision

echo "== 2/5 ultralytics / supervision / paddleocr / helpers"
uv pip install ultralytics supervision paddleocr tqdm timm "git+https://github.com/ultralytics/CLIP.git"
# paddlex pins opencv-contrib-python==4.10.0.84; make opencv-python identical so the two dists don't fight over cv2/
uv pip install --reinstall "opencv-python==4.10.0.84" "opencv-contrib-python==4.10.0.84" "numpy>=2.0,<2.4"

echo "== 3/5 paddlepaddle-gpu (cu126) from the Paddle index"
# uv gives --extra-index-url priority over --index-url; PyPI only has old paddlepaddle-gpu builds.
uv pip install "paddlepaddle-gpu==3.3.1" "numpy>=2.0,<2.4" \
  --index-url https://pypi.org/simple --extra-index-url "$PADDLE_IDX"

echo "== 4/5 restore torch's nvidia-* pins (paddle's older pins break torch: undefined symbol ncclCommGrow)"
# Same CUDA 12 major -> ABI compatible for paddle. `uv pip check` will still list paddle's == pins; ignore.
python - <<'PY'
import re, subprocess, json, importlib.metadata as md
reqs = [r for r in md.requires("torch") if r.startswith("nvidia-") and "==" in r]
pins = [re.match(r"([\w-]+==[\w.]+)", r).group(1) for r in reqs]
print("re-pinning:", " ".join(pins))
subprocess.run(["uv", "pip", "install", *pins], check=True)
PY

echo "== 5/5 ffmpeg/ffprobe static build with NVENC + NVDEC (BtbN, GPL) into the venv bin"
TMP="$(mktemp -d)"
curl -L --fail -o "$TMP/$FF_ASSET" "$FF_URL"
tar -xJf "$TMP/$FF_ASSET" -C "$TMP"
cp "$TMP"/ffmpeg-*/bin/ffmpeg "$TMP"/ffmpeg-*/bin/ffprobe "$VENV/bin/"
cp "$TMP"/ffmpeg-*/LICENSE.txt "$VENV/bin/FFMPEG-LICENSE.txt" || true
rm -rf "$TMP"

echo "== verify"
python - <<'PY'
import torch, paddle, ultralytics, paddleocr, cv2, timm, clip
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("paddle", paddle.__version__, "cuda", paddle.device.is_compiled_with_cuda())
print("ultralytics", ultralytics.__version__, "paddleocr", paddleocr.__version__, "cv2", cv2.__version__)
from ultralytics.models.sam import SAM3VideoSemanticPredictor  # noqa
PY
ffmpeg -hide_banner -hwaccels | grep -q cuda && ffmpeg -hide_banner -encoders | grep -q h264_nvenc && echo "ffmpeg: cuda hwaccel + h264_nvenc OK"
echo "done. activate with: source $VENV/bin/activate"
