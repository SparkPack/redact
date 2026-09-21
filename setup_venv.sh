#!/usr/bin/env bash
# Build the venv for the MCAP face-redaction pipeline.
#   ./setup_venv.sh [venv_dir]        default: ../.venv-redact
#
# Needs an NVIDIA GPU (driver 560+, CUDA 12.6 wheels) and uv:
#   curl -LsSf https://astral.sh/uv/install.sh | sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="${1:-$HERE/../.venv-redact}"
FF_ASSET="ffmpeg-n8.1-latest-linux64-gpl-8.1.tar.xz"
FF_URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/$FF_ASSET"

command -v uv >/dev/null || { echo "uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

echo "== venv: $VENV"
uv venv --python 3.12 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "== 1/4 torch + torchvision (CUDA 12.6 wheels)"
# torchvision must be present and imported before torch.jit.load, or the EgoBlur export fails with
# "Unknown builtin op: torchvision::nms".
uv pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision

echo "== 2/4 mcap + opencv"
uv pip install mcap mcap-protobuf-support "opencv-python>=4.10" "numpy>=2.0"

echo "== 3/4 ffmpeg/ffprobe static build into the venv bin"
# Distro ffmpeg here had neither NVDEC nor NVENC. This BtbN GPL build has both. Note the pipeline
# defaults to libx264 anyway: GeForce cards cap concurrent NVENC sessions, and at 1920x1200 libx264
# measured slightly faster.
TMP="$(mktemp -d)"
curl -L --fail -o "$TMP/$FF_ASSET" "$FF_URL"
tar -xJf "$TMP/$FF_ASSET" -C "$TMP"
cp "$TMP"/ffmpeg-*/bin/ffmpeg "$TMP"/ffmpeg-*/bin/ffprobe "$VENV/bin/"
cp "$TMP"/ffmpeg-*/LICENSE.txt "$VENV/bin/FFMPEG-LICENSE.txt" || true
rm -rf "$TMP"

echo "== 4/4 verify"
python - <<'PY'
import torch, torchvision, cv2, numpy, mcap, mcap_protobuf  # noqa: F401
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
print("opencv", cv2.__version__, "| numpy", numpy.__version__, "| mcap", mcap.__version__)
PY
"$VENV/bin/ffmpeg" -hide_banner -hwaccels | grep -q cuda && echo "ffmpeg: cuda hwaccel present"

cat <<EOM

done. activate with: source $VENV/bin/activate

Next: the EgoBlur Gen2 face model (gated download, Apache 2.0) - nothing works without it.
  1. https://www.projectaria.com/tools/egoblur -> "Access the models", enter your email
  2. the emailed zip contains ego_blur_face_gen2.jit (400 MB)
  3. mkdir -p "$HERE/weights" && unzip -d "$HERE/weights" ~/Downloads/ego_blur_face_gen2.zip
EOM
