#!/usr/bin/env bash
# Measure the face-detection variants on one MCAP recording. Run with both GPUs idle.
#   ./bench_detect.sh [gpu] [mcap]
set -u
GPU="${1:-1}"
MCAP="${2:-$HOME/cocoapack/card 1/349a2f13-0156-4e64-ac3b-f7f76e619a62_196068ee-4449-48fc-a450-dba9eb31a7a1.mcap}"
OUT="${SCRATCH:-/tmp}/bench"; mkdir -p "$OUT"
cd "$(dirname "$0")"
source ../.venv-redact/bin/activate
W=weights/ego_blur_face_gen2.jit
run () {  # name, extra args...
  local name="$1"; shift
  local t0 t1
  t0=$(date +%s.%N)
  python detect_faces_mcap.py "$MCAP" --face-detector egoblur --egoblur-weights "$W" \
      --scale 1.0 --device "$GPU" --no-sha256 --quiet-detections \
      -o "$OUT/$name.json" "$@" >/dev/null 2>"$OUT/$name.err"
  local rc=$?
  t1=$(date +%s.%N)
  if [ $rc -ne 0 ]; then echo "$name FAILED (see $OUT/$name.err)"; tail -3 "$OUT/$name.err"; return; fi
  python - "$OUT/$name.json" "$name" "$(echo "$t1 - $t0" | bc)" <<'PY'
import json, sys
rep = json.load(open(sys.argv[1])); f = rep["files"][0]
frames = sum(c["frames_decoded"] for c in f["channels"])
faces  = sum(c["frames_with_faces"] for c in f["channels"])
wall   = float(sys.argv[3])
print(f"{sys.argv[2]:<26} {wall:7.1f}s  {frames/wall:7.1f} fps  frames={frames:5d}  face-frames={faces:3d}")
PY
}
printf '%-26s %8s  %11s  %s\n' variant wall throughput detections
run "seq-fp32-stride1"  --no-batch-channels --workers 1
run "batch-fp32-stride1" --batch-channels
run "batch-fp16-stride1" --batch-channels --half
run "batch-fp16-stride3" --batch-channels --half --stride 3
run "batch-fp16-stride5" --batch-channels --half --stride 5
echo; echo "reports in $OUT"
