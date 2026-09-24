#!/usr/bin/env bash
# Fetch the open detector weights BEFORE the offline evaluation.
#   bash weights/download.sh            # YOLOX-S + YOLOX-M (137 MB total)
#   bash weights/download.sh yolox_s    # only one model
# Source: Megvii-BaseDetection/YOLOX release 0.1.1rc0 (Apache-2.0), COCO-pretrained ONNX exports.
# Every file is verified against a pinned SHA-256; nothing is downloaded at inference time.
set -euo pipefail
cd "$(dirname "$0")"
BASE="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"
declare -A SHA=(
  [yolox_s]=c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063
  [yolox_m]=21ff6cfdeb53b013bac2249599e55f00bff3cfdfdab37ed7a4620818c1d15b3f
)
MODELS=("${@:-yolox_s yolox_m}")
for m in ${MODELS[@]}; do
  [[ -n "${SHA[$m]:-}" ]] || { echo "unknown model $m" >&2; exit 2; }
  f="$m.onnx"
  if [[ -f "$f" ]] && echo "${SHA[$m]}  $f" | sha256sum -c --status; then
    echo "ok   $f (already present)"; continue
  fi
  echo "get  $f"
  if command -v curl >/dev/null; then curl -fL --retry 4 -o "$f.part" "$BASE/$f"; else wget -q -O "$f.part" "$BASE/$f"; fi
  echo "${SHA[$m]}  $f.part" | sha256sum -c --status || { echo "checksum mismatch for $f" >&2; rm -f "$f.part"; exit 1; }
  mv "$f.part" "$f"
  echo "ok   $f"
done
du -ch ./*.onnx | tail -1
