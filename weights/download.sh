#!/usr/bin/env bash
# Fetch the open detector weights BEFORE the offline evaluation.
#   bash weights/download.sh            # YOLOX-S + YOLOX-M (137 MB total)
#   bash weights/download.sh yolox_s    # only one model
# Source: Megvii-BaseDetection/YOLOX release 0.1.1rc0 (Apache-2.0), COCO-pretrained ONNX exports.
# Every file is verified against a pinned SHA-256; nothing is downloaded at inference time.
# Portable: Linux and macOS (bash 3.2, sha256sum or shasum).
set -euo pipefail
cd "$(dirname "$0")"
BASE="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"

sha_of() {  # pinned checksum per model
  case "$1" in
    yolox_s) echo c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063 ;;
    yolox_m) echo 21ff6cfdeb53b013bac2249599e55f00bff3cfdfdab37ed7a4620818c1d15b3f ;;
    *) return 1 ;;
  esac
}
hash_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1
  else python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$1"; fi
}

if [ "$#" -eq 0 ]; then set -- yolox_s yolox_m; fi
for m in "$@"; do
  want="$(sha_of "$m")" || { echo "unknown model $m" >&2; exit 2; }
  f="$m.onnx"
  if [ -f "$f" ] && [ "$(hash_file "$f")" = "$want" ]; then
    echo "ok   $f (already present)"; continue
  fi
  echo "get  $f"
  if command -v curl >/dev/null 2>&1; then curl -fL --retry 4 -o "$f.part" "$BASE/$f"; else wget -q -O "$f.part" "$BASE/$f"; fi
  if [ "$(hash_file "$f.part")" != "$want" ]; then echo "checksum mismatch for $f" >&2; rm -f "$f.part"; exit 1; fi
  mv "$f.part" "$f"
  echo "ok   $f"
done
du -ch ./*.onnx | tail -1
