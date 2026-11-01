#!/bin/bash
# Pack what colab_budget.ipynb uploads to Google Drive (folder deepshield_budget/):
#   code.zip    the repository's tracked files as they are in the working tree, without the
#               learned watermark checkpoints, plus YuNet and SFace (symlinks, followed);
#   inputs.zip  v3/ for the budget-matched run: every person's clean and wm512 photographs
#               (the swap pairs each person with the next one, and prepare_joint_inputs.py
#               walks everyone), the gallery, joint_inputs/, and the four pilot people's
#               shield512, shield_mist16 and joint8 photographs to re-measure.
# Nothing else is uploaded. Run from anywhere.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:-$ROOT/experiments/lora_defense/colab_upload}
PEOPLE=(hugo_chavez jeb_bush jennifer_lopez nicole_kidman)
mkdir -p "$OUT"
rm -f "$OUT/code.zip" "$OUT/inputs.zip"
cd "$ROOT"
{ git ls-files | grep -v '^models/learned_watermark'
  echo experiments/lora_defense/colab_run_budget.sh
  echo models/face_detection_yunet_2023mar.onnx
  echo models/face_recognition_sface_2021dec.onnx
} | sort -u | zip -q "$OUT/code.zip" -@
cd experiments/lora_defense
list=(v3/clean v3/gallery v3/joint_inputs v3/protected/wm512)
for p in "${PEOPLE[@]}"; do
    list+=(v3/protected/shield512/$p v3/protected/shield_mist16/$p v3/protected/joint8/$p)
done
zip -qr "$OUT/inputs.zip" "${list[@]}"
unzip -l "$OUT/code.zip" | tail -1
unzip -l "$OUT/inputs.zip" | tail -1
cp colab_budget.ipynb "$OUT/"
du -h "$OUT"/*
