#!/bin/bash
# The budget-matched arms on Colab (one T4), for the four pilot people of kaggle_run_v5.sh.
# shield_mist16 moves a pixel up to 16/255 from the clean photograph (8/255 of shield,
# then 8/255 of Mist) and joint8 only 8/255, so "the shield first, Mist after" beating the
# joint optimisation could be the budget alone. Two arms close that:
#   joint16        one PGD with both objectives in 16/255, against shield_mist16;
#   shield4_mist4  4/255 of shield, then 4/255 of Mist, against joint8.
# shield4_512 (the 4/255 shield alone) is measured too, to see whether it still stops the swap.
#
# Everything that has to survive a disconnected runtime lives on Google Drive: the inputs
# (DS_INPUT), the Mist checkpoints, LoRAs, generated images and results (DS_WORK). Every
# step skips what is already there, so after a disconnect the notebook is run again from
# the top and picks up where it stopped. Run from the repository root.
set -euo pipefail
HERE=$(dirname "$(readlink -f "$0")")
: "${DS_INPUT:?}" "${DS_WORK:?}"
export DS_TEMP=${DS_TEMP:-/content/tmp} DS_GPUS=1 DS_CLASS_DIR=$DS_WORK/class_person
PEOPLE=${PEOPLE:-hugo_chavez,jeb_bush,jennifer_lopez,nicole_kidman}
ARMS=joint16,shield4_mist4
V3=$DS_INPUT/v3
mkdir -p "$DS_WORK" "$DS_TEMP"
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$DS_WORK/log.txt"; }
nvidia-smi --query-gpu=name,memory.total --format=csv

# 0. Models: the buffalo_l pack (SCRFD and the w600k_r50 ArcFace the shield attacks) and
#    inswapper, each checked against the digest the local runs used.
BUFFALO=models/insightface/models/buffalo_l
if [ ! -f $BUFFALO/w600k_r50.onnx ]; then
    mkdir -p models/insightface/models
    wget -q -O models/insightface/models/buffalo_l.zip \
        https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
    unzip -q -o models/insightface/models/buffalo_l.zip -d $BUFFALO
fi
echo "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43  $BUFFALO/w600k_r50.onnx" \
    | sha256sum -c -
export ARCFACE_ONNX=$PWD/$BUFFALO/w600k_r50.onnx
[ -f models/inswapper/inswapper_128.onnx ] || python scripts/fetch_inswapper.py

# 1. The 4/255 shield on the watermarked photographs (the shipped shield, smaller budget).
python "$HERE/prepare_joint_inputs.py" --clean "$V3/clean" --work "$V3" \
    --shield-people "$PEOPLE" --shield-epsilon 4
log "shield4_512 ready"

# 2. Mist for both arms, checkpointed after every round.
MIST_PEOPLE=$PEOPLE MIST_VARIANTS=$ARMS MIST_HOURS=${MIST_HOURS:-11} RESUME_TAG=_budget \
    python "$HERE/kaggle_mist.py"
# The swap evaluation reads every arm from one protected/ folder.
for arm in ${ARMS//,/ }; do
    mkdir -p "$V3/protected/$arm"
    cp -rn "$DS_WORK/protected/$arm/." "$V3/protected/$arm/"
done

# 3. A LoRA on each arm's photographs (checkpointed every 100 steps).
LORA_IDENTITIES=$PEOPLE LORA_VARIANTS=$ARMS RESUME_TAG=_budget python "$HERE/kaggle_train_lora.py"

# 4. Sixteen images per LoRA, plus the no-LoRA control.
python "$HERE/generate_local.py" --loras "$DS_WORK/lora" --output "$DS_WORK/generated" \
    --variants $ARMS --device cuda
log "generation done"

# 5. Scoring. Swaps (and watermark, shield frame, PSNR/SSIM) for the new arms and, as a
#    check that this runtime reproduces the earlier numbers, for shield_mist16 and joint8.
mkdir -p "$DS_WORK/results"
python "$HERE/evaluate_joint_pilot.py" --work "$V3" --clean "$V3/clean" --gallery "$V3/gallery" \
    --people "$PEOPLE" --variants joint16,shield4_mist4,shield4_512,shield_mist16,joint8 \
    --output "$DS_WORK/results" --report budget_photos.json
# The LoRA half, scored as evaluate_lora_defense.py scores every earlier run.
BUNDLE=$DS_TEMP/score
rm -rf "$BUNDLE" && mkdir -p "$BUNDLE/generated" "$BUNDLE/protected" "$BUNDLE/faces/clean"
cp -r "$DS_WORK/generated/." "$BUNDLE/generated/"
for arm in ${ARMS//,/ }; do cp -r "$DS_WORK/protected/$arm" "$BUNDLE/protected/"; done
for person in ${PEOPLE//,/ }; do cp -r "$V3/clean/$person" "$BUNDLE/faces/clean/"; done
(cd "$BUNDLE" && zip -qr results.zip generated protected && cd faces && zip -qr ../faces.zip clean)
python scripts/evaluate_lora_defense.py --results "$BUNDLE/results.zip" --faces "$BUNDLE/faces.zip" \
    --gallery "$V3/gallery" --output "$DS_WORK/results/lora"
log "ALL DONE"
