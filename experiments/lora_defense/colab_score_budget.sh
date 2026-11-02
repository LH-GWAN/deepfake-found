#!/bin/bash
# Step 5 of colab_run_budget.sh: score the budget-matched arms once Mist, the LoRAs and the
# generated images are on Drive. Both scorers run ONNX Runtime on the CPU only, so on a
# GPU runtime the T4 sits idle for the hour or more this takes and Colab reclaims it
# (2026-11-02: disconnected 60 minutes in, nothing written). Run this on a CPU runtime
# instead: the photo half and the LoRA half are independent, so they share its two cores,
# and each is skipped when its report is already on Drive. Run from the repository root.
set -euo pipefail
HERE=$(dirname "$(readlink -f "$0")")
: "${DS_INPUT:?}" "${DS_WORK:?}"
export DS_TEMP=${DS_TEMP:-/content/tmp}
PEOPLE=${PEOPLE:-hugo_chavez,jeb_bush,jennifer_lopez,nicole_kidman}
ARMS=${ARMS:-joint16,shield4_mist4}
SHIELD_EPSILON=${SHIELD_EPSILON:-4}
# Reports of another set of arms get their own names, so one already on Drive is not taken
# for theirs: budget_photos.json and lora/ for the first run, <TAG>_photos.json and
# lora_<TAG>/ otherwise.
TAG=${TAG:-budget}
LORA_OUT=$DS_WORK/results/lora$([ "$TAG" = budget ] || echo "_$TAG")
V3=$DS_INPUT/v3
mkdir -p "$DS_WORK/results" "$DS_TEMP"
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$DS_WORK/log.txt"; }
source "$HERE/colab_models.sh"

# Swaps (and watermark, shield frame, PSNR/SSIM) for the new arms and their smaller shield
# alone and, as a check that this runtime reproduces the earlier numbers, for
# shield_mist16 and joint8.
photos() {
    [ -f "$DS_WORK/results/${TAG}_photos.json" ] && return
    python "$HERE/evaluate_joint_pilot.py" --work "$V3" --clean "$V3/clean" --gallery "$V3/gallery" \
        --people "$PEOPLE" --variants "$ARMS,shield${SHIELD_EPSILON}_512,shield_mist16,joint8" \
        --output "$DS_WORK/results" --report "${TAG}_photos.json"
    log "photos scored"
}

# The LoRA half, scored as evaluate_lora_defense.py scores every earlier run: these arms'
# images and the no-LoRA control.
lora() {
    [ -f "$LORA_OUT/lora_defense.json" ] && return
    local bundle=$DS_TEMP/score
    rm -rf "$bundle" && mkdir -p "$bundle/generated" "$bundle/protected" "$bundle/faces/clean"
    for arm in base ${ARMS//,/ }; do cp -r "$DS_WORK/generated/$arm" "$bundle/generated/"; done
    for arm in ${ARMS//,/ }; do cp -r "$DS_WORK/protected/$arm" "$bundle/protected/"; done
    for person in ${PEOPLE//,/ }; do cp -r "$V3/clean/$person" "$bundle/faces/clean/"; done
    (cd "$bundle" && zip -qr results.zip generated protected && cd faces && zip -qr ../faces.zip clean)
    python scripts/evaluate_lora_defense.py --results "$bundle/results.zip" \
        --faces "$bundle/faces.zip" --gallery "$V3/gallery" --output "$LORA_OUT" \
        > "$DS_WORK/results/${TAG}_lora_score.log"
    log "LoRA scored"
}

photos & first=$!
lora & second=$!
status=0
wait $first || status=1
wait $second || status=1
[ $status = 0 ] && log "ALL DONE"
exit $status
