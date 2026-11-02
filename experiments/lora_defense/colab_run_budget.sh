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

# 0. Models.
source "$HERE/colab_models.sh"

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

# 5. Scoring runs on the CPU only and is left to colab_score_budget.sh on a CPU runtime: on
#    this one the idle T4 gets reclaimed before it finishes.
log "GPU steps done: switch to a CPU runtime and run colab_score_budget.sh"
