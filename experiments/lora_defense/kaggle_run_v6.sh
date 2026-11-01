#!/bin/bash
# The one-perturbation test for the other sixteen people on Kaggle ("GPU T4 x2"): Mist v2
# on the shielded photographs (shield_mist16: the watermark, then the swap shield, then
# Mist), which passed the four-person pilot (kaggle_run_v5.sh), then a LoRA on each. Every
# step resumes from /kaggle/working (or from an earlier version's output attached as
# input), and each is first stopped and restarted once in this run to prove it: Mist after
# its first round, the LoRAs at step 150. The Mist deadline (MIST_HOURS) keeps the version
# ending normally; whoever is left then is run by the next version.
set -euo pipefail
HERE=$(dirname "$(readlink -f "$0")")
cd /kaggle/working
pip install -q -U "diffusers>=0.30" "peft>=0.11" uv
# The image's torchao is older than peft accepts, and nothing here uses it.
pip uninstall -q -y torchao
nvidia-smi --query-gpu=name,memory.total --format=csv
PEOPLE=abdullah_gul,al_gore,anna_kournikova,bill_graham,gordon_brown,gray_davis,hamid_karzai
PEOPLE=$PEOPLE,john_kerry,julie_gerberding,megawati_sukarnoputri,pete_sampras,renee_zellweger
PEOPLE=$PEOPLE,roger_federer,sergey_lavrov,tom_hanks,vladimir_putin
VARIANTS=shield_mist16
# Stop here unless every person's shielded photographs came through the upload.
for person in ${PEOPLE//,/ }; do
    found=$(find /kaggle/input -path "*/shield512/$person/*.png" | wc -l)
    [ "$found" -ge 8 ] || { echo "only $found shield512 photos for $person"; exit 1; }
done
echo "shield512 inputs: all 16 people" | tee -a /kaggle/working/log.txt
export MIST_PEOPLE=$PEOPLE MIST_VARIANTS=$VARIANTS MIST_HOURS=${MIST_HOURS:-3.5} RESUME_TAG=_v6

# 1. Mist, one job per GPU, checkpointed after every round.
MIST_STOP_AFTER=1 python "$HERE/kaggle_mist.py"
python "$HERE/kaggle_mist.py"

# 2. A LoRA on each, one process per GPU.
loras() {
    CUDA_VISIBLE_DEVICES=0 LORA_SHARD=0/2 LORA_IDENTITIES=$PEOPLE LORA_VARIANTS=$VARIANTS \
        python "$HERE/kaggle_train_lora.py" & first=$!
    CUDA_VISIBLE_DEVICES=1 LORA_SHARD=1/2 LORA_IDENTITIES=$PEOPLE LORA_VARIANTS=$VARIANTS \
        python "$HERE/kaggle_train_lora.py" & second=$!
    wait $first; wait $second
}
STOP_AT_STEP=150 loras
loras
echo "ALL DONE" | tee -a /kaggle/working/log.txt
