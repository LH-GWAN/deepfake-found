#!/bin/bash
# The anti-LoRA test's last step on Kaggle ("GPU T4 x2"): LoRAs on the photographs
# carrying Mist v2 and, on top, the swap shield of `protect --mode shield`
# (mist16_shield, made locally by evaluate_mist_shield.py), for the people in
# identities_v3.txt and the first three. Same LoRA settings, resumable, and first
# stopped and restarted once to prove it.
set -euo pipefail
HERE=$(dirname "$(readlink -f "$0")")
cd /kaggle/working
pip install -q -U "diffusers>=0.30" "peft>=0.11"
pip uninstall -q -y torchao
nvidia-smi --query-gpu=name,memory.total --format=csv
PEOPLE=tom_hanks,jennifer_lopez,hugo_chavez,$(paste -sd, "$(find /kaggle/input -name identities_v3.txt | head -1)")

loras() {
    CUDA_VISIBLE_DEVICES=0 LORA_SHARD=0/2 LORA_IDENTITIES=$PEOPLE LORA_VARIANTS=mist16_shield \
        python "$HERE/kaggle_train_lora.py" & first=$!
    CUDA_VISIBLE_DEVICES=1 LORA_SHARD=1/2 LORA_IDENTITIES=$PEOPLE LORA_VARIANTS=mist16_shield \
        python "$HERE/kaggle_train_lora.py" & second=$!
    wait $first; wait $second
}
STOP_AT_STEP=150 loras
loras
echo "ALL DONE" | tee -a /kaggle/working/log.txt
