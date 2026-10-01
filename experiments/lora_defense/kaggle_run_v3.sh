#!/bin/bash
# The anti-LoRA test with more people, on Kaggle ("GPU T4 x2"): Mist v2 at 8/255 of
# [0, 1] for the people in identities_v3.txt, then LoRAs on their clean and Mist
# photographs. Every step resumes from /kaggle/working (or from an earlier version's
# output attached as input), and each is first stopped and restarted once to prove it.
set -euo pipefail
HERE=$(dirname "$(readlink -f "$0")")
cd /kaggle/working
pip install -q -U "diffusers>=0.30" "peft>=0.11" uv
# The image's torchao is older than peft accepts, and nothing here uses it.
pip uninstall -q -y torchao
nvidia-smi --query-gpu=name,memory.total --format=csv
PEOPLE=$(paste -sd, "$(find /kaggle/input -name identities_v3.txt | head -1)")

# 1. Mist, one person per GPU, checkpointed after every round.
MIST_STOP_AFTER=1 python "$HERE/kaggle_mist.py"
python "$HERE/kaggle_mist.py"

# 2. LoRAs from the clean and the Mist photographs, one process per GPU.
loras() {
    CUDA_VISIBLE_DEVICES=0 LORA_SHARD=0/2 LORA_IDENTITIES=$PEOPLE LORA_VARIANTS=clean,mist16 \
        python "$HERE/kaggle_train_lora.py" & first=$!
    CUDA_VISIBLE_DEVICES=1 LORA_SHARD=1/2 LORA_IDENTITIES=$PEOPLE LORA_VARIANTS=clean,mist16 \
        python "$HERE/kaggle_train_lora.py" & second=$!
    wait $first; wait $second
}
STOP_AT_STEP=150 loras
loras
echo "ALL DONE" | tee -a /kaggle/working/log.txt
