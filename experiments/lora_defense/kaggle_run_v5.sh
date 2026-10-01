#!/bin/bash
# The one-perturbation pilot on Kaggle ("GPU T4 x2"): can one photograph stop both the
# face swap and the LoRA? For the four people whose LoRA came back most when the shield
# was stacked on Mist, Mist runs on the shielded photographs (shield_mist16: the shield
# first, Mist after) and with the shield's loss inside its own PGD (joint8: one budget of
# 8/255), then LoRAs are trained on both. Every step resumes from /kaggle/working (or from
# an earlier version's output attached as input), and each is first stopped and
# restarted once in this run to prove it: Mist after its first round on each variant,
# the LoRAs at step 150. The Mist deadline (MIST_HOURS) keeps the version ending normally.
set -euo pipefail
HERE=$(dirname "$(readlink -f "$0")")
cd /kaggle/working
pip install -q -U "diffusers>=0.30" "peft>=0.11" uv
# The image's torchao is older than peft accepts, and nothing here uses it.
pip uninstall -q -y torchao
nvidia-smi --query-gpu=name,memory.total --format=csv
# The ArcFace encoder (buffalo_l w600k_r50, the one the shield attacks) came up in 9 MB
# pieces, the browser upload's limit; join them and stop unless it is that exact file.
mkdir -p /kaggle/temp
export ARCFACE_ONNX=/kaggle/temp/w600k_r50.onnx
find /kaggle/input -name 'w600k_r50.onnx.part*' | sort | xargs cat > "$ARCFACE_ONNX"
echo "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43  $ARCFACE_ONNX" \
    | sha256sum -c -
PEOPLE=hugo_chavez,jeb_bush,jennifer_lopez,nicole_kidman
VARIANTS=shield_mist16,joint8
export MIST_PEOPLE=$PEOPLE MIST_VARIANTS=$VARIANTS MIST_HOURS=${MIST_HOURS:-2.5}
export MIST_JOINT_TAU=${MIST_JOINT_TAU:--0.4} RESUME_TAG=_v5

# 1. Mist, one job per GPU, checkpointed after every round.
MIST_STOP_AFTER=1 python "$HERE/kaggle_mist.py"
python "$HERE/kaggle_mist.py"

# 2. LoRAs from both, one process per GPU.
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
