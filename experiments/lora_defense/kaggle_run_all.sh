#!/bin/bash
# The anti-LoRA test's remaining work on Kaggle, in one "Save & Run All" version.
# Each step resumes from /kaggle/working (or from an earlier version's output attached
# as input), and each is first stopped and restarted once to prove the resume works.
set -euo pipefail
HERE=$(dirname "$(readlink -f "$0")")
cd /kaggle/working
pip install -q -U "diffusers>=0.30" "peft>=0.11" uv
# The image's torchao is older than peft accepts, and nothing here uses it.
pip uninstall -q -y torchao
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"

# 1. LoRAs from the photos already protected (clean, Mist 8 and 16, ASPL tom_hanks).
STOP_AT_STEP=150 python "$HERE/kaggle_train_lora.py"
python "$HERE/kaggle_train_lora.py"

# 2. ASPL for the identities still missing, checkpointed every 5 rounds.
ASPL_STOP_AFTER=5 python "$HERE/kaggle_aspl.py"
python "$HERE/kaggle_aspl.py"

# 3. LoRAs for the photos step 2 protected.
python "$HERE/kaggle_train_lora.py"
echo "ALL DONE" | tee -a /kaggle/working/log.txt
