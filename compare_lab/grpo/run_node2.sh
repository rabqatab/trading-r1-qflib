#!/bin/bash
# Node-2 GRPO launcher (run inside nvcr.io/nvidia/pytorch container, cwd /work).
# Kept as a script so the "trl>=0.12" version specs don't get eaten as bash
# redirects in a nested `bash -c "..."`. Extra args ($@) pass through to train.py
# (e.g. --smoke, --wandb).
set -e
pip install -q "trl>=0.12" "peft>=0.13" "datasets>=3.0" "accelerate>=1.0"
pip uninstall -y torchao || true   # container pin version-checks against trl's transformers
python /work/train.py --data /work/data --base-adapter /work/adapter_v1 \
  --out /work/out "$@"
