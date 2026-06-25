#!/bin/bash
# Node-2 SFT launcher (NVIDIA pytorch container, cwd /work). Script form so the
# "trl>=0.12" specs aren't eaten as bash redirects. $@ passes through to train.py
# (e.g. --completion-only --max-length 2048 [--smoke]).
set -e
pip install -q "trl>=0.12" "peft>=0.13" "datasets>=3.0" "accelerate>=1.0"
pip uninstall -y torchao || true   # container pin version-checks against trl's transformers
python /work/train.py --data /work/data --out /work/out "$@"
