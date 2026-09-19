#!/usr/bin/env bash
# Builds a clean conda env for this repo. Run from the repo root:
#   bash scripts/setup_env.sh && conda activate vgs
#
# Why: running in conda `base` with Python 3.13 broke on
#   ImportError: libstdc++.so.6: version `CXXABI_1.3.15' not found
# base's conda-built scipy was compiled against a newer libstdc++ than the system one.
# transformers imports sklearn -> scipy on any `from transformers import ...`, so every
# script dies at import time. A fresh env with pip wheels (manylinux, older ABI) avoids it.
set -euo pipefail

ENV=${ENV:-vgs}
PY=${PY:-3.11}
CUDA=${CUDA:-cu124}          # cu121 for driver <525, cu128 for Blackwell

eval "$(conda shell.bash hook)"
conda create -y -n "$ENV" python="$PY"
conda activate "$ENV"

# newer runtime libs inside the env, so conda-built extensions can't hit the system one
conda install -y -c conda-forge "libstdcxx-ng>=13" "libgcc-ng>=13"

python -m pip install -U pip wheel
python -m pip install "torch>=2.6" --index-url "https://download.pytorch.org/whl/${CUDA}"
python -m pip install -r requirements.txt
python -m pip install charset-normalizer            # silences the requests warning

echo "--- import check ---"
python - <<'PY'
import scipy.spatial, sklearn.metrics, torch, transformers
print("scipy/sklearn ok | torch", torch.__version__, "| transformers", transformers.__version__)
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY

echo "--- tests ---"
python -m pytest -q tests
python scripts/check_env.py

cat <<'EOF'

Next:
  conda activate vgs
  hf auth login                # MedGemma is gated: accept its licence on the model page first
  bash scripts/smoke_16gb.sh   # 50-question sanity check
EOF
