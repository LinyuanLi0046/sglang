#!/usr/bin/env bash
# WeLM v4 OpenCompass accuracy benchmark entrypoint.
# Usage:
#   1. Edit MODEL_PATH / OPENAI_BASE_URL below (or override via env).
#   2. bash run.sh 2>&1 | tee run.log
#
# Outputs will be written to ./outputs.

set -euo pipefail

# Move to the script's own directory so relative paths are stable.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ----- User-configurable variables -----------------------------------------
# Path to the WeLM v4 weight directory. Used both as the sglang `--model-path`
# AND as the local tokenizer path inside the OpenCompass evaluator.
export model_path="${model_path:-{MODEL_PATH}}"

# Base URL of the running sglang server (the `/v1/completions` endpoint).
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://{HOST_IP}:{PORT}/v1/completions}"
# ---------------------------------------------------------------------------

# OpenCompass uses this directory as the dataset cache. First run will
# auto-download the referenced datasets from HuggingFace mirror.
export COMPASS_DATA_CACHE="${COMPASS_DATA_CACHE:-$(pwd)/datasets}"

# WeLM API adapter requires a non-empty OPENAI_API_KEY (value is irrelevant
# when the backend is sglang).
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# Sanity check: refuse to run with the placeholder values.
if [[ "${model_path}" == *"{MODEL_PATH}"* ]]; then
  echo "[run.sh] ERROR: please set \$model_path before running." >&2
  exit 1
fi
if [[ "${OPENAI_BASE_URL}" == *"{HOST_IP}"* || "${OPENAI_BASE_URL}" == *"{PORT}"* ]]; then
  echo "[run.sh] ERROR: please set \$OPENAI_BASE_URL before running." >&2
  exit 1
fi

# Regenerate eval.py from eval_template.py + the model dict.
rm -f eval.py
python3 generate_script.py

opencompass eval.py
