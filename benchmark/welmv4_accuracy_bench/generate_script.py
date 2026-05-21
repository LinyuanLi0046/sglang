"""Generate the final OpenCompass eval.py.

Usage:
    Set environment variables ``model_path`` and ``OPENAI_BASE_URL``, then run:

        python3 generate_script.py

The script reads ``eval_template.py`` (which contains the fixed ``datasets=``
list) and appends a ``models=`` block populated from the environment, writing
the combined config to ``eval.py``.
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE_PATH = HERE / "eval_template.py"
OUTPUT_PATH = HERE / "eval.py"


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.stderr.write(
            f"[generate_script.py] ERROR: environment variable '{name}' is required.\n"
        )
        sys.exit(1)
    return value


model_path = _require_env("model_path")
openai_base_url = _require_env("OPENAI_BASE_URL")

models_block = f"""models = [
    dict(
        abbr=\"welm\",
        type=WeLM,
        path=\"{model_path}\",
        key=\"ENV\",
        openai_api_base=\"{openai_base_url}\",
        max_out_len=4,
        query_per_second=1e6,
        max_seq_len=8192,
        temperature=0.0,
        top_p=1.0,
        tokenizer_path=\"{model_path}\",
        extra_body=None,
        max_workers=1024,
        batch_size=102400,
    )
]
"""

template_content = TEMPLATE_PATH.read_text(encoding="utf-8")
OUTPUT_PATH.write_text(
    template_content + "\n" + models_block,
    encoding="utf-8",
)

print(f"[generate_script.py] wrote {OUTPUT_PATH}")
