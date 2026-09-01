#!/usr/bin/env bash
set -e

BFCL_ROOT="$1"
REGISTRY_KEY="$2"
SERVED_MODEL_NAME="$3"
BASE_URL="${4:-http://localhost:8000/v1}"

python3 - "$BFCL_ROOT" "$REGISTRY_KEY" "$SERVED_MODEL_NAME" <<'PYEOF'
import re, sys
from pathlib import Path

repo_root, key, model_name = sys.argv[1], sys.argv[2], sys.argv[3]
config_path = Path(repo_root) / "bfcl_eval" / "constants" / "model_config.py"
text = config_path.read_text()

if f'"{key}": ModelConfig(' in text:
    sys.exit(0)

import_line = "from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler\n"
if import_line.strip() not in text:
    first_blank = text.index("\n\n")
    text = text[:first_blank] + "\n" + import_line + text[first_blank:]

entry = f'''    "{key}": ModelConfig(
        model_name="{model_name}",
        display_name="{key}",
        url="local",
        org="custom",
        license="apache-2.0",
        model_handler=QwenFCHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=False,
    ),
'''
match = re.search(r"(local_inference_model_map\s*=\s*\{)", text)
insert_at = match.end()
text = text[:insert_at] + "\n" + entry + text[insert_at:]
config_path.write_text(text)
PYEOF

export REMOTE_OPENAI_BASE_URL="${BASE_URL}"
export REMOTE_OPENAI_API_KEY="EMPTY"

pushd "${BFCL_ROOT}" > /dev/null
bfcl generate --model "${REGISTRY_KEY}" --test-category python --num-threads 8
bfcl evaluate --model "${REGISTRY_KEY}" --test-category python
popd > /dev/null
