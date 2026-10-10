#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 ENV_FILE NEW_OUTPUT_DIRECTORY" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
python_bin=${PYTHON_BIN:-python3}

"$python_bin" "$script_dir/render-single-arm-v2.py" --env "$1" --output-dir "$2"
"$python_bin" "$script_dir/validate-single-arm-v2.py" --arm-env "$1" \
  "$2/00-platform.yaml" \
  "$2/05-benchmark-workload-identity.yaml" \
  "$2/10-target.yaml" \
  "$2/20-single-arm-job.yaml"
echo "rendered and validated single arm: $2"
