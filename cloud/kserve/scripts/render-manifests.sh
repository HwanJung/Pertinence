#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 ENV_FILE NEW_OUTPUT_DIRECTORY" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
root_dir=$(cd -- "$script_dir/.." && pwd -P)
python_bin=${PYTHON_BIN:-python3}

"$python_bin" "$script_dir/render_config.py" \
  --env "$1" \
  --output-dir "$2" \
  "$root_dir/manifests/00-platform.yaml.tmpl=00-platform.yaml" \
  "$root_dir/manifests/05-benchmark-workload-identity.yaml.tmpl=05-benchmark-workload-identity.yaml" \
  "$root_dir/manifests/10-workloads.yaml.tmpl=10-workloads.yaml"

"$python_bin" "$script_dir/validate_rendered.py" \
  "$2/00-platform.yaml" \
  "$2/05-benchmark-workload-identity.yaml" \
  "$2/10-workloads.yaml"
echo "rendered and validated: $2"
