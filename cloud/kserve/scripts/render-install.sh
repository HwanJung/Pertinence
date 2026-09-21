#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 CHART_LOCK_ENV NEW_OUTPUT_DIRECTORY" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
python_bin=${PYTHON_BIN:-python3}
"$python_bin" "$script_dir/render_install.py" "$1" "$2"
"$python_bin" "$script_dir/validate_rendered.py" \
  "$2/00-namespaces.yaml" "$2/cert-manager.yaml" "$2/kserve-crd.yaml" "$2/kserve-resources.yaml"
echo "verified chart packages and rendered immutable install manifests: $2"
