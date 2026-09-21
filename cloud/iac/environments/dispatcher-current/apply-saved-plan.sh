#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 SAVED_TFPLAN PROFILE_TFVARS RECEIPT_JSON" >&2
  exit 2
fi
for input in "$1" "$2" "$3"; do
  [[ -f "$input" && ! -L "$input" ]] || { echo "unsafe or missing input: $input" >&2; exit 2; }
done

PLAN=$(realpath -- "$1")
PROFILE=$(realpath -- "$2")
RECEIPT=$(realpath -- "$3")

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
GUARD="$REPO/scripts/terraform_plan_guard.py"
RECEIPT_HELPER="$SCRIPT_DIR/plan-receipt.py"
[[ -f "$GUARD" && ! -L "$GUARD" ]] || { echo "unsafe or missing guard: $GUARD" >&2; exit 2; }
[[ -f "$RECEIPT_HELPER" && ! -L "$RECEIPT_HELPER" ]] || {
  echo "unsafe or missing receipt helper: $RECEIPT_HELPER" >&2
  exit 2
}

PLAN_JSON=$(python3 - "$RECEIPT" <<'PY'
import json
import pathlib
import sys

receipt = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
try:
    value = receipt["plan_json"]["path"]
except (KeyError, TypeError):
    raise SystemExit("receipt does not contain a sealed plan JSON path")
if not isinstance(value, str) or not value:
    raise SystemExit("receipt plan JSON path is invalid")
print(value)
PY
)
[[ -f "$PLAN_JSON" && ! -L "$PLAN_JSON" ]] || {
  echo "unsafe or missing sealed plan JSON: $PLAN_JSON" >&2
  exit 2
}

TEMP_DIR=$(mktemp -d)
trap 'rm -rf -- "$TEMP_DIR"' EXIT
REGENERATED_JSON="$TEMP_DIR/plan.json"
GUARD_OUTPUT="$TEMP_DIR/guard-output.txt"

(
  cd "$SCRIPT_DIR"
  terraform show -json "$PLAN" >"$REGENERATED_JSON"
)
if ! cmp -s -- "$PLAN_JSON" "$REGENERATED_JSON"; then
  echo "sealed plan JSON differs from terraform show -json of the saved plan" >&2
  exit 1
fi
if ! python3 "$GUARD" --mode benchmark-pools-create "$REGENERATED_JSON" \
  >"$GUARD_OUTPUT" 2>&1; then
  cat "$GUARD_OUTPUT" >&2
  exit 1
fi

python3 "$RECEIPT_HELPER" verify \
  --plan "$PLAN" \
  --plan-json "$PLAN_JSON" \
  --profile "$PROFILE" \
  --receipt "$RECEIPT" \
  --guard "$GUARD" \
  --guard-output "$GUARD_OUTPUT"

(
  cd "$SCRIPT_DIR"
  terraform apply "$PLAN"
)
