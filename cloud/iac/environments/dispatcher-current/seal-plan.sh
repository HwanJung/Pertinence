#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 SAVED_TFPLAN PLAN_JSON PROFILE_TFVARS GIT_REVISION NEW_RECEIPT" >&2
  exit 2
fi
for input in "$1" "$2" "$3"; do
  [[ -f "$input" && ! -L "$input" ]] || { echo "unsafe or missing input: $input" >&2; exit 2; }
done
[[ "$4" =~ ^[0-9a-f]{40}$ ]] || { echo "git revision must be a full commit" >&2; exit 2; }
[[ ! -e "$5" ]] || { echo "receipt already exists" >&2; exit 2; }

PLAN=$(realpath -- "$1")
PLAN_JSON=$(realpath -- "$2")
PROFILE=$(realpath -- "$3")
RECEIPT_PARENT=$(realpath -- "$(dirname -- "$5")")
RECEIPT="$RECEIPT_PARENT/$(basename -- "$5")"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
GUARD="$REPO/scripts/terraform_plan_guard.py"
RECEIPT_HELPER="$SCRIPT_DIR/plan-receipt.py"
[[ -f "$GUARD" && ! -L "$GUARD" ]] || { echo "unsafe or missing guard: $GUARD" >&2; exit 2; }
[[ -f "$RECEIPT_HELPER" && ! -L "$RECEIPT_HELPER" ]] || {
  echo "unsafe or missing receipt helper: $RECEIPT_HELPER" >&2
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
  echo "provided plan JSON differs from terraform show -json of the saved plan" >&2
  exit 1
fi
if ! python3 "$GUARD" --mode benchmark-pools-create "$REGENERATED_JSON" \
  >"$GUARD_OUTPUT" 2>&1; then
  cat "$GUARD_OUTPUT" >&2
  exit 1
fi

python3 "$RECEIPT_HELPER" seal \
  --plan "$PLAN" \
  --plan-json "$PLAN_JSON" \
  --profile "$PROFILE" \
  --git-revision "$4" \
  --receipt "$RECEIPT" \
  --guard "$GUARD" \
  --guard-output "$GUARD_OUTPUT"
