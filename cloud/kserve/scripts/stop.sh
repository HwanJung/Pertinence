#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 EXPECTED_KUBE_CONTEXT" >&2
  exit 2
fi
expected_context=$1
actual_context=$(kubectl config current-context)
[[ "$actual_context" == "$expected_context" ]] || {
  echo "refusing stop: current context '$actual_context' != '$expected_context'" >&2
  exit 2
}

namespace=pertinence-serving
services=(
  expert-shufflenet
  expert-mobilenet
  expert-resnet56
  router-dispatcher-000733
)
status=0
for service in "${services[@]}"; do
  if ! kubectl --context "$expected_context" -n "$namespace" annotate \
      inferenceservice "$service" serving.kserve.io/stop=true --overwrite; then
    echo "failed to stop InferenceService/$service; continuing" >&2
    status=1
  fi
done
for job in pertinence-loadgen; do
  if ! kubectl --context "$expected_context" -n "$namespace" patch job "$job" \
      --type=merge -p '{"spec":{"suspend":true}}'; then
    echo "failed to suspend Job/$job; continuing" >&2
    status=1
  fi
done
for service in "${services[@]}"; do
  if ! observed=$(kubectl --context "$expected_context" -n "$namespace" get \
      inferenceservice "$service" \
      -o 'jsonpath={.metadata.annotations.serving\.kserve\.io/stop}'); then
    echo "failed to verify InferenceService/$service stop annotation" >&2
    status=1
    continue
  fi
  if [[ "$observed" != true ]]; then
    echo "InferenceService/$service stop annotation did not persist" >&2
    status=1
  fi
done
for job in pertinence-loadgen; do
  if ! observed=$(kubectl --context "$expected_context" -n "$namespace" get \
      job "$job" -o 'jsonpath={.spec.suspend}'); then
    echo "failed to verify Job/$job suspension" >&2
    status=1
    continue
  fi
  if [[ "$observed" != true ]]; then
    echo "Job/$job suspension did not persist" >&2
    status=1
  fi
done

if (( status != 0 )); then
  echo "stop was incomplete; no object was deleted" >&2
  exit "$status"
fi
echo "four InferenceServices stopped and the loadgen Job suspended; no object was deleted"
