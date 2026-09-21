#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 EXPECTED_KUBE_CONTEXT dispatcher|static V2_REQUEST_JSON" >&2
  exit 2
fi
expected_context=$1
mode=$2
payload=$3
namespace=pertinence-serving
service=pertinence-active

[[ -f "$payload" && ! -L "$payload" ]] || { echo "request JSON must be a regular file" >&2; exit 2; }
case "$mode" in
  dispatcher)
    target=router-dispatcher-000733
    expected_identity=dispatcher-000733
    model_name=dispatcher-000733
    ;;
  static)
    target=router-static-resnet
    expected_identity=static
    model_name=static-resnet-router
    ;;
  *)
    echo "mode must be dispatcher or static" >&2
    exit 2
    ;;
esac

actual_context=$(kubectl config current-context)
[[ "$actual_context" == "$expected_context" ]] || {
  echo "refusing switch: current context '$actual_context' != '$expected_context'" >&2
  exit 2
}
kubectl --context "$expected_context" -n "$namespace" wait \
  --for=condition=Ready --timeout=10m "inferenceservice/$target"

selector_path='{.spec.selector.serving\.kserve\.io/inferenceservice}'
previous=$(kubectl --context "$expected_context" -n "$namespace" \
  get service "$service" -o "jsonpath=$selector_path")
case "$previous" in
  router-dispatcher-000733|router-static-resnet) ;;
  *) echo "refusing switch from unexpected selector: $previous" >&2; exit 1 ;;
esac

rollback_required=true
port_forward_pid=
temporary=$(mktemp -d)
cleanup() {
  if [[ -n "$port_forward_pid" ]]; then
    kill "$port_forward_pid" 2>/dev/null || true
    wait "$port_forward_pid" 2>/dev/null || true
  fi
  if [[ "$rollback_required" == true ]]; then
    kubectl --context "$expected_context" -n "$namespace" \
      patch service "$service" --type=merge \
      -p "{\"spec\":{\"selector\":{\"serving.kserve.io/inferenceservice\":\"$previous\"}}}" \
      >/dev/null || true
    echo "active selector rolled back to $previous" >&2
  fi
  rm -rf -- "$temporary"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

kubectl --context "$expected_context" -n "$namespace" \
  patch service "$service" --type=merge \
  -p "{\"spec\":{\"selector\":{\"serving.kserve.io/inferenceservice\":\"$target\"}}}" \
  >/dev/null

drain_seconds=${DRAIN_SECONDS:-5}
[[ "$drain_seconds" =~ ^([0-9]|[1-5][0-9]|60)$ ]] || {
  echo "DRAIN_SECONDS must be an integer from 0 through 60" >&2
  exit 2
}
sleep "$drain_seconds"

kubectl --context "$expected_context" -n "$namespace" \
  port-forward "service/$service" 18080:80 \
  >"$temporary/port-forward.log" 2>&1 &
port_forward_pid=$!
endpoint_ready=false
for _ in $(seq 1 40); do
  if curl --fail --silent --show-error --max-time 1 \
      http://127.0.0.1:18080/v2/health/ready >/dev/null 2>&1; then
    endpoint_ready=true
    break
  fi
  sleep 0.25
done
kill -0 "$port_forward_pid" 2>/dev/null || {
  cat "$temporary/port-forward.log" >&2
  exit 1
}
[[ "$endpoint_ready" == true ]] || {
  echo "timed out waiting for active alias health readiness" >&2
  exit 1
}

for attempt in $(seq 1 20); do
  response="$temporary/response-$attempt.json"
  curl --fail --silent --show-error --max-time 15 \
    --header 'Connection: close' \
    --header 'Content-Type: application/json' \
    --data-binary "@$payload" \
    "http://127.0.0.1:18080/v2/models/$model_name/infer" >"$response"
  python3 "$(dirname -- "${BASH_SOURCE[0]}")/assert_identity.py" \
    "$response" "$expected_identity"
done

rollback_required=false
echo "active selector switched to $target; 20 fresh responses matched $expected_identity"
