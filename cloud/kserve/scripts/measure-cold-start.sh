#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 EXPECTED_KUBE_CONTEXT WORKLOAD_YAML V2_REQUEST_JSON NEW_OUTPUT_DIR" >&2
  exit 2
fi
expected_context=$1
workload_yaml=$2
payload=$3
output_dir=$4
[[ -f "$workload_yaml" && ! -L "$workload_yaml" ]] || { echo "invalid workload YAML" >&2; exit 2; }
[[ -f "$payload" && ! -L "$payload" ]] || { echo "invalid request JSON" >&2; exit 2; }
[[ ! -e "$output_dir" ]] || { echo "output already exists: $output_dir" >&2; exit 2; }
actual_context=$(kubectl config current-context)
[[ "$actual_context" == "$expected_context" ]] || {
  echo "refusing measurement: current context '$actual_context' != '$expected_context'" >&2
  exit 2
}

mkdir -m 0700 -- "$output_dir"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
port_forward_pid=
measurement_complete=false
cleanup() {
  if [[ -n "$port_forward_pid" ]]; then
    kill "$port_forward_pid" 2>/dev/null || true
    wait "$port_forward_pid" 2>/dev/null || true
  fi
  if [[ "$measurement_complete" != true ]]; then
    "$script_dir/stop.sh" "$expected_context" \
      >"$output_dir/failure-stop.txt" 2>&1 || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

existing_pods=$(kubectl --context "$expected_context" -n pertinence-serving get pods \
  -l app.kubernetes.io/part-of=pertinence --no-headers)
[[ -z "$existing_pods" ]] || {
  echo "cold-start requires zero existing pertinence serving Pods" >&2
  printf '%s\n' "$existing_pods" >&2
  exit 1
}

date --iso-8601=ns >"$output_dir/apply-started-at.txt"
start_ns=$(date +%s%N)
kubectl --context "$expected_context" apply --server-side -f "$workload_yaml" \
  >"$output_dir/apply.txt"
echo 'inferenceservice,ready_at_ns' >"$output_dir/isvc-ready-times.csv"
mapfile -t services < <(python3 - "$workload_yaml" <<'PY'
import pathlib, sys, yaml
for document in yaml.safe_load_all(pathlib.Path(sys.argv[1]).read_text()):
    if isinstance(document, dict) and document.get("kind") == "InferenceService":
        print(document["metadata"]["name"])
PY
)
[[ ${#services[@]} -eq 5 ]] || { echo "expected exactly five InferenceServices" >&2; exit 1; }
for inference_service in "${services[@]}"; do
  kubectl --context "$expected_context" -n pertinence-serving annotate \
    "inferenceservice/$inference_service" serving.kserve.io/stop- \
    >>"$output_dir/unstop.txt" 2>&1
done
for inference_service in "${services[@]}"; do
  kubectl --context "$expected_context" -n pertinence-serving wait \
    --for=condition=Ready --timeout=15m \
    "inferenceservice/$inference_service" >>"$output_dir/wait-ready.txt"
  printf '%s,%s\n' "$inference_service" "$(date +%s%N)" \
    >>"$output_dir/isvc-ready-times.csv"
done
ready_ns=$(date +%s%N)
date --iso-8601=ns >"$output_dir/all-ready-at.txt"
printf 'apply_to_all_ready_ns,%s\n' "$((ready_ns - start_ns))" \
  >"$output_dir/cold-start-summary.csv"
kubectl --context "$expected_context" -n pertinence-serving \
  get inferenceservices,pods -o json \
  >"$output_dir/ready-inventory.json"

services=(expert-shufflenet-predictor expert-mobilenet-predictor expert-resnet56-predictor router-static-resnet-predictor router-dispatcher-000733-predictor)
models=(shufflenetv2_x0_5 mobilenetv2_x0_5 resnet56 static-resnet-router dispatcher-000733)
identities=(shufflenetv2_x0_5 mobilenetv2_x0_5 resnet56 static dispatcher-000733)
echo 'arm,request,started_at,total_seconds,http_code' >"$output_dir/first-20.csv"
for arm_index in "${!services[@]}"; do
  service_name=${services[$arm_index]}
  model_name=${models[$arm_index]}
  expected_identity=${identities[$arm_index]}
  mkdir -m 0700 "$output_dir/$model_name"
  kubectl --context "$expected_context" -n pertinence-serving \
    port-forward "service/$service_name" 18080:80 \
    >"$output_dir/$model_name/port-forward.log" 2>&1 &
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
    cat "$output_dir/$model_name/port-forward.log" >&2
    exit 1
  }
  [[ "$endpoint_ready" == true ]] || {
    echo "timed out waiting for $model_name health readiness" >&2
    exit 1
  }
  for request in $(seq 1 20); do
    response="$output_dir/$model_name/response-$(printf '%02d' "$request").json"
    started_at=$(date --iso-8601=ns)
    measurement=$(curl --silent --show-error --max-time 15 \
      --header 'Connection: close' \
      --header 'Content-Type: application/json' \
      --data-binary "@$payload" \
      --output "$response" \
      --write-out '%{time_total},%{http_code}' \
      "http://127.0.0.1:18080/v2/models/$model_name/infer")
    [[ "$measurement" == *,200 ]] || {
      echo "cold request $model_name/$request failed: $measurement" >&2
      exit 1
    }
    python3 "$script_dir/assert_identity.py" "$response" "$expected_identity"
    printf '%s,%s,%s,%s\n' "$model_name" "$request" "$started_at" "$measurement" \
      >>"$output_dir/first-20.csv"
  done
  kill "$port_forward_pid"
  wait "$port_forward_pid" 2>/dev/null || true
  port_forward_pid=
done
chmod -R go-rwx -- "$output_dir"
measurement_complete=true
echo "cold-start evidence captured without overwriting: $output_dir"
