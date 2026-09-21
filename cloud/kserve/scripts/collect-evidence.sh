#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 EXPECTED_KUBE_CONTEXT NEW_OUTPUT_DIR SAMPLE_COUNT INTERVAL_SECONDS" >&2
  exit 2
fi
expected_context=$1
output_dir=$2
sample_count=$3
interval_seconds=$4
[[ "$sample_count" =~ ^[1-9][0-9]{0,3}$ ]] || { echo "invalid sample count" >&2; exit 2; }
[[ "$interval_seconds" =~ ^([1-9]|[1-5][0-9]|60)$ ]] || {
  echo "interval must be 1 through 60 seconds" >&2
  exit 2
}
[[ ! -e "$output_dir" ]] || { echo "output already exists: $output_dir" >&2; exit 2; }

actual_context=$(kubectl config current-context)
[[ "$actual_context" == "$expected_context" ]] || {
  echo "refusing collection: current context '$actual_context' != '$expected_context'" >&2
  exit 2
}
mkdir -m 0700 -- "$output_dir"
collection_complete=false
on_exit() {
  status=$?
  trap - EXIT
  if [[ "$collection_complete" != true ]]; then
    date --iso-8601=ns >"$output_dir/failed-at.txt" 2>/dev/null || true
    kubectl --context "$expected_context" get nodes -o json \
      >"$output_dir/failure-nodes.json" 2>/dev/null || true
    kubectl --context "$expected_context" -n pertinence-serving get pods -o json \
      >"$output_dir/failure-pods.json" 2>/dev/null || true
    printf '{"schema_version":1,"valid":false,"reason":"evidence_collection_failed","exit_code":%d}\n' \
      "$status" >"$output_dir/INVALID.json" 2>/dev/null || true
    chmod -R go-rwx -- "$output_dir" 2>/dev/null || true
  fi
  exit "$status"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

date --iso-8601=ns >"$output_dir/started-at.txt"
kubectl --context "$expected_context" version -o json >"$output_dir/kubernetes-version.json"
kubectl --context "$expected_context" get nodes -o json >"$output_dir/nodes-start.json"
kubectl --context "$expected_context" -n pertinence-serving get pods -o json \
  >"$output_dir/pods-start.json"
kubectl --context "$expected_context" -n pertinence-serving get inferenceservices -o json \
  >"$output_dir/isvc-start.json"
kubectl --context "$expected_context" -n pertinence-serving get deployments,services,jobs -o json \
  >"$output_dir/workloads-start.json"
kubectl --context "$expected_context" -n kubeflow get workflows.argoproj.io -o json \
  >"$output_dir/kfp-workflows-start.json"

mapfile -t serving_nodes < <(
  kubectl --context "$expected_context" get nodes \
    -l cloud.google.com/gke-nodepool=serving-pool \
    -o 'jsonpath={range .items[*]}{.metadata.name}{"\n"}{end}'
)
mapfile -t loadgen_nodes < <(
  kubectl --context "$expected_context" get nodes \
    -l cloud.google.com/gke-nodepool=loadgen-pool \
    -o 'jsonpath={range .items[*]}{.metadata.name}{"\n"}{end}'
)
nodes=("${serving_nodes[@]}" "${loadgen_nodes[@]}")
[[ ${#serving_nodes[@]} -eq 1 && ${#loadgen_nodes[@]} -eq 1 ]] || {
  echo "expected exactly one serving node and one loadgen node during benchmark" >&2
  exit 1
}

for ((sample = 1; sample <= sample_count; sample++)); do
  prefix=$(printf '%s/sample-%04d' "$output_dir" "$sample")
  date --iso-8601=ns >"$prefix.timestamp.txt"
  kubectl --context "$expected_context" get --raw /apis/metrics.k8s.io/v1beta1/nodes \
    >"$prefix.node-metrics.json"
  kubectl --context "$expected_context" get \
    --raw /apis/metrics.k8s.io/v1beta1/namespaces/pertinence-serving/pods \
    >"$prefix.pod-metrics.json"
  kubectl --context "$expected_context" top node >"$prefix.top-nodes.txt"
  kubectl --context "$expected_context" -n pertinence-serving top pod --containers \
    >"$prefix.top-pods.txt"
  for node in "${nodes[@]}"; do
    [[ "$node" =~ ^[a-z0-9.-]+$ ]] || { echo "unsafe node name: $node" >&2; exit 1; }
    kubectl --context "$expected_context" get \
      --raw "/api/v1/nodes/$node/proxy/metrics/cadvisor" \
      >"$prefix.cadvisor-$node.prom"
  done
  if ((sample < sample_count)); then
    sleep "$interval_seconds"
  fi
done

kubectl --context "$expected_context" get nodes -o json >"$output_dir/nodes-end.json"
kubectl --context "$expected_context" -n pertinence-serving get pods -o json \
  >"$output_dir/pods-end.json"
kubectl --context "$expected_context" -n pertinence-serving get events \
  --sort-by=.lastTimestamp -o json \
  >"$output_dir/events-end.json"
kubectl --context "$expected_context" -n kubeflow get workflows.argoproj.io -o json \
  >"$output_dir/kfp-workflows-end.json"
date --iso-8601=ns >"$output_dir/finished-at.txt"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
python3 "$script_dir/validate_snapshot.py" \
  "$output_dir/nodes-start.json" "$output_dir/nodes-end.json" \
  "$output_dir/pods-start.json" "$output_dir/pods-end.json" \
  "$output_dir/kfp-workflows-start.json" "$output_dir/kfp-workflows-end.json" \
  >"$output_dir/validation.json"
chmod -R go-rwx -- "$output_dir"
collection_complete=true
echo "evidence collected without overwriting: $output_dir"
