#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 EXPECTED_KUBE_CONTEXT GCP_PROJECT_ID" >&2
  exit 2
fi
expected_context=$1
project_id=$2
[[ "$expected_context" =~ ^[A-Za-z0-9._:/-]+$ ]] || { echo "invalid context" >&2; exit 2; }
[[ "$project_id" =~ ^[a-z][a-z0-9-]{4,28}[a-z0-9]$ ]] || { echo "invalid project id" >&2; exit 2; }

actual_context=$(kubectl config current-context)
[[ "$actual_context" == "$expected_context" ]] || {
  echo "refusing audit: current context '$actual_context' != '$expected_context'" >&2
  exit 2
}

failures=0
load_balancers=$(kubectl --context "$expected_context" get service --all-namespaces \
  --field-selector spec.type=LoadBalancer --no-headers)
if [[ -n "$load_balancers" ]]; then
  echo "unexpected LoadBalancer Services:" >&2
  printf '%s\n' "$load_balancers" >&2
  failures=$((failures + 1))
fi

ingresses=$(kubectl --context "$expected_context" get ingress --all-namespaces --no-headers)
if [[ -n "$ingresses" ]]; then
  echo "unexpected Ingress resources:" >&2
  printf '%s\n' "$ingresses" >&2
  failures=$((failures + 1))
fi

gateway_resources=$(kubectl --context "$expected_context" api-resources \
  --api-group=gateway.networking.k8s.io \
  --verbs=list -o name)
if [[ "$gateway_resources" == *"gateways.gateway.networking.k8s.io"* ]]; then
  gateways=$(kubectl --context "$expected_context" \
    get gateways.gateway.networking.k8s.io \
    --all-namespaces --no-headers)
  if [[ -n "$gateways" ]]; then
    echo "unexpected Gateway resources:" >&2
    printf '%s\n' "$gateways" >&2
    failures=$((failures + 1))
  fi
fi

forwarding_rules=$(gcloud compute forwarding-rules list \
  --project "$project_id" \
  --filter="loadBalancingScheme~^EXTERNAL" \
  --format="value(name,region,IPAddress,loadBalancingScheme)")
if [[ -n "$forwarding_rules" ]]; then
  echo "unexpected external forwarding rules in project $project_id:" >&2
  printf '%s\n' "$forwarding_rules" >&2
  failures=$((failures + 1))
fi

if ((failures > 0)); then
  echo "external exposure audit failed ($failures categories)" >&2
  exit 1
fi
echo "external exposure audit passed: no LoadBalancer, Ingress, Gateway, or external forwarding rule"
