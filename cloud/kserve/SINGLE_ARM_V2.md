# Sequential three-arm KServe targets

The fused comparison uses a separate, create-only renderer. It does not change the existing
two-arm deployment contract.

Copy `manifests/single-arm-v2.env.example` outside the repository and seal every value. Use the
same promotion, trace, runtime digest, serving node UID, and machine type in all three copies;
change `ARM_ID`, the unique Job/run IDs, and the output URI for each arm. The approved arm order is
`resnet56-direct`, `dispatcher-000733-fused`, then `dispatcher-000733-distributed`.

```bash
cloud/kserve/scripts/render-single-arm-v2.sh /secure/direct.env /secure/rendered-direct
cloud/kserve/scripts/render-single-arm-v2.sh /secure/fused.env /secure/rendered-fused
cloud/kserve/scripts/render-single-arm-v2.sh /secure/distributed.env /secure/rendered-distributed
```

Each output directory contains only its arm's InferenceServices and one suspended, uniquely named
`benchmark run-single` Job. The renderer rejects an existing output directory, mutable image tags,
sentinel hashes, non-5-RPS benchmark settings, lineage mismatches, and arm/endpoint substitutions.
It also derives immutable promotion and benchmark ConfigMap names from each `BENCHMARK_ID`; the
three arm manifests therefore never update the legacy ConfigMaps or one another's ConfigMaps.
Apply only one target at a time and unsuspend its Job only after the selected InferenceServices are
Ready. The single-arm runner performs the sealed 200-request warm-up before measurement.

The rendered predictor resources deliberately exclude KServe's injected modelcar sidecars. Capture
the actual serving Pods and Nodes after readiness, then produce immutable whole-Pod resource
evidence (the destination must not already exist):

```bash
python cloud/kserve/scripts/validate-single-arm-v2.py \
  --arm-env /secure/fused.env \
  --pod-snapshot /evidence/fused-pods.json \
  --node-snapshot /evidence/nodes.json \
  --resource-evidence /evidence/fused-pod-resources.json \
  /secure/rendered-fused/*.yaml
```

The evidence records every regular container, including the modelcar sidecar, its image and
request/limit, application-container sums, Kubernetes-effective totals (init-container maximum and
Pod overhead included), Pod UID, node name, node UID, and machine type. A serving Pod without a
modelcar sidecar is rejected. Continue to run `scripts/audit-exposure.sh` for the cluster-level
absence check covering public LoadBalancers, Ingresses, Gateways, and external forwarding rules.
