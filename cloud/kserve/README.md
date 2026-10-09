# KServe benchmark assets

이 디렉터리는 direct ResNet56과 dispatcher를 내부 전용 KServe 환경에서 비교하는 배포 자산을
담는다.

## 구성

- `manifests/`: 세 expert, dispatcher router, final loadgen Job, Workload Identity
- `runtime/`: Python 3.11 benchmark/serving runtime image
- `modelcar/`: 세 ONNX model과 dispatcher bundle용 OCI image
- `scripts/`: create-only render, preflight, IAM 검증, exposure audit, 안전 정지

render된 workload는 정확히 4개 InferenceService와 suspended Job 하나를 포함한다. loadgen endpoint는
direct `resnet56` 다음 `dispatcher-000733` 순서다. Ingress, Gateway, HTTPRoute, LoadBalancer Service는
허용하지 않는다.

```bash
cloud/kserve/scripts/render-manifests.sh \
  /secure/path/benchmark-images.env \
  /tmp/pertinence-kserve-rendered
```

상세 계약은 [benchmark plan](../docs/benchmark/Plan.md)과
[실행 절차](../docs/kserve-benchmark-execution.ko.md)를 따른다. 종료 시 `stop.sh`로 네
InferenceService를 stop하고 loadgen Job을 suspend한다. 기존 object나 artifact를 삭제하지 않는다.
