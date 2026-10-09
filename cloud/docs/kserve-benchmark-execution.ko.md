# KServe 비교 benchmark 실행 절차

## 1. 로컬 검증

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check .
```

`experiments/cifar10/configs/benchmark.yaml`이 final trace 하나, arm 순서
`resnet56 -> dispatcher-000733`, arm별 200 warm-up, 2,000 measured requests, 2 RPS를 선언하는지
확인한다.

## 2. build와 promotion

검증된 build-input manifest의 11개 선택 object generation과 SHA-256만 재사용한다. 변경된 코드를
포함하는 runtime, 세 expert modelcar, dispatcher modelcar를 새 고유 tag로 한 번 build한다. 생성된
digest를 사용해 새 immutable promotion을 만들고, 기존 image나 promotion을 덮어쓰지 않는다.

## 3. IAM과 render

benchmark GSA에는 다음 두 conditional binding만 허용한다.

- `cifar10-final-20260824` trace prefix의 `roles/storage.objectViewer`
- 새 benchmark run prefix의 `roles/storage.objectCreator`

```bash
cloud/kserve/scripts/verify-benchmark-iam.py \
  --project "$PROJECT" --bucket "$RUN_BUCKET" --gsa "$BENCHMARK_GSA" \
  --run-set-id "$RUN_SET_ID" --benchmark-id "$BENCHMARK_ID" \
  --gsa-policy "$GSA_POLICY_JSON" --bucket-policy "$BUCKET_POLICY_JSON" \
  --service-account-keys "$GSA_KEYS_JSON" --output "$IAM_RECEIPT_JSON"

cloud/kserve/scripts/render-manifests.sh "$IMAGES_ENV" "$RENDERED_DIR"
```

render 검증은 4개 InferenceService, Job 하나, endpoint 두 개의 정확한 순서와 외부 노출 object 0을
검사한다.

## 4. cluster preflight와 배포

quota, context/project, KServe addon, serving/loadgen node pool, active KFP workflow, 외부 노출을
read-only로 확인한다. manifest를 server-side dry-run한 뒤 적용하고 모든 endpoint Ready를 기다린다.
Job은 이 시점까지 `suspend: true`여야 한다.

```bash
kubectl --context "$KUBE_CONTEXT" apply --server-side --dry-run=server -f "$RENDERED_DIR"
kubectl --context "$KUBE_CONTEXT" apply --server-side -f "$RENDERED_DIR"
kubectl --context "$KUBE_CONTEXT" -n pertinence-serving wait \
  --for=condition=Ready --timeout=15m inferenceservice --all
```

## 5. 측정과 게시

active KFP workflow가 0이고 Pod restart count가 기준값과 같은지 다시 확인한 뒤 Job의 suspend를
해제한다. runner는 direct ResNet56 warm-up/측정, dispatcher warm-up/측정을 순서대로 수행한다.
sidecar는 5초마다 cAdvisor evidence를 기록한다.

sidecar 완료 후 같은 Job에서 request matrix, offline 기준, latency와 serving 자원을 분석해
`summary.json`을 만든다. 그 뒤에만 결과 directory 전체를 새 GCS prefix에
`ifGenerationMatch=0`으로 한 번 게시한다.

필수 결과는 `run.json`, `markers.jsonl`, `requests.jsonl`, `requests.parquet`, `evidence.jsonl`,
`resources.json`, `validity.json`, `summary.json`, `loadgen-complete.json`이다. 실패 시 생성된 파일과
`INVALID.json`을 보존한다.

## 6. 종료

```bash
cloud/kserve/scripts/stop.sh "$KUBE_CONTEXT"
```

네 InferenceService stop annotation, Job suspend, serving/loadgen/pipeline pool 0을 확인한다. KFP
deployment와 기존 artifact는 보존한다. 최종 결과와 안전 정지가 모두 확인되기 전에는 Progress를
완료로 표시하지 않는다.
