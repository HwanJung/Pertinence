# Benchmark Workload Identity boundary

Kubernetes 정본은 `benchmark-runner` KSA와 대상 GSA annotation만 정의한다. 실제 Google IAM
binding은 project/bucket 소유자가 create-only로 별도 적용하고 실행 기록에 policy etag와 명령
출력을 남긴다. JSON service-account key는 만들거나 mount하지 않는다.

요구되는 권한은 다음 두 conditional bucket binding뿐이다.

- immutable trace prefix: `roles/storage.objectViewer`, condition
  `resource.name.startsWith("projects/_/buckets/pertinence-509116-pertinence-runs/objects/pertinence/benchmarks/traces/TRACE_ID/")`
- 새 run prefix: `roles/storage.objectCreator`, condition
  `resource.name.startsWith("projects/_/buckets/pertinence-509116-pertinence-runs/objects/pertinence/benchmarks/runs/BENCHMARK_ID/")`

GSA impersonation은 `roles/iam.workloadIdentityUser`를 다음 member 하나에만 부여한다.

```text
serviceAccount:PROJECT_ID.svc.id.goog[pertinence-serving/benchmark-runner]
```

`objectAdmin`, bucket-wide writer, project Editor/Owner는 금지한다. `objectCreator`는 overwrite와
delete를 허용하지 않으므로 create-only 결과 보존에 맞는다. 실제 적용 전에 trace/run prefix와
GSA/KSA principal을 run manifest와 대조하고, 적용 뒤 `gcloud storage buckets get-iam-policy`와
`gcloud iam service-accounts get-iam-policy`로 read-only 검증한다.

정책 적용 전후 snapshot은 `verify-benchmark-iam.py`로 검증한다. 검증기는 cloud API를 호출하지
않으며 전달된 JSON 파일만 읽고 create-only receipt를 기록한다. 조건 title은 각각
`trace-final-RUN_SET_ID`, `trace-c101-RUN_SET_ID`, `run-final-RUN_SET_ID`,
`run-c101-RUN_SET_ID`이고, expression은 대상 object prefix를 작은따옴표로 감싼 정확한
`resource.name.startsWith('.../')` 형식이어야 한다.

```bash
cloud/kserve/scripts/verify-benchmark-iam.py \
  --project "$PROJECT" \
  --bucket "$RUN_BUCKET" \
  --gsa "$BENCHMARK_GSA" \
  --run-set-id "$RUN_SET_ID" \
  --benchmark-id-final "$BENCHMARK_ID_FINAL" \
  --benchmark-id-c101 "$BENCHMARK_ID_C101" \
  --gsa-policy "$GSA_POLICY_JSON" \
  --bucket-policy "$BUCKET_POLICY_JSON" \
  --service-account-keys "$GSA_KEYS_JSON" \
  --output "$IAM_RECEIPT_JSON"
```

`--service-account-keys`에는 `gcloud iam service-accounts keys list --format=json`의 read-only
출력을 전달한다. `USER_MANAGED` key가 하나라도 있으면 실패한다. Google이 관리하는
`SYSTEM_MANAGED` key metadata는 허용한다.
