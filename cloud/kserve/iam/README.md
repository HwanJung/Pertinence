# Benchmark Workload Identity boundary

`benchmark-runner` KSA는 전용 GSA를 Workload Identity로 사용한다. JSON service-account key는 만들거나
mount하지 않는다. GSA bucket 권한은 다음 두 conditional binding만 허용한다.

- final trace prefix: `roles/storage.objectViewer`
- 새 benchmark run prefix: `roles/storage.objectCreator`

`objectAdmin`, bucket-wide writer, project Editor/Owner는 금지한다. `objectCreator`는 overwrite/delete를
허용하지 않으므로 create-only 결과 게시와 맞는다.

```bash
cloud/kserve/scripts/verify-benchmark-iam.py \
  --project "$PROJECT" --bucket "$RUN_BUCKET" --gsa "$BENCHMARK_GSA" \
  --run-set-id "$RUN_SET_ID" --benchmark-id "$BENCHMARK_ID" \
  --gsa-policy "$GSA_POLICY_JSON" --bucket-policy "$BUCKET_POLICY_JSON" \
  --service-account-keys "$GSA_KEYS_JSON" --output "$IAM_RECEIPT_JSON"
```

조건 title은 `trace-final-RUN_SET_ID`, `run-RUN_SET_ID`이고 expression은 승인된 object prefix에 대한
정확한 `resource.name.startsWith('.../')`여야 한다.
