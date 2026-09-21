# KServe benchmark 로컬 세션 게이트

`cloud/kserve/scripts/session-gate.py`는 20단계 실행의 첫 로컬 gate다. 네트워크나 cloud API를
호출하지 않으며, 모든 사람 승인·검토 파일·image inventory를 검증한 뒤에만 새 evidence root를
만든다. 경로가 이미 있거나 symlink이면 재사용하지 않는다.

## 입력

입력 파일은 shell로 실행하지 않는 엄격한 `NAME=value` 형식이다. 빈 값, 공백, 중복, 알 수 없는
변수와 `REPLACE`, `__REQUIRED__` 같은 placeholder는 거부한다. 다음 값을 모두 제공해야 한다.

- 신원: `EXPECTED_PRINCIPAL`, project `pertinence-509116`에 pre-existing한 `BENCHMARK_GSA`,
  `APPROVED_BY`, UTC `APPROVED_AT`
- 새 실행: 절대 경로 `EVIDENCE_ROOT`, 서로 다른 DNS label `RUN_SET_ID`,
  `BENCHMARK_ID_FINAL`, `BENCHMARK_ID_C101`, 고유 `BUILD_TAG`, `APPROVED_PROMOTION_ID`
- 사람 승인: 독립 `APPROVED_FINAL_LABEL_SHA256`, finite non-negative
  `GOLDEN_LOGITS_ATOL`, `DIRTY_WORKTREE_APPROVED=true`, 승인한
  `WORKTREE_SNAPSHOT_SHA256`. gate는 현재 HEAD 대비 tracked 수정·삭제와 모든 non-ignored untracked
  파일의 경로, type, mode, size, bytes를 다시 canonical inventory로 계산해 이 값과 exact 비교한다.
- 검토 코드: `REVIEWED_INTEGRATION_DRIVER`와 `REVIEWED_PARITY_DRIVER`의 절대 경로 및 각각의
  `_SHA256`
- immutable destination: 서로 다른 `IMMUTABLE_BUILD_BUNDLE_PREFIX`, `EVALUATION_NPZ_URI`. 둘 다
  glob 없는 `gs://pertinence-509116-pertinence-runs/pertinence/...` exact path여야 하며 각각
  `BUILD_TAG`, `RUN_SET_ID` path segment로 분리되어야 한다.
- image 조사: `CONTROLLER_IMAGE_INVENTORY`, `BASE_IMAGE_INVENTORY`의 절대 경로 및 각각의
  `_SHA256`

두 inventory는 `schema_version=1`, UTC `queried_at`, 공백 없는 `crane_version`, `images`만 갖는
strict JSON object다. 각 image도 `source`, `digest`, `os`, `architecture` 외 field를 허용하지 않는다.
controller inventory는 계획에 명시된 cert-manager 5개와 KServe 6개 source tag를 정확히 한 번씩,
base inventory는 `python:3.11-slim`, `busybox:1.37.0`을 정확히 한 번씩 포함해야 한다. 모든 digest는
`sha256:<64 lowercase hex>`, platform은 `linux`/`amd64`여야 한다.

```json
{
  "schema_version": 1,
  "queried_at": "2026-09-21T01:00:00Z",
  "crane_version": "v0.20.7",
  "images": [
    {
      "source": "python:3.11-slim",
      "digest": "sha256:<linux-amd64-child-digest>",
      "os": "linux",
      "architecture": "amd64"
    }
  ]
}
```

## 실행

승인자는 먼저 현재 worktree inventory와 canonical SHA를 얻어 내용 전체를 검토한다.

```bash
.venv/bin/python cloud/kserve/scripts/session-gate.py --print-worktree-snapshot \
  > /secure/review/worktree-snapshot.json
```

출력의 `sha256`을 승인한 뒤 `WORKTREE_SNAPSHOT_SHA256`에 넣는다. 파일이 추가·수정·삭제되면 gate가
계산한 값이 달라져 실행이 중단된다.

```bash
.venv/bin/python cloud/kserve/scripts/session-gate.py \
  --env /secure/approved/session.env
```

일부 또는 전체 값은 반복 가능한 `--set NAME=value`로 줄 수 있다. 같은 변수를 env와 CLI 양쪽에
주면 모호성을 피하기 위해 실패한다. 성공하면 계획의 evidence directory skeleton을 mode `0700`으로
만들고 `00-session/session-gate-receipt.json`을 mode `0600`, `O_EXCL`로 봉인한다. receipt의
`cloud_calls_performed`는 `false`다. 실패하면 `EVIDENCE_ROOT`를 만들지 않는다.
