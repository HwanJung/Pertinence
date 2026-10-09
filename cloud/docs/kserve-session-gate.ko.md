# KServe benchmark 로컬 세션 게이트

`cloud/kserve/scripts/session-gate.py`는 cloud mutation 전에 승인된 로컬 입력을 검증하고 새 evidence
root를 create-only로 만든다. 네트워크나 cloud API는 호출하지 않는다.

입력은 실행하지 않는 `NAME=value` 파일이다. principal/GSA, `RUN_SET_ID`, 단일 `BENCHMARK_ID`,
build tag/promotion ID, 승인자와 UTC 승인 시각, final label hash, integration/parity driver hash,
generation-scoped build/evaluation GCS path, controller/base image inventory, 승인된 worktree snapshot을
검증한다. placeholder, tag-only image, sentinel digest, symlink, 기존 output path는 거부한다.

```bash
.venv/bin/python cloud/kserve/scripts/session-gate.py --print-worktree-snapshot \
  > /secure/review/worktree-snapshot.json

.venv/bin/python cloud/kserve/scripts/session-gate.py \
  --env /secure/approved/session.env
```

성공하면 private evidence directory와 `00-session/session-gate-receipt.json`을 생성한다. benchmark
dataset directory는 승인된 `BENCHMARK_ID` 하나만 만든다. 세션 게이트를 반복 실행해 기존 directory를
재봉인하지 않으며, 입력이 바뀌면 새 ID와 새 evidence root를 사용한다.
