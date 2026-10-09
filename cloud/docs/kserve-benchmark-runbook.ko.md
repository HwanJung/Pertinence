# KServe 비교 benchmark runbook

실행 정본은 [Plan](benchmark/Plan.md), 상태 정본은 [Progress](benchmark/Progress.md), 세부 명령은
[실행 절차](kserve-benchmark-execution.ko.md)다.

## 안전 경계

- workload image/modelcar는 digest로 고정한다.
- 세 expert와 dispatcher router는 replica 1, 내부 전용으로만 배포한다.
- loadgen Job은 suspended 상태로 먼저 배포하고 preflight 완료 후 한 번만 시작한다.
- benchmark GSA에는 final trace read와 새 run prefix create 권한만 부여한다.
- 기존 GCS object, image, GSA, IAM binding, KFP deployment를 삭제하지 않는다.
- 결과 게시와 로컬 output은 create-only이며 실패한 raw evidence도 보존한다.

## 실행 순서

1. 저장소 검증과 build input generation/hash 검증
2. 새 runtime/modelcar build와 promotion 생성
3. cluster/quota/IAM/내부 노출 preflight
4. 4개 InferenceService와 suspended Job 배포
5. Ready 및 active KFP workflow 0 확인
6. direct ResNet56 측정 후 dispatcher 측정
7. sidecar 완료, summary 생성, GCS create-only 게시
8. Job/InferenceService stop 및 node pool 0 확인

어느 단계든 요청 오류, identity mismatch, schedule miss, matrix 불완전, offline 기준 불일치 또는
Pod restart가 발견되면 run을 invalid로 남기고 안전 정지한다.
