# Operator-Selected Cloud-PERTINENCE

## 1. 아이디어 요약

PERTINENCE 논문에서처럼 여러 전문가 모델 구성과 학습 설정으로 dispatcher를
미리 생성한다. 각 dispatcher의 시스템 정확도와 평균 MFLOPs를 측정해 전체
결과의 global Pareto front를 만든다.

클라우드가 부하나 모델 availability를 근거로 dispatcher를 자동 변경하지
않는다. 대신 운영자가 Pareto front를 확인하고 비용과 정확도 요구에 맞는
dispatcher를 선택하면, 클라우드 제어면이 검증된 dispatcher bundle을 안전하게
활성화한다.

```text
Offline plane
  모델 구성 정의
    → 논문 방식으로 dispatcher 학습 및 탐색
    → 독립 평가
    → 모든 결과를 합쳐 global Pareto front 생성
    → Dispatcher Registry 등록

Online plane
  운영자가 Pareto point 선택
    → 필요한 expert와 bundle 검증
    → 새 dispatcher 준비
    → active dispatcher 전환
    → 기존 버전 drain 또는 rollback 대기
```

이 접근의 핵심은 자동 최적화가 아니라 **논문의 여러 accuracy–compute
운용점을 클라우드에서 선택하고 교체할 수 있는 운영 구조**다.

## 2. 문제의식

PERTINENCE는 입력마다 적절한 전문가를 선택하지만, 실제 배포에서는 어떤
전문가 집합과 어떤 dispatcher 해를 사용할지 먼저 결정해야 한다. 논문은 여러
모델 조합과 penalty 설정에서 서로 다른 accuracy–compute 운용점을 만든다.

기존 방식에서는 하나의 운용점을 골라 배포한 뒤 다른 운용점으로 바꾸려면
다음 작업을 수동으로 다시 수행해야 한다.

- dispatcher 가중치 교체
- FC 출력 index와 expert endpoint 대응 확인
- 필요한 전문가 모델의 배포 상태 확인
- 새 버전 검증
- 장애 발생 시 이전 버전 복구

따라서 학습된 dispatcher들을 실행 가능한 bundle로 관리하고, 운영자가
Pareto front를 보고 선택한 운용점을 안전하게 활성화하는 제어면이 필요하다.

## 3. 목표와 비목표

### 목표

- PERTINENCE의 dispatcher 구조와 학습 방법을 유지한다.
- 여러 모델 구성과 NSGA-II 해를 하나의 global Pareto front로 통합한다.
- 운영자가 원하는 dispatcher를 명시적으로 선택할 수 있게 한다.
- dispatcher와 expert catalog의 불일치 없이 안전하게 전환한다.
- 전환 실패 시 이전 dispatcher로 rollback할 수 있게 한다.
- offline 결과와 online 실행 결과가 일치하는지 검증한다.

### 비목표

- 요청 부하에 따른 자동 dispatcher 선택
- Pod 또는 모델별 autoscaling
- 실시간 queue를 dispatcher 입력에 추가하는 `D(x, s)`
- 운영 상태를 이용한 요청별 expert identity 변경
- PERTINENCE loss 또는 NSGA-II 자체의 개선

## 4. 원 논문과의 관계

활성화된 각 dispatcher는 다음과 같이 원 논문의 방법을 따른다.

- 독립적으로 사전학습된 전문가 모델
- frozen feature extractor
- 학습 가능한 단일 FC head
- cheapest-correct route label
- 어떤 전문가도 정답이 아닐 때 가장 저렴한 모델로 보내는 fallback
- 비대칭 penalty와 INS/ISNS/ENS weighting
- system accuracy와 평균 MFLOPs의 다목적 탐색
- dispatcher와 선택된 expert를 포함한 실행 비용 계산

운영자가 dispatcher `d`를 선택한 뒤 다음 변경 전까지 online 함수는 고정된다.

\[
D_{active}(x)=D_d(x)
\]

클라우드 상태가 요청별로 expert 선택에 개입하지 않으므로 online dispatcher는
`D(x, s)`가 되지 않는다. 클라우드 확장은 dispatcher 알고리즘이 아니라 배포할
PERTINENCE 인스턴스를 선택하고 교체하는 control plane에 있다.

엄밀하게는 운영자가 다른 dispatcher를 활성화하는 시점에 시스템 함수가
변한다. 따라서 하나의 영구 고정된 `D(x)`를 재현한다고 표현하기보다는,
**각 deployment epoch에서 논문 방식으로 학습된 하나의 `D(x)`를 실행한다**고
설명한다.

## 5. Dispatcher 구성과 global Pareto front

하나의 dispatcher 구성 `c`를 다음처럼 정의한다.

\[
c=(F, \mathcal M, P, W, \theta)
\]

- `F`: frozen feature extractor
- `M`: routing candidate expert 집합과 순서
- `P`: penalty matrix
- `W`: weighting scheme
- `theta`: 학습된 FC parameter

각 구성은 독립 평가 데이터에서 다음 값을 가진다.

\[
(A_c, C_c)=(\text{system accuracy},\text{average system MFLOPs})
\]

구성 `a`가 `b`보다 정확도가 낮지 않고 비용이 높지 않으며 둘 중 하나에서
엄격히 우수하면 `a`가 `b`를 지배한다.

\[
A_a \ge A_b,\qquad C_a \le C_b
\]

현재 1차 실험에서는 고정된 4-model routing 구성의 탐색 결과를 합친 후
non-dominated dispatcher만 dynamic global Pareto front에 포함한다. 정적
10-model Pareto 경계는 비교용 별도 계열이며 dispatcher 후보 집합에 합치지
않는다.

```python
solutions = run_pertinence_search(fixed_four_model_configuration)
evaluated = evaluate_on_untouched_split(solutions)

global_front = nondominated(
    evaluated,
    maximize="system_accuracy",
    minimize="average_system_mflops",
)

plot_reference_boundary(static_pareto_models_10)
plot_dynamic_front(global_front)
```

Pareto front의 주축은 원 논문과 동일하게 accuracy와 MFLOPs로 유지한다.
클라우드에서 측정한 latency, 모델 상주 메모리와 배포 가능 여부는 학습 목표에
섞지 않고 운영자가 판단할 보조 정보로 표시한다.

## 6. 탐색할 모델 구성

첫 실험의 routing 구성은 다음 4개 모델로 고정한다.

| 순서 | 후보 모델 | 역할 |
|---:|---|---|
| 1 | ShuffleNetV2 x0.5 | feature extractor 및 최저비용 anchor |
| 2 | MobileNetV2 x0.5 | 경량 중간점 |
| 3 | ResNet56 | 중고성능 중간점 |
| 4 | RepVGG-A1 | 최고성능 anchor |

accuracy–MFLOPs 그림에는 현재 공개 zoo의 non-dominated 모델 10종을 정적
Pareto 경계로 계속 표시한다. 그러나 나머지 6종은 routing label, FC 출력,
penalty matrix, NSGA-II chromosome이나 online expert catalog에는 포함하지
않는다. 따라서 각 dispatcher bundle은 동일한 4개 expert mapping을 가지며,
운용점은 penalty, weighting과 학습된 FC parameter 차이에서 나온다.

Feature extractor는 모든 dispatcher에서 ShuffleNetV2 x0.5로 고정한다.
이 덕분에 운용점 전환 시 expert catalog는 그대로 두고 FC head와 관련
metadata만 교체하면 된다.

구성마다 feature extractor까지 달라지는 실험은 다음 문제가 있으므로 후속
범위로 분리한다.

- dispatcher 연산량과 latency가 동시에 변함
- backbone을 새로 적재해야 함
- feature dimension이 달라질 수 있음
- FC head 교체 효과와 extractor 교체 효과가 섞임

## 7. Dispatcher Registry와 bundle

Pareto point 하나를 FC 파일 하나로 취급해서는 안 된다. FC 출력 index와 실제
expert model ID가 함께 바뀌기 때문이다. 다음 항목을 하나의 immutable bundle로
관리한다.

```text
dispatcher-d003-v1/
├── fc_head.pt
├── manifest.json
├── feature_extractor.json
├── expert_catalog.json
├── preprocessing.json
├── penalty_matrix.json
└── validation_metrics.json
```

`manifest.json`의 최소 항목은 다음과 같다.

```json
{
  "dispatcher_id": "d003",
  "version": 1,
  "feature_extractor": "shufflenetv2_x0_5",
  "feature_extractor_hash": "...",
  "expert_catalog_hash": "...",
  "preprocessing_hash": "...",
  "system_accuracy": 0.938,
  "average_system_mflops": 180.0,
  "required_experts": [
    "shufflenetv2_x0_5",
    "mobilenetv2_x1_0",
    "resnet56"
  ]
}
```

`expert_catalog.json`은 FC 출력과 endpoint의 의미를 고정한다.

```json
{
  "0": {
    "model_id": "shufflenetv2_x0_5",
    "service": "expert-shufflenet-v2-05"
  },
  "1": {
    "model_id": "mobilenetv2_x1_0",
    "service": "expert-mobilenet-v2-10"
  },
  "2": {
    "model_id": "resnet56",
    "service": "expert-resnet56"
  }
}
```

Registry는 각 bundle의 artifact 위치, hash, 평가 지표, required expert와 배포
상태를 제공한다. Kubeflow를 도입한다면 Model Registry 계층에 이 metadata를
등록할 수 있지만, 첫 구현은 로컬 manifest와 artifact 디렉터리만으로도
충분하다.

## 8. 운영자 화면

운영 화면의 중심은 global Pareto plot이다.

```text
System accuracy
  ▲
  │                         ● D4
  │                   ● D3
  │             ● D2
  │       ● D1
  └──────────────────────────────▶ Average MFLOPs
```

각 점을 선택하면 다음 정보를 표시한다.

- dispatcher ID와 version
- system accuracy와 평균 MFLOPs
- feature extractor
- required expert 목록
- expert별 route 비율
- no-correct 비율
- dispatcher overhead
- 전체 checkpoint 메모리
- 보조 online latency 측정값
- 현재 expert endpoint 준비 상태
- 현재 활성 여부와 이전 배포 이력

운영자는 `Activate`를 눌러 원하는 point를 선택한다. required expert가 준비되지
않았거나 artifact hash 검증이 실패하면 버튼을 비활성화하거나 전환을
거부한다. 이는 자동 최적화가 아니라 잘못된 배포를 막는 safety gate다.

## 9. 전환 프로토콜

운영자가 `D_old`에서 `D_new`로 전환할 때 다음 절차를 사용한다.

1. `D_new` bundle과 모든 artifact hash를 검증한다.
2. required expert endpoint가 모두 Ready인지 확인한다.
3. feature extractor와 preprocessing compatibility를 확인한다.
4. 새 dispatcher를 inactive 상태로 적재한다.
5. health check와 선택적인 shadow 요청을 실행한다.
6. 새 요청이 참조하는 active bundle ID를 `D_new`로 전환한다.
7. 각 요청은 시작할 때 읽은 bundle ID를 응답 완료까지 유지한다.
8. 기존 dispatcher는 drain 후 일정 시간 rollback 대상으로 보존한다.
9. 오류율 또는 health check가 임계값을 넘으면 `D_old`로 되돌린다.

단일 runtime MVP에서는 여러 FC head를 미리 메모리에 올리고 active pointer만
바꿀 수 있다. 여러 dispatcher replica를 운영할 때는 모든 replica가 새 bundle을
준비했다는 확인을 받은 뒤 동일한 active version을 사용하도록 해야 한다.

전환 로그에는 최소한 다음을 남긴다.

```text
requested_by
old_dispatcher_id
new_dispatcher_id
requested_at
activated_at
required_expert_status
validation_result
rollback_reason
```

각 inference 요청에도 `dispatcher_id`, `dispatcher_version`, `selected_model_id`를
기록한다.

## 10. Kubeflow의 선택적 역할

Kubeflow는 runtime에서 요청별 dispatcher를 선택하는 구성 요소가 아니라,
offline dispatcher 생성과 등록을 자동화하는 데 사용한다.

```text
모델 구성 입력
  → prediction/correctness cache 선택
  → cheapest-correct label 재생성
  → FC 학습과 NSGA-II
  → untouched split 평가
  → Pareto metadata 생성
  → Dispatcher Registry 등록
```

초기 재현에서는 Kubeflow를 사용하지 않고 기존 Python 학습 코드, bundle
manifest, 간단한 관리 API로 구현한다. 이 경로가 동작한 뒤 동일 단계를
Kubeflow Pipeline component로 옮긴다. 이 순서로 진행하면 PERTINENCE 재현
문제와 MLOps 플랫폼 설치 문제를 분리할 수 있다.

## 11. 비교군

제안 방식의 가치를 확인하기 위한 최소 비교군은 다음과 같다.

| 방식 | 설명 |
|---|---|
| `Fixed-Low` | 가장 저비용 dispatcher 하나를 계속 사용 |
| `Fixed-High` | 가장 정확한 dispatcher 하나를 계속 사용 |
| `Restart-Switch` | 운영자가 선택할 때 기존 프로세스를 종료하고 새 dispatcher 시작 |
| `Operator-Selected` | registry에서 검증된 bundle을 준비한 뒤 active version 전환 |

알고리즘 결과는 각 dispatcher의 accuracy–MFLOPs로 비교하고, 클라우드 운영
결과는 전환 시간과 실패 요청 수로 비교한다.

## 12. 평가 지표

### PERTINENCE 지표

- system Top-1 accuracy
- 평균 system MFLOPs
- route hit rate
- expert별 선택 비율
- ideal route 대비 confusion matrix
- no-correct 비율

### 운영 지표

- dispatcher bundle 적재 시간
- 활성 version 전환 시간
- 전환 중 실패 요청 수와 오류율
- 전환 중 잘못된 model ID routing 횟수
- in-flight 요청의 version 일관성
- rollback 시간과 성공 여부
- offline 평가와 online accuracy/MFLOPs의 차이
- 전환 전후 p95/p99 latency

## 13. 재현 가능한 전환 실험

자동 부하 변화 없이 고정된 요청 trace와 수동 전환 시점을 사용한다.

```text
0–5분    D_low 활성
5–10분   운영자가 D_balanced 선택
10–15분  운영자가 D_accurate 선택
15–20분  D_low로 rollback
```

모든 구간에 같은 CIFAR-10 sample 순서를 반복한다. 실험 시작 전 bundle과 expert
endpoint를 준비해 cold start의 영향을 별도 측정으로 분리한다.

추가 장애 실험에서는 다음 상황을 통제해 발생시킨다.

- required expert 하나가 Ready가 아닐 때 activation 거부
- 잘못된 expert catalog hash를 가진 bundle 거부
- 새 dispatcher health check 실패 후 rollback
- 전환 중 들어온 요청이 시작 시점의 version을 끝까지 유지하는지 검증

## 14. 예상 주장과 한계

### 예상 주장

1. 서로 다른 PERTINENCE 모델 구성과 탐색 해를 하나의 global Pareto front로
   통합할 수 있다.
2. 운영자는 논문의 accuracy–compute trade-off를 직접 확인하고 deployment
   epoch마다 원하는 운용점을 선택할 수 있다.
3. versioned bundle과 전환 프로토콜을 사용하면 dispatcher와 expert catalog의
   일관성을 유지하면서 운용점을 교체할 수 있다.
4. online routing에는 논문 방식의 고정된 dispatcher를 사용하므로 실시간
   클라우드 상태를 입력으로 추가하지 않는다.

### 한계

- dispatcher 선택을 자동 최적화하지 않는다.
- 운영자의 선택이 실제 workload에 최적인지는 보장하지 않는다.
- 사전에 학습하지 않은 모델 조합은 선택할 수 없다.
- 많은 모델 조합을 탐색하면 offline 계산 비용이 커진다.
- MFLOPs Pareto 순서가 실제 cloud latency나 요금 순서와 같다고 보장할 수 없다.
- 단순 UI와 hot-swap만으로는 새로운 routing 알고리즘 기여라고 보기 어렵다.

따라서 연구 기여는 새로운 dispatcher 알고리즘보다는 **PERTINENCE의 Pareto
운용점을 재현 가능하게 패키징하고 선택·배포·rollback하는 cloud 운영
방법**으로 한정해 주장한다.

## 15. MVP 범위

첫 구현은 다음 범위로 제한한다.

1. 공통 feature extractor 하나를 고정한다.
2. 대표 expert 구성 3개를 정의한다.
3. 구성별로 1개 이상의 dispatcher 해를 학습한다.
4. 모든 해의 global Pareto front를 JSON과 정적 plot으로 생성한다.
5. 각 point를 versioned bundle로 저장한다.
6. CLI 또는 간단한 API로 active dispatcher를 선택한다.
7. 새 요청부터 선택된 bundle을 사용하도록 전환한다.
8. 전환·실패·rollback 통합 테스트를 작성한다.

MVP에서는 Kubeflow, autoscaling, 실시간 telemetry 기반 선택과 복잡한 web UI를
제외한다. 이후 필요할 때 Kubeflow Pipeline과 Registry, dashboard를 순서대로
추가한다.

## 16. 연구 질문

- 여러 모델 구성에서 나온 dispatcher들을 통합했을 때 어떤 global
  accuracy–MFLOPs Pareto front가 형성되는가?
- 운영자가 선택한 dispatcher의 offline 결과가 online 환경에서도 재현되는가?
- bundle 단위 전환이 프로세스 재시작 방식보다 오류와 중단 시간을 줄이는가?
- 공통 feature extractor를 사용할 때 FC head 교체 비용은 전체 inference
  latency에 비해 충분히 작은가?
- MFLOPs 기준으로 선택한 각 운용점의 실제 latency와 모델 상주 메모리는 어떻게
  달라지는가?

## 17. 제안 명칭과 한 문장 정의

권장 명칭은 **Operator-Selected Cloud-PERTINENCE** 또는
**Pareto-Managed PERTINENCE**다.

> 여러 전문가 구성에서 학습된 PERTINENCE dispatcher의 global
> accuracy–compute Pareto front를 제공하고, 운영자가 선택한 운용점을 검증된
> versioned bundle로 클라우드에서 안전하게 활성화·교체·rollback하는 운영
> 구조다.

## 참고 문서

- [`docs/pertinence-paper-notes.ko.md`](../docs/pertinence-paper-notes.ko.md)
- [`docs/reproduction-plan.ko.md`](../docs/reproduction-plan.ko.md)
