# PERTINENCE CIFAR-10 재현 계획 및 실행 계약

## 결론과 범위

이 구현은 PERTINENCE arXiv 2507.01695v3의 알고리즘을 그대로 따르되,
현재 `chenyaofo/pytorch-cifar-models` 공개 CIFAR-10 체크포인트에서 얻은
accuracy–MAdds Pareto 경계 10종은 정적 비교 기준으로 유지하고, 실제
PERTINENCE dispatcher의 routing 후보는 그중 대표 4종만 사용한다.

두 모델 집합의 역할은 섞지 않는다.

- Pareto 경계 10종: 정적 단일 모델의 accuracy–MFLOPs 기준선, 표와 그림,
  최종 결과 비교에 사용한다.
- Routing 후보 4종: expert prediction cache, cheapest-correct label, FC 출력,
  penalty matrix, NSGA-II와 실제 동적 실행에만 사용한다.

이는 논문 Figure 6(c)의 숫자를 그대로 다시 얻는 실험은 아니다. 논문은
현재 공개 zoo에 없는 ResNet8/14와 현재 Pareto front에서 제외되는
VGG11-BN/VGG16-BN을 사용했다. 따라서 아래 두 층을 분리한다.

- 방법론 충실 요소: cheapest-correct label, frozen extractor, single FC,
  literal penalty loss, 세 weighting, NSGA-II, system-level accuracy/cost,
  feature 공유.
- 공개 자산 적응 요소: 현재 Pareto 경계 10종, 그중 선택한 routing 후보
  4종, ShuffleNetV2 x0.5 extractor, 공개 checkpoint의 전처리와 catalog
  MAdds.

## 단계별 계획

### 1. 환경과 자산 고정

- Python/PyTorch/torchvision/THOP/pymoo 버전을 `pyproject.toml`에 pin한다.
- upstream source를 commit
  `786c16252c0fc58ee9adac063f8337cc4a7a497a`에 고정한다.
- CIFAR-10 원본 tar, source archive와 Pareto 경계 10종의 checkpoint 출처를
  manifest에 고정한다. 실제 cache inference와 dispatcher 실행에서는 아래
  4개 checkpoint만 로드한다.
- 모든 파일은 전체 SHA-256으로 검증하고 결과 manifest를 남긴다.
- 모델 로더는 local source와 local checkpoint만 허용한다. import, dry-run,
  model load 과정에서 암묵적 네트워크 접근은 없다.

### 2. 데이터 분할과 전처리

chenyaofo 공개 학습 코드의 validation transform을 그대로 사용한다.

```text
ToTensor()
Normalize(
  mean=(0.4914, 0.4822, 0.4465),
  std=(0.2023, 0.1994, 0.2010),
)
```

Dispatcher feature에는 augmentation을 적용하지 않는다. 논문은 해당
여부를 공개하지 않았고, 고정 feature cache와 재현 가능한 sample ID가
우선이기 때문이다.

```text
official train 50,000 -> dispatcher_train 50,000
official test  10,000 -> ga_search 8,000 + final_evaluation 2,000
```

논문은 training/test/미사용 validation 역할을 구분하고 2,000장 validation
분석을 보고하지만 정확한 ID는 공개하지 않았다. 따라서 seed 20260824의
결정론적 permutation과 manifest SHA-256을 재현 가정으로 고정한다.

### 3. Pareto 경계와 routing 전문가 cache

정적 Pareto 경계는 기존 비용 오름차순 10종을 그대로 사용한다.

1. ShuffleNetV2 x0.5
2. MobileNetV2 x0.5
3. ShuffleNetV2 x1.0
4. MobileNetV2 x0.75
5. MobileNetV2 x1.0
6. ResNet44
7. ResNet56
8. RepVGG-A0
9. RepVGG-A1
10. RepVGG-A2

실제 routing 후보는 다음 4종으로 고정한다.

| 순서 | 모델 | Top-1 | MFLOPs | 역할 |
|---:|---|---:|---:|---|
| 1 | ShuffleNetV2 x0.5 | 90.13% | 21.80 | extractor, 최저비용 anchor |
| 2 | MobileNetV2 x0.5 | 92.88% | 55.94 | 경량 중간점 |
| 3 | ResNet56 | 94.37% | 251.50 | 중고성능 중간점 |
| 4 | RepVGG-A1 | 94.89% | 1,702.66 | 최고성능 anchor |

Pareto 경계 10종의 정적 수치는 고정 catalog/CSV에서 읽는다. Routing 후보
4종만 frozen/eval/no-grad로 처리해 다음을 split별로 저장한다.

- stable sample index와 원래 CIFAR class
- 모델 순서와 sample별 predicted class
- sample별 correctness
- cheapest-correct route label
- extractor final Linear 직전 feature
- checkpoint/source/config/split fingerprint

4개 routing 후보의 baseline Top-1이 upstream 보고값과 허용 오차 내에
들어오기 전에는 dispatcher 학습을 시작하지 않는다. RepVGG는 checkpoint 원형의
multi-branch 모델을 쓰며 deploy fusion은 별도 실험으로 분리한다.

나머지 6종은 routing label, FC class, penalty gene 및 평균 동적 비용에
포함하지 않는다. 최종 accuracy–MFLOPs 그림에는 10개 정적 모델 경계와
4-model PERTINENCE 해의 Pareto front를 별도 계열로 겹쳐 그린다.

### 4. Ideal dispatcher 검증

각 sample에서 4개 routing 후보 중 정답인 전문가의 비용 index가 가장 작은
것을 route label로
한다. 어떤 전문가도 정답이 아니면 index 0으로 보낸다. Oracle의 task
accuracy, 평균 비용, route 비율, no-correct 비율을 먼저 산출해 캐시와 모델
순서를 검증한다.

### 5. 고정 penalty 파이프라인 검증

GA 전에 방향성 고정 행렬로 한 개 FC를 학습한다.

```text
underestimation (true보다 싼 모델 예측): 10
overestimation  (true보다 비싼 모델 예측): 0.001
weighting: INS
epochs: 20
optimizer: Adam, lr=1e-3
```

optimizer와 learning rate는 논문 미공개 값이므로 설정에 노출된 재현
가정이다. Loss는 논문 수식을 문자 그대로 적용한다.

```python
base = cross_entropy(logits, route, reduction="none")
pred = logits.argmax(1)
loss = (class_weight[route] * P[route, pred] * base).mean()
```

argmax가 정답이면 대각 penalty 0 때문에 해당 sample gradient가 0이 된다.
이를 일반적인 differentiable cost-sensitive loss로 바꾸지 않는다.

### 6. NSGA-II 탐색

- genome: `N*N + 1`; routing 후보 N=4이므로 17 genes
- penalty genes: [0, 100], decode 시 diagonal 0
- 마지막 gene: INS/ISNS/ENS
- population 50, generations 50
- SBX eta 20, probability 0.9
- polynomial mutation eta 25
- individual마다 새 FC를 초기화해 20 epochs 학습
- objectives: `[-system_top1, average_system_mflops]`
- search 중 평가한 모든 해를 저장하고 전체에서 non-dominated front 산출

논문이 mutation probability와 categorical gene 연산을 공개하지 않아 pymoo
기본 mutation probability와 마지막 실수 gene의 `[0,1), [1,2), [2,3]`
floor-bin decode를 사용한다. 이 선택은 run metadata에 저장한다. 정적
Pareto 경계의 나머지 6종은 chromosome이나 search objective 계산에 넣지
않는다.

### 7. 독립 최종 평가

GA가 선택한 non-dominated 해만 untouched 2,000장에 적용한다. Search가
평가한 각 Pareto FC state를 그대로 저장해 이 단계에서 복원하므로,
재학습하거나 penalty를 다시 고르지 않는다. Baseline/fixed/search 명령은
final cache를 열지 않으며 final 명령에서 처음 접근한다.

필수 보고 지표:

- system Top-1 (선택된 전문가의 CIFAR 정답률)
- route hit rate (보조 지표)
- 이미지당 평균 MFLOPs
- ideal-vs-predicted route confusion matrix
- 전문가별 선택 비율
- no-correct 비율
- 4개 routing 후보의 동일 split 단일 전문가 baseline과 oracle
- catalog 기준 정적 10-model Pareto 경계와의 accuracy–MFLOPs 비교

PERTINENCE 비용은 `extractor + FC + selected expert`다. 선택 expert가 extractor와
같으면 이미 얻은 logits를 재사용해 backbone 비용을 두 번 더하지 않는다.
모든 catalog MAdds는 한 곳에서만 `1 MAC = 2 FLOPs`로 바꾼다.

정적 10-model 경계와 동적 PERTINENCE 해는 비용 정의가 다르므로 결과에
명시한다. 정적 점은 해당 expert 단독 비용이고, 동적 점은 dispatcher
overhead를 포함한 평균 비용이다.

## 논문 미공개 가정

다음은 원문만으로 확정할 수 없으며 config/run metadata로 보존한다.

- 정확한 split IDs와 모든 random seed
- FC optimizer, learning rate, batch size, scheduler, initialization
- dispatcher train feature augmentation 여부
- ENS beta와 class-weight normalization
- polynomial mutation probability
- categorical weighting gene 처리
- penalty discretization

본문의 penalty `[0,100]`, step `[0.5,1]` 설명과 Table IV의 0.001 계열 값도
서로 맞지 않는다. 기본 GA는 연속 [0,100], 고정 재현 경로는 0.001을
허용해 두 경우를 섞지 않는다.

## 실행하지 않은 범위

이번 구성 작업에서는 다음 계산을 실행하지 않는다.

- 60,000장에 대한 4개 routing 전문가 GPU 추론
- feature/prediction cache 생성
- FC 학습
- 50 x 50 x 20 epoch NSGA-II
- final 2,000장 결과 산출

자산 다운로드·해시 검증, 정적 검사와 synthetic unit test는 실험 결과를
생성하는 작업이 아니므로 환경 준비 검증 범위에 포함한다.
