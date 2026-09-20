# PERTINENCE 방법론과 재현 기준

## 문서 범위

이 문서는 다음 논문의 방법론, 저장소 공통 용어와 논문만으로 확정할 수 없는 재현 가정을
정리한다. 특정 dataset의 명령과 결과는 각 experiment 문서에서 다룬다.

- **PERTINENCE: Input-based Opportunistic Neural Network Dynamic Execution**
- Omkar Shende, Gayathri Ananthanarayanan, Marcello Traiola
- [arXiv 2507.01695 v3](https://arxiv.org/abs/2507.01695)
- [IEEE Access DOI 10.1109/ACCESS.2026.3707342](https://doi.org/10.1109/ACCESS.2026.3707342)

MFLOPs, 실제 latency, 클라우드 비용은 서로 다른 측정값이다. 논문 재현에서는 계산량을,
GKE serving 확장에서는 별도 online benchmark를 사용해야 하며 같은 결과로 섞지 않는다.

## 핵심 구조

PERTINENCE는 동일한 task를 수행하는 여러 frozen expert 중 입력별로 실행할 모델을 선택한다.
Expert를 공동 학습하거나 구조를 바꾸지 않고, frozen feature extractor와 작은 dispatcher만
학습한다.

```text
input
  ├─ frozen feature extractor ── feature ── linear dispatcher ── expert index
  └───────────────────────────────────────────────────────────── selected expert
```

Expert는 비용 오름차순으로 정렬한다. Sample마다 정답을 내는 가장 싼 expert의 index를
`cheapest-correct` label로 사용하고, 어느 expert도 정답이 아니면 가장 싼 fallback expert로
보낸다. Dispatcher route accuracy는 보조 지표이고, 선택된 expert가 실제 task 정답을 내는
system accuracy가 주 지표다.

## 학습 계약

Dispatcher는 classifier 직전 feature를 입력으로 받는 단일 fully connected layer다. 현재 구현은
다음 항목을 고정하거나 실행 계약에 기록한다.

- expert checkpoint, source revision, preprocessing과 class index 의미;
- dataset split ID와 split manifest hash;
- feature dimension, expert order와 동일 단위의 expert cost;
- Python/NumPy/PyTorch seed와 deterministic 설정;
- optimizer, learning rate, batch size와 epoch;
- weighting 방식과 penalty matrix;
- code, package, container image identity.

Final holdout은 search artifact의 입력이 아니다. Search에서 선택된 Pareto checkpoint를 final에서
재학습하지 않고 그대로 복원하며, final 결과로 hyperparameter나 배포 대상을 다시 고르지 않는다.

## Class weighting과 비대칭 loss

논문은 route-class imbalance를 보정하기 위해 INS, ISNS, ENS weighting을 제시한다. 현재 구현은
관측 class weight의 평균을 1로 normalize하고, 학습 표본이 없는 route class의 weight는 0으로
둔다. 미관측 class에는 직접 supervised signal이 없다는 점을 결과 제한으로 보고한다.

Sample `i`의 route label을 `y_i`, dispatcher prediction을 `p_i`, class weight를 `w`, penalty
matrix를 `P`라 하면 구현은 논문의 argmax-indexed 형태를 그대로 사용한다.

```text
loss_i = w[y_i] * P[y_i, p_i] * cross_entropy(logits_i, y_i)
```

대각 penalty는 0이다. 더 싼 expert로 잘못 보내는 underestimation과 더 비싼 expert로 보내는
overestimation의 비용을 독립적으로 둘 수 있다. 이를 일반적인 differentiable cost-sensitive
loss로 바꾸지 않는다.

## NSGA-II와 비용

목적함수는 system Top-1을 최대화하고 평균 system cost를 최소화하는 것이다.

```text
objectives = [-system_top1, average_system_cost]
```

Expert가 `N`개면 chromosome은 `N × N` penalty gene과 weighting gene 하나로 구성한다. 현재
CIFAR-10 설정은 expert 3개이므로 10 genes다. 기본 탐색은 population 50, generations 50,
SBX crossover와 polynomial mutation을 사용하며 individual마다 새 dispatcher를 20 epoch
학습한다.

CIFAR 실험의 cost는 catalog MAdds를 한 곳에서만 `1 MAC = 2 FLOPs`로 변환한다. Dynamic cost에는
feature extractor, dispatcher FC와 selected expert 비용이 포함된다. Feature extractor와 selected
expert가 같으면 이미 계산한 logits를 재사용해 backbone 비용을 중복 계산하지 않는다.

## 논문과 공개 자산 사이의 경계

이 저장소는 방법론 충실 재현이지 논문 그림의 숫자를 그대로 되찾는 numeric reproduction이
아니다. 논문의 일부 checkpoint와 정확한 split ID, optimizer, mutation probability 등은 공개되지
않았다. 따라서 다음 값은 config와 run metadata에 명시적으로 고정한다.

- 정확한 split seed와 sample ID;
- FC optimizer, learning rate, initialization과 batch size;
- feature augmentation 여부;
- ENS beta와 class-weight normalization;
- mutation probability와 categorical weighting gene decode;
- penalty 범위와 discretization;
- runtime 및 asset fingerprint.

현재 CIFAR-10은 공개 model zoo의 열 모델을 정적 비교 경계로 유지하되, 실제 routing에는
ShuffleNetV2 x0.5, MobileNetV2 x0.5, ResNet56 세 모델만 사용한다. 정적 model point는 standalone
비용이고 dynamic point는 dispatcher overhead를 포함하므로 그림과 표에서 둘의 정의를 명시한다.

## 저장소의 실행 경로

- [CIFAR-10 로컬 재현](cifar10/docs/local-reproduction.ko.md): native PyTorch/CUDA cache와 CLI 실행.
- [Portable pipeline](../pipelines/pertinence/README.md): Parquet/ONNX bundle과 일곱 component 계약.
- [검증된 GKE 실행](../cloud/docs/runs/2026-09-20-dispatcher-gke.ko.md): 실제 KFP production run.
- [Cloud 아키텍처](../cloud/docs/architecture.ko.md): 배포된 GKE/KFP 경계.
