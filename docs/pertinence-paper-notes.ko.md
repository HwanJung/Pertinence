# PERTINENCE 원본 논문 정리 및 재현 설계

## 1. 문서 범위

이 문서는 다음 논문의 최신 공개본을 기준으로 방법론, 필요한 이론, 구현 요소, 실험 설정과 재현 시 주의점을 정리한다.

- 제목: **PERTINENCE: Input-based Opportunistic Neural Network Dynamic Execution**
- 저자: Omkar Shende, Gayathri Ananthanarayanan, Marcello Traiola
- arXiv: [2507.01695 v3](https://arxiv.org/abs/2507.01695), 2026-06-24 개정
- 원문: [HTML](https://arxiv.org/html/2507.01695), [PDF](https://arxiv.org/pdf/2507.01695)
- 관련 DOI: [10.1109/ACCESS.2026.3707342](https://doi.org/10.1109/ACCESS.2026.3707342)

이 문서는 원본 논문의 내용을 보존하기 위한 문서다. MFLOPs를 클라우드 비용이나 서비스 지연시간으로 대체하는 확장 설계는 포함하지 않는다.

## 2. 핵심 아이디어

PERTINENCE는 동일한 작업을 수행하는 여러 사전학습 모델 중에서 각 입력을 올바르게 처리할 수 있는 가장 가벼운 모델을 런타임에 선택한다.

일반적인 단일 모델 추론은 입력 난이도와 관계없이 모든 입력에 같은 계산량을 사용한다. 하지만 작은 모델로도 맞힐 수 있는 쉬운 입력이 많고, 큰 모델이 필요한 입력은 일부에 불과하다. PERTINENCE는 이 차이를 이용한다.

전체 구조는 다음과 같다.

```text
Input image
    │
    ├─ Frozen neural feature extractor
    │       │
    │       └─ Feature vector
    │               │
    │               └─ Trainable FC layer
    │                       │
    │                       └─ Selected model index
    │
    └──────────────────────────> Selected pretrained model
                                      │
                                      └─ Task prediction
```

전문가 모델들은 서로 독립적으로 이미 학습되어 있으며 고정된다. 이미지 분류 실험에서 학습 대상은 dispatcher의 단일 fully connected layer뿐이다.

## 3. 기존 접근과의 차이

### 3.1 정적 경량화와의 차이

Pruning, quantization, knowledge distillation, NAS는 주로 학습 또는 배포 시점에 모델을 경량화한다. 배포된 모델은 입력별 난이도와 무관하게 같은 구조를 실행하는 경우가 많다.

PERTINENCE는 기존 모델을 변경하지 않고 입력마다 실행할 모델을 바꾼다.

### 3.2 Early-exit 및 slimmable network와의 차이

Early-exit은 하나의 backbone 중간에 여러 classifier를 추가하고, 충분히 확신하면 조기에 종료한다. Slimmable network는 런타임에 채널 폭 등을 변경한다. 두 방식 모두 특별한 구조와 공동 학습이 필요하다.

PERTINENCE는 구조적으로 독립적인 기존 사전학습 모델을 그대로 사용할 수 있다.

### 3.3 Mixture of Experts와의 차이

전통적인 MoE는 전문가와 router를 함께 학습하거나 데이터 영역별 전문가를 구성하는 경우가 많다. PERTINENCE의 전문가는 서로 독립적인 기존 모델이며 재학습하지 않는다. 또한 목적이 정확도 향상에 한정되지 않고 계산량 최소화를 명시적으로 포함한다.

## 4. 문제 정의

사전학습 모델 집합을 다음과 같이 둔다.

\[
\mathcal M=\{M_1,M_2,\ldots,M_N\}
\]

각 모델의 속성은 다음과 같다.

- \(\alpha_i\): 모델 \(M_i\)의 정확도
- \(\phi_i\): 모델 \(M_i\)의 계산 비용
- \(\mathcal C(M_i,x)\): 모델 \(M_i\)가 입력 \(x\)를 맞히면 1, 아니면 0

모델은 일반적으로 낮은 비용에서 높은 비용 순서로 정렬한다.

Dispatcher는 다음 함수다.

\[
D_p:\mathcal X\rightarrow\mathcal M
\]

이상적인 선택은 입력을 맞힐 수 있는 모델 중 비용이 가장 작은 모델이다.

\[
D_p(x)=M_i,\qquad
i=\arg\min_j\{\phi_j\mid\mathcal C(M_j,x)=1\}
\]

시스템 전체의 두 목적은 다음과 같다.

\[
\max\ \alpha_{sys}
=
\frac{1}{|\mathcal X|}
\sum_{k=1}^{|\mathcal X|}
\mathcal C(D_p(x_k),x_k)
\]

\[
\min\ \phi_{sys}
=
\frac{1}{|\mathcal X|}
\sum_{k=1}^{|\mathcal X|}
\phi_{D_p(x_k)}
\]

실제 \(\phi_{sys}\)에는 선택된 전문가뿐 아니라 feature extractor와 FC layer의 연산도 포함한다.

## 5. 모델 상보성과 이상적 dispatcher

큰 모델이 작은 모델의 정답 집합을 항상 포함하는 것은 아니다. 서로 다른 모델은 서로 다른 입력에서 오류를 낼 수 있다. 이 상보성 때문에 입력별로 모델을 올바르게 선택하면 가장 정확한 단일 모델보다 시스템 정확도가 높아질 수 있다.

논문의 CIFAR-10 동기 예시는 ResNet8, ResNet14, ResNet20을 사용한다.

| ResNet8 | ResNet14 | ResNet20 | 이미지 수 |
|---|---|---|---:|
| 정답 | 정답 | 정답 | 6,320 |
| 정답 | 정답 | 오답 | 228 |
| 정답 | 오답 | 정답 | 256 |
| 정답 | 오답 | 오답 | 143 |
| 오답 | 정답 | 정답 | 1,759 |
| 오답 | 정답 | 오답 | 237 |
| 오답 | 오답 | 정답 | 527 |
| 오답 | 오답 | 오답 | 530 |

이상적 dispatcher는 다음과 같이 선택한다.

- ResNet8: 6,947개
- ResNet14: 1,996개
- ResNet20: 527개
- 어떤 모델도 맞히지 못한 530개: 계산량 최소화를 위해 ResNet8 선택

논문은 이 이상적 선택이 ResNet20만 사용할 때보다 정확도를 높이면서 연산량을 크게 줄일 수 있음을 보인다.

## 6. 전문가 모델 선택

전문가 후보는 accuracy–MFLOPs 공간에서 Pareto front에 있는 모델을 우선한다.

모델 \(A\)가 모델 \(B\)보다 정확도가 같거나 높고 비용이 같거나 낮으며, 두 항목 중 하나가 엄격히 우수하면 \(A\)가 \(B\)를 지배한다. 다른 모델에 지배되는 모델은 원칙적으로 후보에서 제외한다.

단, 모델의 전체 정확도와 비용만으로는 입력별 오류 상보성을 알 수 없다. 실제 구현에서는 Pareto filtering 후 모델별 correctness overlap도 함께 확인해야 한다.

## 7. Dispatcher 학습 데이터 생성

Dispatcher의 target은 원래 이미지 클래스가 아니다. 각 이미지에 대해 올바르게 분류하는 모델 중 가장 저렴한 모델의 인덱스다.

### 7.1 Label 생성 알고리즘

```python
for sample in dataset:
    correct_models = [
        model_id
        for model_id in models_sorted_by_cost
        if expert_prediction[model_id, sample] == class_label[sample]
    ]

    if correct_models:
        route_label[sample] = correct_models[0]
    else:
        route_label[sample] = cheapest_model_id
```

어떤 모델도 맞히지 못한 경우 가장 저렴한 모델로 보내는 fallback은 논문의 동기 예시에 나온 정책이다. 형식적 수식에는 빈 집합 처리 방식이 명시되지 않으므로 구현에서 명시적으로 처리해야 한다.

### 7.2 캐시해야 할 데이터

- 데이터셋 sample ID
- 원래 class label
- 각 전문가의 predicted class 또는 logits
- 각 전문가의 sample별 correctness
- 생성된 route label
- feature extractor 출력
- 모델별 MACs와 FLOPs
- checkpoint 출처와 hash
- 입력 전처리 설정

전문가 예측과 feature를 미리 캐시하면 GA의 각 개체에서 FC layer만 반복 학습할 수 있다.

## 8. Dispatcher 구조

이미지 분류 실험의 dispatcher는 다음 두 부분으로 구성된다.

1. 사전학습된 frozen neural feature extractor
2. 학습 가능한 단일 fully connected layer

```python
features = frozen_backbone(image)
route_logits = linear(features)
selected_model = route_logits.argmax(dim=1)
```

FC layer의 입력 차원은 backbone feature 차원이고 출력 차원은 routing 후보 모델 수다.

Feature extractor로 선택한 모델보다 정확도가 낮은 모델은 일반적으로 routing 후보에서 제외한다. 논문의 설명은 feature extractor를 실행한 뒤 더 낮은 정확도의 모델을 추가로 실행하는 것이 비효율적이라는 판단에 기반한다.

## 9. Route class imbalance

작은 모델이 많은 이미지를 맞히기 때문에 route label은 심하게 불균형하다. 논문의 CIFAR-10 4-model 예시는 다음과 같다.

| Route model | 학습 샘플 비율 | 구분 |
|---|---:|---|
| ResNet8 | 68.46% | Majority |
| ResNet14 | 20.40% | Minority 1 |
| ShuffleNetV2 ×0.5 | 9.47% | Minority 2 |
| VGG16-BN | 1.67% | Minority 3 |

논문은 다음 세 가지 weighting scheme을 탐색한다.

### 9.1 INS

\[
w_c=\frac{1}{n_c}
\]

### 9.2 ISNS

\[
w_c=\frac{1}{\sqrt{n_c}}
\]

### 9.3 ENS

Class-Balanced Loss의 effective number를 사용한다.

\[
w_c=\frac{1-\beta}{1-\beta^{n_c}}
\]

논문은 ENS의 \(\beta\), weight normalization 방식은 명시하지 않는다.

## 10. 비대칭 penalty matrix

Dispatcher의 오류는 두 종류로 나뉜다.

### 10.1 Underestimation

큰 모델이 필요한 이미지를 작은 모델로 보낸다.

- 이상적 dispatcher 대비 정확도가 하락할 수 있다.
- 일반적으로 큰 penalty를 부여한다.

### 10.2 Overestimation

작은 모델로 충분한 이미지를 큰 모델로 보낸다.

- 계산량이 낭비된다.
- 정확도는 대체로 유지되지만, 모델의 정답 집합이 완전히 중첩되지 않으므로 항상 유지된다고 보장할 수는 없다.

세 개 routing class의 penalty matrix는 다음과 같다.

\[
P=
\begin{bmatrix}
0&p_{01}&p_{02}\\
p_{10}&0&p_{12}\\
p_{20}&p_{21}&0
\end{bmatrix}
\]

\(P[i,j]\)는 실제 route class가 \(i\)일 때 \(j\)로 예측한 오류의 penalty다. 대각선은 0이다.

## 11. 논문에 명시된 custom loss

샘플 \(k\)의 loss는 다음과 같다.

\[
L_k=
\begin{cases}
0,
&y_{true,k}=y_{pred,k}\\
L_{base,k}\cdot P[y_{true,k},y_{pred,k}],
&y_{true,k}\ne y_{pred,k}
\end{cases}
\]

전체 loss는 다음과 같다.

\[
L=\frac1M\sum_{k=1}^{M}L_k
\]

Class weighting까지 포함한 직접적인 구현은 다음과 같다.

```python
pred_route = route_logits.argmax(dim=1)
base_ce = cross_entropy(route_logits, true_route, reduction="none")

penalty = penalty_matrix[true_route, pred_route]
sample_weight = class_weights[true_route]

loss = (sample_weight * penalty * base_ce).mean()
```

이 구현은 논문의 수식을 그대로 옮긴 것이다. 다음 특성이 있다.

- `argmax`로 선택한 penalty는 미분되지 않는다.
- 현재 argmax가 정답이면 diagonal penalty가 0이므로 gradient도 0이다.
- 일반적인 differentiable cost-sensitive cross-entropy와 다르다.

원본 재현에서는 위 방식을 우선 사용하고, differentiable 대안은 별도의 개선 실험으로 분리해야 한다.

## 12. 다목적 최적화와 NSGA-II

각 GA 개체는 하나의 FC layer 학습 설정을 의미한다.

```text
Individual
  ├─ N × N penalty genes
  └─ 1 weighting-scheme gene: INS / ISNS / ENS
```

논문은 routing 대상이 \(N\)개일 때 chromosome 수를 \(N^2+1\)로 설명한다. 예를 들어 대상 모델이 3개면 10개 gene이다.

목적 함수는 다음 두 값이다.

```python
objective_1 = -system_top1_accuracy
objective_2 = average_system_FLOPs
```

개체 평가 절차:

```text
1. Penalty matrix와 weighting scheme decode
2. 새로운 FC layer 초기화
3. Cached training features로 FC layer 학습
4. Search/fitness split에서 routing 수행
5. 선택된 전문가의 cached prediction으로 시스템 정확도 계산
6. Dispatcher와 선택 모델을 포함한 평균 FLOPs 계산
7. NSGA-II에 두 objective 반환
8. Non-dominated sorting, selection, crossover, mutation
```

### 12.1 논문에 명시된 GA 설정

| 항목 | 값 |
|---|---:|
| Population size | 50 |
| Generations | 50 |
| FC training epochs per individual | 20 |
| Crossover | Simulated Binary Crossover |
| SBX eta | 20 |
| Crossover probability | 0.9 |
| Mutation | Polynomial Mutation |
| Mutation eta | 25 |
| Penalty 범위 | [0, 100] |
| Weighting scheme | INS, ISNS, ENS |

논문은 각 모델 조합마다 GA를 한 번 실행하고, 전체 탐색 중 평가된 해를 순위화한 뒤 non-dominated 해를 보고한다.

## 13. 연산량 산정

논문은 [THOP: PyTorch-OpCounter](https://github.com/ultralytics/thop)을 이용해 MACs, parameters와 FLOPs를 계산한다.

일반적인 입력의 시스템 비용은 다음과 같다.

\[
\phi_{sys}(x)
=
\phi_{feature}
+\phi_{FC}
+\phi_{selected\ expert}
\]

단, feature extractor가 routing 후보 자체이고 dispatcher가 같은 모델을 선택했다면 feature computation을 공유한다.

\[
\phi_{sys}(x)=\phi_{feature}+\phi_{FC}
\]

같은 backbone을 dispatcher와 전문가 추론에서 두 번 실행하면 안 된다.

### 13.1 Dispatcher overhead

| Dataset | Feature extractor | 일반 overhead/image | Extractor 자체가 선택됐을 때 추가 비용 |
|---|---|---:|---:|
| CIFAR-10 | ResNet8 | 5.94 MFLOPs | 0.04 MFLOPs |
| CIFAR-10 | ResNet14 | 13.50 MFLOPs | 0.04 MFLOPs |
| CIFAR-10 | ShuffleNetV2 | 24.00 MFLOPs | 0.10 MFLOPs |
| CIFAR-100 | ShuffleNetV2 | 24.17 MFLOPs | 0.10 MFLOPs |
| Tiny-ImageNet | ResNet50 | 73.54 MFLOPs | 해당 없음 |

MAC과 FLOP의 환산 규칙은 구현 전체에서 고정해야 한다. `1 MAC = 1 operation`과 `1 MAC = 2 FLOPs`를 혼용하면 결과가 정확히 두 배 차이 날 수 있다.

## 14. 공통 실험 환경

- CPU: 48-core Intel Xeon Gold 5220R, 2.2 GHz
- GPU: NVIDIA A100 80GB PCIe
- 연산량 측정: THOP
- Top-1 accuracy 사용
- 모든 PERTINENCE 연산량에 dispatcher와 선택된 전문가 비용 포함
- Validation set 전체의 이미지당 평균 연산량 보고

## 15. CIFAR-10 실험

### 15.1 모델 풀

논문은 [chenyaofo/pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models)의 공개 CIFAR 모델을 인용한다.

- ResNet8
- ResNet14
- ShuffleNetV2 ×0.5
- VGG11-BN
- VGG16-BN

공개성에 관한 재현 주의점:

- 인용된 `chenyaofo/pytorch-cifar-models`의 현재 CIFAR-10 표에는 ShuffleNetV2 x0.5, VGG11-BN, VGG16-BN 가중치는 있지만 ResNet8과 ResNet14 엔트리는 없다.
- 논문 Table I의 정답 조합을 합산하면 ResNet8은 69.47%(6,947/10,000), ResNet14는 85.44%(8,544/10,000)다. 논문 보고 연산량은 각각 5.94, 13.50 MFLOPs다.
- 따라서 ResNet8/14의 정확한 재현에는 저자 체크포인트나 누락된 구조·학습 레시피를 확보해야 한다. 공개 저장소의 표준 ResNet20 체크포인트로 대체하면 정확도와 연산량이 논문 값과 크게 달라진다.

### 15.2 Feature extractor 및 routing 조합

ResNet8 feature extractor:

- `{ResNet8, VGG16-BN}`
- `{ResNet14, VGG16-BN}`
- `{ShuffleNetV2 ×0.5, VGG16-BN}`
- `{ResNet8, ShuffleNetV2 ×0.5, VGG16-BN}`
- `{ResNet14, ShuffleNetV2 ×0.5, VGG16-BN}`
- `{ResNet8, ResNet14, ShuffleNetV2 ×0.5, VGG16-BN}`

ResNet14 feature extractor:

- `{ResNet14, VGG16-BN}`
- `{ShuffleNetV2 ×0.5, VGG16-BN}`
- `{ResNet14, ShuffleNetV2 ×0.5, VGG16-BN}`

ShuffleNetV2 ×0.5 feature extractor:

- `{ShuffleNetV2 ×0.5, VGG16-BN}`

### 15.3 대표 재현 실험: Figure 6(c)

```text
Dataset: CIFAR-10
Feature extractor: ResNet8
Routing candidates: ShuffleNetV2 ×0.5, VGG16-BN
Underestimation penalty P[VGG16, ShuffleNet]: 10
Overestimation penalty P[ShuffleNet, VGG16]: 0.001
Weighting scheme: INS
```

최신 본문에서 일관된 목표값:

| 구성 | Top-1 accuracy | Average MFLOPs |
|---|---:|---:|
| VGG16-BN 단독 | 95.0% | 약 629.76 |
| PERTINENCE 대표 해 | 95.2% | 약 401.17 |

이 결과는 VGG16-BN 대비 정확도 0.2%p 향상과 약 36.3%의 연산 감소에 해당한다.

### 15.4 Dispatcher 강건성 분석

논문은 ResNet8 feature extractor와 `{ShuffleNetV2 ×0.5, VGG16-BN}` 후보를 사용한 두 해를 2,000개 validation 이미지에서 분석한다.

| 해 | VGG→VGG | VGG→ShuffleNet | ShuffleNet→VGG | ShuffleNet→ShuffleNet |
|---|---:|---:|---:|---:|
| Ideal | 114 | 0 | 0 | 1,886 |
| Solution 1 | 81 | 33 | 1,145 | 741 |
| Solution 2 | 54 | 60 | 768 | 1,118 |

- VGG→ShuffleNet: underestimation, 정확도 손실
- ShuffleNet→VGG: overestimation, 연산 낭비
- Solution 1 dispatcher hit rate: 41.1%
- Solution 2 dispatcher hit rate: 58.6%

Hit rate가 높지 않아도 overestimation이 task accuracy를 대체로 보존하므로 전체 시스템이 유용한 accuracy–cost trade-off를 만들 수 있다.

## 16. CIFAR-100 실험 요약

Feature extractor는 ShuffleNetV2 ×0.5다.

후보 모델 풀:

- ShuffleNetV2 ×0.5
- MobileNetV2 ×0.5 또는 ×0.75
- ShuffleNetV2 ×1.0
- MobileNetV2 ×1.4
- RepVGG-A1
- RepVGG-A2

보고된 주요 routing 조합:

- `{MobileNetV2 ×0.75, RepVGG-A2}`
- `{ShuffleNetV2 ×0.5, MobileNetV2 ×1.4}`
- `{ShuffleNetV2 ×0.5, MobileNetV2 ×0.75, RepVGG-A2}`
- `{ShuffleNetV2 ×0.5, MobileNetV2 ×1.4, RepVGG-A2}`

PERTINENCE는 약 76.25–77.3% 정확도와 269.93–324.48 MFLOPs 범위의 운용점을 보고한다.

## 17. Tiny-ImageNet 실험 요약

Feature extractor는 ImageNet ILSVRC 2012 pretrained ResNet50이다.

후보 ViT:

- DeiT-S
- Swin-Small
- DeiT-B
- Swin-Base
- CaiT-S36
- DeiT-B-distilled
- ViT-L

보고된 routing 조합:

- `{DeiT-S, DeiT-B, ViT-L}`
- `{DeiT-S, Swin-Base, DeiT-B-distilled, ViT-L}`

대표 결과로 ViT-L의 88.05%보다 높은 88.14%를 얻으면서 연산량을 13.65% 줄인 해를 보고한다.

## 18. 교통 영상 실험 요약

실시간 도로 점유도 추정에 네 개 YOLO 모델을 사용한다.

| 모델 | MFLOPs/frame | 정확도 |
|---|---:|---:|
| TinyYOLOv3 | 2,726 | 63.68% |
| YOLOv3-320 | 19,331 | 64.12% |
| YOLOv3-416 | 32,670 | 73.20% |
| YOLOv3-608 | 69,787 | 75.41% |

실험 환경:

- Validation frames: 1,800
- Frame rate: 15 FPS
- Deadline: 66.67 ms/frame
- Platform: NVIDIA Jetson AGX Orin 32GB
- TensorRT: 8.5.2.2
- PyTorch: 1.12
- Dispatcher backbone: ResNet8, 173 MFLOPs/frame
- Optimizer: Adam
- Learning rate: \(10^{-3}\)
- Batch size: 8
- Epochs: 40

이 실험에서는 ResNet8과 FC layer를 처음부터 함께 학습한다.

### 18.1 Static dispatcher interval

Dispatcher를 매 \(D\) frame마다 실행하고 그 사이에는 이전에 선택한 YOLO를 재사용한다.

\[
DispatcherOverheadPerFrame=\frac{173}{D}
\]

예를 들어 \(D=15\)이면 평균 dispatcher 비용은 약 11.53 MFLOPs/frame이고, 측정된 dispatcher latency 14.82ms는 평균 약 0.99ms/frame이 된다.

### 18.2 Dynamic scene-change trigger

연속 frame 간 absolute pixel difference와 histogram difference를 결합한다.

\[
\theta
=0.2\cdot PixelDifference
+0.8\cdot HistogramDifference
\]

\[
\theta>0.016
\]

이면 dispatcher를 다시 실행한다.

논문은 500개 이상의 weight–threshold 조합을 grid search했고 선택된 동적 설정에서 다음을 보고한다.

- Accuracy: 78.27%
- Per-image time: 56.07ms
- Change detector overhead: 평균 1.349ms/frame

## 19. 메모리 분석

논문은 모든 후보 모델과 dispatcher를 메모리에 유지하는 비용도 보고한다.

| Dataset | 단일 최고 모델 메모리 | PERTINENCE 메모리 | 증가량 |
|---|---:|---:|---:|
| CIFAR-10 | 1,552.56MB | 1,657.54MB | 104.98MB |
| CIFAR-100 | 1,717.21MB | 1,743.07MB | 25.86MB |
| Tiny-ImageNet | 3,742.64MB | 4,542.47MB | 799.83MB |
| Traffic | 1,567.60MB | 4,182.02MB | 2,614.42MB |

연산량 절감과 달리 여러 모델을 상주시켜야 하므로 메모리는 증가한다. 이 trade-off는 배포 설계에서 별도로 평가해야 한다.

## 20. 구현 모듈 권장 구조

```text
src/
  data/
    datasets.py
    splits.py
    transforms.py
  experts/
    registry.py
    checkpoints.py
    evaluator.py
  dispatcher/
    feature_extractor.py
    linear_router.py
    labels.py
    weighting.py
    penalty_loss.py
  optimization/
    chromosome.py
    objectives.py
    nsga2.py
    pareto.py
  metrics/
    flops.py
    routing.py
    system_accuracy.py
  runtime/
    dynamic_executor.py
  experiments/
    cifar10.py
configs/
artifacts/
tests/
```

## 21. 권장 재현 순서

### 단계 1: 환경 고정

- Python, PyTorch, CUDA, cuDNN, THOP 버전 기록
- checkpoint URL과 SHA-256 기록
- normalization과 입력 크기 기록
- 모든 random seed 기록
- MAC/FLOP 환산 규칙 고정

### 단계 2: 전문가 baseline 검증

각 모델에 대해 다음을 먼저 재현한다.

- Top-1 accuracy
- sample별 예측
- sample별 correctness
- MACs/FLOPs
- parameter 수

Baseline이 논문과 다르면 dispatcher를 구현하기 전에 checkpoint와 preprocessing부터 수정한다.

### 단계 3: Ideal dispatcher 검증

- Cheapest-correct label 생성
- No-correct fallback 검증
- 모델별 label 분포 검증
- 이론상 최대 system accuracy와 최소 평균 비용 계산

### 단계 4: Feature 및 예측 캐시

Frozen backbone feature와 모든 전문가 예측을 split별로 캐시한다.

### 단계 5: 고정 penalty 재현

GA 전에 CIFAR-10 Figure 6(c)의 알려진 설정으로 전체 파이프라인을 검증한다.

### 단계 6: NSGA-II 탐색

각 개체마다 FC를 새로 초기화하고 20 epoch 학습한 뒤 실제 system accuracy와 평균 FLOPs를 fitness로 사용한다.

### 단계 7: 독립 final split 평가

Search split에서 얻은 non-dominated 해만 final split에서 평가한다.

### 단계 8: 실제 runtime 검증

Offline fitness에서는 cached expert prediction을 사용할 수 있지만 runtime benchmark에서는 dispatcher가 선택한 모델 하나만 실행해야 한다.

## 22. 검증 체크리스트

- [ ] 전문가 checkpoint와 preprocessing이 일치한다.
- [ ] 모델별 baseline accuracy가 논문과 유사하다.
- [ ] 모델 비용 정렬이 올바르다.
- [ ] Route label은 원래 CIFAR class가 아닌 cheapest-correct model ID다.
- [ ] 어떤 모델도 맞히지 못하면 가장 저렴한 모델을 선택한다.
- [ ] CIFAR-10 4-model label 비율이 논문 값과 유사하다.
- [ ] Feature extractor는 frozen/eval 상태다.
- [ ] 학습 대상은 FC layer뿐이다.
- [ ] Penalty matrix 인덱스는 `[true_route, predicted_route]`다.
- [ ] INS, ISNS, ENS를 각각 구현했다.
- [ ] Dispatcher 연산량이 전체 비용에 포함된다.
- [ ] Feature extractor와 선택 모델이 같을 때 연산을 공유한다.
- [ ] System accuracy는 dispatcher route accuracy가 아니라 선택된 전문가의 task accuracy다.
- [ ] Average FLOPs는 sample별 실제 선택 비용의 평균이다.
- [ ] Pareto dominance가 정확도 최대화, 비용 최소화 방향으로 계산된다.
- [ ] GA search split과 final evaluation split이 분리되어 있다.
- [ ] Offline 캐시 평가와 실제 dynamic runtime 결과가 일치한다.

## 23. 논문만으로 확정할 수 없는 재현 공백

원본 논문만으로 알고리즘의 구조는 구현할 수 있지만 숫자를 완전히 동일하게 재현하려면 다음 정보가 추가로 필요하다.

- CIFAR train/search/final split의 정확한 sample ID
- 논문에서 말하는 test와 validation의 구체적인 분할 방식
- CIFAR-10 강건성 분석에 사용한 2,000개 이미지 선정 방식
- 이미지 분류 FC layer의 optimizer, learning rate, batch size, scheduler
- FC 초기화 및 random seed
- Data augmentation과 normalization의 정확한 구성
- ENS의 \(\beta\) 값과 class-weight normalization 방식
- Polynomial mutation probability
- Categorical weighting-scheme gene 처리 방식
- Penalty gene의 실제 discretization 방식
- Tiny-ImageNet ViT checkpoint의 정확한 출처
- 교통 데이터와 YOLO checkpoint의 정확한 배포 자산
- Dynamic change detector의 histogram bin, 색공간, metric normalization
- 전력, latency, memory 측정의 warm-up 및 sampling 절차

## 24. 논문 내부의 주의할 불일치

### 24.1 Penalty 범위

GA penalty를 `[0,100]`, step size를 `[0.5,1]`로 설명하지만 대표 Table IV에는 `0.001`, `0.005`, `0.05`가 등장한다. 보고된 대표 해와 설명된 탐색 공간이 직접 일치하지 않는다.

### 24.2 MAC/FLOP 단위

최신 본문의 VGG16-BN 및 PERTINENCE 비용과 일부 표의 수치가 약 두 배 차이 난다. 과거 버전의 MAC/FLOP 환산 수치가 표에 남은 것으로 보인다. 최신 본문의 `VGG16-BN 약 629.76 MFLOPs`, `PERTINENCE 약 401.17 MFLOPs`를 주 재현 목표로 사용하는 것이 일관적이다.

### 24.3 Loss 수식

논문의 loss는 predicted argmax를 이용해 penalty를 선택하고 정답 argmax의 loss를 0으로 만든다. 이 방식은 일반적인 cost-sensitive cross-entropy와 다르며 최적화상 특이점이 있다. 원본 수식을 먼저 구현하고 개선 loss와 혼합하지 않아야 한다.

### 24.4 데이터 분할 명칭

논문은 GA fitness를 test set에서 측정하고, GA나 학습에 사용되지 않은 validation set으로 최종 평가한다고 설명한다. 일반적인 train/validation/test 명명과 다를 수 있으므로 코드에서는 다음과 같이 의미가 명확한 이름을 사용하는 편이 안전하다.

```text
expert_train
dispatcher_train
ga_search
final_evaluation
```

## 25. CIFAR-10 구현의 첫 완료 기준

이 프로젝트에서 가장 먼저 목표로 삼을 최소 재현 단위는 다음이다.

```text
Dataset: CIFAR-10
Feature extractor: frozen ResNet8
Candidates: ShuffleNetV2 ×0.5, VGG16-BN
Dispatcher: single Linear layer
Weighting: INS
P[VGG16, ShuffleNet] = 10
P[ShuffleNet, VGG16] = 0.001
```

완료 조건:

1. 전문가 baseline이 논문과 유사하다.
2. Cheapest-correct routing label이 생성된다.
3. Fixed-penalty dispatcher가 학습된다.
4. Underestimation과 overestimation confusion matrix가 계산된다.
5. Dispatcher overhead를 포함한 평균 FLOPs가 계산된다.
6. VGG16-BN 단독보다 유리한 accuracy–cost 해를 얻는다.
7. 같은 코드를 NSGA-II 개체 평가 함수로 사용할 수 있다.

이 최소 단위를 통과하면 PERTINENCE의 핵심인 input-based routing, class imbalance, asymmetric penalty, system-level accuracy, dynamic compute accounting이 모두 구현된 것이다.
