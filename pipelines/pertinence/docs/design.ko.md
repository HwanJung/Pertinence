# PERTINENCE 학습 파이프라인의 Kubeflow Pipelines 적용 설계

## 1. 문서 상태와 결론

- 기준일: 2026-09-06
- 대상: 현재 저장소의 CIFAR-10 PERTINENCE 재현 파이프라인
- 권장 기준: Kubeflow Pipelines(KFP) Runtime/SDK 2.17.x의 같은 minor 버전
- 1차 구현 전략: 기존 CLI와 검증 계약을 보존하는 container component 기반 순차 DAG
- 저장소 변경 범위: 이 문서는 설계만 정의하며 KFP 코드, 컨테이너, 클러스터 리소스는 아직 구현하지 않는다.

가장 중요한 결정은 다음과 같다.

1. `build_cache`, `baseline`, `fixed`, `search`, `final`의 현재 경계를 KFP component로 유지한다.
2. NSGA-II는 처음부터 2,500개 task로 분산하지 않고 한 개의 장시간 search component로 실행한다.
3. 대용량 값은 KFP artifact, hash와 작은 상태 값은 parameter로 전달한다.
4. `final_evaluation` cache는 search component의 입력 계약에서 완전히 제외한다.
5. KFP 기본 cache에만 의존하지 않고 현재 코드의 SHA-256/fingerprint 검증을 함께 유지한다.
6. 실행 때마다 다운로드하지 않도록 asset bootstrap과 학습 pipeline을 분리한다.
7. 모든 container image와 외부 asset URI는 immutable digest 또는 content hash로 고정한다.

이 방식은 KFP 도입 때문에 논문 재현 의미론이 바뀌는 것을 막으면서, 각 단계의 재실행,
artifact lineage, 자원 격리, UI 관측성을 얻는 최소 위험 경로다.

## 2. 조사 결과 요약

### 2.1 적용할 KFP 기준

공식 release와 API 문서상 기준일의 최신 KFP release는 2.17.0이다. KFP v2는 Python
DSL을 IR YAML로 compile하며 backend가 component별 Kubernetes Pod를 실행한다. 신규
pipeline은 v1 DSL이 아니라 `@dsl.pipeline`, `@dsl.component` 또는
`@dsl.container_component`를 사용하는 v2 방식으로 작성한다.

본 설계는 다음 버전 정책을 권장한다.

- KFP Runtime과 SDK를 같은 `2.17.x` minor에 고정한다.
- compiler 환경에는 `kfp==2.17.*`와 호환되는 `kfp-kubernetes`를 lock한다.
- runtime image에는 KFP SDK를 넣지 않고도 실행 가능한 container component를 우선한다.
- 클러스터가 2.15 미만이면 Pipeline Run Workspace를 사용하지 않는다. 이 설계의 1차
  구현은 workspace 없이 object-store artifact만으로도 동작해야 한다.

근거:

- [KFP 2.17 releases](https://github.com/kubeflow/pipelines/releases)
- [KFP v2 API reference](https://www.kubeflow.org/docs/components/pipelines/reference/api/kubeflow-pipeline-api-spec/)
- [KFP version compatibility](https://www.kubeflow.org/docs/components/pipelines/reference/version-compatibility/)
- [KFP v2 migration](https://www.kubeflow.org/docs/components/pipelines/user-guides/migration/)

### 2.2 Component, parameter, artifact

KFP component는 원격에서 실행되는 하나의 container 작업이며, component instance인
task 사이의 의존성과 입출력으로 DAG가 구성된다. 작은 문자열, 수치, boolean, list,
dict는 JSON으로 직렬화되는 parameter에 적합하다. dataset, tensor cache, checkpoint,
전체 JSON 결과, 그림은 typed artifact로 전달해야 한다.

Artifact는 component 내부에서 `.path`, `.uri`, `.metadata`를 제공한다. 출력 `.path`에
쓴 파일은 backend가 pipeline root의 object store로 보존하고, parameter/artifact
전달 이력은 ML Metadata에 기록된다. 따라서 현재의 `artifacts/cifar10/...` 로컬 경로를
component 간에 공유하는 대신 명시적 artifact edge로 바꿔야 한다.

근거:

- [KFP component 개념](https://www.kubeflow.org/docs/components/pipelines/concepts/component/)
- [KFP data types와 lineage](https://www.kubeflow.org/docs/components/pipelines/user-guides/data-handling/data-types/)
- [Parameter 전달](https://www.kubeflow.org/docs/components/pipelines/user-guides/data-handling/parameters/)
- [Artifact 작성과 추적](https://www.kubeflow.org/docs/components/pipelines/user-guides/data-handling/artifacts/)

### 2.3 Pipeline root와 workspace

Pipeline root는 run artifact를 저장하는 SeaweedFS, S3 또는 GCS 경로다. Artifact metadata는
SQL/ML Metadata에, 실제 파일은 pipeline root에 저장된다. 학습 pipeline의 정식 산출물은
PVC의 임의 경로가 아니라 이 object-store artifact여야 한다.

KFP 2.15부터 Pipeline Run Workspace가 제공되어 큰 artifact를 매 단계 object store와
주고받는 비용을 줄일 수 있다. 다만 workspace는 Kubernetes/PVC 결합도를 높이고 실행
종료 후의 보존 정책을 별도로 다뤄야 한다. CIFAR-10 1차 구현은 object-store artifact를
정본으로 두고, 전송 시간이 실제 병목임을 측정한 뒤 workspace를 run-local scratch로만
도입한다. 특히 pipeline 전체에 같은 workspace를 자동 mount하면 `final_evaluation.pt`가
search pod에도 보일 수 있으므로, holdout 격리를 확인하기 전에는 pipeline-wide workspace를
사용하지 않는다.

근거:

- [Pipeline root 개념](https://www.kubeflow.org/docs/components/pipelines/concepts/pipeline-root/)
- [Pipeline root 설정](https://www.kubeflow.org/docs/components/pipelines/user-guides/data-handling/pipeline-root/)
- [Object store 설정](https://www.kubeflow.org/docs/components/pipelines/operator-guides/configure-object-store/)
- [Artifact와 Pipeline Run Workspace](https://www.kubeflow.org/docs/components/pipelines/user-guides/data-handling/artifacts/)

### 2.4 Cache, retry, control flow, 자원

KFP task cache는 기본적으로 활성화되며 같은 component와 같은 입력/parameter의 성공한
출력을 재사용한다. `.set_caching_options()`로 task별 정책을 명시할 수 있다. 이 프로젝트는
입력 content의 hash와 code/image digest가 바뀌면 반드시 새 실행으로 인식되어야 하므로
mutable tag나 mutable URI를 입력으로 사용하면 안 된다.

Task에는 CPU/memory request와 limit, accelerator type/count, retry를 설정할 수 있다.
`dsl.ParallelFor`는 독립 task의 fan-out/fan-in에는 적합하지만, 현재 NSGA-II처럼 다음 세대가
이전 세대 selection 결과에 의존하는 탐색을 그대로 병렬화해 주지는 않는다.

근거:

- [KFP caching](https://www.kubeflow.org/docs/components/pipelines/user-guides/core-functions/caching/)
- [Task 구성과 retry/resource](https://www.kubeflow.org/docs/components/pipelines/user-guides/components/compose-components-into-pipelines/)
- [조건, loop, exit handling](https://www.kubeflow.org/docs/components/pipelines/user-guides/core-functions/control-flow/)
- [Kubernetes-specific 기능과 PVC](https://www.kubeflow.org/docs/components/pipelines/user-guides/core-functions/platform-specific-features/)

### 2.5 실행과 운영 단위

Pipeline definition은 compile된 IR YAML로 versioning하고, 한 번의 실행은 run, 관련 run은
experiment로 묶는다. UI에서 DAG, task log, artifact, 실행 이력을 확인할 수 있다. 반복
실행은 recurring run으로 만들 수 있지만 이 연구 pipeline은 비용이 크고 final holdout
운영 규칙이 있으므로 기본값은 수동 또는 CI 승인형 단발 run으로 둔다.

로컬 KFP runner는 개발 피드백에는 유용하지만 cache, retry, 자원 지정, 인증과 일부 control
flow를 재현하지 못한다. 따라서 compile test와 container smoke 뒤 실제 namespace에서 작은
remote smoke run이 반드시 필요하다.

근거:

- [Pipeline과 IR compile](https://www.kubeflow.org/docs/components/pipelines/user-guides/core-functions/compile-a-pipeline/)
- [Run과 recurring run](https://www.kubeflow.org/docs/components/pipelines/concepts/run/)
- [KFP local execution의 제한](https://www.kubeflow.org/docs/components/pipelines/user-guides/core-functions/execute-kfp-pipelines-locally/)

### 2.6 Kubernetes 통합과 보안

KFP의 platform-neutral DSL에 없는 service account, secret, PVC, node affinity/toleration은
`kfp-kubernetes` extension으로 설정할 수 있다. GPU scheduling에는 cluster의 NVIDIA driver와
device plugin이 선행되어야 한다. Secret은 단순 base64 값이므로 workload identity를 우선하고,
불가피한 Secret에는 encryption at rest와 최소 권한 RBAC를 적용한다.

운영/다중 사용자 배포에서는 standalone 개발 구성을 외부에 노출하지 않고 Kubeflow 배포판의
namespace/RBAC 구성을 사용한다. 다만 공식 multi-user 문서가 설명하는 격리 한계를 고려해,
민감한 final artifact 보안은 KFP Profile만이 아니라 object-store ACL로도 강제한다.

근거:

- [kfp-kubernetes 2.17 API](https://kfp-kubernetes.readthedocs.io/en/kfp-kubernetes-2.17/source/kubernetes.html)
- [KFP installation](https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/)
- [KFP multi-user isolation](https://www.kubeflow.org/docs/components/pipelines/operator-guides/multi-user/)
- [Kubernetes GPU scheduling](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
- [Kubernetes Secret 모범 사례](https://kubernetes.io/docs/concepts/security/secrets-good-practices/)

## 3. 현재 파이프라인 분석

현재 저장소는 논문 방법을 공개 CIFAR-10 asset에 맞춘 재현이다. 정적 accuracy-cost
비교에는 Pareto 경계 10종을 사용하지만 실제 routing, label, FC 출력, NSGA-II에는 다음
4개 expert만 사용한다.

1. `shufflenetv2_x0_5`: feature extractor이자 최저비용 fallback
2. `mobilenetv2_x0_5`
3. `resnet56`
4. `repvgg_a1`

현재 실행 순서와 계약은 다음과 같다.

| 현재 단계 | 구현 진입점 | 핵심 입력 | 핵심 출력/검증 |
|---|---|---|---|
| asset 준비/검증 | `scripts/prepare_assets.py` | asset manifest | dataset/source/checkpoint 12개 SHA-256 검증 |
| production preflight | `scripts/preflight.py` | config, asset, dataset, source, CUDA | dependency, disk, split plan, 4 expert `[2,10]` smoke |
| prediction cache | `scripts/build_cache.py` | 위 asset과 3 split | train/search/final `.pt`, full test expert accuracy gate |
| baseline/oracle | `pertinence baseline` | train/search cache | expert Top-1, route 분포, cheapest-correct oracle |
| fixed dispatcher | `pertinence fixed` | train/search cache | 고정 penalty head, loss, search metrics |
| NSGA-II | `pertinence search` | train/search cache | 2,500 evaluation, search Pareto와 FC state |
| final 평가 | `pertinence final` | final cache, search 결과 | retained head 복원, 2,000장 평가, checkpoints |
| 분석/보고 | CIFAR-10 analysis script | search/final/static catalog | 검증된 표, CSV, JSON, SVG/PNG |

주요 불변식은 다음과 같다.

- `dispatcher_train=50,000`, `ga_search=8,000`, `final_evaluation=2,000`
- split seed와 experiment seed는 `20260824`
- route label은 cheapest correct expert이며 아무 expert도 맞히지 못하면 index 0이다.
- 비용은 `extractor + FC + selected expert`이고 extractor 선택 시 backbone 계산을 공유한다.
- catalog MAdds는 한 곳에서만 `1 MAC = 2 FLOPs`로 바꾼다.
- search는 train/search cache만 열며 final component에서 처음 retained head를 final cache에 적용한다.
- final에서는 FC를 재학습하지 않고 search가 보존한 state를 복원한다.

근거가 되는 저장소 문서/코드는 다음과 같다.

- 전체 단계: `README.md`의 `Staged workflow`
- 실행 및 합격 조건: `experiments/cifar10/docs/runbook.ko.md`
- 논문 재현 계약: `experiments/cifar10/docs/reproduction-plan.ko.md`
- config 제약: `src/pertinence/config.py`
- cache payload/fingerprint: `src/pertinence/cache.py`
- search와 final 입력 분리: `src/pertinence/cli.py`
- NSGA-II 목적과 세대 의존성: `src/pertinence/optimization.py`
- seed와 deterministic 설정: `src/pertinence/reproducibility.py`

## 4. 목표와 비목표

### 목표

- 기존 로컬 runbook을 재현 가능한 KFP DAG로 치환한다.
- 논문 방법과 현재 공개-asset 적응 가정을 그대로 보존한다.
- 각 cache, model, result의 생성자와 입력 lineage를 KFP UI/ML Metadata에서 추적한다.
- GPU는 필요한 단계에서만 할당한다.
- 성공한 고비용 단계를 안전하게 재사용하고 실패 지점부터 재실행할 수 있게 한다.
- final holdout이 search/selection 경로에 노출되지 않도록 interface와 권한을 분리한다.
- 기존 완료 결과를 golden regression으로 이용할 수 있게 한다.

### 1차 구현의 비목표

- 논문에 없는 ResNet8/14를 만들어 Figure 6(c)의 수치를 직접 복제하지 않는다.
- NSGA-II 알고리즘이나 penalty loss를 개선하지 않는다.
- 2,500개 individual을 KFP task로 분산하지 않는다.
- Katib로 NSGA-II를 교체하지 않는다. Katib의 trial 모델은 현재의 세대별 다목적 GA
  상태와 동일하지 않다.
- final 결과를 보고 배포 head를 새로 선택하지 않는다.
- online serving/KServe 배포를 이번 pipeline의 필수 범위에 넣지 않는다.

## 5. 대상 아키텍처

### 5.1 두 pipeline으로 분리

```text
관리자/승인형 asset-bootstrap pipeline (필요할 때만)
  download immutable URLs
    -> verify every SHA-256
    -> safely extract archives
    -> publish versioned AssetBundle URI

연구자용 pertinence-training pipeline (매 실험 run)
  import immutable config + AssetBundle
    -> validate-config
    -> build-split-manifest
    -> verify-assets
    -> gpu-preflight
    -> build-prediction-caches
         ├─ train + search cache -> baseline-oracle ─┐
         ├─ train + search cache -> fixed-gate ──────┼─> nsga2-search ─┐
         └─ final cache (격리) ─────────────────────────────────────────┼─> final-evaluate
                  train/search cache는 검증 입력으로 final에도 전달 ───┘
                                                               -> analyze-report
```

Asset bootstrap을 분리하는 이유는 GPU task에서 인터넷 다운로드를 하지 않고, 여러 run이
동일한 검증 자산을 재사용하며, 자산 갱신 권한을 학습 실행 권한과 분리하기 위해서다. 이미
동일한 bundle이 object store에 있으면 training pipeline은 `dsl.importer`로 가져오고 다시
hash만 검증한다.

테스트와 lint는 runtime DAG가 아니라 image build 전 CI gate에 둔다. CI를 통과한 image
digest만 pipeline version에 넣는다. 필요하면 별도의 diagnostics pipeline에서 재실행할 수
있지만, 동일 image의 모든 연구 run에서 `pytest`를 반복하지 않는다.

### 5.2 Pipeline 입력 parameter

| 이름 | 타입 | 예시/정책 |
|---|---|---|
| `config_uri` | `str` | versioned object URI, mutable `latest` 금지 |
| `config_sha256` | `str` | 제출 전에 계산한 64-hex digest |
| `asset_bundle_uri` | `str` | content-addressed 또는 versioned URI |
| `asset_bundle_sha256` | `str` | bundle 전체 digest |
| `experiment_name` | `str` | `cifar10-chenyaofo-pareto` |
| `run_mode` | `str` | `smoke` 또는 `full`; production 결과는 `full`만 인정 |
| `force_recompute_token` | `str` | 기본 빈 값; 승인된 강제 재계산 때만 변경 |

Container image는 실행 parameter로 받지 않고 compile 시 digest로 고정한다. GPU type,
service account, node selector도 일반 연구 parameter가 아니라 environment별 pipeline factory
설정으로 관리한다. 이를 자유 입력으로 열면 재현성과 권한 통제가 약해진다.

`run_mode=smoke`는 production과 같은 config/asset으로 preflight까지만 실행한다. 데이터 크기나
NSGA-II 횟수를 몰래 줄인 결과를 full 결과처럼 저장하지 않는다. 조건부 output 복잡성을 피하려면
동일 component를 조합한 `pertinence-smoke` pipeline을 별도로 compile해도 된다.

### 5.3 DSL 구성 골격

아래는 edge와 정책을 설명하는 의사 코드이며 실제 구현에서는 pinned KFP 2.17 SDK로 compile
test를 통과시켜야 한다.

```python
@dsl.pipeline(name="pertinence-cifar10")
def pertinence_pipeline(config_uri: str, config_sha256: str,
                        asset_bundle_uri: str, asset_bundle_sha256: str,
                        force_recompute_token: str = ""):
    config = dsl.importer(
        artifact_uri=config_uri, artifact_class=dsl.Artifact, reimport=False)
    assets = dsl.importer(
        artifact_uri=asset_bundle_uri, artifact_class=dsl.Artifact, reimport=False)

    spec = validate_config(
        config=config.output, expected_sha256=config_sha256)
    splits = build_split_manifest(spec=spec.outputs["normalized_config"])
    verified = verify_assets(
        assets=assets.output,
        expected_sha256=asset_bundle_sha256,
        spec=spec.outputs["normalized_config"])

    preflight = gpu_preflight(
        assets=assets.output,
        spec=spec.outputs["normalized_config"],
        splits=splits.output).after(verified)
    preflight.set_caching_options(enable_caching=False)

    caches = build_prediction_caches(
        assets=assets.output,
        spec=spec.outputs["normalized_config"],
        splits=splits.output,
        force_recompute_token=force_recompute_token).after(preflight)

    baseline = baseline_oracle(
        train_cache=caches.outputs["train"],
        search_cache=caches.outputs["search"])
    fixed = fixed_dispatcher_gate(
        train_cache=caches.outputs["train"],
        search_cache=caches.outputs["search"])
    search = nsga2_search(
        train_cache=caches.outputs["train"],
        search_cache=caches.outputs["search"]).after(baseline, fixed)

    final = final_evaluate(
        train_cache=caches.outputs["train"],
        search_cache=caches.outputs["search"],
        final_cache=caches.outputs["final"],
        search_result=search.outputs["result"],
        states=search.outputs["states"])
    analyze_report(search=search.outputs["result"], final=final.outputs["report"])
```

Resource, retry, service account, node selector modifier는 작은 helper 함수로 모아 environment별
차이를 한 곳에서 적용한다. Component 본문은 Kubernetes API를 직접 호출하지 않는다.

## 6. Component 상세 설계

아래 자원은 최초 요청/상한의 출발점이며 staging profiling 후 조정한다. GPU 단계는 기본
1 GPU다. 논문 장비와 wall-time을 비교하려면 A100 80GB node pool을 고정하고, 정확도와
MFLOPs 재현만 목적이면 검증된 CUDA 호환 GPU를 사용할 수 있다.

| 순서 | Component | Artifact 입력 → 출력 | 초기 자원 | Cache / retry | 성공 gate |
|---:|---|---|---|---|---|
| 1 | `validate-config` | imported config → normalized config, spec report | 0.5 CPU, 2Gi | on / 0 | schema, 모델 순서, split, seed, 1 MAC=2 FLOPs |
| 2 | `build-split-manifest` | normalized config → `SplitManifest` | 0.5 CPU, 2Gi | on / 0 | 50,000/8,000/2,000, overlap 0, stable index/hash |
| 3 | `verify-assets` | AssetBundle, normalized config → asset report/fingerprint | 2 CPU, 8Gi | on / 1 | 12 asset, source revision, 모든 SHA 일치 |
| 4 | `gpu-preflight` | AssetBundle, config, split → preflight report/metrics | 4 CPU, 16Gi, 1 GPU | **off** / **0** | CUDA, dataset 크기, disk, 4 expert finite `[2,10]` |
| 5 | `build-prediction-caches` | bundle, config, split → 3 `Dataset` cache, build report | 8 CPU, 32Gi, 1 GPU | on / 1 | expert delta gate, shape/order/hash, 세 split 모두 commit |
| 6 | `baseline-oracle` | train/search cache → JSON report, scalar metrics | 2 CPU, 8Gi | on / 0 | cache fingerprint, oracle/route 통계 finite |
| 7 | `fixed-dispatcher-gate` | train/search cache → `Model`, JSON, metrics | 4 CPU, 16Gi, 1 GPU | on / 0 | 20 epochs, loss/parameters finite |
| 8 | `nsga2-search` | train/search cache → search result, Pareto state bundle, metrics | 8 CPU, 32Gi, 1 GPU | on / **0** | 50×50 완료, 2,500 unique IDs, Pareto/state 검증 |
| 9 | `final-evaluate` | train/search/**final** cache, search result/states → final front models/report | 4 CPU, 16Gi, 1 GPU | on / 1 | retained ID 전부 복원, 2,000장, fingerprint 일치 |
| 10 | `analyze-report` | search/final/static catalog → CSV, figures, HTML/Markdown | 2 CPU, 8Gi | on / 0 | 독립 Pareto/ID/hash 재검증 후 보고서 생성 |
| 11 | `cleanup-notify` | task status → 작은 status artifact | 0.2 CPU, 512Mi | off / 1 | 성공/실패 상태 기록; 연구 결과를 변경하지 않음 |

`preflight`는 현재 node의 GPU/driver/disk 상태를 검사하므로 cache하면 안 된다. `search`는
중간 resume을 구현하기 전 retry가 전체 6~7시간 작업을 다시 시작하므로 자동 retry를 0으로
둔다. Preflight와 expert accuracy gate처럼 입력/환경 오류를 나타내는 gate도 자동 retry하지
않는다. Cache build의 object-store 일시 오류에 한해 1회 retry할 수 있으며 부분 output은 task
성공 전 KFP artifact로 publish하지 않는다.

`baseline-oracle`과 `fixed-dispatcher-gate`는 cache build 뒤 병렬로 실행할 수 있지만,
`nsga2-search`는 두 gate가 성공한 뒤 시작한다. 이를 data edge 또는 `.after()`로 명시한다.

## 7. Artifact 계약

### 7.1 공통 규칙

모든 artifact metadata에 가능한 범위에서 다음을 공통 기록한다.

- `schema_version`
- `kind`
- `created_at_utc`
- `code_git_sha`
- `container_image_digest`
- `config_fingerprint`
- `asset_fingerprint`
- `split_fingerprint` 또는 `cache_fingerprints`
- `experiment_seed`, `split_seed`
- Python/PyTorch/torchvision/THOP/pymoo/CUDA/cuDNN version
- GPU model과 Kubernetes node name

KFP metadata만 믿지 않고 결과 파일 내부에도 fingerprint를 유지한다. Object URI가 같아도
내용이 바뀌는 운영 오류를 검출하기 위해 파일 SHA-256을 별도 기록한다.

### 7.2 Artifact별 schema

| Artifact | KFP 타입 | 파일 내용 | 필수 metadata |
|---|---|---|---|
| `ExperimentSpec` | `Artifact` | normalized YAML/JSON | config SHA, schema, code/image digest |
| `AssetBundle` | `Artifact` | dataset, pinned source, checkpoints 또는 manifest+immutable URIs | 각 asset expected/actual SHA, source revision |
| `SplitManifest` | `Dataset` | split별 stable sample index와 manifest JSON | seed, 50,000/8,000/2,000, overlap 0, SHA |
| `PreflightReport` | `Metrics` + JSON `Artifact` | 현재 preflight report | status, device, dataset sizes, output shapes |
| `PredictionCache` | `Dataset` | split별 `.pt` | split, samples, model order, extractor, cache/split hash |
| `BaselineReport` | `Metrics` + JSON `Artifact` | expert/oracle/route 통계 | cache hashes, accuracy/cost 단위 |
| `FixedDispatcher` | `Model` | `state_dict` | feature dim, model order, penalty, weighting, seed |
| `SearchResult` | `Artifact` | evaluations/metrics/chromosomes JSON | 정확한 evaluation 수, search front IDs |
| `ParetoStateBundle` | `Model` | ID별 dispatcher state와 index manifest | search result SHA, ID→checkpoint SHA mapping |
| `FinalFrontBundle` | `Model` | 복원 검증된 head들 | final cache hash, search ID mapping |
| `FinalReport` | `Metrics`, `ClassificationMetrics`, JSON | final metrics/confusion/selection 비율 | final은 사후 평가임을 표시 |
| `ExperimentReport` | `HTML` 또는 `Markdown` | 표, 그림 link, 제한점 | search/final/static 비교의 서로 다른 모집단 명시 |

기존 cache payload의 아래 필드는 그대로 보존한다.

```text
schema_version, split, fingerprint,
asset_fingerprint, config_fingerprint, split_fingerprint,
model_order, feature_extractor,
targets[S], sample_indices[S], expert_predictions[S,4],
features[S,D], route_labels[S]
```

### 7.3 현재 CLI에서 반드시 고칠 artifact 경로 문제

현재 CLI는 directory와 side-effect 경로에 의존한다.

- `fixed.json` 옆에 `fixed.pt`를 쓴다.
- `final.json`의 suffix를 제거한 directory에 여러 checkpoint를 쓴다.
- `search.json` 안에 Pareto dispatcher state를 JSON list로 포함한다.
- final JSON에는 pod-local checkpoint 경로가 들어간다.

KFP에서는 선언한 output만 lineage와 lifecycle 관리 대상이 된다. 따라서 component wrapper를
만들기 전에 CLI를 다음처럼 명시적 output 계약으로 바꾼다.

```text
pertinence baseline --train-cache ... --search-cache ... --report-output ...
pertinence fixed    --train-cache ... --search-cache ... --report-output ... --model-output ...
pertinence search   --train-cache ... --search-cache ... --report-output ... --states-output-dir ...
pertinence final    --final-cache ... --search-result ... --states-input-dir ...
                    --report-output ... --models-output-dir ...
```

이때 final은 `optimization.evaluate_final_solution()`처럼 다시 학습하는 경로가 아니라 현재
`pertinence.cli final`처럼 저장한 state를 복원하는 경로를 사용한다.

현재 `make_splits()` 결과는 각 cache 안에 포함되지만 별도 split manifest 파일로 실제 보존되지
않는다. `save_splits()`를 pipeline 경로에 연결하고, cache builder가 임의로 split을 다시 만드는
대신 같은 `SplitManifest`를 입력으로 받도록 변경한다.

## 8. Final holdout 격리 설계

### 8.1 DAG 수준

`nsga2-search` component signature에는 오직 다음만 둔다.

```text
ExperimentSpec
dispatcher_train PredictionCache
ga_search PredictionCache
```

통합 `cache-dir`이나 final URI를 전달하지 않는다. `final_evaluate`만
`final_evaluation PredictionCache`를 입력받는다. 현재 final 구현의 fingerprint와 feature-dimension
검증을 보존하기 위해 final task는 train/search cache도 받지만, 반대 방향으로 final cache가
search task에 전달되는 edge는 없다. Search 완료 전 final artifact가 생성되어 있을 수는 있지만
search pod에 stage되거나 공용 workspace로 노출되지 않아야 한다.

Cache build가 official test 10,000장 전체로 고정 expert accuracy gate를 수행하는 것은 기존
계약의 예외다. 이 gate 결과는 checkpoint/source/transform 무결성 확인에만 쓰고 search
fitness나 candidate selection으로 출력하지 않는다.

### 8.2 권한 수준

강한 격리가 필요하면 DAG wiring만으로 부족하다. Search service account가 object store
전체를 읽을 수 있으면 final URI를 추측할 수 있기 때문이다.

- `train/`과 `search/` prefix는 search service account가 읽을 수 있다.
- `final/` prefix는 cache builder가 쓰고 final evaluator만 읽을 수 있다.
- Search와 final task는 서로 다른 Kubernetes service account를 사용한다.
- static object-store credential secret을 pipeline parameter나 환경변수 값으로 직접 넘기지
  않는다. Workload identity 또는 service-account 연동을 우선한다.
- KFP launcher 자체의 artifact staging 권한과 component container의 직접 object-store 권한을
  구분해 점검한다.

### 8.3 통계적 사용 규칙

- Search Pareto front는 `ga_search`에서만 결정한다.
- Final은 search-retained 모든 head의 일반화 성능을 한 번 기술한다.
- Final에서 다시 계산한 Pareto front나 비용 상한별 최고점을 새 hyperparameter 선택에 쓰지
  않는다.
- 배포 head가 필요하면 final을 보기 전에 search metric 기반 정책을 고정하거나, 별도 validation
  또는 새로운 test set을 확보한다.

## 9. NSGA-II 실행 설계

### 9.1 1차: 단일 search component

현재 `pymoo.minimize` 호출과 `ElementwiseProblem._evaluate()`를 그대로 한 container에서
실행한다. 이 선택은 다음 의미론을 보존한다.

- 세대별 selection/crossover/mutation 순서
- evaluation ID의 순차 부여
- 모든 individual에서 동일한 FC 초기화와 minibatch seed
- 전체 evaluation에서 계산한 non-dominated front
- retained evaluation의 정확한 dispatcher state

KFP task 병렬화를 하지 않아도 cache build와 search를 서로 다른 자원/실행 단위로 분리하고,
실패 이후 final만 재실행하는 효과는 얻는다.

### 9.2 가장 먼저 추가할 복구 기능

현재 search는 세대 중간 resume을 지원하지 않는다. 장시간 task를 운영하기 전에 매 generation
끝에 다음을 원자적으로 checkpoint하는 기능을 추가한다.

- 완료 generation index
- population과 objective/state
- pymoo algorithm state 또는 동등한 복원 상태
- Python, NumPy, PyTorch CPU/CUDA RNG state
- 다음 evaluation ID와 누적 evaluation records
- 현재 Pareto IDs와 dispatcher states
- config/cache/code/image fingerprints

동일 fingerprint에서만 resume을 허용한다. Checkpoint는 run-local PVC에 주기적으로 쓰고,
세대 경계마다 durable object store에도 publish한다. Resume 구현 전에는 preemptible/spot GPU를
사용하지 않는다.

### 9.3 2차: 제한적 병렬화 조건

병렬화는 먼저 알고리즘 결과 동등성 test를 만든 뒤 검토한다. 가능한 구조는 generation을
controller task로 두고 그 generation의 individual 평가만 fan-out하는 것이다. 그러나 KFP의
단순 `ParallelFor`로 바꾸면 다음 문제가 생긴다.

- 다음 세대 후보는 현재 세대 전체 결과가 있어야 생성된다.
- 50 generation을 DAG에 정적으로 펼치면 task가 약 2,500개가 된다.
- 각 pod가 동일한 train/search tensor를 다시 내려받아 시작 비용과 object-store 부하가 커진다.
- 병렬 완료 순서가 evaluation ID와 floating-point/RNG 순서를 바꿀 수 있다.
- GPU 50개 동시 요청은 FC head의 작은 연산에 비해 비효율적일 수 있다.

따라서 2차 최적화 우선순위는 KFP fan-out보다 먼저 한 pod 안의 bounded worker 또는 vectorized
training을 검토하는 것이다. 변경 전후에 모든 chromosome, objective, retained ID와 핵심
metric을 비교하고, 결과가 달라지면 별도 실험 방법으로 versioning한다.

## 10. Cache, idempotency, 재실행

KFP cache와 도메인 cache의 역할을 구분한다.

| 계층 | 역할 |
|---|---|
| KFP task cache | 성공한 component 전체를 재실행하지 않고 output artifact lineage를 재사용 |
| PERTINENCE cache fingerprint | tensor 내부의 asset/config/split/model-order 의미가 같은지 검증 |
| 파일 SHA-256 | object store 전송/운영 중 실제 byte 무결성 확인 |

Task별 정책은 명시적으로 코드에 남긴다.

- `validate-config`, `verify-assets`, `build-cache`, `baseline`, `fixed`, `search`, `final`, `analysis`:
  입력이 immutable이면 cache 활성화
- `gpu-preflight`, cleanup/status: cache 비활성화
- 강제 재실행은 `force_recompute_token`을 해당 task의 입력으로 연결해 cache key를 바꾼다.
- Mutable image tag, 같은 URI의 덮어쓰기, timestamp를 몰래 읽는 component는 금지한다.
- 기존 output과 fingerprint가 다르면 덮어쓰지 않고 실패시키며 새 run/prefix를 사용한다.

현재 `experiment_config_fingerprint()`는 fixed/search 설정과 output 경로를 포함한 raw config 전체를
hash한다. 이 때문에 penalty나 generation 수만 바꿔도 60,000장 prediction cache가 무효화된다.
구현 시 fingerprint를 다음 계층으로 분리한다.

1. `asset_fingerprint`: routing에 필요한 dataset/source/4 checkpoint와 별도의 static catalog 자산
2. `data_fingerprint`: transform, split IDs/seed
3. `prediction_cache_fingerprint`: asset + data + model order + extractor + code/image schema
4. `dispatcher_fingerprint`: prediction cache + optimizer/loss/seed
5. `search_fingerprint`: dispatcher + NSGA-II 설정
6. `final_fingerprint`: retained state + final cache + metric schema

정적 비교에만 쓰는 나머지 6개 checkpoint의 변화가 routing cache를 불필요하게 무효화하지 않게
한다. 기존 broad fingerprint는 schema migration 동안 읽기 호환을 유지한다.

Retry는 idempotent한 단계에만 둔다. 특히 KFP retry는 실패 task의 미완성 output을 이어서 쓰는
기능이 아니므로, search checkpoint/resume과 혼동하면 안 된다.

## 11. 컨테이너와 Kubernetes 설계

### 11.1 Image

권장 image 구성은 다음과 같다.

- Python 3.11
- CUDA와 `torch==2.5.1` 호환 runtime
- 저장소 package와 현재 pin된 dependency
- 실행에 필요한 CA certificate와 non-root user
- OCI label: git SHA, build timestamp, source URL, license
- SBOM과 vulnerability scan 결과
- registry에서는 tag가 아니라 `image@sha256:...`로 참조

CPU/GPU image를 처음부터 두 개로 나누면 의존성 차이로 결과가 달라질 수 있다. 1차는 하나의
검증된 image로 시작하고, 크기/시작 시간이 병목일 때 동일 lockfile 기반 CPU image를 분리한다.

Asset과 학습 결과를 image layer에 넣지 않는다. Upstream model source는 현재 재현 계약대로
고정 source artifact로 받거나, image에 넣는다면 image digest와 source SHA를 둘 다 결과에
기록한다.

### 11.2 Pod와 scheduling

- GPU task: `nvidia.com/gpu: 1`, GPU node selector, 필요 시 taint toleration
- CPU/memory request와 limit를 모두 설정해 scheduling과 OOM 원인을 명확히 한다.
- cache build/search에는 충분한 ephemeral storage 또는 전용 run workspace를 제공한다.
- search는 완료 결과가 있기 전까지 긴 graceful termination과 충분한 wall time을 둔다.
- 모든 task는 non-root, 제한된 service account, namespace quota 아래에서 실행한다.
- `/tmp`와 artifact staging 경로만 writable로 두는 read-only root filesystem을 우선 검토한다.

KFP의 platform-neutral task API로 CPU/memory/GPU/retry를 지정하고, service account, PVC,
secret, node scheduling 같은 Kubernetes 전용 기능은 `kfp-kubernetes` modifier로 격리한다.

초기 resource profile은 다음처럼 request/limit을 분리한다. `ephemeral-storage`는 KFP 기본
task API가 아니라 Kubernetes pod 설정 또는 전용 scratch volume으로 적용할 수 있다.

| Profile | CPU request/limit | Memory request/limit | Ephemeral request/limit | GPU | 권장 timeout |
|---|---:|---:|---:|---:|---:|
| small CPU | 500m / 1 | 1Gi / 2Gi | 2Gi / 5Gi | 0 | 15분 |
| report CPU | 2 / 4 | 4Gi / 8Gi | 5Gi / 10Gi | 0 | 30분 |
| GPU preflight | 4 / 8 | 16Gi / 32Gi | 10Gi / 20Gi | 1 | 30분 |
| GPU cache | 8 / 12 | 32Gi / 48Gi | 10Gi / 20Gi | 1 | 3시간 |
| GPU fixed/final | 4 / 8 | 16Gi / 32Gi | 5Gi / 10Gi | 1 | 2시간 |
| GPU search | 8 / 12 | 32Gi / 48Gi | 10Gi / 20Gi | 1 | 12시간 |

실제 관측치를 수집한 뒤 requests를 p95 사용량에 여유를 더한 값으로 낮추거나 높인다. Memory
limit 초과는 OOM으로 task를 종료하므로 단순히 낮은 값으로 고정하지 않는다.

### 11.3 Secret과 network

- Registry pull, pipeline root, asset store 접근은 workload identity를 우선한다.
- 불가피한 secret은 Kubernetes Secret reference로 주입하고 값 자체를 parameter/log에 남기지 않는다.
- Training namespace의 기본 egress는 차단하고 object store, registry, KFP/metadata endpoint만 허용한다.
- 외부 HTTPS 다운로드는 asset-bootstrap service account에만 허용한다.
- Component log에는 signed URL, token, 환경변수 전체 dump를 남기지 않는다.

OSS KFP에서 container component와 호환되는 전통적 `Input[Artifact]`/`Output[Artifact]` 문법을
사용한다. 새 Pythonic artifact 반환 문법은 backend/component 종류에 따른 지원 차이가 있으므로
이 pipeline의 이식성 기준으로 삼지 않는다.

KFP multi-user metadata가 민감한 데이터에 대해 namespace 이상의 완전한 보안 경계를 제공한다고
가정하지 않는다. Final sample ID, label, credential 같은 민감 정보는 ML Metadata custom property나
UI용 metric에 기록하지 않고 접근 통제된 artifact 본문에만 둔다.

## 12. 관측성과 결과 보고

KFP UI의 scalar `Metrics`에는 최소한 다음을 기록한다.

- asset/expert accuracy gate 상태
- split별 sample 수와 cache fingerprint의 짧은 prefix
- baseline expert Top-1과 oracle accuracy/MFLOPs
- fixed search accuracy/MFLOPs와 마지막 loss
- search evaluation 수, retained 수, accuracy/cost 범위
- final solution 수, accuracy/cost 범위, no-correct 비율

전체 loss 배열, 2,500 evaluation, confusion matrix 묶음은 작은 parameter로 넘기지 않고 JSON/CSV/
HTML artifact로 저장한다. 대표 solution의 confusion matrix만 `ClassificationMetrics`로 보여주고,
전체 frontier는 HTML/plot artifact로 제공한다.

주 지표는 route hit rate가 아니라 선택 expert의 system Top-1과 average system MFLOPs다. Route
accuracy가 낮더라도 overestimation은 task accuracy를 유지할 수 있으므로 route accuracy는 보조
지표로 표기한다.

정적 10-model catalog point와 동적 final point는 accuracy 평가 모집단과 비용 정의가 다르다.
그림에 함께 표시할 수는 있지만, 지배 관계의 확정은 같은 final 2,000장에서 평가한 4개 routing
expert와 dynamic system 사이에서만 한다.

## 13. 재현성과 lineage

Run 이름만으로 재현성을 주장하지 않는다. 다음 identity tuple을 모든 핵심 artifact에서 확인한다.

```text
(pipeline version,
 component spec digest,
 container image digest,
 code git SHA,
 config SHA/fingerprint,
 asset manifest/bundle SHA,
 source revision,
 checkpoint SHAs,
 split manifest SHA,
 experiment seed,
 framework/CUDA versions,
 GPU model)
```

현재 `seed_everything()`의 Python/NumPy/PyTorch/CUDA seed, cuDNN deterministic 설정,
`CUBLAS_WORKSPACE_CONFIG`, deterministic algorithms 설정은 container에서도 유지한다. Kubernetes
retry나 향후 병렬화가 individual seed를 임의로 바꾸지 않게 한다.

Experiment는 예를 들어 `pertinence-cifar10`으로 두고, pipeline version은
`gitsha-imageDigest-configSchema` 형식으로 만든다. Run display name에는 dataset, 날짜, 짧은 config
hash를 넣되 이름이 identity를 대신하지는 않는다.

## 14. CI/CD 설계

### Pull request

1. `pytest`, `ruff` 실행
2. KFP component/pipeline unit test
3. pipeline IR compile 및 type check
4. compile 결과에서 image가 digest로 고정됐는지 검사
5. search component input에 final artifact/URI가 없는지 구조 검사
6. container build, SBOM, vulnerability scan
7. container 내부 CLI smoke test

### Merge/tag

1. image를 immutable digest로 push/sign
2. 해당 digest를 사용해 IR YAML 재compile
3. IR과 component specs를 build artifact로 보존
4. KFP에 새 pipeline version upload
5. staging namespace에서 `run_mode=smoke` remote run
6. 승인 후 full run 제출

### Release gate

- SDK/compiler version과 runtime compatibility 확인
- 모든 task image digest pin 확인
- pipeline root 쓰기/읽기와 artifact UI rendering 확인
- GPU scheduling과 production preflight 통과
- 실패 task retry 및 KFP cache-hit smoke
- service account가 허용된 object-store prefix만 접근하는지 확인

## 15. 구현 파일 배치 제안

```text
pipelines/
  pertinence/
    components.py          # container component interface
    pipeline.py            # training DAG
    Dockerfile             # shared component image build
    requirements-components.lock
    requirements-kfp.lock  # compiler 전용
    component_specs/       # 필요 시 독립 YAML specs
    docs/
    tests/
      test_compile.py
      test_dag_contract.py
  assets/
    pipeline.py            # 승인형 asset bootstrap
src/pertinence/
  pipeline_cli.py          # artifact path ↔ component core adapter
```

애플리케이션 dependency와 compiler dependency를 분리한다. `kfp`를 현재 학습 package의 필수
runtime dependency로 추가하지 않는다.

## 16. 구현 순서

### Phase 0: 계약 고정

- 기존 완료 run의 config/cache/search/final hash와 핵심 metric을 golden fixture로 정리
- cache/result schema version 문서화
- final leakage와 artifact side-effect를 검사하는 test 추가

### Phase 1: CLI 입출력 정규화

- directory 암묵 계약을 명시적 파일/artifact 경로로 변경
- search result와 Pareto model state 분리
- pod-local 절대 경로를 portable relative path/URI로 변경
- 모든 출력에 file SHA와 code/image identity 추가

### Phase 2: Container component와 compile

- 현재 CLI를 호출하는 얇은 wrapper 구현
- CPU/GPU 자원, cache, retry, service account 설정
- IR compile/type/DAG contract test
- asset-bootstrap pipeline 구현

### Phase 3: Staging 검증

- tiny/smoke config로 remote pipeline 성공
- 동일 입력 2회 실행 시 의도한 task만 cache hit인지 확인
- 고의 hash mismatch, corrupt cache, CUDA 미할당, expert gate 실패 test
- final artifact가 search pod에 mount/download되지 않는지 확인

### Phase 4: Full equivalence run

- 동일 immutable asset/config/image에서 full run
- cache tensor와 route label 비교
- 2,500 chromosome/evaluation ID/objective 비교
- retained search ID/state 비교
- final 핵심 metric과 confusion/selected count 비교
- 차이가 있으면 KFP 이관과 알고리즘 변경을 분리해 원인 분석

### Phase 5: 운영 강화

- generation checkpoint/resume
- object-store prefix ACL과 task별 service account
- artifact retention/lifecycle policy
- 필요할 때만 workspace 또는 제한적 병렬화 실험

## 17. 인수 기준

### 기능

- Pipeline IR이 pinned KFP compiler로 type error 없이 생성된다.
- Full DAG가 사람의 중간 파일 복사 없이 완료된다.
- 단계별 합격 조건이 실패하면 downstream task는 실행되지 않는다.
- Search는 정확히 train/search cache만 입력받는다.
- Final은 search-retained 모든 state를 재학습 없이 복원한다.
- 분석 결과가 정적 10-model과 routing 4-model의 역할을 섞지 않는다.

### 데이터와 재현성

- 12개 asset hash와 source revision이 일치한다.
- split은 50,000/8,000/2,000이고 search/final overlap은 0이다.
- Cache의 model order, extractor, sample ID, tensor shape, 모든 fingerprint가 일치한다.
- Expert accuracy delta가 config의 `[-0.25pp, +0.75pp]` gate를 통과한다.
- Fixed/search loss와 parameter, 모든 objective/metric이 finite다.
- Search evaluation ID는 unique하고 수가 2,500이다.
- 독립 Pareto 재계산 결과와 retained flag가 일치한다.
- Final solution ID 집합과 search-retained ID 집합이 정확히 같다.
- Confusion matrix와 selected count의 합이 각각 2,000이다.

### 운영과 보안

- GPU task만 GPU를 요청한다.
- Preflight는 매 run 실제 실행된다.
- 동일 immutable 입력의 2차 run에서 의도한 cache hit가 발생한다.
- Mutable URI/tag가 pipeline spec에 없다.
- Search service account는 final prefix를 읽을 수 없다.
- Secret 값이 IR YAML, parameter, log, artifact metadata에 나타나지 않는다.
- 중단된 task의 부분 output이 정식 artifact로 노출되지 않는다.

## 18. 위험과 대응

| 위험 | 영향 | 대응 |
|---|---|---|
| Search pod 선점/노드 장애 | 6~7시간 재계산 | 초기 non-preemptible GPU, generation checkpoint/resume 구현 |
| KFP cache가 mutable URI를 재사용 | 오래된 결과 혼입 | content hash parameter, immutable URI/image digest, 내부 fingerprint 재검증 |
| 대형 artifact 전송 | 시작/종료 지연 | 먼저 계측, object store 정본 유지, 필요 시 2.15+ workspace |
| Final leakage | 과적합된 결과 보고 | artifact edge와 service account/prefix ACL 동시 분리 |
| CLI side-effect output | lineage 누락 | 모든 파일을 명시적 Output artifact로 선언 |
| 무리한 individual 병렬화 | GA/RNG 의미론 변화, 비용 증가 | 단일 search로 이관 후 equivalence test 기반 별도 최적화 |
| GPU/driver 차이 | 성능/결과 차이 | image/framework/GPU metadata, node pool 고정, production preflight |
| Static/dynamic Pareto 혼합 | 잘못된 우월성 주장 | 같은 split 비교만 정량 결론에 사용, 다른 모집단은 설명적 overlay |

## 19. 구현 전에 확정할 환경 의사결정

다음 값은 저장소만으로 결정할 수 없으며 플랫폼 운영자가 확정해야 한다.

1. KFP 배포 형태와 실제 Runtime patch version
2. Pipeline root의 SeaweedFS/S3/GCS URI와 retention/versioning 정책
3. GPU vendor resource name, node label, quota, A100 사용 여부
4. StorageClass와 ReadWriteMany 지원 여부
5. Namespace, service account, workload identity, final prefix ACL
6. Container registry와 image signing/scanning 정책
7. 최대 task wall time와 non-preemptible GPU 제공 여부
8. 최종 artifact 보존 기간과 experiment 삭제 시 데이터 처리 정책

이 값들은 알고리즘 parameter가 아니라 `dev/staging/prod`별 배포 overlay로 관리한다.

## 20. 추적성 메모

이 설계의 Kubeflow 부분은 공식 KFP 문서와 release를 조사해 작성했고, 프로젝트 부분은
저장소의 논문 노트, 재현 계획, runbook, 완료 결과와 실제 Python 구현을 대조했다. 독립 검토에서
특히 다음 사항을 재확인했다.

- Search를 단일 component로 먼저 옮겨야 세대/RNG/evaluation ID 의미론을 보존한다.
- Search에는 final artifact를 전달하지 않아야 하며 강한 격리는 service account/ACL도 필요하다.
- Final에서는 저장된 dispatcher state를 복원해야 하며 재학습 API를 사용하면 안 된다.
- 단일 최고 model이 아니라 Pareto frontier 자체가 기본 결과 artifact다.
- Route hit rate가 아니라 system Top-1과 average MFLOPs가 주 평가 축이다.

현재 완료된 동일 fingerprint 실행은 full equivalence의 참고 기준으로 search 2,500 evaluations,
search-retained state 243개, final sample 2,000개와 search 약 6시간 52분을 기록한다. 이 값은
asset/config/image identity가 같을 때만 golden regression으로 사용하며, 다른 GPU나 schema의
일반적인 pass/fail threshold로 확대하지 않는다.
