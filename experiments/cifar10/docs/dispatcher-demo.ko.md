# Kubernetes Dispatcher 이미지 업로드 시연

기존 GKE `pertinence-dispatcher-260919-1708`의 CIFAR-10 `dispatcher-000733`과 세 expert 모델을
사용한다. 추론 구성은 Hybrid이며 Router 내부에 ShuffleNet과 Dispatcher를 로딩한다.
ShuffleNet 선택 시 내부 logits를 재사용하고 MobileNet·ResNet56 선택 시 해당 외부 서비스만
binary로 한 번 호출한다. 브라우저와 업로드 API는 하나의 컨테이너이며, API가 내부 KServe V2 router를 호출한다.
실행 코드는 [demo 패키지](../../../src/pertinence/demo/), 배포 도구는
[cloud/demo](../../../cloud/demo/)에 있다. 업로드 이미지는 메모리에서만 처리한다.

## 나중에 다시 실행하기 (기존 Hybrid 이미지 재사용)

저장소 root에서 아래 명령을 실행한다. 이미 빌드한 이미지와 모델 자산을 재사용하므로 재빌드는
필요 없다. 종료된 session 경로는 재사용하지 않고 매번 새 경로를 만든다.

```bash
kubectl config use-context gke_pertinence-509116_asia-northeast3-a_pertinence-dispatcher-260919-1708

demo_session_path="artifacts/cifar10/runs/demo/session-$(date +%Y%m%d-%H%M%S)"
.venv/bin/python cloud/demo/manage.py start "$demo_session_path" \
  --image-file artifacts/cifar10/runs/demo/build-hybrid-20261018/image.txt

kubectl -n pertinence-serving port-forward --address 127.0.0.1 \
  service/pertinence-demo 8088:80
```

`http://localhost:8088`에 접속한다. 시작 도구가 serving node 1대를 올리고 Hybrid Router와
외부 expert 두 개, 웹/API를 생성한다. 모델 준비까지 수 분 걸릴 수 있다.

시연이 끝나면 port-forward를 `Ctrl+C`로 닫고 **같은 터미널**에서 종료한다.

```bash
.venv/bin/python cloud/demo/manage.py stop "$demo_session_path"
```

다른 터미널에서는 `$demo_session_path` 대신 시작할 때 사용한 실제 session 경로를 지정한다.
종료 시 해당 session 리소스를 삭제하고 serving node를 시작 전 상태로 복원한다.
이미지 digest·모델·검증 결과는 보존된다.

## 준비와 빌드

Repository root에서 실행한다. 로컬 Python 3.11을 권장하며 3.12도 지원한다.
gcloud 인증, 기존 클러스터 접근 권한과 Artifact Registry/Cloud Build 권한이 필요하다.
추가 IAM 권한을 자동으로 부여하거나 기존 클러스터를 재생성하지 않는다.

```bash
.venv/bin/python -m pip install -r cloud/demo/requirements.lock
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pytest tests/test_demo.py tests/test_demo_deployment.py
.venv/bin/ruff check .

.venv/bin/python cloud/demo/build.py artifacts/cifar10/runs/demo/my-build
```

`build.py`는 최소 소스만 Cloud Build에 올리고 `submitted.json`, `build.json`, `image.txt`를
기록한다. `image.txt`는 배포에 사용할 digest 고정 이미지다. 기본 base image는 기존 검증된
Python 3.11 slim digest이며 `--base-image`로 다른 고정 digest를 지정할 수 있다.
출력 디렉터리는 새 경로여야 한다. 빌드 실패 시 `submitted.json`의 build ID로 로그를 확인한다.

## 시작과 브라우저 접속

```bash
kubectl config use-context gke_pertinence-509116_asia-northeast3-a_pertinence-dispatcher-260919-1708

.venv/bin/python cloud/demo/manage.py start artifacts/cifar10/runs/demo/my-session \
  --image-file artifacts/cifar10/runs/demo/my-build/image.txt

kubectl -n pertinence-serving port-forward --address 127.0.0.1 \
  service/pertinence-demo 8088:80
```

브라우저에서 `http://localhost:8088`을 열어 JPEG/PNG 한 장을 업로드한다. 파일 상한은 5MiB,
해상도 상한은 2천만 픽셀이다. EXIF 방향 보정, RGB 변환, 32×32 bilinear resize 후 기존 CIFAR
정규화 함수를 적용한다. 예측 클래스, 상위 3개 softmax 점수, 선택 expert, 단계별 시간과 API 전체
시간을 표시한다. 모델은 CIFAR-10의 10개 클래스만 구분하며 softmax 점수는 정확도 지표가 아니다.

시작 도구는 기존 expert 이미지/hash와 sealed promotion을 대조하고 시작 전 상태를 `state.json`에
보존한다. `serving-pool`이 0대면 1대로 기동한다. MobileNet·ResNet56은 기존 모델 이미지를
재사용해 데모 전용 서비스로 생성한다(CPU 1·메모리 1Gi). binary 요청을 처리하고 HTTP/1.0으로
요청마다 연결을 닫으며 재시도하지 않는다. 기존 실험용 서비스의 설정과 정지 상태는 유지한다.
Router는 CPU 2·메모리 2Gi를 사용하며 별도 ShuffleNet 서비스는 생성하지 않는다. Hybrid modelcar는
`dispatcher-000733-hybrid-modelcar@sha256:2c133fb51aef75d5e01f49e403e6ae4fe7bd7fa94962b93503d67e89e499bf17`
로 고정한다. `/mnt/models/evaluated`와 `/mnt/models/experts/shufflenetv2_x0_5/model.onnx`를
기존 promotion·SHA-256 검증 절차로 로딩한다. sealed promotion 자체는 변경하지 않는다.
웹/API·두 expert·router·promotion ConfigMap은 데모 전용이며 기존 데모 객체가 있으면 중단한다.
모델 로딩 중에는 최대 10분씩 readiness를 기다린다.

API는 `POST /api/predict`의 multipart `file` 필드를 받는다. 응답에는 `request_id`, `model_id`,
`prediction`, `top_k`, `logits`, `route`, `expert_id`, `extractor_reused`, `serving_mode`, `timings`가 있다.
API는 router 응답의 `serving_mode=hybrid`를 검증한다.
`timings`의 단위는 ms이며 `total_ms`는 업로드 수신부터 결과 변환까지의 서버 처리시간이다.
동시 추론은 1건, 전체 deadline은 30초이며 재시도하지 않는다. `/health/live`는 API 자체 상태,
`/health/ready`는 Hybrid router와 두 외부 expert의 준비 상태를 확인한다.

입력 오류는 400, 파일·해상도 초과는 413, 형식 오류는 415, 동시 요청은 429,
추론 서비스 오류는 502, timeout은 504다. 오류 뒤 같은 화면에서 다시 실행할 수 있다.
공개 URL, LoadBalancer, Ingress는 생성하지 않는다. serving node 실행 중에는 컴퓨팅 비용이 발생한다.

## 실제 추론 일치 검증

웹 port-forward를 유지한 채 다른 터미널에서 router도 연결한다.

```bash
kubectl -n pertinence-serving port-forward --address 127.0.0.1 \
  service/demo-dispatcher-000733-predictor 18088:80
```

```bash
.venv/bin/python cloud/demo/verify.py \
  --trace artifacts/cifar10/runs/kserve/expert-once-20261017-1435/raw/trace/trace.npz \
  --output artifacts/cifar10/runs/demo/my-verification

cloud/kserve/scripts/audit-exposure.sh \
  gke_pertinence-509116_asia-northeast3-a_pertinence-dispatcher-260919-1708 \
  pertinence-509116
```

검증 도구는 Hybrid 응답 모드와 고정 trace의 처음 10장을 PNG로 복원하고, 업로드 전처리가 원래 tensor와 정확히
같은지 확인한다. 웹/API 경유와 직접 V2 호출의 logits·route가 일치하는지 검사하고 손상 이미지
오류 뒤 정상 재실행과 12초 유휴 뒤 첫 요청의 성공도 확인한다. 출력 PNG는 브라우저 시연에 사용할 수 있다.
`verification.json`은 검증 결과와 요청별 응답을 보존한다. `--baseline`에 distributed 검증 결과를
지정하면 같은 PNG·tensor인지 확인하고 route 일치와 logits의 `rtol=1e-5, atol=1e-6` 일치를 검사한다. 실제 환경에 해당 trace가 없으면
별도로 검증된 로컬 trace 경로를 지정한다. 벤치마크나 학습 Job은 실행하지 않는다.

## 실행 중인 distributed session 전환

새 웹 이미지를 먼저 빌드한 뒤 기존 session에 적용한다. 짧은 접속 중단이 발생한다.

```bash
.venv/bin/python cloud/demo/manage.py upgrade-hybrid \
  artifacts/cifar10/runs/demo/session-20261018-final \
  --image-file artifacts/cifar10/runs/demo/build-hybrid-20261018/image.txt \
  --trace artifacts/cifar10/runs/kserve/expert-once-20261017-1435/raw/trace/trace.npz
```

`upgrade-hybrid/`에 변경 전 state, 실제 객체·이미지 digest·소유 label, pool 상태와 복원용 manifest를
먼저 보존한다. 기존 distributed 구성으로 고정 이미지 10장의 기준을 수집하고, session 소유의
Router와 웹 Deployment만 교체한다. 임시 loopback port-forward로 새 Pod에 연결해 Hybrid 모드,
logits·route 기준 일치, 손상 이미지 후 복구, 12초 유휴 후 요청을 확인한다. 성능 개선은 보장하지 않는다.
검증 실패 시 기존 웹 이미지와 distributed Router spec을 복원하며 ShuffleNet 서비스를 유지한다.
복구 중에도 종료 명령은 기존 session의 전체 객체 목록을 사용한다.
검증 통과 후 Hybrid state를 기록하고 데모 전용 ShuffleNet 서비스를 삭제한 뒤 리소스 목록을 확정한다.
삭제 도중 중단되면 `verified-cleanup-pending` 상태와 전체 정리 목록이 남으며 같은 `stop` 명령으로
정리한다. 이전 session에 대한 재전환은 백업 충돌을 막기 위해 거부한다.

전환 후 기존 웹/router port-forward를 종료하고 위 접속 명령으로 다시 연결한다.

## 종료와 복구

port-forward 터미널에서 Ctrl+C로 연결을 닫은 후 시작할 때 사용한 state 경로로 종료한다.

```bash
.venv/bin/python cloud/demo/manage.py stop artifacts/cifar10/runs/demo/my-session
```

이번 session label을 가진 데모 객체만 삭제한다. 기존 expert의 설정과 stop annotation은 바꾸지 않는다.
처음 serving pool이 0대였다면 workload 종료를 확인하고 0대로 돌린다. autoscaling 설정은
변경하지 않는다. 기존 router, KFP, PVC, GCS artifact와 registry 이미지는 보존한다.
이미 종료된 session의 stop은 아무것도 변경하지 않는다.

시작 중 실패하거나 연결이 끊겨도 같은 stop 명령으로 복구한다. 시작 전 `state.json`이 먼저
기록되므로 중간 실패도 복구할 수 있다. serving node에 별도 workload가 남아 있으면 자동 복원을
중단한다. 모델 Pod 종료 대기 timeout이면 잠시 후
같은 stop 명령을 재실행한다. `state.json`과 `pool-after.json`으로 종료 상태를 확인한다.

## 2026-10-18 검증 실행

- 최종 Cloud Build: `ccece4c6-4842-49fd-8f2a-670cb5b20976` (`SUCCESS`).
- 웹 이미지: `asia-northeast3-docker.pkg.dev/pertinence-509116/pertinence/demo@sha256:fa11a3298766a9c64a05d046de48aa1de47293fb8ef828dd6a05f547ca595dfa`.
- 실행 state: `artifacts/cifar10/runs/demo/session-20261018-final/state.json`.
- 실제 추론: 고정 이미지 10장 logits·route 정확히 일치, 손상 이미지 오류 복구 및 12초 유휴 후
  첫 요청 성공. 증적은 `artifacts/cifar10/runs/demo/verification-20261018-isolated/`에 있다.
- Chromium: 실제 업로드, 선택 경로, API 오류 후 재실행, readiness 실패 표시, 모바일 너비 검증
  통과. 화면과 결과는 `artifacts/cifar10/runs/demo/browser/`에 있다.
- 신규 회귀 테스트 35개, Ruff, pip check, 외부 노출 감사 통과. 전체 회귀에서 기존
  `pipelines/pertinence/implementation-guide.ko.md`의 `...` 링크가 문서 검사를 실패시키는 것을
  확인했다. 해당 기존 파일은 이번 작업에서 변경하지 않았다.
- 첫 검증 session의 종료에서 기존 stop 상태·serving pool 0대·autoscaling 0..1 복원을 확인했다.
  최종 session은 시연용으로 serving node 1대와 데모 서비스를 실행 상태로 유지한다.

로컬 8080 포트가 사용 중이어서 이 실행은 `http://localhost:8088`을 사용한다. 연결이 종료되면
다음 명령으로 다시 연다.

```bash
kubectl -n pertinence-serving port-forward --address 127.0.0.1 \
  service/pertinence-demo 8088:80
```

시연이 끝나면 다음 명령으로 이번 session의 데모 서비스와 serving node를 정리한다.

```bash
.venv/bin/python cloud/demo/manage.py stop artifacts/cifar10/runs/demo/session-20261018-final
```

## 2026-10-18 Hybrid 전환 결과

- Cloud Build `6f91f116-9445-4d9a-95c6-8830be77dc79`: `SUCCESS`.
- 웹 이미지: `asia-northeast3-docker.pkg.dev/pertinence-509116/pertinence/demo@sha256:7e1089e353f9159463140049a261cf61bb1aab74be0f8066f2a1363c8e23e46b`.
- 기존 `session-20261018-final`에서 `upgrade-hybrid`를 실행했고 상태는 `complete`,
  `serving_mode=hybrid`다. `http://localhost:8088`로 새 Pod에 다시 연결했다.
- 변경 전 자산·객체·session·pool과 distributed 기준 결과는
  `artifacts/cifar10/runs/demo/session-20261018-final/upgrade-hybrid/`에 보존했다.
  `distributed/verification.json`, `hybrid/verification.json`, `final-audit.json`을 확인한다.
- 고정 이미지 10장 모두 distributed와 Hybrid의 logits가 정확히 일치했다(최대 절대 차이 0).
  route는 `[1, 1, 2, 2, 1, 2, 2, 2, 2, 2]`로 동일했다. 업로드와 직접 V2 호출도 정확히
  일치했고 손상 이미지 후 재실행, 12초 유휴 후 첫 요청이 성공했다. route 0은 로컬 runner와
  실제 HTTP expert 두 개를 사용하는 회귀 테스트에서 원격 호출 0회·내부 logits 재사용을 검증했다.
- Hybrid 관련 회귀 테스트 66개, Ruff, pip check와 외부 노출 감사가 통과했다. 전체 회귀는
  462개 통과·기존 문서 링크 오류 1개 실패였다(후속 추가 테스트 전 집계).
  실패 위치는 기존 `pipelines/pertinence/implementation-guide.ko.md`의 `...` 링크다.
- Chromium의 실제 업로드·외부 선택 경로·Router 내부 경계·오류 후 복구·readiness 실패 표시·
  모바일 너비 검증이 통과했다. 증적은 `artifacts/cifar10/runs/demo/browser-hybrid/`에 있다.
- 데모 전용 ShuffleNet 서비스는 검증 후 제거했다. 기존 실험용 expert·router의 spec과 stop
  annotation, sealed promotion은 동일하며 serving node 1대를 유지한다. Router와 외부 expert
  두 개가 실행 중이고 최종 `/api/predict` 응답의 `serving_mode`는 `hybrid`다.
- 종료는 위와 동일한 session의 `stop` 명령을 사용한다. 시작 전 pool이 0대였다는 기록과
  `resize_requested`를 보존했으므로 소유 리소스를 정리한 뒤 원래 pool 상태를 복원한다.

## 2026-10-18 시연 종료

사용자 요청으로 `session-20261018-final`을 종료했다. 데모 소유 리소스와 port-forward를
제거했고 serving pool 0대·serving node 없음·`state.json`의 `stopped=true`를 확인했다.
이미지와 모델, 백업·검증 결과는 보존했다. 재실행은 문서 맨 위의 간단 실행 절차를 사용한다.
