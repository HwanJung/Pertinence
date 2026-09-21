# Pinned controller installation

Chart는 version과 공식 release `.tgz`의 절대 경로 및 SHA-256을 함께 고정한다. OCI registry
가용성에 의존하지 않으며 운영자가 별도로 받은 release package bytes를 그대로 검증한다.

`render-install.sh`는 지정된 local archive의 SHA-256을 비교한 뒤 manifest를 render한다. install을
직접 수행하지 않으며 기존 release를 upgrade하지 않는다. 세 namespace도 명시적으로 render한다.
render 결과의 모든 Pod container/initContainer image가 digest-pinned인지 검사한다.

정본 lock example에는 KServe v0.20.0 GitHub release asset에 게시된 CRD/resources package
SHA-256과, 2026-09-20에 공식 OCI에서 직접 받은 cert-manager v1.20.4 package SHA-256
`aa916e8…cd03`이 들어 있다. cert-manager OCI manifest digest는
`sha256:86e65e22…e4f0`으로 별도 확인했다. Controller image digest의 all-zero sentinel은 실제
linux/amd64 digest를 입력하기 전까지 의도적으로 render를 막는다.

```bash
cloud/kserve/scripts/render-install.sh /secure/path/chart-lock.env /tmp/kserve-install-rendered
kubectl apply --server-side --dry-run=server \
  -f /tmp/kserve-install-rendered/cert-manager.yaml
kubectl apply --server-side --dry-run=server \
  -f /tmp/kserve-install-rendered/kserve-crd.yaml
kubectl apply --server-side --dry-run=server \
  -f /tmp/kserve-install-rendered/kserve-resources.yaml
```

적용 순서는 cert-manager, cert-manager webhook Ready, KServe CRD, KServe resources,
KServe controller Ready이다. KServe/cert-manager controller의 node selector가 `system-pool`인지,
KServe ConfigMap의 Standard/disableIngress/modelcar 값이 의도대로 render됐는지 diff로 확인한다.
KServe runtime-configs chart는 사용하지 않는다. 본 benchmark는 digest-pinned custom runtime만
사용한다.
