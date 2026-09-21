# OCI modelcar input layout

입력 bundle에는 builder가 허용하는 파일만 둔다. builder가 임시 Docker context의 `bundle/`을
만들고 `inventory.json`을 추가한다.

- expert: `model.onnx`, `model.sha256`
- dynamic router: `contract.json`, `search-result.json`, `final-report.json`, `manifest.json`과
  `dispatcher-000733`의 FP32 NPZ checkpoint 하나. builder는 전체 lineage, SHA, key, shape와 finite
  값을 검증한다.

modelcar에는 `promotion.json`을 넣지 않는다. 네 modelcar digest가 나온 뒤 promotion을 봉인하고
immutable ConfigMap으로 router에 mount한다. static router는 modelcar를 사용하지 않는다.
`build-modelcar.sh KIND BUNDLE_DIR ...`는 symlink와 allowlist 밖 파일을 거부하며 push 직후 registry가
반환한 digest reference를 출력한다.
