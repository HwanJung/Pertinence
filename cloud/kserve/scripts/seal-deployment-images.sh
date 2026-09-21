#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 EXPECTED_REPOSITORY BUILD_TAG OUTPUT_JSON" >&2
  exit 2
fi
repository=$1
tag=$2
output=$3
[[ "$repository" =~ ^[^[:space:]@:]+([/:][^[:space:]@:]+)+$ ]] || { echo "invalid repository" >&2; exit 2; }
[[ "$tag" =~ ^[A-Za-z0-9_.-]+$ && "$tag" != latest ]] || { echo "invalid build tag" >&2; exit 2; }
[[ ! -e "$output" ]] || { echo "output already exists" >&2; exit 2; }

images=(serving-runtime shufflenetv2_x0_5-modelcar mobilenetv2_x0_5-modelcar resnet56-modelcar dispatcher-000733-modelcar)
temporary=$(mktemp -d)
trap 'rm -rf -- "$temporary"' EXIT
for name in "${images[@]}"; do
  tagged="$repository/$name:$tag"
  docker pull --platform linux/amd64 "$tagged" >/dev/null
  digest=$(docker image inspect --format '{{index .RepoDigests 0}}' "$tagged")
  [[ "$digest" == "$repository/$name@sha256:"* ]] || { echo "unexpected repository digest: $digest" >&2; exit 1; }
  architecture=$(docker image inspect --format '{{.Architecture}}' "$tagged")
  user=$(docker image inspect --format '{{.Config.User}}' "$tagged")
  [[ "$architecture" == amd64 && "$user" == 65532:65532 ]] || {
    echo "$name must be linux/amd64 and USER 65532:65532" >&2
    exit 1
  }
  printf '%s\t%s\t%s\n' "$name" "$digest" "$user" >>"$temporary/images.tsv"
done

runtime="$repository/serving-runtime:$tag"
versions=$(docker run --rm --platform linux/amd64 --entrypoint python "$runtime" -c \
  'import json,platform,numpy,onnxruntime; print(json.dumps({"python":platform.python_version(),"numpy":numpy.__version__,"onnxruntime":onnxruntime.__version__},sort_keys=True))')
docker run --rm --platform linux/amd64 "$runtime" --help >/dev/null
python3 - "$temporary/images.tsv" "$versions" "$output" <<'PY'
import hashlib, json, os, pathlib, sys
rows = []
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    name, image, user = line.split("\t")
    rows.append({"name": name, "image": image, "architecture": "linux/amd64", "user": user})
versions = json.loads(sys.argv[2])
if not versions["python"].startswith("3.11.") or versions != {
    "python": versions["python"], "numpy": "1.26.4", "onnxruntime": "1.20.1"
}:
    raise SystemExit("runtime dependency versions differ from the deployment contract")
body = {"schema_version": 1, "images": rows, "runtime_versions": versions, "cli_smoke": True}
body["fingerprint"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
payload = (json.dumps(body, sort_keys=True, indent=2) + "\n").encode()
descriptor = os.open(
    pathlib.Path(sys.argv[3]),
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
    0o600,
)
with os.fdopen(descriptor, "wb") as stream:
    stream.write(payload)
    stream.flush()
    os.fsync(stream.fileno())
PY
