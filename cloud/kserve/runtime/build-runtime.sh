#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 REPOSITORY:BUILD_TAG PYTHON_3_11_IMAGE@sha256:DIGEST" >&2
  exit 2
fi
tagged_image=$1
base_image=$2
[[ "$tagged_image" =~ ^[^[:space:]@]+:[A-Za-z0-9_.-]+$ && "$tagged_image" != *:latest ]] || {
  echo "target must be an explicit, non-latest tag" >&2
  exit 2
}
[[ "$base_image" =~ ^python:3\.11[^@]*@sha256:[0-9a-f]{64}$ ]] || {
  echo "base must be a digest-pinned Python 3.11 image" >&2
  exit 2
}
[[ "$base_image" != *@sha256:0000000000000000000000000000000000000000000000000000000000000000 ]] || {
  echo "replace the all-zero base image digest" >&2
  exit 2
}

repository_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)
docker build \
  --platform linux/amd64 \
  --file "$repository_root/cloud/kserve/runtime/Dockerfile" \
  --build-arg "PYTHON_BASE_IMAGE=$base_image" \
  --tag "$tagged_image" \
  "$repository_root"
docker push "$tagged_image"
digest_ref=$(docker image inspect --format '{{join .RepoDigests "\n"}}' "$tagged_image" \
  | awk 'index($0, "@sha256:") {print; exit}')
[[ "$digest_ref" =~ @sha256:[0-9a-f]{64}$ ]] || {
  echo "push completed but no immutable RepoDigest was found" >&2
  exit 1
}
printf '%s\n' "$digest_ref"
