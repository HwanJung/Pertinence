#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 expert|dynamic-router BUNDLE_DIR REPOSITORY:BUILD_TAG BUSYBOX_IMAGE@sha256:DIGEST" >&2
  exit 2
}

[[ $# -eq 4 ]] || usage
kind=$1
context_dir=$2
tagged_image=$3
base_image=$4

case "$kind" in
  expert|dynamic-router) ;;
  *) usage ;;
esac
[[ "$tagged_image" =~ ^[^[:space:]@]+:[A-Za-z0-9_.-]+$ ]] || {
  echo "build target must be an explicit, non-latest tag" >&2
  exit 2
}
[[ "$tagged_image" != *:latest ]] || { echo "latest tag is forbidden" >&2; exit 2; }
[[ "$base_image" =~ @sha256:[0-9a-f]{64}$ ]] || {
  echo "base image must be digest pinned" >&2
  exit 2
}
[[ "$base_image" != *@sha256:0000000000000000000000000000000000000000000000000000000000000000 ]] || {
  echo "replace the all-zero base image digest" >&2
  exit 2
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
temporary=$(mktemp -d)
trap 'rm -rf -- "$temporary"' EXIT
python3 "$script_dir/prepare_bundle.py" "$kind" "$context_dir" "$temporary/context"
docker build \
  --platform linux/amd64 \
  --file "$script_dir/Dockerfile" \
  --build-arg "MODELCAR_BASE_IMAGE=$base_image" \
  --tag "$tagged_image" \
  "$temporary/context"
docker push "$tagged_image"

digest_ref=$(docker image inspect --format '{{join .RepoDigests "\n"}}' "$tagged_image" \
  | awk 'index($0, "@sha256:") {print; exit}')
[[ "$digest_ref" =~ @sha256:[0-9a-f]{64}$ ]] || {
  echo "push completed but no immutable RepoDigest was found" >&2
  exit 1
}
printf '%s\n' "$digest_ref"
