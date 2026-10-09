#!/usr/bin/env python3
"""Render allowlisted ${NAME} templates without executing an env file."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import re
from pathlib import Path

import yaml


VARIABLE_RE = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")
DIGEST_RE = re.compile(r"^[^\s@]+@sha256:([0-9a-f]{64})$")
GSA_RE = re.compile(r"^[a-z][a-z0-9-]{4,29}@[a-z][a-z0-9-]{4,28}\.iam\.gserviceaccount\.com$")


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{number}: expected NAME=value")
        key, value = line.split("=", 1)
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ValueError(f"{path}:{number}: invalid variable name")
        if key in values:
            raise ValueError(f"{path}:{number}: duplicate {key}")
        if not value or value != value.strip() or any(char.isspace() for char in value):
            raise ValueError(f"{path}:{number}: {key} must be one non-empty token")
        values[key] = value
    return values


def validate_inputs(values: dict[str, str], required: set[str]) -> None:
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError(f"missing variables: {', '.join(missing)}")
    extra = sorted(values.keys() - required)
    if extra:
        raise ValueError(f"unexpected variables: {', '.join(extra)}")

    for key in required:
        value = values[key]
        if key.endswith("_IMAGE") or key.endswith("_MODELCAR"):
            match = DIGEST_RE.fullmatch(value)
            if not match or match.group(1) == "0" * 64:
                raise ValueError(f"{key} must be a non-sentinel digest-pinned image")
        elif key == "PROMOTION_FINGERPRINT":
            if not re.fullmatch(r"[0-9a-f]{64}", value) or value == "0" * 64:
                raise ValueError(f"{key} must be a non-sentinel canonical fingerprint")
        elif key.endswith("_SHA256"):
            if not re.fullmatch(r"[0-9a-f]{64}", value) or value == "0" * 64:
                raise ValueError(f"{key} must be a non-sentinel SHA-256")
        elif key == "BENCHMARK_GSA" and not GSA_RE.fullmatch(value):
            raise ValueError("BENCHMARK_GSA is not a valid GSA address")
        elif key == "PROMOTION_JSON_BASE64":
            try:
                promotion = json.loads(base64.b64decode(value, validate=True))
            except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("PROMOTION_JSON_BASE64 must encode valid JSON") from error
            fingerprint = promotion.get("fingerprint") if isinstance(promotion, dict) else None
            body = {item: child for item, child in promotion.items() if item != "fingerprint"}
            encoded = json.dumps(
                body, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
            if fingerprint != hashlib.sha256(encoded).hexdigest() or fingerprint != values.get(
                "PROMOTION_FINGERPRINT"
            ):
                raise ValueError("promotion ConfigMap content or fingerprint is invalid")
            expected_images = {
                "shufflenetv2_x0_5": values.get("SHUFFLENET_MODELCAR"),
                "mobilenetv2_x0_5": values.get("MOBILENET_MODELCAR"),
                "resnet56": values.get("RESNET56_MODELCAR"),
            }
            promoted_experts = {
                item.get("id"): item.get("modelcar_image")
                for item in promotion.get("experts", [])
                if isinstance(item, dict)
            }
            if (
                promotion.get("runtime_image") != values.get("RUNTIME_IMAGE")
                or promotion.get("dispatcher", {}).get("modelcar_image")
                != values.get("DISPATCHER_MODELCAR")
                or promoted_experts != expected_images
            ):
                raise ValueError("promotion image digests differ from deployment inputs")
        elif key == "BENCHMARK_CONFIG_BASE64":
            try:
                config = yaml.safe_load(base64.b64decode(value, validate=True))
            except (binascii.Error, UnicodeDecodeError, yaml.YAMLError) as error:
                raise ValueError("BENCHMARK_CONFIG_BASE64 must encode valid YAML") from error
            expected_arms = ["resnet56", "dispatcher-000733"]
            if not isinstance(config, dict) or config.get("arms") != expected_arms:
                raise ValueError("benchmark config arms/order is not approved")
        elif key == "BENCHMARK_ID":
            if not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]{0,61}[a-z0-9])?", value):
                raise ValueError(f"{key} must be a DNS-safe identifier")
        elif key == "BENCHMARK_TRACE_URI":
            if not re.fullmatch(
                r"gs://pertinence-509116-pertinence-runs/pertinence/benchmarks/traces/[a-z0-9.-]+",
                value,
            ):
                raise ValueError("BENCHMARK_TRACE_URI uses an unexpected prefix")
        elif key == "BENCHMARK_OUTPUT_URI":
            if not re.fullmatch(
                r"gs://pertinence-509116-pertinence-runs/pertinence/benchmarks/runs/[a-z0-9.-]+",
                value,
            ):
                raise ValueError("BENCHMARK_OUTPUT_URI uses an unexpected prefix")


def render(env_path: Path, output_dir: Path, mappings: list[tuple[Path, str]]) -> None:
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")

    sources = [(path, path.read_text(encoding="utf-8")) for path, _ in mappings]
    required = {name for _, text in sources for name in VARIABLE_RE.findall(text)}
    values = read_env(env_path)
    validate_inputs(values, required)

    rendered: list[tuple[str, str]] = []
    for (path, text), (_, output_name) in zip(sources, mappings, strict=True):
        result = VARIABLE_RE.sub(lambda match: values[match.group(1)], text)
        if VARIABLE_RE.search(result):
            raise ValueError(f"unresolved variable in {path}")
        documents = list(yaml.safe_load_all(result))
        if not documents or all(document is None for document in documents):
            raise ValueError(f"no YAML documents in {path}")
        if any(isinstance(document, dict) and document.get("kind") == "NetworkPolicy"
               for document in documents):
            raise ValueError("NetworkPolicy is forbidden while cluster enforcement is disabled")
        rendered.append((output_name, result))

    output_dir.mkdir(mode=0o700, parents=False)
    for name, result in rendered:
        destination = output_dir / name
        destination.write_text(result, encoding="utf-8")
        destination.chmod(0o600)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "mapping",
        nargs="+",
        help="TEMPLATE=OUTPUT_NAME (OUTPUT_NAME must not contain a slash)",
    )
    args = parser.parse_args()
    mappings: list[tuple[Path, str]] = []
    for mapping in args.mapping:
        source, separator, name = mapping.partition("=")
        if not separator or not name or "/" in name or name in {".", ".."}:
            parser.error(f"invalid mapping: {mapping}")
        mappings.append((Path(source).resolve(strict=True), name))
    try:
        render(args.env.resolve(strict=True), args.output_dir.resolve(), mappings)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
