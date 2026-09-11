"""Strict manifest parsing and immutable contract values."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any

import yaml


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject ambiguous duplicate mapping keys."""


def _mapping(loader: UniqueKeyLoader, node: yaml.MappingNode) -> dict:
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str) or key in result:
            raise ValueError("manifest keys must be unique strings")
        result[key] = loader.construct_object(value_node)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def parse_manifest(text: str) -> dict:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("manifest keys must be unique strings")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        # YAML 1.1 treats JSON's 9e-05 spelling as a string. Parse JSON first to
        # preserve exact numeric types in canonical fingerprints.
        value = json.loads(text, object_pairs_hook=unique_pairs, parse_constant=invalid_constant)
    except json.JSONDecodeError:
        value = yaml.load(text, Loader=UniqueKeyLoader)
    if not isinstance(value, dict):
        raise ValueError("manifest must be a mapping")
    return value


def fields(value: dict, required: set[str], optional: set[str] | None = None) -> None:
    if not isinstance(value, dict):
        raise ValueError("expected a mapping")
    missing = required - value.keys()
    unknown = value.keys() - required - (optional or set())
    if missing or unknown:
        raise ValueError(
            f"invalid manifest fields: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def version(value: dict) -> None:
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("unsupported schema_version; expected 1")


def integer(value: Any, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def number(value: Any, name: str, minimum: float = 0) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return float(value)


def name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"{label} must be a non-empty portable identifier")
    return value


def digest(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("SHA-256 must contain 64 lowercase hexadecimal characters")
    return value


@dataclass(frozen=True)
class Manifest:
    """Store canonical JSON so callers cannot mutate a validated contract."""

    _json: str

    @classmethod
    def from_dict(cls, value: dict):
        cls.validate(value)
        return cls(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))

    @classmethod
    def from_text(cls, text: str):
        return cls.from_dict(parse_manifest(text))

    @staticmethod
    def validate(value: dict) -> None:
        version(value)

    def to_dict(self) -> dict:
        return json.loads(self._json)
