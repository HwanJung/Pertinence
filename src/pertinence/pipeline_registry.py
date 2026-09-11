"""Transactional local Registry adapter for evaluated Search Pareto dispatchers."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Protocol

from .pipeline_artifacts import read_sealed, safe_path, seal, write_json
from .pipeline_search import validate_models, validate_search
from .pipeline_validation import load_contract
from .reproducibility import canonical_json_hash, sha256_file


class RegistryAdapter(Protocol):
    """Implement one atomic idempotent batch operation for a future remote backend."""

    def register_many(self, records: list[dict]) -> list[dict]: ...


class SQLiteRegistry:
    """Persist metadata and checkpoint bytes together under a UNIQUE idempotency key."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def register_many(self, records: list[dict]) -> list[dict]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS dispatcher_versions (
                    idempotency_key TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    version_id TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL CHECK (status IN ('candidate', 'evaluated')),
                    metadata_json TEXT NOT NULL,
                    checkpoint BLOB NOT NULL
                )
            """)
            registered = []
            for record in records:
                key = record["idempotency_key"]
                metadata = json.dumps(record["metadata"], sort_keys=True, allow_nan=False)
                checkpoint = record["checkpoint"]
                existing = connection.execute(
                    "SELECT metadata_json, checkpoint FROM dispatcher_versions WHERE idempotency_key=?",
                    (key,),
                ).fetchone()
                if existing is not None and existing != (metadata, checkpoint):
                    raise ValueError(
                        "Registry idempotency key already exists with different content"
                    )
                model_id = record["model_id"]
                version_id = f"version-{key}"
                connection.execute(
                    "INSERT OR IGNORE INTO dispatcher_versions VALUES (?, ?, ?, ?, ?, ?)",
                    (key, model_id, version_id, "evaluated", metadata, checkpoint),
                )
                registered.append(
                    {
                        "model_id": model_id,
                        "version_id": version_id,
                        "idempotency_key": key,
                        "status": "evaluated",
                    }
                )
            connection.commit()
            return registered
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()


def register_pareto_dispatchers(
    evaluated_bundle: Path,
    *,
    registration_report: Path,
    registry: RegistryAdapter,
) -> list[dict]:
    contract = load_contract(evaluated_bundle / "contract.json")
    bundle = read_sealed(evaluated_bundle / "manifest.json")
    search = read_sealed(evaluated_bundle / "search-result.json")
    report = read_sealed(evaluated_bundle / "final-report.json")
    records = validate_search(search, contract)
    validate_models(evaluated_bundle, bundle, search, contract)
    if (
        bundle.get("kind") != "evaluated-dispatchers"
        or bundle.get("status") != "evaluated"
        or bundle["final_report_fingerprint"] != report["fingerprint"]
        or report["contract_fingerprint"] != contract["fingerprint"]
        or report["search_fingerprint"] != search["fingerprint"]
        or report["front_id"] != search["front_id"]
        or [item["evaluation_id"] for item in report["solutions"]] != search["pareto_ids"]
    ):
        raise ValueError("evaluated bundle/report lineage or Pareto IDs mismatch")
    registrations = []
    for model, solution in zip(bundle["models"], report["solutions"]):
        if (
            model["model_id"] != solution["model_id"]
            or model["checkpoint_sha256"] != solution["checkpoint_sha256"]
            or solution["search_metrics"] != records[model["evaluation_id"]]["metrics"]
        ):
            raise ValueError("evaluated solution differs from checkpoint/search record")
        checkpoint = safe_path(evaluated_bundle, model["checkpoint"])
        key = canonical_json_hash(
            {
                "run_contract_fingerprint": contract["fingerprint"],
                "front_id": search["front_id"],
                "evaluation_id": model["evaluation_id"],
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        )
        registrations.append(
            {
                "idempotency_key": key,
                "model_id": f"dispatcher-{contract['fingerprint']}-{model['evaluation_id']}",
                "checkpoint": checkpoint.read_bytes(),
                "metadata": {
                    "contract": contract,
                    "front_id": search["front_id"],
                    "model": model,
                    "search": records[model["evaluation_id"]],
                    "evaluation": solution,
                    "final_cache_fingerprint": report["final_cache_fingerprint"],
                    "final_report_fingerprint": report["fingerprint"],
                },
            }
        )
    registered = registry.register_many(registrations)
    write_json(
        registration_report,
        seal(
            {
                "schema_version": 1,
                "kind": "registry-report",
                "contract_fingerprint": contract["fingerprint"],
                "versions": registered,
            }
        ),
    )
    return registered
