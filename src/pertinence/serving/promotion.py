"""Revalidate evaluated lineage before an explicit serving promotion is used."""

from __future__ import annotations

from pathlib import Path

from ..contracts.run import NormalizedRunContract
from ..pipeline.artifacts import read_sealed, safe_path, verify_artifact
from .contracts import PromotionManifestV1
from .runtime import load_dispatcher_npz


def validate_promotion_bundle(
    promotion: dict | PromotionManifestV1, evaluated_bundle: Path
) -> dict:
    """Validate the contract/search/final/model chain for exactly dispatcher-000733."""

    value = (
        promotion.to_dict()
        if isinstance(promotion, PromotionManifestV1)
        else PromotionManifestV1.from_dict(promotion).to_dict()
    )
    root = Path(evaluated_bundle)
    contract = NormalizedRunContract.from_dict(read_sealed(root / "contract.json")).to_dict()
    search = read_sealed(root / "search-result.json")
    final = read_sealed(root / "final-report.json")
    bundle = read_sealed(root / "manifest.json")
    if search.get("kind") != "search-result" or search.get("contract_fingerprint") != contract[
        "fingerprint"
    ]:
        raise ValueError("search result contract lineage mismatch")
    records = {item["evaluation_id"]: item for item in search.get("evaluations", [])}
    if len(records) != len(search.get("evaluations", [])):
        raise ValueError("search evaluation IDs are not unique")

    lineage = value["lineage"]
    if lineage != {
        "contract_fingerprint": contract["fingerprint"],
        "search_fingerprint": search["fingerprint"],
        "final_fingerprint": final["fingerprint"],
    }:
        raise ValueError("promotion lineage differs from evaluated artifacts")
    if value["evaluated_bundle"]["fingerprint"] != bundle["fingerprint"]:
        raise ValueError("promotion evaluated bundle fingerprint mismatch")
    if (
        bundle.get("kind") != "evaluated-dispatchers"
        or bundle.get("status") != "evaluated"
        or bundle.get("contract_fingerprint") != contract["fingerprint"]
        or bundle.get("search_fingerprint") != search["fingerprint"]
        or bundle.get("front_id") != search["front_id"]
        or bundle.get("final_report_fingerprint") != final["fingerprint"]
        or final.get("contract_fingerprint") != contract["fingerprint"]
        or final.get("search_fingerprint") != search["fingerprint"]
        or final.get("front_id") != search["front_id"]
    ):
        raise ValueError("evaluated bundle has invalid final lineage")
    if contract["dimensions"]["D"] != 1024 or contract["dimensions"]["K"] != 3:
        raise ValueError("promotion requires evaluated dimensions D=1024 and K=3")

    expert_order = contract["config"]["routing"]["expert_ids"]
    promoted_order = [item["id"] for item in value["experts"]]
    if promoted_order != expert_order:
        raise ValueError("promoted expert order differs from the evaluated contract")
    contract_experts = {item["id"]: item for item in contract["experts"]["experts"]}
    for promoted in value["experts"]:
        if contract_experts[promoted["id"]]["model_sha256"] != promoted["model_sha256"]:
            raise ValueError("promoted expert model digest differs from the evaluated contract")

    dispatcher_id = value["dispatcher"]["id"]
    models = [item for item in bundle["models"] if item["model_id"] == dispatcher_id]
    solutions = [item for item in final["solutions"] if item["model_id"] == dispatcher_id]
    if len(models) != 1 or len(solutions) != 1:
        raise ValueError("approved dispatcher is not uniquely present in the evaluated bundle")
    model = models[0]
    solution = solutions[0]
    expected_sha = value["dispatcher"]["checkpoint_sha256"]
    if model["checkpoint_sha256"] != expected_sha or solution["checkpoint_sha256"] != expected_sha:
        raise ValueError("approved dispatcher checkpoint digest mismatch")
    if model["dimensions"] != {"D": 1024, "K": 3} or model["expert_order"] != expert_order:
        raise ValueError("approved dispatcher dimensions or expert order mismatch")
    if model["evaluation_id"] not in records or model["evaluation_id"] not in search["pareto_ids"]:
        raise ValueError("approved dispatcher is not a retained search solution")
    if solution["search_metrics"] != records[model["evaluation_id"]]["metrics"]:
        raise ValueError("approved dispatcher search metrics differ from evaluated search")
    checkpoint = safe_path(root, model["checkpoint"])
    if checkpoint.suffix != ".npz":
        raise ValueError("approved dispatcher checkpoint must be NPZ")
    verify_artifact(checkpoint, expected_sha)
    load_dispatcher_npz(checkpoint)
    return {
        "promotion_fingerprint": value["fingerprint"],
        "bundle_fingerprint": bundle["fingerprint"],
        "dispatcher_id": dispatcher_id,
        "checkpoint": model["checkpoint"],
        "checkpoint_sha256": expected_sha,
        "dimensions": model["dimensions"],
        "expert_order": expert_order,
    }
