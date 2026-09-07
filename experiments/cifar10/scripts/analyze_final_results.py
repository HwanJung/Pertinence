#!/usr/bin/env python3
"""Validate completed artifacts and build final CSV, JSON, and comparison figures."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = WORKSPACE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pertinence-matplotlib")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator  # noqa: E402

from pertinence.analysis import analyze_payloads  # noqa: E402
from pertinence.config import load_config  # noqa: E402
from pertinence.experts import EXPERTS_BY_NAME  # noqa: E402


def _atomic_json(value: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _load_static_front(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    front = [row for row in rows if row["pareto"].strip().lower() == "true"]
    if len(front) != 10:
        raise ValueError("static catalog CSV must contain exactly 10 Pareto rows")
    return front


def _write_csv(rows: list[dict[str, Any]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [key for key in rows[0] if key != "chromosome"] + ["chromosome_json"]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda item: (
                float(item["final_mflops"]),
                -float(item["final_accuracy_pct"]),
                int(item["search_evaluation_id"]),
            ),
        ):
            output = {key: value for key, value in row.items() if key != "chromosome"}
            output["chromosome_json"] = json.dumps(row["chromosome"], separators=(",", ":"))
            writer.writerow(output)


def _unique_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[float, float], dict[str, Any]] = {}
    for row in rows:
        if not row["final_pareto"]:
            continue
        key = (round(float(row["final_mflops"]), 9), round(float(row["final_accuracy_pct"]), 9))
        previous = unique.get(key)
        if previous is None or int(row["search_evaluation_id"]) < int(
            previous["search_evaluation_id"]
        ):
            unique[key] = row
    return sorted(unique.values(), key=lambda row: float(row["final_mflops"]))


def _plot(analysis: dict[str, Any], destination_stem: Path) -> None:
    summary = analysis["summary"]
    rows = analysis["rows"]
    front = _unique_front(rows)
    style = (
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "seaborn-whitegrid"
    )
    plt.style.use(style)
    plt.rcParams["svg.hashsalt"] = "pertinence-cifar10-final"
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4), dpi=160)

    for ax in axes:
        ax.scatter(
            [row["final_mflops"] for row in rows],
            [row["final_accuracy_pct"] for row in rows],
            s=16,
            color="#E6A57E",
            alpha=0.28,
            linewidths=0,
            label=f"Search-retained solutions ({len(rows)})",
            zorder=1,
        )
        ax.plot(
            [row["final_mflops"] for row in front],
            [row["final_accuracy_pct"] for row in front],
            color="#D1495B",
            linewidth=1.8,
            alpha=0.85,
            zorder=3,
        )
        ax.scatter(
            [row["final_mflops"] for row in front],
            [row["final_accuracy_pct"] for row in front],
            s=25,
            color="#D1495B",
            edgecolors="white",
            linewidths=0.4,
            label=f"Final non-dominated objectives ({len(front)})",
            zorder=4,
        )
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.set_xlabel("MFLOPs per image (log scale)")
        ax.set_ylabel("Top-1 accuracy (%)")

    same_split = summary["same_split_expert_baselines"]
    axes[0].scatter(
        [item["standalone_mflops"] for item in same_split],
        [item["final_accuracy_pct"] for item in same_split],
        marker="D",
        s=65,
        color="#2878B5",
        edgecolors="white",
        linewidths=0.7,
        label="Single experts (same final split)",
        zorder=5,
    )
    for item in same_split:
        axes[0].annotate(
            item["model"],
            (item["standalone_mflops"], item["final_accuracy_pct"]),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=7.5,
            color="#164A70",
        )
    oracle = summary["final_oracle"]
    axes[0].scatter(
        [oracle["average_mflops"]],
        [oracle["accuracy_pct"]],
        marker="*",
        s=180,
        color="#2A9D8F",
        edgecolors="white",
        linewidths=0.8,
        label="Cheapest-correct oracle",
        zorder=6,
    )
    axes[0].set_title("Untouched final split: dynamic system vs routing experts")
    axes[0].set_ylim(89.5, 99.2)
    axes[0].legend(loc="lower right", fontsize=8)

    catalog = summary["catalog_static_pareto"]
    axes[1].plot(
        [item["standalone_mflops"] for item in catalog],
        [item["catalog_accuracy_pct"] for item in catalog],
        color="#5B6770",
        linewidth=1.5,
        linestyle="--",
        alpha=0.8,
        zorder=2,
    )
    axes[1].scatter(
        [item["standalone_mflops"] for item in catalog],
        [item["catalog_accuracy_pct"] for item in catalog],
        marker="^",
        s=54,
        color="#5B6770",
        edgecolors="white",
        linewidths=0.6,
        label="Catalog static Pareto (reported full-test)",
        zorder=5,
    )
    above_catalog = [
        row for row in front if row["above_catalog_interpolated_front"]
    ]
    axes[1].scatter(
        [row["final_mflops"] for row in above_catalog],
        [row["final_accuracy_pct"] for row in above_catalog],
        marker="o",
        s=65,
        facecolors="none",
        edgecolors="#2A9D8F",
        linewidths=1.2,
        label=f"Above interpolated catalog boundary ({len(above_catalog)})",
        zorder=6,
    )
    axes[1].set_title("Descriptive overlay with 10-model catalog boundary")
    axes[1].set_ylim(89.5, 95.5)
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].text(
        0.02,
        0.98,
        "Accuracy sources differ; use the left panel for controlled comparisons.",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#5B6770",
    )

    fig.suptitle("PERTINENCE CIFAR-10 final accuracy–compute results", fontsize=14)
    fig.tight_layout()
    destination_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        destination_stem.with_suffix(".png"),
        bbox_inches="tight",
        metadata={"Software": "pertinence-reproduction"},
    )
    fig.savefig(
        destination_stem.with_suffix(".svg"),
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "pertinence-reproduction"},
    )
    plt.close(fig)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--search-result", type=Path, required=True)
    parser.add_argument("--final-result", type=Path, required=True)
    parser.add_argument("--static-csv", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--solutions-csv", type=Path, required=True)
    parser.add_argument("--figure-stem", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        config = load_config(arguments.config)
        search = json.loads(arguments.search_result.read_text(encoding="utf-8"))
        final = json.loads(arguments.final_result.read_text(encoding="utf-8"))
        static_front = _load_static_front(arguments.static_csv)
        analysis = analyze_payloads(
            search,
            final,
            candidate_names=config.candidates,
            candidate_mflops=[EXPERTS_BY_NAME[name].mflops for name in config.candidates],
            static_pareto_rows=static_front,
        )
        for row in analysis["rows"]:
            checkpoint = Path(row["dispatcher_checkpoint"])
            if not checkpoint.is_file():
                raise FileNotFoundError(f"dispatcher checkpoint is missing: {checkpoint}")
        _atomic_json(analysis["summary"], arguments.summary_output)
        _write_csv(analysis["rows"], arguments.solutions_csv)
        _plot(analysis, arguments.figure_stem)
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"final analysis failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(analysis["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
