#!/usr/bin/env python3
"""Build the CIFAR-10 Pareto front for chenyaofo/pytorch-cifar-models.

The source model zoo reports Top-1/Top-5 accuracy, parameter count, and MAdds
for a single public CIFAR-10 checkpoint per model.  The primary compute axis in
this analysis is therefore the reported MAdds value.  A strict MFLOPs column is
also emitted using the project-wide convention ``1 MAC = 2 FLOPs``; multiplying
every x value by two does not change Pareto membership.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pertinence-matplotlib")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = EXPERIMENT_ROOT / "data" / "model_catalog.csv"
FIGURE_DIR = EXPERIMENT_ROOT / "figures"
SOURCE_URL = "https://github.com/chenyaofo/pytorch-cifar-models#model-zoo"
SNAPSHOT_DATE = "2026-08-24"


# model, Top-1 (%), Top-5 (%), parameters (M), MAdds (M)
# Values are transcribed from the repository's CIFAR-10 Model Zoo table.
RAW_ROWS = [
    ("resnet20", 92.60, 99.81, 0.27, 40.81),
    ("resnet32", 93.53, 99.77, 0.47, 69.12),
    ("resnet44", 94.01, 99.77, 0.66, 97.44),
    ("resnet56", 94.37, 99.83, 0.86, 125.75),
    ("vgg11_bn", 92.79, 99.72, 9.76, 153.29),
    ("vgg13_bn", 94.00, 99.77, 9.94, 228.79),
    ("vgg16_bn", 94.16, 99.71, 15.25, 313.73),
    ("vgg19_bn", 93.91, 99.64, 20.57, 398.66),
    ("mobilenetv2_x0_5", 92.88, 99.86, 0.70, 27.97),
    ("mobilenetv2_x0_75", 93.72, 99.79, 1.37, 59.31),
    ("mobilenetv2_x1_0", 93.79, 99.73, 2.24, 87.98),
    ("mobilenetv2_x1_4", 94.22, 99.80, 4.33, 170.07),
    ("shufflenetv2_x0_5", 90.13, 99.70, 0.35, 10.90),
    ("shufflenetv2_x1_0", 92.98, 99.73, 1.26, 45.00),
    ("shufflenetv2_x1_5", 93.55, 99.77, 2.49, 94.26),
    ("shufflenetv2_x2_0", 93.81, 99.79, 5.37, 187.81),
    ("repvgg_a0", 94.39, 99.82, 7.84, 489.08),
    ("repvgg_a1", 94.89, 99.83, 12.82, 851.33),
    ("repvgg_a2", 94.98, 99.82, 26.82, 1850.10),
]

EXPECTED_FRONT = {
    "shufflenetv2_x0_5",
    "mobilenetv2_x0_5",
    "shufflenetv2_x1_0",
    "mobilenetv2_x0_75",
    "mobilenetv2_x1_0",
    "resnet44",
    "resnet56",
    "repvgg_a0",
    "repvgg_a1",
    "repvgg_a2",
}


def family_of(model: str) -> str:
    if model.startswith("resnet"):
        return "ResNet"
    if model.startswith("vgg"):
        return "VGG"
    if model.startswith("mobilenet"):
        return "MobileNetV2"
    if model.startswith("shufflenet"):
        return "ShuffleNetV2"
    if model.startswith("repvgg"):
        return "RepVGG"
    raise ValueError(f"Unknown model family: {model}")


def is_pareto(row: dict[str, object], rows: list[dict[str, object]]) -> bool:
    """Return whether no model is cheaper and at least as accurate (or better)."""
    compute = float(row["madds_m"])
    accuracy = float(row["top1_accuracy_pct"])
    return not any(
        float(other["madds_m"]) <= compute
        and float(other["top1_accuracy_pct"]) >= accuracy
        and (
            float(other["madds_m"]) < compute
            or float(other["top1_accuracy_pct"]) > accuracy
        )
        for other in rows
    )


def build_rows() -> list[dict[str, object]]:
    rows = [
        {
            "model": model,
            "family": family_of(model),
            "top1_accuracy_pct": top1,
            "top5_accuracy_pct": top5,
            "params_m": params,
            "madds_m": madds,
            "mflops": round(2.0 * madds, 2),
            "pareto": False,
            "catalog": "chenyaofo/pytorch-cifar-models",
            "source_url": SOURCE_URL,
            "snapshot_date": SNAPSHOT_DATE,
        }
        for model, top1, top5, params, madds in RAW_ROWS
    ]
    for row in rows:
        row["pareto"] = is_pareto(row, rows)

    actual_front = {str(row["model"]) for row in rows if row["pareto"]}
    if actual_front != EXPECTED_FRONT:
        raise AssertionError(
            f"Unexpected Pareto front: expected {sorted(EXPECTED_FRONT)}, "
            f"got {sorted(actual_front)}"
        )
    return sorted(rows, key=lambda row: float(row["madds_m"]))


def write_csv(rows: list[dict[str, object]]) -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATA_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_figures(rows: list[dict[str, object]]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    front = [row for row in rows if row["pareto"]]
    dominated = [row for row in rows if not row["pareto"]]

    style = (
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "seaborn-whitegrid"
    )
    plt.style.use(style)
    fig, ax = plt.subplots(figsize=(13.5, 8), dpi=160)

    ax.scatter(
        [float(row["madds_m"]) for row in dominated],
        [float(row["top1_accuracy_pct"]) for row in dominated],
        s=62,
        color="#8A98A6",
        alpha=0.68,
        edgecolors="white",
        linewidths=0.7,
        label=f"Dominated ({len(dominated)})",
        zorder=2,
    )
    ax.plot(
        [float(row["madds_m"]) for row in front],
        [float(row["top1_accuracy_pct"]) for row in front],
        color="#E4572E",
        linewidth=2.2,
        alpha=0.82,
        zorder=3,
    )
    ax.scatter(
        [float(row["madds_m"]) for row in front],
        [float(row["top1_accuracy_pct"]) for row in front],
        s=86,
        color="#E4572E",
        edgecolors="white",
        linewidths=0.9,
        label=f"Pareto front ({len(front)})",
        zorder=4,
    )

    label_offsets = {
        "shufflenetv2_x0_5": (7, -3, "left", "top"),
        "mobilenetv2_x0_5": (7, 7, "left", "bottom"),
        "shufflenetv2_x1_0": (7, -8, "left", "top"),
        "mobilenetv2_x0_75": (7, 7, "left", "bottom"),
        "mobilenetv2_x1_0": (7, -8, "left", "top"),
        "resnet44": (7, 7, "left", "bottom"),
        "resnet56": (7, 7, "left", "bottom"),
        "repvgg_a0": (-7, 7, "right", "bottom"),
        "repvgg_a1": (-7, 7, "right", "bottom"),
        "repvgg_a2": (-7, -7, "right", "top"),
    }
    for row in front:
        dx, dy, horizontal, vertical = label_offsets[str(row["model"])]
        ax.annotate(
            str(row["model"]),
            (float(row["madds_m"]), float(row["top1_accuracy_pct"])),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.4,
            color="#4A2A20",
            ha=horizontal,
            va=vertical,
            zorder=5,
        )

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_xlabel("Compute per CIFAR-10 image (million multiply-adds, log scale)")
    ax.set_ylabel("Reported Top-1 accuracy (%)")
    ax.set_title("CIFAR-10 accuracy–compute Pareto front")
    ax.text(
        0.01,
        0.015,
        "Source: chenyaofo/pytorch-cifar-models · 19 public checkpoints · snapshot 2026-08-24",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#58636E",
    )
    ax.legend(loc="lower right", frameon=True)
    ax.margins(x=0.07, y=0.08)
    fig.tight_layout()

    for extension in ("png", "svg"):
        fig.savefig(
            FIGURE_DIR / f"model_pareto.{extension}",
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    rows = build_rows()
    write_csv(rows)
    write_figures(rows)
    front = [row for row in rows if row["pareto"]]
    print(f"Wrote {len(rows)} models to {DATA_PATH}")
    print(f"Pareto front ({len(front)}):")
    for row in front:
        print(
            f"  {row['model']}: {float(row['top1_accuracy_pct']):.2f}% Top-1, "
            f"{float(row['madds_m']):.2f} MAdds, {float(row['mflops']):.2f} MFLOPs"
        )


if __name__ == "__main__":
    main()
