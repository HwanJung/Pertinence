#!/usr/bin/env python3
"""Build the CIFAR-100 Pareto front for chenyaofo/pytorch-cifar-models.

The source model zoo reports Top-1/Top-5 accuracy, parameter count, and MAdds
for a single public CIFAR-100 checkpoint per model. The primary compute axis in
this analysis is therefore the reported MAdds value. An MFLOPs column is also
emitted using the project-wide convention ``1 MAC = 2 FLOPs``; multiplying
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
SOURCE_URL = "https://github.com/chenyaofo/pytorch-cifar-models#cifar-100"
SNAPSHOT_DATE = "2026-09-06"


# model, Top-1 (%), Top-5 (%), parameters (M), MAdds (M)
# Values are transcribed from the repository's CIFAR-100 Model Zoo table.
RAW_ROWS = [
    ("resnet20", 68.83, 91.01, 0.28, 40.82),
    ("resnet32", 70.16, 90.89, 0.47, 69.13),
    ("resnet44", 71.63, 91.58, 0.67, 97.44),
    ("resnet56", 72.63, 91.94, 0.86, 125.75),
    ("vgg11_bn", 70.78, 88.87, 9.80, 153.34),
    ("vgg13_bn", 74.63, 91.09, 9.99, 228.84),
    ("vgg16_bn", 74.00, 90.56, 15.30, 313.77),
    ("vgg19_bn", 73.87, 90.13, 20.61, 398.71),
    ("mobilenetv2_x0_5", 70.88, 91.72, 0.82, 28.08),
    ("mobilenetv2_x0_75", 73.61, 92.61, 1.48, 59.43),
    ("mobilenetv2_x1_0", 74.20, 92.82, 2.35, 88.09),
    ("mobilenetv2_x1_4", 75.98, 93.44, 4.50, 170.23),
    ("shufflenetv2_x0_5", 67.82, 89.93, 0.44, 10.99),
    ("shufflenetv2_x1_0", 72.39, 91.46, 1.36, 45.09),
    ("shufflenetv2_x1_5", 73.91, 92.13, 2.58, 94.35),
    ("shufflenetv2_x2_0", 75.35, 92.62, 5.55, 188.00),
    ("repvgg_a0", 75.22, 92.93, 7.96, 489.19),
    ("repvgg_a1", 76.12, 92.71, 12.94, 851.44),
    ("repvgg_a2", 77.18, 93.51, 26.94, 1850.22),
]

EXPECTED_FRONT = {
    "shufflenetv2_x0_5",
    "mobilenetv2_x0_5",
    "shufflenetv2_x1_0",
    "mobilenetv2_x0_75",
    "mobilenetv2_x1_0",
    "mobilenetv2_x1_4",
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
        "mobilenetv2_x0_75": (-7, -8, "right", "top"),
        "mobilenetv2_x1_0": (7, 8, "left", "bottom"),
        "mobilenetv2_x1_4": (7, 7, "left", "bottom"),
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
    ax.set_xlabel("Compute per CIFAR-100 image (million multiply-adds, log scale)")
    ax.set_ylabel("Reported Top-1 accuracy (%)")
    ax.set_title("CIFAR-100 accuracy–compute Pareto front")
    ax.text(
        0.01,
        0.015,
        "Source: chenyaofo/pytorch-cifar-models · 19 public checkpoints · snapshot 2026-09-06",
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
