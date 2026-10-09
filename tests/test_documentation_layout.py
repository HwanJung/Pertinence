from __future__ import annotations

from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"!?\[[^]]*\]\(([^)]+)\)")


def test_documentation_has_one_current_structure() -> None:
    required = (
        "experiments/method.ko.md",
        "experiments/cifar10/docs/local-reproduction.ko.md",
        "experiments/cifar10/docs/local-4expert-results.ko.md",
        "cloud/docs/architecture.ko.md",
        "cloud/docs/runbook.ko.md",
        "cloud/docs/roadmap.ko.md",
        "cloud/docs/benchmark/Plan.md",
        "cloud/docs/benchmark/Progress.md",
        "cloud/docs/runs/2026-09-20-dispatcher-gke.ko.md",
        "cloud/kfp/kfp-launcher-configmap.yaml",
        "cloud/kfp/service-account.yaml",
    )
    forbidden = (
        "docs",
        "pipelines/pertinence/docs",
        "pipelines/pertinence/gke",
        "cloud/dispatcher-training-run-plan.ko.md",
        "cloud/dispatcher-training-run-report.ko.md",
        "cloud/gke-kfp-architecture-report.ko.md",
        "cloud/iac/IMPLEMENTATION_REPORT.ko.md",
        "experiments/cifar10/docs/reproduction-plan.ko.md",
        "experiments/cifar10/docs/results.ko.md",
        "experiments/cifar10/docs/runbook.ko.md",
    )

    for relative_path in required:
        assert (ROOT / relative_path).is_file(), relative_path
    for relative_path in forbidden:
        assert not (ROOT / relative_path).exists(), relative_path


def test_historical_local_results_are_run_labelled() -> None:
    required = (
        "experiments/cifar10/data/local-4expert-final-solutions.csv",
        "experiments/cifar10/figures/local-4expert-final-accuracy-mflops.png",
        "experiments/cifar10/figures/local-4expert-final-accuracy-mflops.svg",
    )
    ambiguous = (
        "experiments/cifar10/data/final_solutions.csv",
        "experiments/cifar10/figures/final_accuracy_mflops.png",
        "experiments/cifar10/figures/final_accuracy_mflops.svg",
    )

    for relative_path in required:
        assert (ROOT / relative_path).is_file(), relative_path
    for relative_path in ambiguous:
        assert not (ROOT / relative_path).exists(), relative_path


def test_local_markdown_links_resolve() -> None:
    for markdown in ROOT.rglob("*.md"):
        if any(part in {".git", ".venv", "artifacts"} for part in markdown.parts):
            continue
        for match in MARKDOWN_LINK.finditer(markdown.read_text(encoding="utf-8")):
            raw_target = match.group(1).strip().split(maxsplit=1)[0].strip("<>")
            if raw_target.startswith("#"):
                continue
            parsed = urlsplit(raw_target)
            if parsed.scheme or parsed.netloc:
                continue
            relative_path = unquote(parsed.path)
            if not relative_path:
                continue
            target = (markdown.parent / relative_path).resolve()
            assert target.is_relative_to(ROOT), f"link escapes repository: {markdown}: {raw_target}"
            assert target.exists(), f"broken link: {markdown}: {raw_target}"


def test_current_benchmark_docs_do_not_describe_superseded_scope() -> None:
    current = (
        "cloud/docs/benchmark/Plan.md",
        "cloud/docs/benchmark/Progress.md",
        "cloud/docs/kserve-benchmark-runbook.ko.md",
        "cloud/docs/kserve-benchmark-execution.ko.md",
        "experiments/cifar10/docs/kserve-benchmark-protocol.ko.md",
    )
    forbidden = ("single-target", "CIFAR-10.1", "five-arm", "P00", "P21")

    for relative_path in current:
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        for phrase in forbidden:
            assert phrase not in text, f"{relative_path}: stale phrase {phrase}"
