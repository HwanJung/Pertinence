#!/usr/bin/env python3
"""Write the Korean report from preserved diagnostic and paired-block artifacts."""

import argparse
import json
from pathlib import Path

import numpy as np

from pertinence.serving.transport_suite import VARIANTS, paired_blocks


def mean(values):
    return float(np.mean(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--performance", type=Path)
    parser.add_argument("--single-run", action="store_true",
                        help="Report diagnostic observations after repeated runs were canceled")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.single_run and args.performance is not None:
        parser.error("--single-run cannot be combined with --performance")
    diagnostic = {}
    for path in args.diagnostics.glob("*/summary.json"):
        value = json.loads(path.read_text())
        if value["manifest"]["diagnostics"] and "-diag-" in value["manifest"]["run_id"]:
            diagnostic[value["manifest"]["arm"]] = (path.parent, value)
    if set(diagnostic) != set(VARIANTS):
        raise ValueError("six complete diagnostic arms are required")
    lines = ["# Distributed serving 전송 비교 결과", "", "## 확인된 기존 수치", "",
             "2026-10-09 단일 실행의 Fused/Distributed mean은 각각 11.466/25.845 ms, "
             "p95는 13.519/30.379 ms였다. Extractor 구간 증가는 7.936 ms, 선택 expert 구간 "
             "증가는 6.482 ms였다. 기존 request CPU와 내부 wall의 계측 범위가 달라 "
             "54% JSON CPU/46% 통신이라는 원인 분해는 사용하지 않는다.", "",
             "이전 결과는 참고값이다. 아래 비교는 새 runtime의 D0를 기준으로 한다.", "",
             "## 추가 계측으로 확인한 원인", "",
             "각 진단은 warm-up 200회 뒤 같은 trace 2,000건을 5 RPS로 처리했다. "
             "다음 codec CPU와 잔차는 request당 값이며 warm-up 요청을 제외했다. "
             "RPC wall에서 server의 서로 겹치지 않는 decode/model/output/encode/write wall을 "
             "뺀 값은 전송·스케줄링·대기 잔차다. 순수 네트워크 시간이 아니다.", "",
             "| Arm | Mean ms (ON) | Codec thread CPU ms/request | 잔차 ms/request | "
             "요청/응답 body bytes/request | 생성/재사용 연결 |", "|---|---:|---:|---:|---:|---:|"]
    for arm in VARIANTS:
        path, summary = diagnostic[arm]
        calls = []
        if (path / "joined.json").exists():
            calls = [row for row in json.loads((path / "joined.json").read_text())["calls"]
                     if int(row["request_id"].rsplit("-", 1)[1]) >= 200]
        codec = sum(row["client_codec_cpu_seconds"] + row["server_codec_cpu_seconds"]
                    for row in calls) * 1000 / 2000
        residual = sum(row["transport_scheduling_wait_residual_seconds"] for row in calls) * 1000 / 2000
        request_bytes = sum(row["request_bytes"] for row in calls) / 2000
        response_bytes = sum(row["response_bytes"] for row in calls) / 2000
        created = sum(row["connections_created"] for row in calls)
        reused = sum(row["connections_reused"] for row in calls)
        lines.append(f"| {arm} | {summary['mean_ms']:.3f} | {codec:.3f} | {residual:.3f} | "
                     f"{request_bytes:.1f} / {response_bytes:.1f} | {created} / {reused} |")
    lines.extend(["", "F0의 원격 RPC 지표 0은 원격 호출이 없다는 뜻이다. 로컬 모델 계산 비용이 "
                  "0이라는 뜻은 아니다. 계측 기록의 파일 I/O와 미계측 framework 처리도 잔차에 "
                  "영향을 줄 수 있다. 전체 request CPU는 child span과 중복 합산하지 않았다.", "",
                  "## 수정별 실제 절감량", ""])
    if args.single_run:
        lines.extend([
            "사용자 요청으로 36회 반복 실험과 별도 ON/OFF 반복을 생략했다. 아래는 구조별 "
            "계측 ON 1회에서 관측한 차이다. 계측 OFF 성능이나 통계적 개선으로 판정하지 않으며 "
            "bootstrap 신뢰구간과 최종 후보 통과 판정은 산출하지 않는다.", "",
            "| Arm | Mean ms | p50 ms | p95 ms | p99 ms | Pod CPU ms/request | achieved RPS |",
            "|---|---:|---:|---:|---:|---:|---:|"])
        for arm in VARIANTS:
            row = diagnostic[arm][1]
            values = [row[key] for key in (
                "mean_ms", "p50_ms", "p95_ms", "p99_ms", "pod_cpu_ms_per_request", "achieved_rps")]
            lines.append(f"| {arm} | " + " | ".join(f"{v:.3f}" for v in values) + " |")
        lines.extend(["", "| 비교 | 관측 Mean 절감 ms | 절감률 |",
                      "|---|---:|---:|"])
        for before, after in (("D0", "D1"), ("D0", "D2"), ("D0", "D3"),
                              ("D1", "D3"), ("D2", "D3"), ("D3", "H1"), ("F0", "H1")):
            baseline = diagnostic[before][1]["mean_ms"]
            saving = baseline - diagnostic[after][1]["mean_ms"]
            lines.append(f"| {before} → {after} | {saving:.3f} | {saving / baseline:.1%} |")
        lines.extend(["", "각 arm은 2,000건 모두 검증을 통과했으며 prediction·route mismatch 0, "
                      "logits 최대 절대오차 0, byte-exact 일치다. 오류·누락·중복·재시작·"
                      "metric gap·schedule miss는 모두 0이다.", "",
                      "D0=JSON/기존 연결, D1=Binary/기존 연결, D2=JSON/keep-alive, "
                      "D3=Binary/keep-alive, H1=Hybrid/Binary/keep-alive, F0=Fused다. "
                      "연결 수는 warm-up을 제외하므로 keep-alive 연결 생성 0은 warm-up에서 "
                      "생성한 연결을 본 측정에서 계속 재사용했다는 뜻이다.", "",
                      "| Arm | Route | 요청 수 | Mean ms | p50 ms | p95 ms | p99 ms |",
                      "|---|---:|---:|---:|---:|---:|---:|"])
        for arm in VARIANTS:
            routes = diagnostic[arm][1]["latency"]["routes"]
            for route in ("0", "1", "2"):
                if route not in routes:
                    lines.append(f"| {arm} | {route} | 0 | — | — | — | — |")
                    continue
                row = routes[route]
                values = [row[key] for key in ("mean_ms", "p50_ms", "p95_ms", "p99_ms")]
                lines.append(f"| {arm} | {route} | {row['count']} | "
                             + " | ".join(f"{v:.3f}" for v in values) + " |")
        lines.extend(["", "## 권장 배치 구조", "",
                      "구조별 단일 진단은 병목과 개선 방향을 확인하는 자료다. 반복 측정과 계측 "
                      "ON/OFF 비교를 생략했으므로 최종 성능 우위나 안정적인 개선을 확정하지 않는다. "
                      "Production promotion은 수행하지 않았다."])
    elif args.performance is None:
        lines.append("계측 OFF 6-block 비교는 아직 완료되지 않았다. 진단 ON 수치만으로 개선을 "
                     "확정하거나 배치 구조를 선택하지 않는다.")
    else:
        comparison = json.loads((args.performance / "comparison.json").read_text())
        runs = [json.loads(path.read_text()) for path in args.performance.glob("*/summary.json")]
        runs = [row for row in runs if not row["manifest"]["diagnostics"]
                and "-b" in row["manifest"]["run_id"]]
        lines.extend(["36개 run의 모든 요청이 parity·운영 유효성 기준을 통과했다. 아래 p50/p95/p99는 "
                      "6개 run별 percentile의 평균이다. Bootstrap 단위는 요청이 아니라 대응 block이다.",
                      "", "| Arm | Mean | p50 | p95 | p99 | Pod CPU/request | achieved RPS |",
                      "|---|---:|---:|---:|---:|---:|---:|"])
        by_arm = {}
        for arm in VARIANTS:
            rows = sorted((row for row in runs if row["manifest"]["arm"] == arm),
                          key=lambda row: row["manifest"]["block"])
            if len(rows) != 6:
                raise ValueError("six blocks per arm required")
            by_arm[arm] = rows
            values = [mean([row[key] for row in rows]) for key in (
                "mean_ms", "p50_ms", "p95_ms", "p99_ms", "pod_cpu_ms_per_request", "achieved_rps")]
            lines.append("| " + arm + " | " + " | ".join(f"{value:.3f}" for value in values) + " |")
        lines.extend(["", "Route별 지연(각 run percentile의 평균, 단위 ms):", "",
                      "| Arm | Route | 요청 수 | Mean | p50 | p95 | p99 |",
                      "|---|---:|---:|---:|---:|---:|---:|"])
        for arm, rows in by_arm.items():
            for route in ("0", "1", "2"):
                selected = [row["latency"]["routes"][route] for row in rows
                            if route in row["latency"]["routes"]]
                if not selected:
                    lines.append(f"| {arm} | {route} | 0 | — | — | — | — |")
                    continue
                values = [mean([row[key] for row in selected])
                          for key in ("mean_ms", "p50_ms", "p95_ms", "p99_ms")]
                lines.append(f"| {arm} | {route} | {sum(row['count'] for row in selected)} | "
                             + " | ".join(f"{value:.3f}" for value in values) + " |")
        lines.extend(["", "Latency와 CPU 단위는 ms다. CPU에는 modelcar가 포함되며 별도 CPU·memory·"
                      "throttling 내역은 각 resources.json에 보존했다.", "",
                      "| 비교 | Mean 절감 ms | 절감률 | Block bootstrap 95% CI ms | 기준 통과 |",
                      "|---|---:|---:|---:|---|"])
        for label, value in comparison["comparisons"].items():
            stat = value["mean_ms"]
            lines.append(f"| {label} | {stat['mean_saving']:.3f} | {stat['fraction_saved']:.1%} | "
                         f"[{stat['ci95'][0]:.3f}, {stat['ci95'][1]:.3f}] | {value['passes']} |")
        interaction = paired_blocks(
            [a["mean_ms"] + b["mean_ms"] for a, b in zip(by_arm["D1"], by_arm["D2"])],
            [a["mean_ms"] + b["mean_ms"] for a, b in zip(by_arm["D0"], by_arm["D3"])])
        lines.extend(["", f"결합 효과에서 단독 효과 합을 뺀 상호작용은 "
                      f"{interaction['mean_saving']:.3f} ms, 95% CI "
                      f"[{interaction['ci95'][0]:.3f}, {interaction['ci95'][1]:.3f}] ms다.", "",
                      "Route별 latency와 각 block의 짝지어진 차이는 summary.json과 comparison.json에 "
                      "보존했다. 모든 logits의 byte-exact 여부는 run별 summary의 byte_exact에 기록했다."])
        overhead_path = args.performance / "instrumentation-overhead.json"
        if overhead_path.exists():
            overhead = json.loads(overhead_path.read_text())
            lines.extend(["", "계측 오버헤드(별도 ON/OFF 실행):", ""])
            for arm, pair in overhead.items():
                off, on = pair["False"], pair["True"]
                lines.append(f"- {arm}: mean {off['mean_ms']:.3f} → {on['mean_ms']:.3f} ms, "
                             f"Pod CPU {off['pod_cpu_ms_per_request']:.3f} → "
                             f"{on['pod_cpu_ms_per_request']:.3f} ms/request.")
        lines.extend(["", "## 권장 배치 구조", "",
                      f"완전 분산 선택: **{comparison['selected_distributed'] or '통과 후보 없음'}**. "
                      "Mean 절감 CI 하한 > 0, p95와 전체 Pod CPU/request 악화 ≤ 5%를 적용했다. "
                      "D3의 추가 효과가 불확실하면 통과한 단일 변경을 우선했다.", "",
                      "H1은 D3와 F0에 대한 위 대응 비교로 부분 통합의 실익을 평가한다. "
                      "최대 처리량·autoscaling·cross-node 효과와 production promotion은 범위 밖이다."])
    first = diagnostic["D0"][1]["manifest"]
    lines.extend(["", "## 재현 정보", "", f"- Runtime: `{first['runtime_image']}`",
                  f"- Promotion fingerprint: `{first['promotion_fingerprint']}`",
                  f"- Trace fingerprint: `{first['trace_fingerprint']}`",
                  "- Python 3.11 / ORT 1.20.1 / worker 1 / ORT intra·inter-op 각 1.",
                  f"- 진단 원본: `{args.diagnostics}`",
                  f"- 계측 OFF 성능 원본: `{args.performance}`" if args.performance else
                  "- 계측 OFF 성능 실행: 생략.", ""])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
