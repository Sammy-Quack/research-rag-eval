#!/usr/bin/env python3
"""
RAG Evaluation Compressor + Visualizer

Usage:
    python rag_eval_analyzer.py results.json
    python rag_eval_analyzer.py results.json --out evaluation_report
    python rag_eval_analyzer.py results.json --threshold 0.80

Expected raw format:
[
  {
    "eval_id": "...",
    "question": "...",
    "answer": "...",
    "scores": {
      "faithfulness": 0.9,
      "context_precision": 0.8,
      "context_recall": 0.95,
      "response_relevancy": 0.9
    },
    "latency_seconds": 20.1,
    "token_usage": {
      "prompt_tokens": 3000,
      "completion_tokens": 100,
      "total_tokens": 3100
    }
  }
]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

import matplotlib.pyplot as plt


DEFAULT_METRICS = [
    "faithfulness",
    "context_precision",
    "context_recall",
    "response_relevancy",
]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def load_results(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        # Support common wrappers such as {"results": [...]}.
        for key in ("results", "evaluations", "data", "items"):
            if isinstance(data.get(key), list):
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError("Expected a JSON list of evaluation records.")

    return data


def safe_float(value: Any) -> float | None:
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def extract_metric_names(records: list[dict[str, Any]]) -> list[str]:
    names = set()
    for r in records:
        names.update((r.get("scores") or {}).keys())

    # Keep the important RAG metrics first, then any custom metrics.
    ordered = [m for m in DEFAULT_METRICS if m in names]
    ordered.extend(sorted(names - set(ordered)))
    return ordered


def build_compact(records: list[dict[str, Any]], source_name: str) -> dict[str, Any]:
    metrics = extract_metric_names(records)

    metric_values = {
        m: [
            x for x in (safe_float(r.get("scores", {}).get(m)) for r in records)
            if x is not None
        ]
        for m in metrics
    }

    metric_means = {
        m: round(mean(v), 6) if v else None
        for m, v in metric_values.items()
    }

    metric_ranges = {
        m: {
            "min": round(min(v), 6),
            "max": round(max(v), 6),
            "mean": round(mean(v), 6),
            "median": round(median(v), 6),
        }
        for m, v in metric_values.items()
        if v
    }

    latency = [
        x for x in (safe_float(r.get("latency_seconds")) for r in records)
        if x is not None
    ]

    prompt_tokens = []
    completion_tokens = []
    total_tokens = []

    compact_per_eval = []

    for r in records:
        usage = r.get("token_usage") or {}

        p = safe_float(usage.get("prompt_tokens"))
        c = safe_float(usage.get("completion_tokens"))
        t = safe_float(usage.get("total_tokens"))

        if p is not None:
            prompt_tokens.append(p)
        if c is not None:
            completion_tokens.append(c)
        if t is not None:
            total_tokens.append(t)

        scores = {}
        for m in metrics:
            value = safe_float((r.get("scores") or {}).get(m))
            if value is not None:
                scores[m] = round(value, 6)

        compact_per_eval.append({
            "id": str(r.get("eval_id", "")),
            "scores": scores,
            "latency_seconds": round(safe_float(r.get("latency_seconds")) or 0, 4),
            "total_tokens": int(t or 0),
        })

    strongest = None
    weakest = None
    valid_means = {k: v for k, v in metric_means.items() if v is not None}

    if valid_means:
        strongest = max(valid_means, key=valid_means.get)
        weakest = min(valid_means, key=valid_means.get)

    summary = {
        "metrics": valid_means,
        "performance": {
            "avg_latency_seconds": round(mean(latency), 4) if latency else None,
            "median_latency_seconds": round(median(latency), 4) if latency else None,
            "p95_latency_seconds": round(percentile(latency, 0.95), 4) if latency else None,
            "avg_prompt_tokens": round(mean(prompt_tokens), 2) if prompt_tokens else None,
            "avg_completion_tokens": round(mean(completion_tokens), 2) if completion_tokens else None,
            "avg_total_tokens": round(mean(total_tokens), 2) if total_tokens else None,
        },
        "diagnosis": {
            "strongest_metric": strongest,
            "weakest_metric": weakest,
            "metric_ranges": metric_ranges,
        },
    }

    return {
        "schema_version": "1.0",
        "source": {
            "file": source_name,
            "samples": len(records),
        },
        "summary": summary,
        "per_eval": compact_per_eval,
    }


def save_json(obj: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_metric_means(compact: dict[str, Any], out: Path) -> None:
    metrics = compact["summary"]["metrics"]
    names = list(metrics)
    values = [metrics[m] for m in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, values)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("RAG Evaluation — Mean Metric Scores")
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(out / "01_metric_means.png", dpi=180)
    plt.close(fig)


def plot_metric_distributions(compact: dict[str, Any], out: Path) -> None:
    per_eval = compact["per_eval"]
    metrics = compact["summary"]["metrics"]

    data = [
        [r["scores"][m] for r in per_eval if m in r["scores"]]
        for m in metrics
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data, tick_labels=list(metrics), showmeans=True)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("RAG Evaluation — Metric Distributions")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(out / "02_metric_distributions.png", dpi=180)
    plt.close(fig)


def plot_latency(compact: dict[str, Any], out: Path) -> None:
    rows = compact["per_eval"]
    x = range(1, len(rows) + 1)
    y = [r["latency_seconds"] for r in rows]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, y, marker="o", markersize=3)
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Latency (seconds)")
    ax.set_title("RAG Evaluation — Latency per Evaluation")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(out / "03_latency.png", dpi=180)
    plt.close(fig)


def plot_tokens(compact: dict[str, Any], out: Path) -> None:
    rows = compact["per_eval"]
    x = range(1, len(rows) + 1)
    y = [r["total_tokens"] for r in rows]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, y, marker="o", markersize=3)
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Total tokens")
    ax.set_title("RAG Evaluation — Token Usage per Evaluation")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(out / "04_token_usage.png", dpi=180)
    plt.close(fig)


def plot_latency_vs_tokens(compact: dict[str, Any], out: Path) -> None:
    rows = compact["per_eval"]
    x = [r["total_tokens"] for r in rows]
    y = [r["latency_seconds"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, alpha=0.7)
    ax.set_xlabel("Total tokens")
    ax.set_ylabel("Latency (seconds)")
    ax.set_title("RAG Evaluation — Latency vs Total Tokens")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(out / "05_latency_vs_tokens.png", dpi=180)
    plt.close(fig)


def write_markdown_report(compact: dict[str, Any], out: Path) -> None:
    s = compact["summary"]
    lines = [
        "# RAG Evaluation Report",
        "",
        f"**Samples:** {compact['source']['samples']}",
        f"**Source:** `{compact['source']['file']}`",
        "",
        "## Mean metrics",
        "",
        "| Metric | Mean | Min | Median | Max |",
        "|---|---:|---:|---:|---:|",
    ]

    for name, stats in s["diagnosis"]["metric_ranges"].items():
        lines.append(
            f"| {name} | {stats['mean']:.4f} | {stats['min']:.4f} | "
            f"{stats['median']:.4f} | {stats['max']:.4f} |"
        )

    p = s["performance"]
    lines += [
        "",
        "## Performance",
        "",
        f"- Average latency: **{p['avg_latency_seconds']} s**",
        f"- Median latency: **{p['median_latency_seconds']} s**",
        f"- P95 latency: **{p['p95_latency_seconds']} s**",
        f"- Average prompt tokens: **{p['avg_prompt_tokens']}**",
        f"- Average completion tokens: **{p['avg_completion_tokens']}**",
        f"- Average total tokens: **{p['avg_total_tokens']}**",
        "",
        "## Automatic diagnosis",
        "",
        f"- Strongest metric: **{s['diagnosis']['strongest_metric']}**",
        f"- Weakest metric: **{s['diagnosis']['weakest_metric']}**",
        "",
        "## Generated graphs",
        "",
        "1. `01_metric_means.png` — average quality scores",
        "2. `02_metric_distributions.png` — score spread and outliers",
        "3. `03_latency.png` — latency by evaluation",
        "4. `04_token_usage.png` — total tokens by evaluation",
        "5. `05_latency_vs_tokens.png` — efficiency relationship",
    ]

    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Raw RAG evaluation JSON")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: <input>_analysis)",
    )
    args = parser.parse_args()

    input_path = args.input
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    out = args.out or input_path.with_name(input_path.stem + "_analysis")
    out.mkdir(parents=True, exist_ok=True)

    records = load_results(input_path)
    compact = build_compact(records, input_path.name)

    save_json(compact, out / "compact_evaluation.json")
    write_markdown_report(compact, out)

    plot_metric_means(compact, out)
    plot_metric_distributions(compact, out)
    plot_latency(compact, out)
    plot_tokens(compact, out)
    plot_latency_vs_tokens(compact, out)

    print(f"Processed {len(records)} evaluations.")
    print(f"Output: {out.resolve()}")
    print(f"Compact JSON: {(out / 'compact_evaluation.json').resolve()}")
    print("Graphs: 01_metric_means.png ... 05_latency_vs_tokens.png")


if __name__ == "__main__":
    main()
