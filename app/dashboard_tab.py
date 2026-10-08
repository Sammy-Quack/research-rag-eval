import json
import math
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
FINAL_SUFFIX = "__llama3_2_3b_8k"
METRICS = [
    "faithfulness",
    "context_precision",
    "context_recall",
    "response_relevancy",
]
METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "context_precision": "Context precision",
    "context_recall": "Context recall",
    "response_relevancy": "Response relevancy",
}


def _finite_scores(records: list[dict], metric: str) -> list[float]:
    values = []
    for record in records:
        value = record.get("scores", {}).get(metric)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            values.append(float(value))
    return values


def build_summary(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    rows = []
    for path in sorted(results_dir.glob(f"*{FINAL_SUFFIX}.json")):
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"{path.name} must contain a JSON list of evaluation records.")

        config = path.stem[: -len(FINAL_SUFFIX)]
        row = {"config": config, "n": len(records)}
        for metric in METRICS:
            scores = _finite_scores(records, metric)
            row[metric] = sum(scores) / len(scores) if scores else None

        latencies = [
            record["latency_seconds"]
            for record in records
            if isinstance(record, dict)
            and isinstance(record.get("latency_seconds"), (int, float))
            and math.isfinite(record["latency_seconds"])
        ]
        row["avg_latency_s"] = sum(latencies) / len(latencies) if latencies else None
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["n", *METRICS, "avg_latency_s"]).rename_axis("config")
    return pd.DataFrame(rows).set_index("config")


@st.cache_data
def load_summary() -> pd.DataFrame:
    return build_summary()


def _report_directories(results_dir: Path = RESULTS_DIR) -> list[Path]:
    return sorted(
        path
        for path in results_dir.glob(f"*{FINAL_SUFFIX}_analysis")
        if path.is_dir()
    )


def render_dashboard_tab() -> None:
    st.header("Ablation results")
    try:
        summary = load_summary()
    except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
        st.error(f"Could not load evaluation results: {exc}")
        return

    if summary.empty:
        st.info(f"No final evaluation results found in `{RESULTS_DIR}`.")
        return

    formatters = {metric: "{:.3f}" for metric in METRICS}
    formatters["avg_latency_s"] = "{:.1f}"
    st.dataframe(
        summary.style.format(formatters, na_rep="n/a"),
        width="stretch",
    )
    st.subheader("Quality metrics")
    st.bar_chart(summary[METRICS].rename(columns=METRIC_LABELS))
    st.caption(
        "Each final configuration uses the same 70 questions, generator "
        "`llama3.2-3b-8k`, and judge `qwen2.5-14b-8k`. The `none` baseline has "
        "no retrieved context, so only response relevancy applies. Latency was "
        "measured under background load; compare quality, not speed."
    )

    st.subheader("Per-configuration report")
    reports = _report_directories()
    if not reports:
        st.info(
            "Per-configuration reports are not available yet. Generate them "
            "with `eval/results/rag_eval_analyzer.py`."
        )
        return

    selected = st.selectbox(
        "Configuration",
        reports,
        format_func=lambda path: path.name.removesuffix("_analysis"),
    )
    report_path = selected / "REPORT.md"
    if report_path.is_file():
        st.markdown(report_path.read_text(encoding="utf-8"))

    images = sorted(selected.glob("*.png"))
    for start in range(0, len(images), 2):
        columns = st.columns(2)
        for column, image in zip(columns, images[start : start + 2]):
            column.image(str(image), caption=image.stem.replace("_", " "))
