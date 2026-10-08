import json

from app.dashboard_tab import FINAL_SUFFIX, METRICS, build_summary


def test_summary_includes_final_results_and_excludes_legacy_results(tmp_path):
    final_path = tmp_path / f"none{FINAL_SUFFIX}.json"
    final_path.write_text(
        json.dumps(
            [
                {
                    "scores": {"response_relevancy": 0.5, "faithfulness": float("nan")},
                    "latency_seconds": 10,
                },
                {
                    "scores": {"response_relevancy": 0.7, "faithfulness": None},
                    "latency_seconds": 20,
                },
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "old_groq_run.json").write_text("[]", encoding="utf-8")

    summary = build_summary(tmp_path)

    assert list(summary.index) == ["none"]
    assert summary.loc["none", "n"] == 2
    assert summary.loc["none", "response_relevancy"] == 0.6
    assert summary.loc["none", "faithfulness"] is None
    assert summary.loc["none", "avg_latency_s"] == 15
    assert all(metric in summary.columns for metric in METRICS)


def test_summary_is_empty_without_final_result_files(tmp_path):
    summary = build_summary(tmp_path)

    assert summary.empty
