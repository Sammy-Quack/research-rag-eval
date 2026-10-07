# RAG Evaluation Toolkit

Converts a large RAG evaluation JSON into:

- `compact_evaluation.json` — model-friendly representation
- `REPORT.md` — human-readable summary
- `01_metric_means.png` — average metric scores
- `02_metric_distributions.png` — metric distributions/outliers
- `03_latency.png` — latency per evaluation
- `04_token_usage.png` — total token usage per evaluation
- `05_latency_vs_tokens.png` — latency/token relationship

## Install

```bash
pip install matplotlib
```

## Run

```bash
python rag_eval_analyzer.py your_results.json
```

Or:

```bash
python rag_eval_analyzer.py your_results.json --out my_analysis
```

## Find zero-valued metrics

Run the standalone scanner to check every JSON file directly in the project
root. It writes `zero_metric_results.csv` with the source filename, evaluation
ID, and metric name for each score equal to `0.0`:

```bash
python find_zero_metrics.py
```

You can scan another directory or choose a different CSV output path:

```bash
python find_zero_metrics.py --directory path\to\results --output zero_metrics.csv
```

The script expects a list of records containing `eval_id`, `scores`,
`latency_seconds`, and `token_usage`. It also supports common wrappers such
as `{"results": [...]}`.

## Why this structure?

Keep the original JSON for debugging. Feed `compact_evaluation.json` to the
general model. This removes the long question/answer text while retaining
the information needed to reason about evaluation quality, latency, and cost.
