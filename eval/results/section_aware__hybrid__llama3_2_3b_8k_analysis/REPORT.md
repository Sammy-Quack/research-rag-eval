# RAG Evaluation Report

**Samples:** 70
**Source:** `section_aware__hybrid__llama3_2_3b_8k.json`

## Mean metrics

| Metric | Mean | Min | Median | Max |
|---|---:|---:|---:|---:|
| faithfulness | 0.8584 | 0.0000 | 1.0000 | 1.0000 |
| context_precision | 0.8798 | 0.0000 | 1.0000 | 1.0000 |
| context_recall | 0.9857 | 0.0000 | 1.0000 | 1.0000 |
| response_relevancy | 0.7824 | 0.0000 | 0.9069 | 1.0000 |

## Performance

- Average latency: **26.415 s**
- Median latency: **24.75 s**
- P95 latency: **36.8132 s**
- Average prompt tokens: **2978.91**
- Average completion tokens: **140.63**
- Average total tokens: **3119.54**

## Automatic diagnosis

- Strongest metric: **context_recall**
- Weakest metric: **response_relevancy**

## Generated graphs

1. `01_metric_means.png` — average quality scores
2. `02_metric_distributions.png` — score spread and outliers
3. `03_latency.png` — latency by evaluation
4. `04_token_usage.png` — total tokens by evaluation
5. `05_latency_vs_tokens.png` — efficiency relationship