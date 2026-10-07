# RAG Evaluation Report

**Samples:** 70
**Source:** `sentence__hybrid__llama3_2_3b_8k.json`

## Mean metrics

| Metric | Mean | Min | Median | Max |
|---|---:|---:|---:|---:|
| faithfulness | 0.8703 | 0.3333 | 1.0000 | 1.0000 |
| context_precision | 0.8825 | 0.2000 | 1.0000 | 1.0000 |
| context_recall | 0.9857 | 0.0000 | 1.0000 | 1.0000 |
| response_relevancy | 0.7550 | 0.0000 | 0.9053 | 1.0000 |

## Performance

- Average latency: **22.4496 s**
- Median latency: **22.1325 s**
- P95 latency: **29.1938 s**
- Average prompt tokens: **3021.63**
- Average completion tokens: **148.94**
- Average total tokens: **3170.57**

## Automatic diagnosis

- Strongest metric: **context_recall**
- Weakest metric: **response_relevancy**

## Generated graphs

1. `01_metric_means.png` — average quality scores
2. `02_metric_distributions.png` — score spread and outliers
3. `03_latency.png` — latency by evaluation
4. `04_token_usage.png` — total tokens by evaluation
5. `05_latency_vs_tokens.png` — efficiency relationship