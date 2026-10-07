# RAG Evaluation Report

**Samples:** 70
**Source:** `fixed_size__hybrid__llama3_2_3b_8k.json`

## Mean metrics

| Metric | Mean | Min | Median | Max |
|---|---:|---:|---:|---:|
| faithfulness | 0.8625 | 0.0000 | 1.0000 | 1.0000 |
| context_precision | 0.8717 | 0.0000 | 1.0000 | 1.0000 |
| context_recall | 0.9714 | 0.0000 | 1.0000 | 1.0000 |
| response_relevancy | 0.7511 | 0.0000 | 0.9141 | 1.0000 |

## Performance

- Average latency: **27.5951 s**
- Median latency: **25.711 s**
- P95 latency: **40.6467 s**
- Average prompt tokens: **3164.96**
- Average completion tokens: **145.09**
- Average total tokens: **3310.04**

## Automatic diagnosis

- Strongest metric: **context_recall**
- Weakest metric: **response_relevancy**

## Generated graphs

1. `01_metric_means.png` — average quality scores
2. `02_metric_distributions.png` — score spread and outliers
3. `03_latency.png` — latency by evaluation
4. `04_token_usage.png` — total tokens by evaluation
5. `05_latency_vs_tokens.png` — efficiency relationship