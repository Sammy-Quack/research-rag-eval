# RAG Evaluation Report

**Samples:** 70
**Source:** `section_aware__dense__llama3_2_3b_8k.json`

## Mean metrics

| Metric | Mean | Min | Median | Max |
|---|---:|---:|---:|---:|
| faithfulness | 0.8080 | 0.0000 | 0.8990 | 1.0000 |
| context_precision | 0.8443 | 0.0000 | 1.0000 | 1.0000 |
| context_recall | 0.9000 | 0.0000 | 1.0000 | 1.0000 |
| response_relevancy | 0.7196 | 0.0000 | 0.9131 | 1.0000 |

## Performance

- Average latency: **33.7054 s**
- Median latency: **22.9765 s**
- P95 latency: **48.1218 s**
- Average prompt tokens: **2965.91**
- Average completion tokens: **139.44**
- Average total tokens: **3105.36**

## Automatic diagnosis

- Strongest metric: **context_recall**
- Weakest metric: **response_relevancy**

## Generated graphs

1. `01_metric_means.png` — average quality scores
2. `02_metric_distributions.png` — score spread and outliers
3. `03_latency.png` — latency by evaluation
4. `04_token_usage.png` — total tokens by evaluation
5. `05_latency_vs_tokens.png` — efficiency relationship