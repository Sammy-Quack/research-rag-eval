# Phase 6: Demo + Dashboard Build Guide

Two different things need building. Treat them as separate concerns:

1. **The live query app.** A visitor types a question, picks a configuration, and
   gets an answer with sources, latency and token count. This needs the real
   backend running: Ollama, the Chroma index, the BM25 index. It cannot be a
   static page.
2. **The evaluation dashboard.** The ablation table and graphs. This is static
   data: the numbers only change when an evaluation is re-run, so it needs no
   live backend at all.

That split decides the deployment story (see "Deployment") and keeps the
public-facing part cheap to host.

---

## Recommended approach: Streamlit with two tabs

Streamlit fits both needs with the least code: widgets for the query form,
`st.dataframe` / `st.bar_chart` / `st.image` for the dashboard.

```
app/
├── streamlit_app.py     # entry point, two tabs
├── query_tab.py         # live demo (needs Ollama + indexes)
├── dashboard_tab.py     # eval results (needs only eval/results/ + report images)
└── dashboard_only.py    # optional slim entry point for public hosting
```

### 1. Entry point: `app/streamlit_app.py`

```python
import sys
from pathlib import Path

# make `from src...` importable no matter where streamlit is launched from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from query_tab import render_query_tab
from dashboard_tab import render_dashboard_tab

st.set_page_config(page_title="Autonomous Agents RAG", layout="wide")

tab_query, tab_dashboard = st.tabs(["Ask a question", "Evaluation dashboard"])

with tab_query:
    render_query_tab()

with tab_dashboard:
    render_dashboard_tab()
```

### 2. Query tab: `app/query_tab.py`

It calls `src.pipeline.answer_question`, the same function the CLI uses.

**Cache the retriever.** Streamlit re-runs the script on every interaction. If
the retriever is rebuilt each time, BGE-M3 (about 2.3 GB) reloads on every
question. That exact bug inflated latency by roughly 3x during the evaluation
work. `st.cache_resource` fixes it.

```python
import streamlit as st

from src.pipeline import answer_question, get_retriever


# max_entries=2: each cached retriever holds its own copy of BGE-M3 (~2.3 GB RAM),
# so don't let a visitor clicking through every strategy/mode combo exhaust memory
@st.cache_resource(max_entries=2, show_spinner="Loading retrieval index (first query only)...")
def load_retriever(strategy: str, mode: str):
    return get_retriever(strategy, mode)


def render_query_tab():
    st.header("Ask the corpus a question")
    st.caption("Runs on local models via Ollama. Expect roughly 20-40 seconds per answer.")

    col1, col2 = st.columns(2)
    strategy = col1.selectbox("Chunking strategy", ["section_aware", "sentence", "fixed_size"])
    mode = col2.selectbox("Retrieval mode", ["hybrid", "dense", "none"],
                          help="'none' = no retrieval, the model answers from its own knowledge (the baseline)")

    query = st.text_input("Your question")

    if st.button("Ask", disabled=not query):
        retriever = None if mode == "none" else load_retriever(strategy, mode)

        with st.spinner("Retrieving and generating..."):
            result = answer_question(query, strategy, mode, retriever=retriever)

        st.subheader("Answer")
        st.write(result["answer"])

        if result["chunks_used"]:
            st.subheader("Sources")
            for i, chunk in enumerate(result["chunks_used"], start=1):
                label = f"[{i}] {chunk['paper_id']} | {chunk.get('section') or 'unknown'}"
                with st.expander(label):
                    st.markdown(f"[arXiv: {chunk['paper_id']}](https://arxiv.org/abs/{chunk['paper_id']})")
                    st.write(chunk["text"])

        m1, m2 = st.columns(2)
        m1.metric("Latency", f"{result['latency_seconds']:.1f}s")
        if result["token_usage"] and result["token_usage"].get("total_tokens"):
            m2.metric("Tokens", result["token_usage"]["total_tokens"])
```

Notes:

- The live app only needs the **generation** model (`llama3.2-3b-8k`) and BGE-M3.
  The 14B judge model is used only by the evaluation scripts, so the demo is
  lighter than an evaluation run.
- Ollama serves requests essentially one at a time. Multiple simultaneous
  visitors queue. Fine for a demo, not for traffic.
- The arXiv link works for new-style IDs (`2503.23633`). Check any old-style
  IDs in your corpus if you have them.

### 3. Dashboard tab: `app/dashboard_tab.py`

Two layers: a combined comparison computed straight from the result JSONs, then
your existing per-config reports and graphs.

```python
import json
import math
from pathlib import Path

import pandas as pd
import streamlit as st

RESULTS_DIR = Path("eval/results")
REPORTS_DIR = Path("eval/reports")      # wherever your existing report script writes REPORT.md + PNGs
FINAL_SUFFIX = "__llama3_2_3b_8k"       # only final results; ignores superseded Groq-era / contaminated files
METRICS = ["faithfulness", "context_precision", "context_recall", "response_relevancy"]


@st.cache_data
def load_summary() -> pd.DataFrame:
    rows = []
    for path in sorted(RESULTS_DIR.glob(f"*{FINAL_SUFFIX}.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        row = {"config": path.stem.removesuffix(FINAL_SUFFIX), "n": len(data)}
        for m in METRICS:
            vals = [r["scores"].get(m) for r in data]
            vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
            row[m] = sum(vals) / len(vals) if vals else None   # NaN/None excluded; baseline has only relevancy
        row["avg_latency_s"] = sum(r["latency_seconds"] for r in data) / len(data)
        rows.append(row)
    return pd.DataFrame(rows).set_index("config")


def render_dashboard_tab():
    st.header("Ablation results")

    summary = load_summary()
    st.dataframe(summary.style.format("{:.3f}", na_rep="n/a"), use_container_width=True)
    st.bar_chart(summary[METRICS])
    st.caption(
        "Same 70 questions for every configuration; generator llama3.2-3b-8k, judge qwen2.5-14b-8k. "
        "'none' is the no-retrieval baseline, so only response_relevancy applies to it. "
        "Latency was measured with other applications running; compare quality, not speed."
    )

    st.subheader("Per-configuration report")
    if REPORTS_DIR.exists():
        configs = sorted(p.name for p in REPORTS_DIR.iterdir() if p.is_dir())
        if configs:
            selected = st.selectbox("Configuration", configs)
            report_dir = REPORTS_DIR / selected
            report_md = report_dir / "REPORT.md"
            if report_md.exists():
                st.markdown(report_md.read_text(encoding="utf-8"))
            cols = st.columns(2)
            for i, img in enumerate(sorted(report_dir.glob("*.png"))):
                cols[i % 2].image(str(img))
    else:
        st.info("Per-configuration reports not found. Run the reporting script first.")
```

The `load_summary` logic was tested against synthetic result files (a NaN score,
a stale non-final file, and the baseline row). NaN scores are excluded rather
than poisoning the mean, the stale file is skipped by its filename suffix, and
the baseline row shows `n/a` for metrics that don't apply to it.

To make graphs interactive later, replace `st.image` with `st.pyplot(fig)` and
call your existing matplotlib functions directly. Reading saved PNGs is simpler
to start with because it reuses your script unchanged.

### 4. Optional slim entry point: `app/dashboard_only.py`

For public hosting without the heavy ML stack.

```python
import streamlit as st
from dashboard_tab import render_dashboard_tab

st.set_page_config(page_title="RAG evaluation dashboard", layout="wide")
render_dashboard_tab()
```

Put a small `app/requirements.txt` next to it containing only:

```
streamlit
pandas
```

---

## Running locally

```powershell
pip install streamlit
streamlit run app/streamlit_app.py
```

Ollama must be running (`ollama list` should work) with `llama3.2-3b-8k` built.

---

## Deployment: what is and isn't possible

The query tab depends on a local Ollama install, your custom models and a local
Chroma index. **No free cloud host provides that.** Verified specifics:

- **Streamlit Community Cloud** is free and runs `streamlit run` from the
  repository root. It reads dependencies from `requirements.txt` in the repo
  root **or** next to the entrypoint file. A dashboard-only app with its own
  small `app/requirements.txt` therefore avoids installing torch and
  sentence-transformers.
- **Hugging Face Spaces.** The original roadmap assumed a free CPU Space. The
  current Hub docs say Gradio and Docker Spaces require a paid plan, and only
  **Static** Spaces are free. The Spaces configuration reference still lists a
  `streamlit` SDK, but the overview lists only Gradio, Docker and static HTML.
  Check the "Create new Space" screen before planning around it.
- **GitHub Pages** is free for public repos and suits a static dashboard.

Options, in order of recommendation:

1. **Dashboard-only public deploy (recommended).** Deploy `app/dashboard_only.py`
   to Streamlit Community Cloud. The repo must contain `eval/results/*.json` and
   the report PNGs (commit them; they are results, not regenerable junk). Show
   the live query tab through screenshots or a short screen recording in the
   README instead.
2. **Local-only full demo.** Run both tabs on your machine for interviews or a
   video walkthrough. No cost, and honest for a project built around local models.
3. **Swap the query backend for a hosted API, public demo only.** Point the
   generator at a cloud API for the public link while keeping all evaluation
   results generated locally. Real work, and only worth it if a public live
   demo specifically matters.

If you ever host the query tab publicly, show shortened excerpts (for example
the first 300 characters) plus the arXiv link rather than full chunk text.

---

## Other UI options

| Option | Live app + graphs in one UI? | Effort | Free hosting | Notes |
|---|---|---|---|---|
| **Streamlit** (recommended) | Yes, via tabs | Low | Streamlit Community Cloud (dashboard only, see above) | Best fit for the mixed live + static need |
| **Gradio Blocks** | Yes (`gr.Tab`, `gr.Image`, `gr.Plot`) | Low | Hugging Face Gradio Spaces now need a paid plan | Strongest for a single input-to-output demo; also a good fit if you only want the query app |
| **Dash (Plotly)** | Yes | Medium to high | Self-host | Callback model, most control over interactivity; overkill here |
| **Panel (HoloViz)** | Yes | Medium | Self-host | Good with matplotlib/bokeh, smaller community |
| **NiceGUI** | Yes | Medium | Self-host | App-like UI, less dashboard-oriented |
| **Static HTML on GitHub Pages** | Graphs only | Low | Yes | Render `REPORT.md` and the PNGs into one `docs/index.html`; enable Pages from `/docs`. Public link with no extra accounts |
| **Notebook rendered on GitHub** | Graphs only | Lowest | Yes (GitHub renders `.ipynb`) | The original roadmap's `notebooks/results_analysis.ipynb`; commit it with outputs saved |
| **FastAPI + one HTML page** | Yes | High | Self-host | Full control, most code; only if you want a custom front end |

Sensible combination: **Streamlit locally for the full demo, plus a static
dashboard (GitHub Pages or the notebook) as the always-available public link.**
