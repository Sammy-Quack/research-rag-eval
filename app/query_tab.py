from urllib.parse import quote

import streamlit as st

from src.pipeline import answer_question, get_retriever


@st.cache_resource(
    max_entries=2,
    show_spinner="Loading retrieval index (first query only)...",
)
def load_retriever(strategy: str, mode: str):
    return get_retriever(strategy, mode)


def render_query_tab() -> None:
    st.header("Ask the corpus a question")
    st.caption(
        "Runs locally through Ollama. Expect roughly 20-40 seconds per answer."
    )

    with st.form("query_form"):
        strategy_col, mode_col = st.columns(2)
        strategy = strategy_col.selectbox(
            "Chunking strategy",
            ["section_aware", "sentence", "fixed_size"],
        )
        mode = mode_col.selectbox(
            "Retrieval mode",
            ["hybrid", "dense", "none"],
            help=(
                "'none' is the no-retrieval baseline: the model answers "
                "from its own knowledge."
            ),
        )
        query = st.text_input("Your question")
        submitted = st.form_submit_button("Ask", disabled=not query.strip())

    if not submitted:
        return

    retriever = None if mode == "none" else load_retriever(strategy, mode)
    with st.spinner("Retrieving and generating..."):
        result = answer_question(query.strip(), strategy, mode, retriever=retriever)

    st.subheader("Answer")
    st.write(result["answer"])

    if result["chunks_used"]:
        st.subheader("Sources")
        for index, chunk in enumerate(result["chunks_used"], start=1):
            paper_id = str(chunk["paper_id"])
            section = chunk.get("section") or "unknown"
            with st.expander(f"[{index}] {paper_id} | {section}"):
                st.markdown(
                    f"[arXiv: {paper_id}](https://arxiv.org/abs/{quote(paper_id, safe='.')})"
                )
                st.write(chunk["text"])

    latency_col, token_col = st.columns(2)
    latency_col.metric("Latency", f"{result['latency_seconds']:.1f}s")
    token_usage = result.get("token_usage")
    if token_usage and token_usage.get("total_tokens") is not None:
        token_col.metric("Tokens", token_usage["total_tokens"])
