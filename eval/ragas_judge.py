"""Ragas judge LLM setup -- wraps Groq (via its OpenAI-compatible endpoint)
for use as the evaluator/judge in Ragas metrics.

Deliberately a DIFFERENT model (gpt-oss-120b) than the one being evaluated
(qwen3.6-27b, used by the actual RAG pipeline in src/pipeline.py) -- using
the same model to both generate an answer and judge its own answer risks
self-preference bias in the scores, the exact thing the original project
plan called out to avoid.

Uses the older LangchainLLMWrapper pattern (not the newer llm_factory) and
requires ragas pinned below 0.4 -- ragas 0.4.x has a confirmed upstream bug
(unconditional import of Google's ChatVertexAI at module load, broken for
every non-Google-Cloud user) tracked at
https://github.com/vibrantlabsai/ragas/issues/2753 and duplicates. Install:
    pip install "ragas<0.4" langchain_openai
"""

import os

from eval import _ragas_compat  # noqa: F401 -- must import before ragas, see that file
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

JUDGE_MODEL = "openai/gpt-oss-120b"


def get_judge_llm():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set -- same key used elsewhere in this project.")

    chat = ChatOpenAI(
        model=JUDGE_MODEL,
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    return LangchainLLMWrapper(chat)