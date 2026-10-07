"""Ragas judge LLM setup -- pointed at a local Ollama model, eliminating
the daily token quota entirely (compute-bound on your own GPU, not a
shared cloud budget).

Deliberately still a DIFFERENT model than the pipeline's generation model
(qwen2.5:14 for generation in ollama_llm.py) -- using
the same model to both generate an answer and judge its own answer risks
self-preference bias in the scores.

Model history, in case this needs revisiting:
  - phi4-mini: ~0-20% success rate on Ragas' structured-output parsing
    (its internal NLI-based faithfulness checking needs reliably
    well-formed JSON-shaped output).
  - qwen2.5:7b: ~40% success rate -- real improvement (Qwen2.5 is
    specifically documented for structured-output reliability), but not
    enough to trust for a full run.
  - qwen2.5:14b (current): research suggests Qwen2.5 handles this
    reliably starting around 14B. Much slower on 4GB VRAM (mostly
    CPU-bound), genuinely unverified until run for real.
"""

from eval import _ragas_compat  # noqa: F401 -- must import before ragas, see that file
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

JUDGE_MODEL = "qwen2.5-14b-8k"
OLLAMA_BASE_URL = "http://localhost:11434/v1"


def get_judge_llm():
    chat = ChatOpenAI(
        model=JUDGE_MODEL,
        api_key="ollama",  # ignored by Ollama's OpenAI-compatible endpoint
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
    )
    return LangchainLLMWrapper(chat)