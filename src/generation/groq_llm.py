"""Groq-hosted LLM for answer generation -- the free, open-source-model
option (no cost, no card). Requires GROQ_API_KEY. Uses Groq's official
Python client (their API is OpenAI-compatible under the hood, but the
native client is simpler than routing through the openai package).

Reasoning-capable models (e.g. Qwen3's thinking mode) emit their internal
deliberation wrapped in <think>...</think> tags as part of the same content
string as the final answer -- stripped out here before returning, since the
raw trace should never be what Ragas evaluates or what a user sees as "the
answer" in Phase 5. Also sets an explicit, generous max_tokens: reasoning
models can burn thousands of tokens on deliberation before ever reaching
the answer, and the default budget was observed to truncate a real answer
mid-sentence during testing.
"""

import os
import re

from groq import Groq

MODEL_NAME = "qwen/qwen3.6-27b"
MAX_TOKENS = 4096

THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think_block(text: str) -> str:
    return THINK_BLOCK_PATTERN.sub("", text).strip()


class GroqLLM:
    name = "groq_qwen_3_6_27b"  # no slash -- gets used in filenames/keys in Phase 5

    def __init__(self, model: str = MODEL_NAME):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Get a free key (no card needed) at "
                "https://console.groq.com/keys, then set it as an env var."
            )
        self._client = Groq(api_key=api_key)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # low: grounded factual answers, not creative variation
            max_tokens=MAX_TOKENS,
        )
        choice = response.choices[0]

        if choice.finish_reason == "length":
            print("  WARNING: response was cut off by the token limit "
                  "(finish_reason='length') -- answer below may be incomplete.")

        return strip_think_block(choice.message.content)