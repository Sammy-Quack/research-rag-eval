"""Ollama-hosted LLM -- fully local generation, zero API cost, zero rate
limits, since it's bound by your own GPU/CPU rather than a shared cloud
quota. Requires Ollama running locally (usually automatic after install)
with the model already built with the larger context window:
    ollama create llama3.2-3b-8k -f Modelfile_gen.txt

Deliberately a DIFFERENT model than the judge (llama3.2-3b-8k here vs
qwen2.5-14b-8k for judging in eval/ragas_judge.py) -- using the same model
for both roles risks self-preference bias in the judge's scores.
"""

import re

from openai import OpenAI

MODEL_NAME = "llama3.2-3b-8k"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think_block(text: str) -> str:
    return THINK_BLOCK_PATTERN.sub("", text).strip()


class OllamaLLM:
    name = "llama3_2_3b_8k"  # no slash/dot -- gets used in filenames/keys

    def __init__(self, model: str = MODEL_NAME, base_url: str = OLLAMA_BASE_URL):
        # Ollama's OpenAI-compatible endpoint ignores the API key entirely --
        # any non-empty string works, "ollama" is just the usual convention
        self._client = OpenAI(api_key="ollama", base_url=base_url)
        self.model = model

    def _call_api(self, system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        choice = response.choices[0]
        text = strip_think_block(choice.message.content)
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
            "completion_tokens": response.usage.completion_tokens if response.usage else None,
            "total_tokens": response.usage.total_tokens if response.usage else None,
        }
        return text, usage

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        text, _usage = self._call_api(system_prompt, user_prompt)
        return text

    def generate_with_usage(self, system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        return self._call_api(system_prompt, user_prompt)