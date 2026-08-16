"""OpenAI text-embedding-3-large backend — the commercial-API comparison
point against BGE-M3 for your ablation table. Entirely optional: nothing
else in this project needs it. Only wire it in once OPENAI_API_KEY is set
and you're intentionally ready to spend a small amount of API credit.
"""

import os

from openai import OpenAI

MODEL_NAME = "text-embedding-3-large"


class OpenAIEmbedder:
    name = "openai_3_large"
    dimension = 3072

    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. This embedder is optional — use "
                "BGEM3Embedder for now, or set the key to enable this one."
            )
        self._client = OpenAI(api_key=api_key)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # API caps batch size in practice; chunk defensively regardless of input size
        all_embeddings: list[list[float]] = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self._client.embeddings.create(model=MODEL_NAME, input=batch)
            all_embeddings.extend(item.embedding for item in response.data)
        return all_embeddings