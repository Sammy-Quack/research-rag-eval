"""BGE-M3 embedding backend via sentence-transformers. Self-hosted, no API
key needed. First run downloads ~2.3GB of model weights from Hugging Face —
be patient and on decent wifi. Auto-detects and uses a CUDA GPU if available
and torch was installed with CUDA support; falls back to CPU otherwise —
always prints which one it picked, so it's visible rather than assumed.
"""

import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"


class BGEM3Embedder:
    name = "bge_m3"
    dimension = 1024

    def __init__(self, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            print("  WARNING: device='cuda' requested but torch.cuda.is_available() is "
                  "False — falling back to CPU. Your torch install likely lacks CUDA support.")
            device = "cpu"

        gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else None
        print(f"  BGE-M3 using device: {device}" + (f" ({gpu_name})" if gpu_name else ""))

        self._model = SentenceTransformer(MODEL_NAME, device=device)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()