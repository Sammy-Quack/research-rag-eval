from src.embedding import bge_m3


def test_embedder_forwards_local_files_only(monkeypatch):
    captured = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs):
            captured["model_name"] = model_name
            captured.update(kwargs)

    monkeypatch.setattr(bge_m3.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(bge_m3, "SentenceTransformer", FakeSentenceTransformer)

    bge_m3.BGEM3Embedder(local_files_only=True)

    assert captured["model_name"] == bge_m3.MODEL_NAME
    assert captured["device"] == "cpu"
    assert captured["local_files_only"] is True
