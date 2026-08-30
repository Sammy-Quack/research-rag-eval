"""Compatibility shim for a confirmed upstream ragas bug (all versions
checked so far, 0.3.9 through 0.4.x): ragas/llms/base.py has an
unconditional import of Google's ChatVertexAI at module load time --

    from langchain_community.chat_models.vertexai import ChatVertexAI

-- which fails on any current install, since langchain-community was
officially sunset (May 2026) and no longer ships that submodule at all.
This has nothing to do with anything in this project; we never use Google
Vertex AI anywhere.

Fix: inject a fake module satisfying that one import line BEFORE ragas
tries to run it, so ragas's own broken import line finds something and
never actually needs the real (nonexistent) package.

Import this FIRST, before importing anything from ragas, in every file
that touches ragas. Tracked upstream at
https://github.com/vibrantlabsai/ragas/issues/2753 -- once fixed there,
delete this file and its imports.
"""

import sys
import types

if "langchain_community.chat_models.vertexai" not in sys.modules:
    _fake_module = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # never actually instantiated -- just needs to exist
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "This is a compatibility stub for a broken ragas import. "
                "Google Vertex AI is not actually available in this project."
            )

    _fake_module.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _fake_module