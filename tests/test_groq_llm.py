def test_strips_realistic_think_block():
    from src.generation.groq_llm import strip_think_block
    sample = (
        "<think>\nLots of reasoning here.\nChecking constraints.\n</think>\n\n"
        "Based on the excerpts, agents are evaluated using task success metrics [1]."
    )
    result = strip_think_block(sample)
    assert result == "Based on the excerpts, agents are evaluated using task success metrics [1]."


def test_passthrough_when_no_think_block():
    from src.generation.groq_llm import strip_think_block
    text = "Just a plain answer with no think block at all."
    assert strip_think_block(text) == text


def test_empty_string():
    from src.generation.groq_llm import strip_think_block
    assert strip_think_block("") == ""


def test_multiline_answer_after_think_block_preserved():
    from src.generation.groq_llm import strip_think_block
    text = "<think>reasoning</think>Line one.\nLine two.\nLine three."
    result = strip_think_block(text)
    assert "reasoning" not in result
    assert "Line one" in result and "Line three" in result


def test_name_has_no_slash():
    from src.generation.groq_llm import GroqLLM
    assert "/" not in GroqLLM.name