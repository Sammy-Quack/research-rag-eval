from src.generation.prompt import build_prompt


def make_chunk(paper_id, section, text):
    return {"chunk_id": f"{paper_id}::0", "paper_id": paper_id, "section": section, "text": text}


def test_excerpts_numbered_from_one():
    chunks = [make_chunk("p1", "Intro", "first text"), make_chunk("p2", "Method", "second text")]
    prompt = build_prompt("a question", chunks)
    assert "[1]" in prompt
    assert "[2]" in prompt


def test_paper_id_and_section_included():
    chunks = [make_chunk("2404.04289", "Background", "some content")]
    prompt = build_prompt("q", chunks)
    assert "2404.04289" in prompt
    assert "Background" in prompt


def test_query_included_in_prompt():
    chunks = [make_chunk("p1", "Intro", "text")]
    prompt = build_prompt("How are agents evaluated?", chunks)
    assert "How are agents evaluated?" in prompt


def test_missing_section_falls_back_to_unknown():
    chunk = {"chunk_id": "p1::0", "paper_id": "p1", "text": "text"}  # no "section" key
    prompt = build_prompt("q", [chunk])
    assert "unknown" in prompt