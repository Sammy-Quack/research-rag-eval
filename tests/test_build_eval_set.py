from eval.build_eval_set import has_meta_reference, parse_json_response, stratified_sample


def make_chunk(chunk_id, paper_id, word_count=200, section="Method"):
    return {"chunk_id": chunk_id, "paper_id": paper_id, "word_count": word_count, "section": section, "text": "..."}


def test_stratified_sample_respects_target_count():
    chunks = [make_chunk(f"c{i}", f"p{i}") for i in range(10)]
    sampled = stratified_sample(chunks, target_count=5)
    assert len(sampled) == 5


def test_stratified_sample_excludes_low_word_count():
    chunks = [make_chunk("c1", "p1", word_count=10), make_chunk("c2", "p1", word_count=200)]
    sampled = stratified_sample(chunks, target_count=10, min_word_count=80)
    ids = [c["chunk_id"] for c in sampled]
    assert "c1" not in ids
    assert "c2" in ids


def test_stratified_sample_excludes_preamble():
    chunks = [make_chunk("c1", "p1", section="preamble"), make_chunk("c2", "p1", section="Method")]
    sampled = stratified_sample(chunks, target_count=10)
    ids = [c["chunk_id"] for c in sampled]
    assert "c1" not in ids


def test_stratified_sample_caps_per_paper():
    chunks = [make_chunk(f"c{i}", "p1") for i in range(5)]
    sampled = stratified_sample(chunks, target_count=10)
    assert len(sampled) == 2  # only 1 paper available, capped at 2 per paper


def test_stratified_sample_deterministic_with_same_seed():
    chunks = [make_chunk(f"c{i}", f"p{i}") for i in range(20)]
    sample1 = stratified_sample(chunks, target_count=5, seed=42)
    sample2 = stratified_sample(chunks, target_count=5, seed=42)
    assert [c["chunk_id"] for c in sample1] == [c["chunk_id"] for c in sample2]


def test_parse_json_response_plain_json():
    result = parse_json_response('{"question": "Q?", "reference_answer": "A."}')
    assert result == {"question": "Q?", "reference_answer": "A."}


def test_parse_json_response_strips_markdown_fence():
    text = '```json\n{"question": "Q?", "reference_answer": "A."}\n```'
    result = parse_json_response(text)
    assert result == {"question": "Q?", "reference_answer": "A."}


def test_parse_json_response_invalid_returns_none():
    assert parse_json_response("not json at all") is None


def test_parse_json_response_missing_keys_returns_none():
    assert parse_json_response('{"question": "Q?"}') is None


def test_has_meta_reference_catches_excerpt():
    q = ("What indirect avenues, beyond the direct creation of biological weapons, does "
         "the excerpt identify as ways AI systems might increase biosecurity risks?")
    assert has_meta_reference(q) is True


def test_has_meta_reference_catches_according_to_the_text():
    assert has_meta_reference("According to the text, how are COMPAS scores mapped?") is True


def test_has_meta_reference_catches_the_source():
    assert has_meta_reference("What does the source say about scene graphs?") is True


def test_has_meta_reference_false_on_clean_question():
    assert has_meta_reference("What indirect risks can AI systems pose to biosecurity?") is False


def test_has_meta_reference_false_on_unrelated_wording():
    q = "How does the modified teachability agent prevent storage of similar memories?"
    assert has_meta_reference(q) is False