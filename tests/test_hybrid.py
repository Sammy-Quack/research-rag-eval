from src.retrieval.hybrid import reciprocal_rank_fusion, RRF_K


def make_chunk(chunk_id):
    return {"chunk_id": chunk_id, "text": f"text for {chunk_id}"}


def test_chunk_at_top_of_both_lists_ranks_first():
    list_a = [make_chunk("x"), make_chunk("y"), make_chunk("z")]
    list_b = [make_chunk("x"), make_chunk("z"), make_chunk("y")]
    fused = reciprocal_rank_fusion(list_a, list_b)
    assert fused[0][0]["chunk_id"] == "x"


def test_preserves_all_unique_chunks_across_lists():
    list_a = [make_chunk("a"), make_chunk("b")]
    list_b = [make_chunk("b"), make_chunk("c")]
    fused = reciprocal_rank_fusion(list_a, list_b)
    ids = {chunk["chunk_id"] for chunk, score in fused}
    assert ids == {"a", "b", "c"}


def test_chunk_appearing_in_both_lists_outranks_single_list_chunk():
    # "b" appears in both lists; "a" appears only once, even at a better rank
    list_a = [make_chunk("a"), make_chunk("b")]
    list_b = [make_chunk("c"), make_chunk("b")]
    fused = reciprocal_rank_fusion(list_a, list_b)
    fused_ids = [chunk["chunk_id"] for chunk, score in fused]
    assert fused_ids.index("b") < fused_ids.index("a")


def test_score_matches_manual_rrf_calculation():
    list_a = [make_chunk("x")]
    fused = reciprocal_rank_fusion(list_a, k=RRF_K)
    expected_score = 1 / (RRF_K + 1)
    assert abs(fused[0][1] - expected_score) < 1e-9


def test_empty_input_returns_empty():
    assert reciprocal_rank_fusion([]) == []


def test_single_list_preserves_original_order():
    list_a = [make_chunk("first"), make_chunk("second"), make_chunk("third")]
    fused = reciprocal_rank_fusion(list_a)
    assert [chunk["chunk_id"] for chunk, score in fused] == ["first", "second", "third"]