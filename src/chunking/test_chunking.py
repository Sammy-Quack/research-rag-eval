from src.chunking.fixed_size import chunk_fixed_size
from src.chunking.section_aware import chunk_section_aware
from src.chunking.sentence_based import chunk_sentence_based


# ---------- fixed_size ----------

def test_fixed_size_empty_text_returns_no_chunks():
    assert chunk_fixed_size("", "p1") == []


def test_fixed_size_short_text_returns_one_chunk():
    text = "one two three four five"
    chunks = chunk_fixed_size(text, "p1", chunk_size_words=10, overlap_words=2)
    assert len(chunks) == 1
    assert chunks[0]["word_count"] == 5
    assert chunks[0]["text"] == text


def test_fixed_size_splits_long_text_into_multiple_chunks():
    text = " ".join(f"word{i}" for i in range(100))
    chunks = chunk_fixed_size(text, "p1", chunk_size_words=30, overlap_words=5)
    assert len(chunks) > 1
    # every chunk except possibly the last respects the size budget
    for chunk in chunks[:-1]:
        assert chunk["word_count"] == 30


def test_fixed_size_overlap_actually_overlaps():
    text = " ".join(f"word{i}" for i in range(100))
    chunks = chunk_fixed_size(text, "p1", chunk_size_words=30, overlap_words=5)
    first_words = chunks[0]["text"].split()
    second_words = chunks[1]["text"].split()
    assert first_words[-5:] == second_words[:5]


def test_fixed_size_char_offsets_are_valid_into_source_text():
    text = " ".join(f"word{i}" for i in range(50))
    chunks = chunk_fixed_size(text, "p1", chunk_size_words=20, overlap_words=3)
    for chunk in chunks:
        assert text[chunk["char_start"]:chunk["char_end"]] == chunk["text"]


def test_fixed_size_rejects_overlap_not_smaller_than_chunk_size():
    try:
        chunk_fixed_size("a b c", "p1", chunk_size_words=10, overlap_words=10)
        assert False, "expected AssertionError"
    except AssertionError:
        pass


# ---------- sentence_based ----------

def test_sentence_based_empty_text_returns_no_chunks():
    assert chunk_sentence_based("", "p1") == []


def test_sentence_based_splits_on_sentence_boundaries():
    text = "First sentence here. Second sentence here. Third one too."
    chunks = chunk_sentence_based(text, "p1", chunk_size_words=1000)
    # small budget-free case: everything fits in one chunk
    assert len(chunks) == 1
    assert "First sentence" in chunks[0]["text"]
    assert "Third one too" in chunks[0]["text"]


def test_sentence_based_respects_word_budget():
    sentences = [f"Sentence number {i} has five words." for i in range(30)]
    text = " ".join(sentences)
    chunks = chunk_sentence_based(text, "p1", chunk_size_words=40, overlap_sentences=1)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["word_count"] <= 40 or chunk["word_count"] == len(chunk["text"].split())


def test_sentence_based_oversized_single_sentence_does_not_hang():
    # one sentence alone far exceeds the budget — must not infinite-loop,
    # must still produce output
    huge_sentence = "Word " * 500 + "end."
    chunks = chunk_sentence_based(huge_sentence, "p1", chunk_size_words=50)
    assert len(chunks) >= 1


def test_sentence_based_no_sentence_terminator_still_returns_chunk():
    text = "no terminal punctuation here just words"
    chunks = chunk_sentence_based(text, "p1")
    assert len(chunks) == 1
    assert chunks[0]["text"] == text


# ---------- section_aware ----------

SAMPLE_PAPER = """John Smith, Jane Doe

Abstract
This paper studies autonomous agents in simulated environments.

1. Introduction
Autonomous agents have become increasingly capable in recent years.
This introduction covers background and motivation for the work.

2. Related Work
Prior work has explored various approaches to agent training.

3. Methodology
We propose a novel training procedure based on reinforcement learning.

4. Results
Our method outperforms baselines across all benchmark tasks.

5. Conclusion
We presented a new approach and demonstrated its effectiveness.

References
[1] Some citation here.
"""


def test_section_aware_detects_known_headers():
    chunks = chunk_section_aware(SAMPLE_PAPER, "p1")
    sections_found = {c["section"] for c in chunks}
    assert "Abstract" in sections_found
    assert "Introduction" in sections_found
    assert "Conclusion" in sections_found


def test_section_aware_falls_back_to_unknown_when_no_headers():
    text = "Just a plain wall of text with no section headers anywhere in it at all."
    chunks = chunk_section_aware(text, "p1")
    assert len(chunks) == 1
    assert chunks[0]["section"] == "unknown"


def test_section_aware_empty_text_returns_no_chunks():
    assert chunk_section_aware("", "p1") == []


def test_section_aware_large_section_gets_subsplit_but_keeps_section_tag():
    big_section = "1. Introduction\n" + ("word " * 900)
    chunks = chunk_section_aware(big_section, "p1")
    assert len(chunks) > 1
    assert all(c["section"] == "Introduction" for c in chunks)


# ---------- cross-strategy ----------

def test_chunk_ids_are_unique_within_each_strategy():
    for chunk_fn in (chunk_fixed_size, chunk_sentence_based):
        text = " ".join(f"word{i}" for i in range(200))
        chunks = chunk_fn(text, "p1", chunk_size_words=30)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))