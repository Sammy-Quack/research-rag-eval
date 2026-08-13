from src.ingest.clean import (
    clean_text,
    collapse_whitespace,
    dehyphenate,
    looks_like_multi_column_artifact,
)


def test_dehyphenate_rejoins_broken_word():
    assert dehyphenate("trans-\nformer") == "transformer"


def test_dehyphenate_ignores_normal_hyphen():
    assert dehyphenate("state-of-the-art") == "state-of-the-art"


def test_collapse_whitespace_trims_and_shrinks():
    assert collapse_whitespace("  hello   world  \n\n\n\nfoo  ") == "hello world\n\nfoo"


def test_clean_text_pipeline():
    raw = "This is a trans-\nformer   model.\n\n\n\nIt works well."
    result = clean_text(raw)
    assert "transformer" in result
    assert "  " not in result


def test_multi_column_artifact_flagged_on_short_lines():
    scrambled = "\n".join(["abc def", "gh ij", "kl mn"] * 10)
    assert looks_like_multi_column_artifact(scrambled) is True


def test_multi_column_artifact_not_flagged_on_normal_prose():
    normal = "This is a normal sentence of reasonable length that spans a full line of text."
    assert looks_like_multi_column_artifact(normal) is False
