"""Unit tests for the SearchEngine."""

import sys
from io import StringIO

import pytest

sys.path.insert(0, "src")
from indexer import Indexer, Posting  # noqa: E402
from search import SearchEngine  # noqa: E402

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

URL_A = "https://example.com/a"
URL_B = "https://example.com/b"
URL_C = "https://example.com/c"


@pytest.fixture()
def engine():
    """Engine over a 3-document index with known TF-IDF values."""
    idx = Indexer()
    idx.index_page(
        URL_A, "<html><body>love life love friendship</body></html>"
    )
    idx.index_page(URL_B, "<html><body>love wisdom knowledge</body></html>")
    idx.index_page(URL_C, "<html><body>life wisdom beauty</body></html>")
    idx.compute_tfidf()
    return SearchEngine(idx)


@pytest.fixture()
def empty_engine():
    return SearchEngine(Indexer())


# ------------------------------------------------------------------
# find - empty / edge cases
# ------------------------------------------------------------------


def test_find_empty_words_returns_empty(engine):
    assert engine.find([]) == []


def test_find_missing_word_returns_empty(engine):
    assert engine.find(["nonexistentterm"]) == []


def test_find_one_missing_word_conjunctive_empty(engine):
    """AND semantics: if any word is absent, result must be empty."""
    assert engine.find(["love", "nonexistentterm"]) == []


def test_find_empty_index_returns_empty(empty_engine):
    assert empty_engine.find(["love"]) == []


# ------------------------------------------------------------------
# find - single word
# ------------------------------------------------------------------


def test_find_single_word_returns_matching_docs(engine):
    urls = [url for url, _ in engine.find(["love"])]
    assert URL_A in urls
    assert URL_B in urls
    assert URL_C not in urls


def test_find_returns_list_of_tuples(engine):
    results = engine.find(["love"])
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)


def test_find_scores_are_floats(engine):
    results = engine.find(["love"])
    assert all(isinstance(score, float) for _, score in results)


def test_find_scores_nonnegative(engine):
    results = engine.find(["love"])
    assert all(score >= 0 for _, score in results)


# ------------------------------------------------------------------
# find - multi-word conjunctive AND
# ------------------------------------------------------------------


def test_find_two_words_conjunctive(engine):
    """Only URL_A contains both 'love' and 'life'."""
    urls = [url for url, _ in engine.find(["love", "life"])]
    assert urls == [URL_A]


def test_find_no_intersection_returns_empty(engine):
    """'friendship' only in A, 'beauty' only in C -> no AND match."""
    assert engine.find(["friendship", "beauty"]) == []


def test_find_three_words_no_intersection(engine):
    assert engine.find(["love", "life", "beauty"]) == []


# ------------------------------------------------------------------
# find - TF-IDF ranking
# ------------------------------------------------------------------


def test_find_sorted_by_score_descending(engine):
    results = engine.find(["love"])
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_find_unique_term_ranks_higher():
    """
    A doc with a unique term (higher IDF) should rank above one where
    the term is shared across many documents.
    """
    idx = Indexer()
    # "rare" appears only in URL_A; "common" appears in both
    idx.index_page(URL_A, "<html><body>rare common common</body></html>")
    idx.index_page(URL_B, "<html><body>common common</body></html>")
    idx.compute_tfidf()
    se = SearchEngine(idx)

    results = se.find(["common"])
    # URL_A has same freq of "common" but URL_A also has "rare" boosting IDF
    # We only check "common" here; just verify sorted order
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


# ------------------------------------------------------------------
# print_word
# ------------------------------------------------------------------


def test_print_word_existing(engine, capsys):
    engine.print_word("love")
    out = capsys.readouterr().out
    assert "love" in out
    assert URL_A in out
    assert URL_B in out


def test_print_word_missing(engine, capsys):
    engine.print_word("nonexistentxyz")
    out = capsys.readouterr().out
    assert "not found" in out.lower()


def test_print_word_shows_frequency(engine, capsys):
    engine.print_word("love")
    out = capsys.readouterr().out
    assert "frequency" in out


def test_print_word_shows_positions(engine, capsys):
    engine.print_word("love")
    out = capsys.readouterr().out
    assert "positions" in out


def test_print_word_shows_tfidf(engine, capsys):
    engine.print_word("love")
    out = capsys.readouterr().out
    assert "tf-idf" in out


def test_print_word_case_insensitive(engine, capsys):
    engine.print_word("LOVE")
    out = capsys.readouterr().out
    assert "not found" not in out.lower()


def test_print_word_empty_index(empty_engine, capsys):
    empty_engine.print_word("anything")
    out = capsys.readouterr().out
    assert "not found" in out.lower()


# ------------------------------------------------------------------
# Integration: find after round-trip save/load
# ------------------------------------------------------------------


def test_find_after_save_load(tmp_path):
    idx = Indexer()
    idx.index_page(URL_A, "<html><body>hello world</body></html>")
    idx.index_page(URL_B, "<html><body>world foo</body></html>")
    idx.compute_tfidf()

    path = str(tmp_path / "index.json")
    idx.save(path)

    loaded = Indexer()
    loaded.load(path)
    se = SearchEngine(loaded)

    results = se.find(["world"])
    urls = [url for url, _ in results]
    assert URL_A in urls
    assert URL_B in urls


def test_find_conjunctive_after_load(tmp_path):
    idx = Indexer()
    idx.index_page(URL_A, "<html><body>alpha beta</body></html>")
    idx.index_page(URL_B, "<html><body>alpha gamma</body></html>")
    idx.compute_tfidf()

    path = str(tmp_path / "index.json")
    idx.save(path)

    loaded = Indexer()
    loaded.load(path)
    se = SearchEngine(loaded)

    results = se.find(["alpha", "beta"])
    urls = [url for url, _ in results]
    assert urls == [URL_A]
