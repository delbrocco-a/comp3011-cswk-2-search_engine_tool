"""Unit tests for the Indexer."""

import json
import math
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, "src")
from indexer import Indexer, Posting

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

SIMPLE_HTML = """
<html>
<head><title>Test Page</title></head>
<body>
<h1>The quick brown fox</h1>
<p>The fox jumps over the lazy dog.</p>
</body>
</html>
"""

SCRIPT_HTML = """
<html>
<body>
<script>var x = 1;</script>
<style>.cls { color: red; }</style>
<p>visible text only</p>
</body>
</html>
"""

URL_A = "https://example.com/a"
URL_B = "https://example.com/b"


@pytest.fixture()
def indexer():
    return Indexer()


@pytest.fixture()
def populated_indexer():
    idx = Indexer()
    idx.index_page(URL_A, "<html><body>hello world hello</body></html>")
    idx.index_page(URL_B, "<html><body>world foo bar</body></html>")
    idx.compute_tfidf()
    return idx


# ------------------------------------------------------------------
# Tokenisation
# ------------------------------------------------------------------


def test_tokenize_lowercases(indexer):
    assert indexer._tokenize("Hello World") == ["hello", "world"]


def test_tokenize_strips_punctuation(indexer):
    tokens = indexer._tokenize("don't stop, believing!")
    assert "don" in tokens
    assert "t" in tokens
    assert "stop" in tokens
    assert "believing" in tokens


def test_tokenize_empty_string(indexer):
    assert indexer._tokenize("") == []


def test_tokenize_numbers(indexer):
    assert "2025" in indexer._tokenize("Year 2025")


def test_tokenize_only_special_chars(indexer):
    assert indexer._tokenize("!!! ???") == []


# ------------------------------------------------------------------
# Text extraction
# ------------------------------------------------------------------


def test_extract_text_removes_script(indexer):
    text = indexer._extract_text(SCRIPT_HTML)
    assert "var x" not in text


def test_extract_text_removes_style(indexer):
    text = indexer._extract_text(SCRIPT_HTML)
    assert "color" not in text


def test_extract_text_keeps_body(indexer):
    text = indexer._extract_text(SCRIPT_HTML)
    assert "visible" in text.lower()


def test_extract_text_empty_html(indexer):
    text = indexer._extract_text("<html></html>")
    assert isinstance(text, str)


# ------------------------------------------------------------------
# index_page
# ------------------------------------------------------------------


def test_index_page_creates_entry(indexer):
    indexer.index_page(URL_A, "<html><body>hello</body></html>")
    assert "hello" in indexer._index


def test_index_page_increments_frequency(indexer):
    indexer.index_page(URL_A, "<html><body>hello hello hello</body></html>")
    assert indexer._index["hello"][URL_A].frequency == 3


def test_index_page_records_positions(indexer):
    indexer.index_page(URL_A, "<html><body>alpha beta alpha</body></html>")
    positions = indexer._index["alpha"][URL_A].positions
    assert len(positions) == 2


def test_index_page_case_insensitive(indexer):
    indexer.index_page(URL_A, "<html><body>Hello HELLO hello</body></html>")
    assert indexer._index["hello"][URL_A].frequency == 3


def test_index_page_updates_doc_length(indexer):
    indexer.index_page(URL_A, "<html><body>one two three</body></html>")
    assert indexer._doc_lengths[URL_A] == 3


def test_index_page_multiple_documents(indexer):
    indexer.index_page(URL_A, "<html><body>shared word</body></html>")
    indexer.index_page(URL_B, "<html><body>shared other</body></html>")
    assert URL_A in indexer._index["shared"]
    assert URL_B in indexer._index["shared"]


def test_total_docs_count(indexer):
    assert indexer.total_docs == 0
    indexer.index_page(URL_A, "<html><body>x</body></html>")
    assert indexer.total_docs == 1
    indexer.index_page(URL_B, "<html><body>y</body></html>")
    assert indexer.total_docs == 2


# ------------------------------------------------------------------
# TF-IDF computation
# ------------------------------------------------------------------


def test_compute_tfidf_sets_nonzero_score(populated_indexer):
    pm = populated_indexer.get_posting_map("hello")
    assert pm is not None
    assert pm[URL_A].tf_idf > 0


def test_compute_tfidf_unique_term_has_higher_idf(populated_indexer):
    # "hello" only in URL_A; "world" in both -> "hello" gets higher IDF
    hello_tfidf = populated_indexer.get_posting_map("hello")[URL_A].tf_idf
    world_tfidf = populated_indexer.get_posting_map("world")[URL_A].tf_idf
    # IDF(hello) = log(2/1) > IDF(world) = log(2/2) = 0
    assert hello_tfidf > world_tfidf


def test_compute_tfidf_shared_term_idf_zero_or_low():
    """Term appearing in every document has IDF = log(N/N) = 0."""
    idx = Indexer()
    idx.index_page(URL_A, "<html><body>shared</body></html>")
    idx.index_page(URL_B, "<html><body>shared</body></html>")
    idx.compute_tfidf()
    # IDF = log(2/2) = 0 -> TF-IDF = 0
    assert idx.get_posting_map("shared")[URL_A].tf_idf == 0.0


def test_compute_tfidf_empty_index(indexer):
    """compute_tfidf on empty indexer must not raise."""
    indexer.compute_tfidf()


# ------------------------------------------------------------------
# Save / Load round-trip
# ------------------------------------------------------------------


def test_save_creates_file(populated_indexer):
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "index.json")
        populated_indexer.save(path)
        assert Path(path).exists()


def test_save_creates_parent_directories(populated_indexer):
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "nested" / "dir" / "index.json")
        populated_indexer.save(path)
        assert Path(path).exists()


def test_save_load_round_trip(populated_indexer):
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "index.json")
        populated_indexer.save(path)

        fresh = Indexer()
        fresh.load(path)

        assert fresh.total_docs == populated_indexer.total_docs
        assert fresh.total_terms == populated_indexer.total_terms


def test_save_load_preserves_frequency(populated_indexer):
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "index.json")
        populated_indexer.save(path)
        fresh = Indexer()
        fresh.load(path)
        original = populated_indexer.get_posting_map("hello")[URL_A].frequency
        loaded = fresh.get_posting_map("hello")[URL_A].frequency
        assert original == loaded


def test_save_load_preserves_positions(populated_indexer):
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "index.json")
        populated_indexer.save(path)
        fresh = Indexer()
        fresh.load(path)
        original = populated_indexer.get_posting_map("hello")[URL_A].positions
        loaded = fresh.get_posting_map("hello")[URL_A].positions
        assert original == loaded


def test_save_load_preserves_tfidf(populated_indexer):
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "index.json")
        populated_indexer.save(path)
        fresh = Indexer()
        fresh.load(path)
        orig = populated_indexer.get_posting_map("hello")[URL_A].tf_idf
        loaded = fresh.get_posting_map("hello")[URL_A].tf_idf
        assert abs(orig - loaded) < 1e-9


def test_load_missing_file_raises(indexer):
    with pytest.raises(FileNotFoundError):
        indexer.load("/nonexistent/path/index.json")


def test_save_valid_json(populated_indexer):
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "index.json")
        populated_indexer.save(path)
        with open(path) as fh:
            data = json.load(fh)
        assert "metadata" in data
        assert "index" in data
        assert "doc_lengths" in data


# ------------------------------------------------------------------
# get_posting_map
# ------------------------------------------------------------------


def test_get_posting_map_existing_word(populated_indexer):
    pm = populated_indexer.get_posting_map("hello")
    assert pm is not None
    assert URL_A in pm


def test_get_posting_map_missing_word(populated_indexer):
    assert populated_indexer.get_posting_map("nonexistentxyz") is None


def test_get_posting_map_case_insensitive(populated_indexer):
    assert populated_indexer.get_posting_map("HELLO") is not None
    assert populated_indexer.get_posting_map("Hello") is not None
