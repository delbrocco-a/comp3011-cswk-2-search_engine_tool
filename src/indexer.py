"""Inverted index with TF-IDF scoring."""

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup


@dataclass
class Posting:
    """
    Per-document statistics for a single term.

    Attributes:
        frequency: Raw occurrence count of the term in this document.
        positions: Ordered list of zero-based word positions.
        tf_idf:    TF-IDF score, computed post-crawl via compute_tfidf().
    """

    frequency: int = 0
    positions: list[int] = field(default_factory=list)
    tf_idf: float = 0.0


class Indexer:
    """
    Builds and manages an inverted index with TF-IDF scoring.

    Data structure: dict[term, dict[url, Posting]]
    - Outer dict keyed by normalised term -> O(1) average term lookup.
    - Inner dict keyed by URL -> O(1) average document lookup.

    Space complexity: O(T * D) where T = unique terms, D = total documents.
    """

    def __init__(self) -> None:
        self._index: dict[str, dict[str, Posting]] = {}
        self._doc_lengths: dict[str, int] = {}
        self.base_url: str = ""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_docs(self) -> int:
        """Number of indexed documents."""
        return len(self._doc_lengths)

    @property
    def total_terms(self) -> int:
        """Number of unique terms in the index."""
        return len(self._index)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_page(self, url: str, html: str) -> None:
        """
        Parse html and update the inverted index for url.

        Args:
            url:  Canonical URL of the page (used as document identifier).
            html: Raw HTML content of the page.
        """
        text = self._extract_text(html)
        tokens = self._tokenize(text)
        self._doc_lengths[url] = len(tokens)

        for position, token in enumerate(tokens):
            url_map = self._index.setdefault(token, {})
            posting = url_map.setdefault(url, Posting())
            posting.frequency += 1
            posting.positions.append(position)

    def compute_tfidf(self) -> None:
        """
        Compute TF-IDF scores for every (term, document) pair.

        TF  = term_frequency / document_length  (normalised term frequency)
        IDF = log(total_docs / df)              (inverse document frequency)
        TF-IDF = TF * IDF

        Must be called after all pages have been indexed.
        Complexity: O(T * D) over all term-document pairs.
        """
        n_docs = self.total_docs
        if n_docs == 0:
            return

        for url_map in self._index.values():
            df = len(url_map)
            idf = math.log(n_docs / df)
            for url, posting in url_map.items():
                doc_length = self._doc_lengths.get(url, 1)
                tf = posting.frequency / doc_length
                posting.tf_idf = tf * idf

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Serialise the index to a JSON file.

        Creates parent directories if they do not exist.

        Args:
            path: File path for the output JSON.
        """
        data: dict = {
            "metadata": {
                "total_docs": self.total_docs,
                "total_terms": self.total_terms,
                "base_url": self.base_url,
            },
            "doc_lengths": self._doc_lengths,
            "index": {
                word: {
                    url: {
                        "frequency": p.frequency,
                        "positions": p.positions,
                        "tf_idf": p.tf_idf,
                    }
                    for url, p in url_map.items()
                }
                for word, url_map in self._index.items()
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    def load(self, path: str) -> None:
        """
        Deserialise the index from a previously saved JSON file.

        Args:
            path: Path to the JSON index file.

        Raises:
            FileNotFoundError: If path does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        self.base_url = data.get("metadata", {}).get("base_url", "")
        self._doc_lengths = data.get("doc_lengths", {})
        self._index = {}

        for word, url_map in data.get("index", {}).items():
            self._index[word] = {
                url: Posting(
                    frequency=entry["frequency"],
                    positions=entry["positions"],
                    tf_idf=entry["tf_idf"],
                )
                for url, entry in url_map.items()
            }

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_posting_map(self, word: str) -> Optional[dict[str, Posting]]:
        """
        Return the posting map for a term, or None if not in index.

        Args:
            word: Search term (case-insensitive).
        """
        return self._index.get(word.lower())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenise text into lowercase alphanumeric tokens.

        Uses regex to split on any non-alphanumeric boundary, matching
        the two-pass tokenisation approach from lecture 11.

        Args:
            text: Plain text to tokenise.

        Returns:
            Ordered list of lowercase token strings.
        """
        return re.findall(r"[a-z0-9]+", text.lower())

    def _extract_text(self, html: str) -> str:
        """
        Extract visible text from HTML, stripping markup and metadata.

        Removes script, style, meta, and head elements before extracting
        text, as these contain no searchable content.

        Args:
            html: Raw HTML content.

        Returns:
            Whitespace-separated visible text.
        """
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "meta", "head"]):
            tag.decompose()
        return soup.get_text(separator=" ")
