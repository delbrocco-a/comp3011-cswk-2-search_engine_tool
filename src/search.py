"""Query processing over an inverted index."""

from indexer import Indexer


class SearchEngine:
    """
    Provides print and find operations over an Indexer.

    Search strategy:
      - print: display full inverted list for a single term.
      - find:  conjunctive (AND) retrieval ranked by summed TF-IDF.

    Complexity of find: O(k + |C|) where k = number of query terms,
    |C| = size of the candidate intersection set.
    """

    def __init__(self, indexer: Indexer) -> None:
        """
        Args:
            indexer: A populated Indexer instance (loaded or built).
        """
        self._indexer = indexer

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def print_word(self, word: str) -> None:
        """
        Print the full inverted index entry for word to stdout.

        Displays document count, and for each document: URL, raw
        frequency, first occurrence positions (up to 10), and TF-IDF.

        Args:
            word: The search term (case-insensitive).
        """
        posting_map = self._indexer.get_posting_map(word)
        if not posting_map:
            print(f"'{word}' not found in index.")
            return

        print(f"\nInverted index for '{word.lower()}':")
        print(f"  Documents containing this term: {len(posting_map)}")
        print()
        for url, posting in sorted(posting_map.items()):
            truncated = posting.positions[:10]
            ellipsis = "..." if len(posting.positions) > 10 else ""
            print(f"  URL:       {url}")
            print(f"  frequency: {posting.frequency}")
            print(f"  positions: {truncated}{ellipsis}")
            print(f"  tf-idf:    {posting.tf_idf:.6f}")
            print()

    def find(self, words: list[str]) -> list[tuple[str, float]]:
        """
        Conjunctive search: return pages containing ALL query words.

        Results are ranked in descending order by summed TF-IDF score
        across all query terms, implementing document-at-a-time scoring
        (lecture 13).

        Args:
            words: One or more query terms (case-insensitive).

        Returns:
            List of (url, score) tuples sorted by score descending.
            Empty list if any term is absent or no intersection exists.
        """
        if not words:
            return []

        # Retrieve posting maps; bail early if any term is missing
        posting_maps: list[dict] = []
        for word in words:
            pm = self._indexer.get_posting_map(word)
            if pm is None:
                return []
            posting_maps.append(pm)

        # Conjunctive intersection: O(k * min_df)
        # Start from the shortest list to minimise intersection work
        sorted_maps = sorted(posting_maps, key=len)
        common_urls: set[str] = set(sorted_maps[0].keys())
        for pm in sorted_maps[1:]:
            common_urls &= pm.keys()

        if not common_urls:
            return []

        # Score each surviving document by summing TF-IDF contributions
        scores: dict[str, float] = {
            url: sum(pm.get(url).tf_idf for pm in posting_maps if url in pm)
            for url in common_urls
        }

        return sorted(scores.items(), key=lambda item: item[1], reverse=True)
