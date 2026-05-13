"""Interactive CLI shell for the search engine."""

import logging
import sys
import time
from pathlib import Path

from crawler import WebCrawler
from indexer import Indexer
from search import SearchEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

BASE_URL = "https://quotes.toscrape.com/"
INDEX_PATH = "data/index.json"

HELP_TEXT = """\
Commands:
  build           Crawl the site, build the index, save to disk
  load            Load the index from disk
  print <word>    Print the inverted index entry for <word>
  find <words...> Find pages containing all given words (ranked by TF-IDF)
  help            Show this message
  quit            Exit"""


# ------------------------------------------------------------------
# Command implementations
# ------------------------------------------------------------------


def cmd_build(indexer: Indexer) -> None:
    """Crawl the target site, build and save the inverted index."""
    print(f"Crawling {BASE_URL}")
    print(f"Politeness window: 6 seconds between requests.")
    print("This will take several minutes. Press Ctrl+C to abort.\n")

    crawler = WebCrawler(BASE_URL)
    t_start = time.perf_counter()

    try:
        pages = crawler.crawl()
    except KeyboardInterrupt:
        print("\nCrawl interrupted.")
        return

    if not pages:
        print("No pages were retrieved. Check your network connection.")
        return

    print(f"\nCrawled {len(pages)} pages. Building index...")

    # Reset indexer state before rebuilding
    indexer.__init__()
    indexer.base_url = BASE_URL

    for url, html in pages.items():
        indexer.index_page(url, html)
        print(f"  indexed: {url}")

    print("\nComputing TF-IDF scores...")
    indexer.compute_tfidf()

    indexer.save(INDEX_PATH)
    elapsed = time.perf_counter() - t_start

    print(
        f"\nDone in {elapsed:.1f}s."
        f" {indexer.total_docs} documents, {indexer.total_terms} unique terms."
        f"\nIndex saved to '{INDEX_PATH}'."
    )


def cmd_load(indexer: Indexer) -> None:
    """Load the index from INDEX_PATH."""
    if not Path(INDEX_PATH).exists():
        print(f"Index file not found: '{INDEX_PATH}'. Run 'build' first.")
        return
    try:
        indexer.load(INDEX_PATH)
        print(
            f"Loaded '{INDEX_PATH}':"
            f" {indexer.total_docs} documents, {indexer.total_terms} terms."
        )
    except Exception as exc:
        print(f"Failed to load index: {exc}")


def cmd_print(indexer: Indexer, word: str) -> None:
    """Print the inverted list for a single word."""
    if indexer.total_docs == 0:
        print("Index is empty. Run 'build' or 'load' first.")
        return
    SearchEngine(indexer).print_word(word)


def cmd_find(indexer: Indexer, words: list[str]) -> None:
    """Find and display pages containing all given words, ranked."""
    if indexer.total_docs == 0:
        print("Index is empty. Run 'build' or 'load' first.")
        return
    if not words:
        print("Usage: find <word> [word...]")
        return

    results = SearchEngine(indexer).find(words)
    query_str = " ".join(words)

    if not results:
        print(f"No pages found containing: {query_str}")
        return

    print(f"\nFound {len(results)} page(s) for '{query_str}' (ranked by TF-IDF):\n")
    for rank, (url, score) in enumerate(results, start=1):
        print(f"  {rank}. {url}  (score: {score:.6f})")
    print()


# ------------------------------------------------------------------
# REPL
# ------------------------------------------------------------------


def run() -> None:
    """Start the interactive command shell."""
    indexer = Indexer()
    print("Search Engine Tool  |  type 'help' for commands\n")

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        parts = raw.split()
        cmd, args = parts[0].lower(), parts[1:]

        if cmd == "build":
            cmd_build(indexer)
        elif cmd == "load":
            cmd_load(indexer)
        elif cmd == "print":
            if not args:
                print("Usage: print <word>")
            else:
                cmd_print(indexer, args[0])
        elif cmd == "find":
            cmd_find(indexer, args)
        elif cmd in ("help", "?"):
            print(HELP_TEXT)
        elif cmd in ("quit", "exit", "q"):
            break
        else:
            print(f"Unknown command '{cmd}'. Type 'help' for commands.")


if __name__ == "__main__":
    run()
