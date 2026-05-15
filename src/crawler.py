"""Web crawler with BFS traversal and politeness policy."""

import logging
import time
from collections import deque
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

POLITENESS_SECONDS = 6.0
REQUEST_TIMEOUT = 15


class WebCrawler:
    """
    BFS web crawler constrained to a single domain.

    Complexity: O(P * L) where P = pages crawled, L = links per page.
    Space:      O(P) for visited set and frontier.
    """

    def __init__(
        self, base_url: str, politeness: float = POLITENESS_SECONDS
    ) -> None:
        """
        Args:
            base_url:    Seed URL and domain boundary.
            politeness:  Minimum seconds between successive HTTP requests.
        """
        self.base_url = self._normalize(base_url) or base_url
        self.politeness = politeness
        self._visited: set[str] = set()
        self._frontier: deque[str] = deque()
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "COMP3011-Crawler/1.0"})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def crawl(self) -> dict[str, str]:
        """
        Crawl all in-scope pages reachable from base_url.

        Returns:
            Mapping of {url: html_content} for every successfully fetched page.
        """
        self._frontier.append(self.base_url)

        pages: dict[str, str] = {}
        while self._frontier:
            url = self._frontier.popleft()
            if url in self._visited:
                continue

            logger.info("Fetching %s", url)
            html = self._fetch(url)
            if html is None:
                self._visited.add(url)
                continue

            self._visited.add(url)
            pages[url] = html

            for link in self._extract_links(html, url):
                if link not in self._visited:
                    self._frontier.append(link)

            time.sleep(self.politeness)

        return pages

    @property
    def visited_count(self) -> int:
        """Number of URLs processed (including failed fetches)."""
        return len(self._visited)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch(self, url: str) -> str | None:
        """Fetch a URL. Returns HTML text or None on any error."""
        try:
            response = self._session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.text
        except requests.exceptions.HTTPError as exc:
            logger.warning("HTTP %s for %s", exc.response.status_code, url)
        except requests.exceptions.ConnectionError:
            logger.warning("Connection error for %s", url)
        except requests.exceptions.Timeout:
            logger.warning("Timeout for %s", url)
        except requests.exceptions.RequestException as exc:
            logger.warning("Request failed for %s: %s", url, exc)
        return None

    def _extract_links(self, html: str, base_url: str) -> list[str]:
        """
        Extract and normalise all in-scope href links from a page.

        Args:
            html:     Raw HTML content.
            base_url: Used to resolve relative hrefs.

        Returns:
            List of absolute, normalised, in-scope URLs.
        """
        soup = BeautifulSoup(html, "html.parser")
        links: list[str] = []
        for tag in soup.find_all("a", href=True):
            absolute = urljoin(base_url, tag["href"])
            normalised = self._normalize(absolute)
            if normalised and self._in_scope(normalised):
                links.append(normalised)
        return links

    def _normalize(self, url: str) -> str | None:
        """
        Normalise a URL by removing fragments and enforcing http/https.

        Returns None for unsupported schemes (mailto, javascript, etc.).
        """
        try:
            parsed = urlparse(url)
        except ValueError:
            return None
        if parsed.scheme not in ("http", "https"):
            return None
        return urlunparse(parsed._replace(fragment=""))

    def _in_scope(self, url: str) -> bool:
        """Return True if url shares the same netloc as base_url."""
        return urlparse(url).netloc == urlparse(self.base_url).netloc
