"""Unit and integration tests for the WebCrawler."""

import sys
from unittest.mock import Mock, patch, MagicMock

import pytest
import requests

sys.path.insert(0, "src")
from crawler import WebCrawler  # noqa: E402

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

BASE = "https://quotes.toscrape.com/"

SIMPLE_HTML = """<html><body>
<a href="/page/2/">Next</a>
<a href="/author/Einstein/">Einstein</a>
<a href="https://external.com/">External</a>
</body></html>"""

LEAF_HTML = "<html><body><p>No more links here.</p></body></html>"


@pytest.fixture()
def crawler():
    return WebCrawler(BASE, politeness=0)


# ------------------------------------------------------------------
# Initialisation
# ------------------------------------------------------------------


def test_init_sets_base_url(crawler):
    assert crawler.base_url == BASE


def test_init_politeness_stored(crawler):
    assert crawler.politeness == 0


def test_init_visited_empty(crawler):
    assert crawler.visited_count == 0


# ------------------------------------------------------------------
# URL normalisation
# ------------------------------------------------------------------


def test_normalize_removes_fragment(crawler):
    result = crawler._normalize("https://quotes.toscrape.com/page/1/#top")
    assert "#" not in result


def test_normalize_rejects_mailto(crawler):
    assert crawler._normalize("mailto:user@example.com") is None


def test_normalize_handles_value_error(crawler):
    # urlparse raises ValueError for some malformed URLs on some Python builds;
    # _normalize must return None rather than propagate the exception.
    with patch("crawler.urlparse", side_effect=ValueError("bad url")):
        assert crawler._normalize("not-a-url") is None


def test_normalize_rejects_javascript(crawler):
    assert crawler._normalize("javascript:void(0)") is None


def test_normalize_preserves_https(crawler):
    url = "https://quotes.toscrape.com/"
    assert crawler._normalize(url) == url


def test_normalize_preserves_path(crawler):
    url = "https://quotes.toscrape.com/page/2/"
    assert crawler._normalize(url) == url


# ------------------------------------------------------------------
# Domain scoping
# ------------------------------------------------------------------


def test_in_scope_same_domain(crawler):
    assert crawler._in_scope("https://quotes.toscrape.com/page/2/") is True


def test_in_scope_rejects_external(crawler):
    assert crawler._in_scope("https://external.com/") is False


def test_in_scope_rejects_subdomain(crawler):
    assert crawler._in_scope("https://sub.quotes.toscrape.com/") is False


# ------------------------------------------------------------------
# Link extraction
# ------------------------------------------------------------------


def test_extract_links_returns_absolute(crawler):
    links = crawler._extract_links(SIMPLE_HTML, BASE)
    assert "https://quotes.toscrape.com/page/2/" in links


def test_extract_links_resolves_relative(crawler):
    links = crawler._extract_links(SIMPLE_HTML, BASE)
    assert "https://quotes.toscrape.com/author/Einstein/" in links


def test_extract_links_excludes_external(crawler):
    links = crawler._extract_links(SIMPLE_HTML, BASE)
    assert "https://external.com/" not in links


def test_extract_links_empty_page(crawler):
    assert crawler._extract_links(LEAF_HTML, BASE) == []


# ------------------------------------------------------------------
# HTTP fetch
# ------------------------------------------------------------------


@patch("crawler.requests.Session.get")
def test_fetch_success_returns_text(mock_get, crawler):
    mock_response = Mock()
    mock_response.text = "<html></html>"
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response
    assert crawler._fetch(BASE) == "<html></html>"


@patch("crawler.requests.Session.get")
def test_fetch_http_error_returns_none(mock_get, crawler):
    mock_response = Mock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        response=mock_response
    )
    mock_get.return_value = mock_response
    assert crawler._fetch(BASE) is None


@patch("crawler.requests.Session.get")
def test_fetch_connection_error_returns_none(mock_get, crawler):
    mock_get.side_effect = requests.exceptions.ConnectionError()
    assert crawler._fetch(BASE) is None


@patch("crawler.requests.Session.get")
def test_fetch_timeout_returns_none(mock_get, crawler):
    mock_get.side_effect = requests.exceptions.Timeout()
    assert crawler._fetch(BASE) is None


# ------------------------------------------------------------------
# Full crawl behaviour (mocked network)
# ------------------------------------------------------------------


@patch("crawler.time.sleep")
@patch("crawler.requests.Session.get")
def test_crawl_visits_linked_pages(mock_get, mock_sleep, crawler):
    page2_html = "<html><body><p>Page 2</p></body></html>"

    def fake_get(url, **kwargs):
        resp = Mock()
        resp.raise_for_status = Mock()
        if url == BASE:
            resp.text = SIMPLE_HTML
        elif url == "https://quotes.toscrape.com/page/2/":
            resp.text = page2_html
        elif url == "https://quotes.toscrape.com/author/Einstein/":
            resp.text = LEAF_HTML
        else:
            resp.text = LEAF_HTML
        return resp

    mock_get.side_effect = fake_get
    pages = crawler.crawl()

    assert BASE in pages
    assert "https://quotes.toscrape.com/page/2/" in pages


@patch("crawler.time.sleep")
@patch("crawler.requests.Session.get")
def test_crawl_does_not_revisit(mock_get, mock_sleep, crawler):
    """Each URL must be fetched at most once even if linked multiple times."""
    html_with_self_link = (
        f'<html><body><a href="{BASE}">Self</a></body></html>'
    )
    resp = Mock()
    resp.raise_for_status = Mock()
    resp.text = html_with_self_link
    mock_get.return_value = resp

    crawler.crawl()
    fetch_calls = [c for c in mock_get.call_args_list if c.args[0] == BASE]
    assert len(fetch_calls) == 1


@patch("crawler.time.sleep")
@patch("crawler.requests.Session.get")
def test_crawl_respects_politeness(mock_get, mock_sleep, crawler):
    """time.sleep must be called once per successfully fetched page."""
    resp = Mock()
    resp.raise_for_status = Mock()
    resp.text = LEAF_HTML
    mock_get.return_value = resp

    crawler.crawl()
    assert mock_sleep.called


@patch("crawler.time.sleep")
@patch("crawler.requests.Session.get")
def test_crawl_excludes_failed_pages(mock_get, mock_sleep, crawler):
    """Pages that fail HTTP should not appear in result dict."""
    resp = Mock()
    mock_resp_obj = Mock(status_code=500)
    resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
        response=mock_resp_obj
    )
    mock_get.return_value = resp

    pages = crawler.crawl()
    assert pages == {}


@patch("crawler.time.sleep")
@patch("crawler.requests.Session.get")
def test_crawl_visited_count_after_crawl(mock_get, mock_sleep, crawler):
    resp = Mock()
    resp.raise_for_status = Mock()
    resp.text = LEAF_HTML
    mock_get.return_value = resp

    crawler.crawl()
    assert crawler.visited_count >= 1
