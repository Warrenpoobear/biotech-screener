"""Unit tests for research-only Firecrawl adapter (SDK v2)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tools.firecrawl_research_ingest import (
    FirecrawlResearchAdapter,
    _document_from_scrape,
    _iter_search_hits,
    _search_hit_to_result,
)


def test_iter_search_hits_combines_web_and_news():
    data = MagicMock()
    web = MagicMock(url="https://a.example", title="A", description="da")
    news = MagicMock(url="https://b.example", title="B", description="db")
    data.web = [web]
    data.news = [news]
    data.images = None

    hits = list(_iter_search_hits(data))
    assert hits == [web, news]


def test_search_hit_to_result_from_search_result_web():
    hit = MagicMock(url="https://x.com", title="T", description="D", score=None)
    result = _search_hit_to_result(hit)
    assert result.url == "https://x.com"
    assert result.title == "T"
    assert result.description == "D"


def test_document_from_scrape_success():
    doc = MagicMock()
    doc.markdown = "# body"
    doc.warning = None
    doc.metadata = MagicMock(title="Page", description="Sum", error=None)
    title, description, markdown, err = _document_from_scrape(doc)
    assert err is None
    assert markdown == "# body"
    assert title == "Page"
    assert description == "Sum"


def test_document_from_scrape_metadata_error():
    doc = MagicMock()
    doc.markdown = ""
    doc.metadata = MagicMock(title=None, description=None, error="blocked")
    title, description, markdown, err = _document_from_scrape(doc)
    assert err == "blocked"


@patch.dict("os.environ", {"FIRECRAWL_API_KEY": "fc-test"}, clear=False)  # pragma: allowlist secret
def test_search_delegates_to_v2_client():
    adapter = FirecrawlResearchAdapter(api_key="fc-test")
    mock_web = MagicMock(url="https://statnews.com/x", title="Story", description=None)
    search_data = MagicMock(web=[mock_web], news=[], images=[])

    adapter.client.search = MagicMock(return_value=search_data)
    results = adapter.search("biotech trial", limit=5)

    adapter.client.search.assert_called_once_with("biotech trial", limit=5)
    assert len(results) == 1
    assert results[0].url == "https://statnews.com/x"


@patch.dict("os.environ", {}, clear=True)
def test_adapter_requires_api_key():
    with pytest.raises(ValueError, match="FIRECRAWL_API_KEY"):
        FirecrawlResearchAdapter(api_key=None)
