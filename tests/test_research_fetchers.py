"""Deep Research page fetch chain: CRW -> crawl4ai -> Firecrawl -> Jina -> built-in."""

import src.research_fetchers as rf
import src.search as search_mod

LONG = "x" * (rf.MIN_CONTENT_CHARS + 10)


def _setup(monkeypatch, crawl4ai="http://c4a:8080", firecrawl="fc-key", mode="auto", crw=None):
    calls = []
    # Keep the chain hermetic: no real network via Jina, CRW only when a test asks for it.
    monkeypatch.setenv("JINA_READER", "off")
    if crw:
        monkeypatch.setenv("CRW_URL", crw)
    else:
        monkeypatch.delenv("CRW_URL", raising=False)
    monkeypatch.setattr(rf, "_discover_backends", lambda: (crawl4ai, firecrawl))
    monkeypatch.setattr(rf, "_fetcher_mode", lambda: mode)
    monkeypatch.setattr(
        search_mod, "fetch_webpage_content",
        lambda url, timeout: calls.append("builtin") or {"success": True, "content": "b", "url": url},
    )
    import services.search.content as content_mod
    monkeypatch.setattr(content_mod, "_public_http_url", lambda url: True)
    return calls


def test_crawl4ai_first(monkeypatch):
    calls = _setup(monkeypatch)
    monkeypatch.setattr(rf, "_fetch_crawl4ai", lambda b, u, t: calls.append("c4a") or rf._page(u, LONG, via="crawl4ai"))
    monkeypatch.setattr(rf, "_fetch_firecrawl", lambda k, u, t: calls.append("fc") or None)
    page = rf.fetch_page_for_research("https://example.com/a")
    assert page["fetched_via"] == "crawl4ai"
    assert calls == ["c4a"]


def test_falls_back_to_firecrawl_then_builtin(monkeypatch):
    calls = _setup(monkeypatch)

    def boom(b, u, t):
        calls.append("c4a")
        raise RuntimeError("down")

    monkeypatch.setattr(rf, "_fetch_crawl4ai", boom)
    monkeypatch.setattr(rf, "_fetch_firecrawl", lambda k, u, t: calls.append("fc") or None)
    page = rf.fetch_page_for_research("https://example.com/a")
    assert page["content"] == "b"
    assert calls == ["c4a", "fc", "builtin"]


def test_pdf_and_builtin_mode_skip_renderers(monkeypatch):
    calls = _setup(monkeypatch)
    monkeypatch.setattr(rf, "_fetch_crawl4ai", lambda b, u, t: calls.append("c4a"))
    rf.fetch_page_for_research("https://example.com/plan.PDF")
    assert calls == ["builtin"]

    calls = _setup(monkeypatch, mode="builtin")
    monkeypatch.setattr(rf, "_fetch_crawl4ai", lambda b, u, t: calls.append("c4a"))
    rf.fetch_page_for_research("https://example.com/a")
    assert calls == ["builtin"]


def test_no_backends_uses_builtin(monkeypatch):
    calls = _setup(monkeypatch, crawl4ai=None, firecrawl=None)
    rf.fetch_page_for_research("https://example.com/a")
    assert calls == ["builtin"]


def test_crawl4ai_mention_enables_web_intent():
    from src.action_intents import classify_tool_intent
    intent = classify_tool_intent("используй crawl4ai: прочитай страницы курсов")
    assert "web" in str(intent).lower()


def test_crw_is_tried_first(monkeypatch):
    calls = _setup(monkeypatch, crw="http://crw:3000")
    monkeypatch.setattr(rf, "_fetch_crawl4ai", lambda b, u, t: calls.append("c4a") or None)
    monkeypatch.setattr(rf, "_fetch_crw", lambda b, u, t: calls.append("crw") or rf._page(u, LONG, via="crw"))
    monkeypatch.setattr(rf, "_fetch_firecrawl", lambda k, u, t: calls.append("fc") or None)
    page = rf.fetch_page_for_research("https://example.com/a")
    assert page["fetched_via"] == "crw"
    assert calls == ["crw"]


def test_thin_crw_falls_through_to_crawl4ai(monkeypatch):
    calls = _setup(monkeypatch, crw="http://crw:3000")
    monkeypatch.setattr(rf, "_fetch_crw", lambda b, u, t: calls.append("crw") or None)
    monkeypatch.setattr(rf, "_fetch_crawl4ai", lambda b, u, t: calls.append("c4a") or rf._page(u, LONG, via="crawl4ai"))
    monkeypatch.setattr(rf, "_fetch_firecrawl", lambda k, u, t: calls.append("fc") or None)
    page = rf.fetch_page_for_research("https://example.com/spa")
    assert page["fetched_via"] == "crawl4ai"
    assert calls == ["crw", "c4a"]


def test_crw_skipped_without_url(monkeypatch):
    calls = _setup(monkeypatch)
    monkeypatch.setattr(rf, "_fetch_crawl4ai", lambda b, u, t: None)
    monkeypatch.setattr(rf, "_fetch_crw", lambda b, u, t: calls.append("crw"))
    monkeypatch.setattr(rf, "_fetch_firecrawl", lambda k, u, t: calls.append("fc") or None)
    rf.fetch_page_for_research("https://example.com/a")
    assert calls == ["fc", "builtin"]


class _Resp:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


def test_fetch_crw_maps_response_and_sends_key(monkeypatch):
    monkeypatch.setenv("CRW_API_KEY", "k1")
    seen = {}

    def _post(url, **kw):
        seen.update(url=url, headers=kw["headers"], json=kw["json"])
        return _Resp({"success": True, "data": {
            "markdown": LONG, "metadata": {"title": "T", "ogImage": "https://i/x.png"}}})

    monkeypatch.setattr(rf.httpx, "post", _post)
    page = rf._fetch_crw("http://crw:3000/", "https://example.com/a", 30)
    assert page["fetched_via"] == "crw" and page["title"] == "T" and page["og_image"] == "https://i/x.png"
    assert seen["url"] == "http://crw:3000/v1/scrape"
    assert seen["headers"]["Authorization"] == "Bearer k1"
    assert seen["json"] == {"url": "https://example.com/a", "formats": ["markdown"]}


def test_fetch_crw_thin_or_failed_returns_none(monkeypatch):
    monkeypatch.delenv("CRW_API_KEY", raising=False)
    monkeypatch.setattr(rf.httpx, "post", lambda *a, **k: _Resp({"success": True, "data": {"markdown": "short"}}))
    assert rf._fetch_crw("http://crw:3000", "https://e.com", 5) is None
    monkeypatch.setattr(rf.httpx, "post", lambda *a, **k: _Resp({"success": False, "error": "x"}))
    assert rf._fetch_crw("http://crw:3000", "https://e.com", 5) is None
