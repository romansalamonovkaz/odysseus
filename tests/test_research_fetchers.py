"""Deep Research page fetch chain: crawl4ai -> Firecrawl -> built-in."""

import src.research_fetchers as rf
import src.search as search_mod

LONG = "x" * (rf.MIN_CONTENT_CHARS + 10)


def _setup(monkeypatch, crawl4ai="http://c4a:8080", firecrawl="fc-key", mode="auto"):
    calls = []
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
