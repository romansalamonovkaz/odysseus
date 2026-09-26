"""Page fetching for Deep Research: CRW -> crawl4ai -> Firecrawl -> Jina Reader -> built-in fetcher.

The built-in fetcher (``src.search.fetch_webpage_content``) is plain httpx +
BeautifulSoup: it cannot run JavaScript, so SPA / JS-heavy pages come back
thin or empty. When a crawl4ai server or a Firecrawl key is configured (as an
MCP server row, or via env), Deep Research renders HTML pages through them
first and only falls back to the built-in fetcher.

PDFs always go to the built-in fetcher — it extracts PDF text locally and
keeps Firecrawl credits for the pages that need a browser.

Configuration (all optional):
  settings ``research_fetcher``: "auto" (default) or "builtin" to disable.
  env ``CRAWL4AI_URL``      — base URL of a crawl4ai server (overrides MCP row).
  env ``FIRECRAWL_API_KEY`` — Firecrawl key (overrides MCP row).
  env ``CRW_URL``           — base URL of a self-hosted CRW server (fastcrw/crw);
                              ``CRW_API_KEY`` is sent as a Bearer token if set.
  env ``JINA_API_KEY``      — optional Jina Reader key (higher rate limits).
                              Jina Reader also works keyless, so it is tried
                              unless ``JINA_READER=off``.
"""

import json
import logging
import os
import time
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

# Below this many characters a rendered page is treated as a failed render
# (login wall, cookie banner only, empty SPA shell) and the next fetcher runs.
MIN_CONTENT_CHARS = 300

_DISCOVERY_TTL = 60.0
_discovery_cache: Tuple[float, Optional[str], Optional[str]] = (0.0, None, None)


def _discover_backends() -> Tuple[Optional[str], Optional[str]]:
    """Return (crawl4ai_base_url, firecrawl_api_key), cached for a minute."""
    global _discovery_cache
    now = time.monotonic()
    if now - _discovery_cache[0] < _DISCOVERY_TTL:
        return _discovery_cache[1], _discovery_cache[2]

    crawl4ai_url = (os.environ.get("CRAWL4AI_URL") or "").strip() or None
    firecrawl_key = (os.environ.get("FIRECRAWL_API_KEY") or "").strip() or None

    if not crawl4ai_url or not firecrawl_key:
        try:
            from src.database import SessionLocal
            from sqlalchemy import text

            db = SessionLocal()
            try:
                rows = db.execute(text(
                    "SELECT name, url, env FROM mcp_servers WHERE is_enabled = 1"
                )).mappings().all()
            finally:
                db.close()
            for row in rows:
                name = (row.get("name") or "").lower()
                if not crawl4ai_url and "crawl4ai" in name and row.get("url"):
                    parsed = urlparse(row["url"])
                    crawl4ai_url = f"{parsed.scheme}://{parsed.netloc}"
                if not firecrawl_key and "firecrawl" in name and row.get("env"):
                    try:
                        env = json.loads(row["env"]) or {}
                    except (TypeError, ValueError):
                        env = {}
                    firecrawl_key = (env.get("FIRECRAWL_API_KEY") or "").strip() or None
        except Exception as e:
            logger.debug(f"research fetcher discovery failed: {e}")

    _discovery_cache = (now, crawl4ai_url, firecrawl_key)
    return crawl4ai_url, firecrawl_key


def _fetcher_mode() -> str:
    try:
        from src.settings import get_setting
        return (get_setting("research_fetcher", "auto") or "auto").strip().lower()
    except Exception:
        return "auto"


def _is_pdf_url(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".pdf")


def _page(url: str, content: str, title: str = "", via: str = "") -> Dict:
    return {"success": True, "url": url, "content": content,
            "title": title, "og_image": "", "fetched_via": via}


def _fetch_crawl4ai(base: str, url: str, timeout: int) -> Optional[Dict]:
    resp = httpx.post(f"{base.rstrip('/')}/md",
                      json={"url": url, "f": "fit"}, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    content = (data.get("markdown") or "").strip()
    if data.get("success") is False or len(content) < MIN_CONTENT_CHARS:
        return None
    return _page(url, content, via="crawl4ai")


def _fetch_firecrawl(api_key: str, url: str, timeout: int) -> Optional[Dict]:
    resp = httpx.post(
        "https://api.firecrawl.dev/v2/scrape",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"url": url, "formats": ["markdown"], "onlyMainContent": True,
              "removeBase64Images": True},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json().get("data") or {}
    content = (data.get("markdown") or "").strip()
    if len(content) < MIN_CONTENT_CHARS:
        return None
    meta = data.get("metadata") or {}
    page = _page(url, content, title=meta.get("title") or "", via="firecrawl")
    page["og_image"] = meta.get("ogImage") or ""
    return page


def _fetch_crw(base: str, url: str, timeout: int) -> Optional[Dict]:
    """Self-hosted CRW (fastcrw/crw, Firecrawl-style): POST /v1/scrape -> data.markdown."""
    headers = {}
    key = (os.environ.get("CRW_API_KEY") or "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    resp = httpx.post(f"{base.rstrip('/')}/v1/scrape", headers=headers,
                      json={"url": url, "formats": ["markdown"]}, timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    data = body.get("data") or {}
    content = (data.get("markdown") or "").strip()
    if body.get("success") is False or len(content) < MIN_CONTENT_CHARS:
        return None
    meta = data.get("metadata") or {}
    page = _page(url, content, title=meta.get("title") or "", via="crw")
    page["og_image"] = meta.get("ogImage") or ""
    return page


def _fetch_jina(url: str, timeout: int) -> Optional[Dict]:
    """Jina Reader (r.jina.ai): URL -> markdown. Works keyless; a key raises limits."""
    headers = {"Accept": "text/plain", "X-Return-Format": "markdown"}
    key = (os.environ.get("JINA_API_KEY") or "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    resp = httpx.get(f"https://r.jina.ai/{url}", headers=headers,
                     timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    text = resp.text.strip()
    title = ""
    if text.startswith("Title:"):
        first, _, rest = text.partition("\n")
        title = first[len("Title:"):].strip()
        marker = "Markdown Content:"
        idx = rest.find(marker)
        text = rest[idx + len(marker):].strip() if idx != -1 else rest.strip()
    if len(text) < MIN_CONTENT_CHARS:
        return None
    return _page(url, text, title=title, via="jina")


def fetch_page_for_research(url: str, timeout: int = 10) -> Dict:
    """Fetch ``url`` for Deep Research. Same result shape as fetch_webpage_content."""
    from src.search import fetch_webpage_content

    if _fetcher_mode() == "builtin" or _is_pdf_url(url):
        return fetch_webpage_content(url, timeout)

    try:
        from services.search.content import _public_http_url
        public = _public_http_url(url)
    except Exception:
        public = urlparse(url).scheme in ("http", "https")
    if not public:
        return fetch_webpage_content(url, timeout)

    crawl4ai_url, firecrawl_key = _discover_backends()
    # Rendering a JS page takes longer than a plain GET.
    render_timeout = max(timeout, 45)

    # Order: CRW first (tiny, fast, static pages), crawl4ai (real browser, JS) picks up the
    # pages CRW returns thin, then metered Firecrawl, Jina as the last resort.
    crw_url = (os.environ.get("CRW_URL") or "").strip() or None
    backends = [("crw", crw_url), ("crawl4ai", crawl4ai_url), ("firecrawl", firecrawl_key)]
    if (os.environ.get("JINA_READER") or "on").strip().lower() != "off":
        backends.append(("jina", "on"))

    for name, base in backends:
        if not base:
            continue
        try:
            if name == "jina":
                page = _fetch_jina(url, render_timeout)
            elif name == "crawl4ai":
                page = _fetch_crawl4ai(base, url, render_timeout)
            elif name == "crw":
                page = _fetch_crw(base, url, render_timeout)
            else:
                page = _fetch_firecrawl(base, url, render_timeout)
            if page:
                logger.info(f"Research fetch via {name}: {url} ({len(page['content'])} chars)")
                return page
            logger.info(f"Research fetch via {name} returned thin content for {url}")
        except Exception as e:
            logger.warning(f"Research fetch via {name} failed for {url}: {e}")

    return fetch_webpage_content(url, timeout)
