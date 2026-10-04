"""POST /api/search must not block the event loop (2026-10-04): the sync search
ran inline in the async route and froze every other request for 3-15 s."""
import asyncio
import inspect

from routes.search import search_routes


def test_do_web_search_runs_search_in_a_thread():
    src = inspect.getsource(search_routes.setup_search_routes)
    body = src[src.index("async def do_web_search"):src.index("async def list_search_providers")]
    assert "asyncio.to_thread(" in body and "comprehensive_web_search," in body
    assert asyncio.to_thread  # stdlib
