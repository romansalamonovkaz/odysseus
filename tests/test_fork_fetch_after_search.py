"""Fork 2026-10-04: after a web search, opening a found page (web_fetch) is not gated
while the run has read no private data; anything else stays gated as upstream."""
from src.tool_capabilities import ToolRunSecurityContext

SEARCH_RESULT = {"results": [{"url": "https://ex.ru", "content": "текст страницы"}], "exit_code": 0}


def _after_search(ctx):
    ctx.observe_tool_result("web_search", SEARCH_RESULT, {"query": "q"})
    assert ctx.external_untrusted_context_seen


def test_fetch_allowed_after_search_without_private_reads():
    ctx = ToolRunSecurityContext()
    _after_search(ctx)
    assert ctx.decision_for("web_fetch", {"url": "https://ex.ru/a"}).allowed


def test_fetch_gated_once_private_data_was_read():
    ctx = ToolRunSecurityContext()
    ctx.observe_tool_result("list_emails", {"emails": [{"subject": "x"}], "exit_code": 0}, {})
    _after_search(ctx)
    assert ctx.private_context_seen
    assert not ctx.decision_for("web_fetch", {"url": "https://ex.ru/a"}).allowed


def test_side_effects_still_gated_after_search():
    ctx = ToolRunSecurityContext()
    _after_search(ctx)
    assert not ctx.decision_for("send_email", {"to": "a@b.c", "body": "x"}).allowed
