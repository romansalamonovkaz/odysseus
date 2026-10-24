"""Fork 2026-10-24: background retry of user stdio MCP servers whose startup
connect failed, and readable connect errors for ExceptionGroups."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from src import mcp_manager as mm


def _row(server_id="fc", transport="stdio", enabled=True):
    return SimpleNamespace(
        id=server_id,
        name="Firecrawl",
        transport=transport,
        command="npx",
        args=json.dumps(["-y", "firecrawl-mcp@3.34.1"]),
        env=json.dumps({}),
        url="http://example.invalid/mcp" if transport != "stdio" else None,
        is_enabled=enabled,
    )


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *_a, **_k):
        return self

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return list(self._rows)


class _FakeDB:
    def __init__(self, rows):
        self._rows = rows

    def query(self, *_a, **_k):
        return _FakeQuery(self._rows)

    def close(self):
        pass


@pytest.fixture
def fast_retries(monkeypatch):
    monkeypatch.setattr(mm, "_MCP_RETRY_SCHEDULE", ((0.01, 1.0), (0.01, 1.0)))


def _install(monkeypatch, manager, rows, outcomes):
    """connect_server fake: pops the next outcome (True/False/"hang") per call."""
    calls = []
    monkeypatch.setattr(mm, "SessionLocal", lambda: _FakeDB(rows))

    async def fake_connect_server(server_id, name, transport, command=None, args=None, env=None, url=None):
        if transport == "stdio":
            manager._cancel_pending_retry(server_id)
        calls.append(server_id)
        outcome = outcomes.pop(0) if outcomes else False
        if outcome == "hang":
            await asyncio.sleep(3600)
        if outcome is True:
            manager._connections[server_id] = {"status": "connected", "name": name}
            manager._sessions[server_id] = object()
        else:
            manager._connections[server_id] = {"status": "error", "error": "boom", "name": name}
        return outcome is True

    monkeypatch.setattr(manager, "connect_server", fake_connect_server)
    return calls


async def _drain(manager, server_id):
    task = manager._connect_tasks.get(server_id)
    if task is not None:
        await asyncio.wait_for(asyncio.shield(task), timeout=5)


@pytest.mark.asyncio
async def test_failed_startup_connect_is_retried_until_connected(monkeypatch, fast_retries):
    manager = mm.McpManager()
    calls = _install(monkeypatch, manager, [_row()], [False, False, True])

    await manager.connect_all_enabled()
    assert "fc" in manager._connect_tasks  # retry scheduled, not awaited
    await _drain(manager, "fc")

    assert calls == ["fc", "fc", "fc"]
    assert manager._connections["fc"]["status"] == "connected"
    assert "fc" not in manager._connect_tasks


@pytest.mark.asyncio
async def test_successful_startup_connect_schedules_no_retry(monkeypatch, fast_retries):
    manager = mm.McpManager()
    calls = _install(monkeypatch, manager, [_row()], [True])

    await manager.connect_all_enabled()

    assert calls == ["fc"]
    assert "fc" not in manager._connect_tasks


@pytest.mark.asyncio
async def test_retries_stop_after_schedule_is_exhausted(monkeypatch, fast_retries):
    manager = mm.McpManager()
    calls = _install(monkeypatch, manager, [_row()], [False, False, False, True])

    await manager.connect_all_enabled()
    await _drain(manager, "fc")

    assert calls == ["fc"] * (1 + len(mm._MCP_RETRY_SCHEDULE))
    assert manager._connections["fc"]["status"] == "error"


@pytest.mark.asyncio
async def test_retry_attempt_timeout_is_recorded_and_next_attempt_runs(monkeypatch):
    monkeypatch.setattr(mm, "_MCP_RETRY_SCHEDULE", ((0.01, 0.05), (0.01, 1.0)))
    manager = mm.McpManager()
    calls = _install(monkeypatch, manager, [_row()], [False, "hang", True])

    await manager.connect_all_enabled()
    await _drain(manager, "fc")

    assert calls == ["fc", "fc", "fc"]
    assert manager._connections["fc"]["status"] == "connected"


@pytest.mark.asyncio
async def test_manual_reconnect_during_retry_cancels_it_single_session(monkeypatch):
    monkeypatch.setattr(mm, "_MCP_RETRY_SCHEDULE", ((0.3, 1.0), (0.3, 1.0)))
    manager = mm.McpManager()
    calls = _install(monkeypatch, manager, [_row()], [False, True])

    await manager.connect_all_enabled()
    retry = manager._connect_tasks["fc"]
    # A manual reconnect while the retry is sleeping.
    assert await manager.connect_server("fc", "Firecrawl", "stdio", "npx", [], {}) is True
    await asyncio.sleep(0.4)

    assert retry.cancelled()
    assert calls == ["fc", "fc"]  # the retry never made its own attempt
    assert "fc" not in manager._connect_tasks


@pytest.mark.asyncio
async def test_disconnect_during_retry_stops_further_attempts(monkeypatch):
    monkeypatch.setattr(mm, "_MCP_RETRY_SCHEDULE", ((0.3, 1.0), (0.3, 1.0)))
    manager = mm.McpManager()
    calls = _install(monkeypatch, manager, [_row()], [False, True])

    await manager.connect_all_enabled()
    retry = manager._connect_tasks["fc"]
    await manager.disconnect_server("fc")
    await asyncio.sleep(0.4)

    assert retry.cancelled()
    assert calls == ["fc"]


@pytest.mark.asyncio
async def test_disabled_or_deleted_row_is_not_retried(monkeypatch, fast_retries):
    manager = mm.McpManager()
    rows = [_row()]
    calls = _install(monkeypatch, manager, rows, [False, True])

    await manager.connect_all_enabled()
    rows[0] = _row(enabled=False)  # disabled in the meantime
    await _drain(manager, "fc")
    assert calls == ["fc"]

    manager2 = mm.McpManager()
    rows2 = [_row()]
    calls2 = _install(monkeypatch, manager2, rows2, [False, True])
    await manager2.connect_all_enabled()
    rows2.clear()  # deleted in the meantime
    await _drain(manager2, "fc")
    assert calls2 == ["fc"]


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["sse", "http"])
async def test_non_stdio_servers_are_never_retried(monkeypatch, fast_retries, transport):
    manager = mm.McpManager()
    calls = _install(monkeypatch, manager, [_row(transport=transport)], [False, True])

    await manager.connect_all_enabled()
    await asyncio.sleep(0.05)

    assert calls == ["fc"]
    assert "fc" not in manager._connect_tasks


@pytest.mark.asyncio
async def test_disconnect_all_cancels_pending_retry_of_never_connected_server(monkeypatch):
    monkeypatch.setattr(mm, "_MCP_RETRY_SCHEDULE", ((0.3, 1.0), (0.3, 1.0)))
    manager = mm.McpManager()
    calls = _install(monkeypatch, manager, [_row()], [False, True])

    await manager.connect_all_enabled()
    retry = manager._connect_tasks["fc"]
    await manager.disconnect_all()
    await asyncio.sleep(0.4)

    assert retry.cancelled()
    assert manager._connect_tasks == {}
    assert calls == ["fc"]


def test_describe_connect_error_flattens_exception_groups():
    group = ExceptionGroup(
        "unhandled errors in a TaskGroup",
        [ExceptionGroup("inner", [ConnectionError("Connection closed")]), ValueError("bad json")],
    )
    text = mm._describe_connect_error(group)
    assert "ConnectionError: Connection closed" in text
    assert "ValueError: bad json" in text
    assert "TaskGroup" not in text


def test_describe_connect_error_plain_and_empty_messages():
    # Plain exceptions keep their message unchanged (test_mcp_manager relies on it).
    assert mm._describe_connect_error(RuntimeError("x")) == "x"
    assert mm._describe_connect_error(TimeoutError()) == "TimeoutError"
    assert mm._describe_connect_error(ExceptionGroup("g", [TimeoutError()])) == "TimeoutError"


@pytest.mark.asyncio
async def test_connect_server_stores_and_logs_the_real_cause(monkeypatch, caplog):
    manager = mm.McpManager()

    async def failing_stdio(*_a, **_k):
        raise ExceptionGroup("unhandled errors in a TaskGroup", [ConnectionError("npm exited 1")])

    monkeypatch.setattr(manager, "_connect_stdio", failing_stdio)
    with caplog.at_level("ERROR"):
        ok = await manager.connect_server("fc", "Firecrawl", "stdio", "npx", ["-y", "firecrawl-mcp"], {})

    assert ok is False
    assert "ConnectionError: npm exited 1" in manager._connections["fc"]["error"]
    assert "ConnectionError: npm exited 1" in caplog.text
