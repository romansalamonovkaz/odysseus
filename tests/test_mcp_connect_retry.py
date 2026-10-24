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
    """Fake the transports (not connect_server), so the real connect_server
    wiring — including its cancel of a pending retry — is exercised. Each
    connect attempt pops the next outcome: True / False / "hang"."""
    calls = []
    monkeypatch.setattr(mm, "SessionLocal", lambda: _FakeDB(rows))

    async def attempt(server_id, name):
        calls.append(server_id)
        outcome = outcomes.pop(0) if outcomes else False
        if outcome == "hang":
            await asyncio.sleep(3600)
        if outcome is True:
            session = object()
            manager._sessions[server_id] = session
            manager._connections[server_id] = {"status": "connected", "name": name}
            return True
        manager._connections[server_id] = {"status": "error", "error": "boom", "name": name}
        return False

    async def fake_stdio(server_id, name, command, args, env):
        return await attempt(server_id, name)

    async def fake_sse(server_id, name, url):
        return await attempt(server_id, name)

    async def fake_http(server_id, name, url, wait=8.0):
        return await attempt(server_id, name)

    monkeypatch.setattr(manager, "_connect_stdio", fake_stdio)
    monkeypatch.setattr(manager, "_connect_sse", fake_sse)
    monkeypatch.setattr(manager, "_start_http_connect", fake_http)
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


@pytest.mark.asyncio
async def test_manual_reconnect_during_hung_startup_connect_makes_no_retry(monkeypatch, fast_retries):
    """Review r1: startup attempt hangs on npx, the user reconnects successfully,
    then the startup attempt times out — it must neither overwrite "connected"
    nor schedule a retry that would open a second session."""
    real_wait_for = asyncio.wait_for

    async def short_wait_for(awaitable, timeout):
        return await real_wait_for(awaitable, timeout=min(timeout, 0.2))

    monkeypatch.setattr(mm.asyncio, "wait_for", short_wait_for)
    manager = mm.McpManager()
    calls = _install(monkeypatch, manager, [_row()], ["hang", True, True])

    startup = asyncio.create_task(manager.connect_all_enabled())
    await asyncio.sleep(0.05)
    assert await manager.connect_server("fc", "Firecrawl", "stdio", "npx", [], {}) is True
    session_b = manager._sessions["fc"]
    await startup
    await asyncio.sleep(0.1)

    assert calls == ["fc", "fc"]
    assert manager._connections["fc"]["status"] == "connected"
    assert manager._sessions["fc"] is session_b
    assert "fc" not in manager._connect_tasks


@pytest.mark.asyncio
async def test_disconnect_all_waits_for_cancelled_connects(monkeypatch):
    monkeypatch.setattr(mm, "_MCP_RETRY_SCHEDULE", ((0.01, 5.0), (0.01, 5.0)))
    manager = mm.McpManager()
    calls = _install(monkeypatch, manager, [_row()], [False, "hang"])

    await manager.connect_all_enabled()
    retry = manager._connect_tasks["fc"]
    await asyncio.sleep(0.05)  # the retry is now inside its hung connect attempt
    assert calls == ["fc", "fc"]
    await manager.disconnect_all()

    assert retry.done()
    assert manager._connect_tasks == {}


def _fake_mcp_stdio(monkeypatch, gates):
    """Real _connect_stdio over a fake stdio_client/ClientSession. Connect #n
    waits on gates[n] (if any) while opening; log records opened/closed n."""
    import mcp
    import mcp.client.stdio as stdio_mod

    log = {"opened": [], "closed": []}

    class _Client:
        def __init__(self, _params):
            log["opened"].append(len(log["opened"]) + 1)
            self.n = log["opened"][-1]

        async def __aenter__(self):
            if self.n in gates:
                await gates[self.n].wait()
            return (self.n, self.n)

        async def __aexit__(self, *_exc):
            log["closed"].append(self.n)
            return False

    class _Session:
        def __init__(self, read_stream, _write_stream):
            self.n = read_stream

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def initialize(self):
            pass

        async def list_tools(self):
            return SimpleNamespace(tools=[])

    monkeypatch.setattr(stdio_mod, "stdio_client", _Client)
    monkeypatch.setattr(mcp, "ClientSession", _Session)
    return log


@pytest.mark.asyncio
async def test_overlapping_connects_keep_one_session_and_close_the_other(monkeypatch):
    manager = mm.McpManager()
    gates = {1: asyncio.Event()}
    log = _fake_mcp_stdio(monkeypatch, gates)

    slow = asyncio.create_task(manager.connect_server("fc", "Firecrawl", "stdio", "npx", [], {}))
    await asyncio.sleep(0.01)
    assert await manager.connect_server("fc", "Firecrawl", "stdio", "npx", [], {}) is True
    gates[1].set()
    assert await slow is True

    assert manager._sessions["fc"].n == 2
    assert list(manager._stacks) == ["fc"]
    assert log["closed"] == [1]
    await manager.disconnect_all()
    assert log["closed"] == [1, 2]


@pytest.mark.asyncio
async def test_retry_racing_an_unfinished_manual_reconnect_orphans_no_session(monkeypatch, fast_retries):
    """Review r2: startup attempt hangs; a manual Reconnect starts and is still
    running when the startup attempt times out, so a retry is scheduled and
    connects first. The manual connect must close its own session, not
    overwrite the retry's and orphan it."""
    real_wait_for = asyncio.wait_for

    async def short_wait_for(awaitable, timeout):
        return await real_wait_for(awaitable, timeout=min(timeout, 0.2))

    monkeypatch.setattr(mm.asyncio, "wait_for", short_wait_for)
    monkeypatch.setattr(mm, "SessionLocal", lambda: _FakeDB([_row()]))
    manager = mm.McpManager()
    gates = {1: asyncio.Event(), 2: asyncio.Event()}  # 1 never opens; 2 = manual, released later
    log = _fake_mcp_stdio(monkeypatch, gates)

    startup = asyncio.create_task(manager.connect_all_enabled())
    await asyncio.sleep(0.05)
    manual = asyncio.create_task(manager.connect_server("fc", "Firecrawl", "stdio", "npx", [], {}))
    await startup  # times out with the manual connect still in flight → retry scheduled
    await asyncio.sleep(0.1)  # the retry (connect #3) has connected
    assert manager._sessions["fc"].n == 3
    gates[2].set()
    assert await manual is True

    assert log["opened"] == [1, 2, 3]
    assert manager._sessions["fc"].n == 3
    assert list(manager._stacks) == ["fc"]
    assert log["closed"] == [2]
    assert manager._connections["fc"]["status"] == "connected"
    await manager.disconnect_all()
    assert log["closed"] == [2, 3]
