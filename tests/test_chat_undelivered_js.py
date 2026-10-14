"""Run the «question never reached the server» helpers under Node (2026-10-14)."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parents[1]
_MODULE = (_REPO / "static" / "js" / "chatUndelivered.js").as_uri()


def _run(script: str) -> dict:
    if not shutil.which("node"):
        pytest.skip("node is not installed")
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_is_undelivered_only_before_headers_and_not_for_user_stop():
    out = _run(f"""
      import {{ isUndeliveredSend }} from {json.dumps(_MODULE)};
      const base = {{ headersSeen: false, accumulated: '', text: 'вопрос', isBackground: false,
                      aborted: true, abortReason: 'recovery', recoverableError: false }};
      console.log(JSON.stringify({{
        recovery: isUndeliveredSend(base),
        staleLocal: isUndeliveredSend({{ ...base, abortReason: 'stale-local' }}),
        timeout: isUndeliveredSend({{ ...base, abortReason: 'timeout' }}),
        userStop: isUndeliveredSend({{ ...base, abortReason: 'user-stop' }}),
        noReason: isUndeliveredSend({{ ...base, abortReason: '' }}),
        headersSeen: isUndeliveredSend({{ ...base, headersSeen: true }}),
        streamed: isUndeliveredSend({{ ...base, accumulated: 'часть ответа' }}),
        background: isUndeliveredSend({{ ...base, isBackground: true }}),
        emptyText: isUndeliveredSend({{ ...base, text: '   ' }}),
        networkError: isUndeliveredSend({{ ...base, aborted: false, recoverableError: true }}),
        otherError: isUndeliveredSend({{ ...base, aborted: false, recoverableError: false }}),
      }}));
    """)
    assert out == {
        "recovery": True, "staleLocal": True, "timeout": True,
        "userStop": False, "noReason": False, "headersSeen": False,
        "streamed": False, "background": False, "emptyText": False,
        "networkError": True, "otherError": False,
    }


def test_question_reached_server_checks_active_run_then_last_user_message():
    out = _run(f"""
      import {{ questionReachedServer }} from {json.dumps(_MODULE)};
      const resp = (status, body) => ({{ ok: status >= 200 && status < 300, status, json: async () => body }});
      const fake = (statusCode, history) => async (url) =>
        url.includes('/stream_status/') ? resp(statusCode, {{}}) : resp(200, {{ history }});
      const hist = [{{ role: 'user', content: 'старый' }}, {{ role: 'assistant', content: 'ответ' }},
                    {{ role: 'user', content: 'новый  вопрос\\n' }}];
      const r = {{}};
      r.activeRun = await questionReachedServer(fake(200, []), '', 's1', 'x');
      r.savedSame = await questionReachedServer(fake(404, hist), '', 's1', 'новый вопрос');
      r.notSaved = await questionReachedServer(fake(404, hist.slice(0, 2)), '', 's1', 'новый вопрос');
      r.emptyHistory = await questionReachedServer(fake(404, []), '', 's1', 'новый вопрос');
      try {{ await questionReachedServer(fake(502, []), '', 's1', 'x'); r.statusError = 'no-throw'; }}
      catch (e) {{ r.statusError = 'threw'; }}
      try {{ await questionReachedServer(async () => {{ throw new TypeError('NetworkError'); }}, '', 's1', 'x'); r.netError = 'no-throw'; }}
      catch (e) {{ r.netError = 'threw'; }}
      console.log(JSON.stringify(r));
    """)
    assert out == {
        "activeRun": True, "savedSame": True, "notSaved": False, "emptyHistory": False,
        "statusError": "threw", "netError": "threw",
    }


def test_chat_js_wires_the_helpers_into_the_send_catch():
    src = (_REPO / "static" / "js" / "chat.js").read_text(encoding="utf-8")
    assert "from './chatUndelivered.js'" in src
    assert "_postHeadersSeen = true;" in src
    # the check must run before the auto-recover / interrupted renderers
    assert src.index("if (isUndeliveredSend({") < src.index("isRecoverableStreamError(err) && _tryAutoRecover(")
