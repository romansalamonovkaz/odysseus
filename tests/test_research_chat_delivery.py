"""Finished Deep Research reports are posted, in full, into the chat that asked for them."""

import pytest

from routes.research.research_routes import _make_chat_delivery


class _Sess:
    def __init__(self, owner="alice", model="m1"):
        self.owner, self.model, self.messages = owner, model, []

    def add_message(self, msg):
        self.messages.append(msg)


class _SM:
    def __init__(self, sess=None):
        self.sess, self.saved = sess, 0

    def get_session(self, sid):
        if self.sess is None:
            raise KeyError(sid)
        return self.sess

    def save_sessions(self):
        self.saved += 1


REPORT = "# Report\n\n" + "full body paragraph. " * 200


def test_posts_full_report_with_sources():
    s = _Sess()
    sm = _SM(s)
    _make_chat_delivery(sm, "chat-1", "alice")("rp-1", REPORT, [{"url": "https://a.example"}], [])
    assert len(s.messages) == 1
    msg = s.messages[0]
    assert msg.role == "assistant" and msg.content == REPORT.strip() or REPORT.strip() in msg.content
    assert len(msg.content) >= len(REPORT.strip()) - 5  # a copy, not a summary
    assert msg.metadata["research"] is True and msg.metadata["research_session_id"] == "rp-1"
    assert msg.metadata["research_sources"] == [{"url": "https://a.example"}]
    assert sm.saved == 1


def test_empty_result_posts_nothing():
    s = _Sess()
    _make_chat_delivery(_SM(s), "chat-1", "alice")("rp-1", "   ", [], [])
    assert s.messages == []


def test_other_users_chat_is_refused():
    s = _Sess(owner="bob")
    sm = _SM(s)
    _make_chat_delivery(sm, "chat-1", "alice")("rp-1", REPORT, [], [])
    assert s.messages == [] and sm.saved == 0


def test_missing_chat_session_does_not_raise():
    _make_chat_delivery(_SM(None), "gone", "alice")("rp-1", REPORT, [], [])


def test_auth_disabled_empty_user_still_delivers():
    s = _Sess(owner="")
    _make_chat_delivery(_SM(s), "chat-1", "")("rp-1", REPORT, [], [])
    assert len(s.messages) == 1


@pytest.mark.asyncio
async def test_trigger_research_sends_chat_session_id(monkeypatch):
    import httpx
    from src.tools.research import do_trigger_research
    seen = {}

    class _R:
        status_code = 200
        text = ""

        def json(self):
            return {"session_id": "rp-9"}

    class _C:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, json=None, headers=None):
            seen["json"] = json
            return _R()

    monkeypatch.setattr(httpx, "AsyncClient", _C)
    out = await do_trigger_research('{"topic": "t"}', owner="alice", session_id="chat-7")
    assert seen["json"]["chat_session_id"] == "chat-7"
    assert "FULL report is posted into this chat" in out["output"]

    out = await do_trigger_research('{"topic": "t"}', owner="alice")
    assert "chat_session_id" not in seen["json"] and "sidebar" in out["output"]


class _Msg:
    def __init__(self, role, content):
        self.role, self.content = role, content


def test_verbatim_user_request_is_appended():
    from routes.research.research_routes import _with_verbatim_user_request
    s = _Sess()
    s.history = [_Msg("user", "проведи исследование про олесю добровольскую"),
                 _Msg("assistant", "```trigger_research ...```")]
    out = _with_verbatim_user_request(_SM(s), "chat-1", "alice", "БАДы Олеся Доброланская")
    assert out.startswith("БАДы Олеся Доброланская")
    assert "«проведи исследование про олесю добровольскую»" in out
    assert "не исправлять" in out


def test_verbatim_skipped_for_other_owner_missing_chat_or_duplicate():
    from routes.research.research_routes import _with_verbatim_user_request
    s = _Sess(owner="bob")
    s.history = [_Msg("user", "x y z")]
    assert _with_verbatim_user_request(_SM(s), "c", "alice", "t") == "t"
    assert _with_verbatim_user_request(_SM(None), "c", "alice", "t") == "t"
    s2 = _Sess()
    s2.history = [_Msg("user", "тема")]
    assert _with_verbatim_user_request(_SM(s2), "c", "alice", "тема") == "тема"


def _capture_email(monkeypatch):
    import threading, httpx
    sent = []

    class _R:
        status_code = 200

    monkeypatch.setattr(httpx, "post", lambda url, json=None, headers=None, timeout=None: sent.append((url, json, headers)) or _R())

    class _SyncThread:
        def __init__(self, target=None, name=None, daemon=None):
            self._t = target

        def start(self):
            self._t()

    monkeypatch.setattr(threading, "Thread", _SyncThread)
    return sent


def test_report_is_emailed_when_configured(monkeypatch):
    sent = _capture_email(monkeypatch)
    monkeypatch.setenv("RESEARCH_EMAIL_TO", "me@example.com")
    monkeypatch.setenv("MAIL_RELAY_URL", "https://relay.example/mail-relay")
    monkeypatch.setenv("MAIL_RELAY_SECRET", "s3")
    s = _Sess()
    topic = "БАДы Добровольская\n\nДословный запрос пользователя: «...»"
    _make_chat_delivery(_SM(s), "chat-1", "alice", topic)("rp-1", "# Отчёт\n\n| a | b |\n|---|---|\n| 1 | 2 |",
                                                          [{"url": "https://src.example", "title": "Src"}], [])
    assert len(sent) == 1
    url, payload, headers = sent[0]
    assert url == "https://relay.example/mail-relay" and headers == {"x-relay-secret": "s3"}
    assert payload["to"] == "me@example.com"
    assert payload["subject"] == "Deep Research: БАДы Добровольская"
    assert "<table>" in payload["body"] and 'href="https://src.example"' in payload["body"]


def test_no_email_without_config_or_when_not_posted(monkeypatch):
    sent = _capture_email(monkeypatch)
    for k in ("RESEARCH_EMAIL_TO", "MAIL_RELAY_URL", "MAIL_RELAY_SECRET"):
        monkeypatch.delenv(k, raising=False)
    _make_chat_delivery(_SM(_Sess()), "chat-1", "alice", "t")("rp-1", REPORT, [], [])
    assert sent == []
    monkeypatch.setenv("RESEARCH_EMAIL_TO", "me@example.com")
    monkeypatch.setenv("MAIL_RELAY_URL", "https://relay.example/mail-relay")
    monkeypatch.setenv("MAIL_RELAY_SECRET", "s3")
    _make_chat_delivery(_SM(_Sess(owner="bob")), "chat-1", "alice", "t")("rp-1", REPORT, [], [])
    assert sent == []  # another user's chat: not posted, not emailed
