"""Email copies of web-chat replies (src/mail_copy.py)."""

import threading

import httpx

from src import mail_copy


class _Msg:
    def __init__(self, role, content):
        self.role, self.content = role, content


class _Sess:
    def __init__(self, name="", history=None):
        self.name, self.history = name, history or []


def _capture(monkeypatch):
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


def _configure(monkeypatch):
    monkeypatch.setenv("CHAT_EMAIL_TO", "me@example.com")
    monkeypatch.setenv("MAIL_RELAY_URL", "https://relay.example/mail-relay")
    monkeypatch.setenv("MAIL_RELAY_SECRET", "s3")


def test_chat_reply_emailed_with_question(monkeypatch):
    sent = _capture(monkeypatch)
    _configure(monkeypatch)
    s = _Sess(name="БАДы и Рэйки", history=[_Msg("user", "кто такая <Олеся>?"), _Msg("assistant", "…")])
    assert mail_copy.email_chat_reply(s, "abcdef123456", "**Ответ** текст") is True
    url, payload, headers = sent[0]
    assert headers == {"x-relay-secret": "s3"} and payload["to"] == "me@example.com"
    assert payload["subject"] == "Odysseus: БАДы и Рэйки"
    assert "кто такая &lt;Олеся&gt;?" in payload["body"]  # question escaped
    assert "<strong>Ответ</strong>" in payload["body"]


def test_untitled_chat_uses_question_as_subject(monkeypatch):
    sent = _capture(monkeypatch)
    _configure(monkeypatch)
    mail_copy.email_chat_reply(_Sess(history=[_Msg("user", "что   такое\nрэйки")]), "s1", "ответ")
    assert sent[0][1]["subject"] == "Odysseus: что такое рэйки"


def test_nothing_sent_when_not_configured_or_empty(monkeypatch):
    sent = _capture(monkeypatch)
    for k in ("CHAT_EMAIL_TO", "MAIL_RELAY_URL", "MAIL_RELAY_SECRET", "TG_RELAY_URL"):
        monkeypatch.delenv(k, raising=False)
    assert mail_copy.email_chat_reply(_Sess(), "s1", "ответ") is False
    _configure(monkeypatch)
    assert mail_copy.email_chat_reply(_Sess(), "s1", "   ") is False
    assert sent == []


def test_relay_failure_does_not_raise(monkeypatch):
    _capture(monkeypatch)
    _configure(monkeypatch)

    def boom(*a, **k):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "post", boom)
    assert mail_copy.email_chat_reply(_Sess(), "s1", "ответ") is True  # queued; failure only logged


def test_save_assistant_response_emails_but_not_incognito(monkeypatch):
    from routes import chat_helpers
    calls = []
    monkeypatch.setattr(mail_copy, "email_chat_reply", lambda sess, sid, text: calls.append((sid, text)))
    monkeypatch.setattr(chat_helpers, "_append_incognito_message", lambda *a, **k: None, raising=False)

    class _S(_Sess):
        model = "m"

        def add_message(self, m):
            self.history.append(m)

    class _SM:
        def save_sessions(self):
            pass

    import core.database as cdb
    monkeypatch.setattr(cdb, "update_session_last_accessed", lambda sid: None)
    chat_helpers.save_assistant_response(_S(), _SM(), "sid-1", "обычный ответ", {})
    assert calls == [("sid-1", "обычный ответ")]
    calls.clear()
    chat_helpers.save_assistant_response(_S(), _SM(), "sid-2", "секрет", {}, incognito=True)
    assert calls == []


def test_chat_reply_also_goes_to_telegram(monkeypatch):
    sent = _capture(monkeypatch)
    _configure(monkeypatch)
    monkeypatch.setenv("TG_RELAY_URL", "https://relay.example/tg-relay")
    s = _Sess(name="Тема", history=[_Msg("user", "вопрос?")])
    mail_copy.email_chat_reply(s, "abcdef123456", "ответ")
    urls = [u for u, _, _ in sent]
    assert urls == ["https://relay.example/mail-relay", "https://relay.example/tg-relay"]
    tg = sent[1]
    assert tg[2] == {"x-relay-secret": "s3"}
    assert tg[1] == {"text": "💬 Odysseus: Тема\n\n❓ вопрос?\n\nответ"}


def test_telegram_only_when_relay_url_set(monkeypatch):
    sent = _capture(monkeypatch)
    monkeypatch.setenv("MAIL_RELAY_SECRET", "s3")
    monkeypatch.delenv("TG_RELAY_URL", raising=False)
    assert mail_copy.send_telegram_copy("x") is False
    monkeypatch.setenv("TG_RELAY_URL", "https://relay.example/tg-relay")
    assert mail_copy.send_telegram_copy("x") is True
    assert len(sent) == 1
