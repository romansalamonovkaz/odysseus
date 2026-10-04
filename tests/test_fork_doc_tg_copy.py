"""Fork 2026-10-05: documents the agent writes in the side window are copied to Telegram."""
from routes import chat_routes


def test_copy_streamed_docs_sends_title_and_body(monkeypatch):
    sent = []
    import src.mail_copy as mc
    monkeypatch.setattr(mc, "send_telegram_copy", lambda text, tag="": sent.append((text, tag)) or True)
    chat_routes._copy_streamed_docs([{"title": "Куда копать", "content": "Карта входов в аналитику"},
                                     {"title": "", "content": "   "}], "abcdef123456")
    assert len(sent) == 1
    assert sent[0][0].startswith("📄 Odysseus — документ «Куда копать»") and "Карта входов" in sent[0][0]
    assert sent[0][1] == "doc-abcdef12"


def test_copy_errors_never_raise(monkeypatch):
    import src.mail_copy as mc
    def boom(*a, **k):
        raise RuntimeError("relay down")
    monkeypatch.setattr(mc, "send_telegram_copy", boom)
    chat_routes._copy_streamed_docs([{"title": "t", "content": "x"}], "s")
