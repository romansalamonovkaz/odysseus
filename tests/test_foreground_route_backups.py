"""Fork 2026-10-25: explicit same-model backup routes (polza.ai → AITUNNEL) for foreground Chat and Agent."""

import asyncio
import json
from types import SimpleNamespace

import pytest

import routes.chat_routes as chat_routes
import src.foreground_model_routing as fmr
import src.llm_core as llm_core
from src.foreground_model_routing import (
    BACKUP_ROUTE_EXTRA_STATUSES,
    FOREGROUND_AVAILABILITY_STATUSES,
    MAX_FOREGROUND_FALLBACKS,
    ForegroundModelPolicy,
    build_foreground_model_candidates,
    build_foreground_route_descriptors,
    build_foreground_routes,
    resolve_foreground_model_policy,
)
from tests.test_foreground_model_routing import _RouteRequest, _chat_stream_endpoint
from tests.test_llm_core_fallback import _POLZA_EMPTY_BALANCE, _run_provider_stream

POLZA = ("https://polza.ai/api/v1/chat/completions", "openai/gpt-6-sol", {"Authorization": "Bearer p"})
AIT = ("https://api.aitunnel.ru/v1/chat/completions", "gpt-6-sol", {"Authorization": "Bearer a"})
NOVITA = ("https://api.novita.ai/v3/openai/chat/completions", "deepseek/deepseek-v4.1-flash", {"Authorization": "Bearer n"})


def _desc(ep):
    return {"endpoint_id": ep, "endpoint_label": ep, "endpoint_cost_tracked": True}


@pytest.fixture
def backups(monkeypatch):
    """Primary route is endpoint 'polza'; 'ait' lists gpt-6-sol; resolution is recorded."""

    seen = {"lookups": [], "resolved": []}
    monkeypatch.setattr(fmr, "resolve_route_descriptor", lambda url, model, headers=None, owner=None: _desc("polza"))
    monkeypatch.setattr(fmr, "resolve_route_descriptor_by_id", lambda *a, **k: None)

    def backup_model_for(endpoint_id, identity):
        seen["lookups"].append((endpoint_id, identity))
        return {"ait": {"gpt-6-sol": "gpt-6-sol"}}.get(endpoint_id, {}).get(identity)

    def resolve(entries, owner=None, *, require_exact_model=False):
        seen["resolved"].append((entries, owner, require_exact_model))
        assert entries == [{"endpoint_id": "ait", "model": "gpt-6-sol"}]
        return [(AIT, _desc("ait"))]

    monkeypatch.setattr(fmr, "_backup_model_for", backup_model_for)
    monkeypatch.setattr(fmr, "resolve_fallback_entries_with_descriptors", resolve)
    return seen


def _policy(**kw):
    base = dict(
        enabled=True,
        fallback_candidates=(NOVITA,),
        fallback_descriptors=(_desc("novita"),),
        route_backups=(("polza", ("ait",)),),
    )
    base.update(kw)
    return ForegroundModelPolicy(**base)


def test_model_identity_drops_one_vendor_segment():
    assert fmr._model_identity("openai/gpt-6-sol") == fmr._model_identity("GPT-6-Sol") == "gpt-6-sol"
    assert fmr._model_identity("google/gemini-4-argon") == "gemini-4-argon"


def test_backup_inserted_after_primary_before_list_with_401_403(backups):
    routes = build_foreground_routes(*POLZA, owner="admin", policy=_policy())
    assert routes.candidates == [POLZA, AIT, NOVITA]
    assert [d["endpoint_id"] for d in routes.descriptors] == ["polza", "ait", "novita"]
    assert routes.fallback_statuses == FOREGROUND_AVAILABILITY_STATUSES | BACKUP_ROUTE_EXTRA_STATUSES
    assert {401, 403} <= routes.fallback_statuses
    assert backups["resolved"][0][1:] == ("admin", True)
    # the thin wrappers return the same lists
    assert build_foreground_model_candidates(*POLZA, owner="admin", policy=_policy()) == routes.candidates
    assert build_foreground_route_descriptors(*POLZA, owner="admin", policy=_policy()) == routes.descriptors


def test_unmapped_primary_is_unchanged(backups, monkeypatch):
    monkeypatch.setattr(fmr, "resolve_route_descriptor", lambda *a, **k: _desc("novita"))
    routes = build_foreground_routes(*NOVITA, policy=_policy(fallback_candidates=(), fallback_descriptors=()))
    assert routes.candidates == [NOVITA]
    assert routes.fallback_statuses == FOREGROUND_AVAILABILITY_STATUSES
    assert backups["lookups"] == []


def test_backup_without_the_model_is_skipped(backups):
    other = ("https://polza.ai/api/v1/chat/completions", "qwen/qwen3.8-27b", {})
    routes = build_foreground_routes(*other, policy=_policy())
    assert routes.candidates == [other, NOVITA]
    assert routes.fallback_statuses == FOREGROUND_AVAILABILITY_STATUSES


def test_allowed_models_excludes_backup(backups):
    routes = build_foreground_routes(*POLZA, policy=_policy(allowed_models=frozenset({"deepseek/deepseek-v4.1-flash"})))
    assert routes.candidates == [POLZA, NOVITA]
    assert backups["resolved"] == []
    allowed = build_foreground_routes(*POLZA, policy=_policy(allowed_models=frozenset({"openai/gpt-6-sol"})))
    assert allowed.candidates == [POLZA, AIT, NOVITA]


def test_disabled_policy_ignores_backups(backups):
    routes = build_foreground_routes(*POLZA, policy=ForegroundModelPolicy(route_backups=(("polza", ("ait",)),)))
    assert routes.candidates == [POLZA]
    assert routes.fallback_statuses == FOREGROUND_AVAILABILITY_STATUSES


def test_backup_equal_to_list_candidate_is_not_duplicated(backups):
    routes = build_foreground_routes(*POLZA, policy=_policy(fallback_candidates=(AIT, NOVITA),
                                                           fallback_descriptors=(_desc("ait"), _desc("novita"))))
    assert routes.candidates == [POLZA, AIT, NOVITA]
    assert len(routes.descriptors) == 3


@pytest.mark.parametrize("prefs, expected_backups, enabled", [
    ({"foreground_fallback_enabled": True, "foreground_route_backups": {"polza": ["ait"]}}, (("polza", ("ait",)),), True),
    ({"foreground_route_backups": {"polza": ["ait"]}}, (), False),  # no consent → strict
    ({"foreground_fallback_enabled": True, "foreground_route_backups": {"polza": "ait", "": ["x"], "p": ["p"]}}, (), False),
])
def test_policy_parses_backups_under_explicit_consent(monkeypatch, prefs, expected_backups, enabled):
    monkeypatch.setattr(fmr, "_load_policy_preferences", lambda owner=None: prefs)
    policy = resolve_foreground_model_policy("admin")
    assert policy.enabled is enabled
    assert policy.route_backups == expected_backups


def test_policy_keeps_list_and_backups_together(monkeypatch):
    monkeypatch.setattr(fmr, "_load_policy_preferences", lambda owner=None: {
        "foreground_fallback_enabled": True,
        "foreground_model_fallbacks": [{"endpoint_id": "novita", "model": "deepseek/deepseek-v4.1-flash"}],
        "foreground_route_backups": {"polza": ["ait"]},
    })
    monkeypatch.setattr(fmr, "resolve_fallback_entries_with_descriptors",
                        lambda entries, owner=None, require_exact_model=False: [(NOVITA, _desc("novita"))])
    policy = resolve_foreground_model_policy("admin", allowed_models={"openai/gpt-6-sol"})
    assert policy.fallback_candidates == ()  # novita model not in allowlist
    assert policy.enabled and policy.route_backups == (("polza", ("ait",)),)
    assert policy.allowed_models == frozenset({"openai/gpt-6-sol"})


def test_backup_model_for_reads_enabled_unhidden_models(monkeypatch):
    ep = SimpleNamespace(cached_models=json.dumps(["gpt-6-sol", "gemini-4-argon"]), pinned_models=None,
                         hidden_models=json.dumps(["gemini-4-argon"]))

    class Q:
        def filter(self, *a, **k):
            return self

        def first(self):
            return ep

    class Db:
        def query(self, *a, **k):
            return Q()

        def close(self):
            pass

    import core.database as database
    monkeypatch.setattr(database, "SessionLocal", Db)
    assert fmr._backup_model_for("ait", "gpt-6-sol") == "gpt-6-sol"
    assert fmr._backup_model_for("ait", "gemini-4-argon") is None


def test_stream_fallback_reaches_backup_on_401_only_with_backup_statuses(backups, monkeypatch):
    async def fake_stream_llm(url, model, messages, headers=None, **kwargs):
        if url == POLZA[0]:
            yield 'event: error\ndata: {"error": "key blocked", "status": 401}\n\n'
            return
        yield 'data: {"delta": "Париж"}\n\n'
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(llm_core, "stream_llm", fake_stream_llm)
    routes = build_foreground_routes(*POLZA, policy=_policy())

    async def run(statuses):
        return [c async for c in llm_core.stream_llm_with_fallback(
            routes.candidates, [{"role": "user", "content": "?"}],
            fallback_statuses=statuses, fallback_on_empty=False,
            candidate_route_descriptors=routes.descriptors,
        )]

    with_backup = "".join(asyncio.run(run(routes.fallback_statuses)))
    assert "Париж" in with_backup and with_backup.count("Париж") == 1
    without = "".join(asyncio.run(run(FOREGROUND_AVAILABILITY_STATUSES)))
    assert "Париж" not in without


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["chat", "agent"])
async def test_chat_and_agent_routes_pass_backup_statuses(monkeypatch, mode, backups):
    captured = {}
    endpoint = _chat_stream_endpoint(monkeypatch, mode, captured)
    monkeypatch.setattr(fmr, "_load_policy_preferences", lambda owner=None: {
        "foreground_fallback_enabled": True,
        "foreground_route_backups": {"polza": ["ait"]},
    })
    monkeypatch.setattr(fmr, "_backup_model_for", lambda endpoint_id, identity: "gpt-6-sol")
    monkeypatch.setattr(fmr, "resolve_fallback_entries_with_descriptors",
                        lambda entries, owner=None, require_exact_model=False: [(AIT, _desc("ait"))])

    async def chat_stream(candidates, messages, **kwargs):
        captured["chat"] = (candidates, kwargs["fallback_statuses"])
        yield 'data: {"delta": "ok"}\n\n'
        yield "data: [DONE]\n\n"

    async def agent_stream(endpoint_url, model, messages, **kwargs):
        captured["agent"] = (kwargs.get("fallbacks"), kwargs["fallback_statuses"])
        yield 'data: {"delta": "ok"}\n\n'
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(chat_routes, "stream_llm_with_fallback", chat_stream)
    monkeypatch.setattr(chat_routes, "stream_agent_loop", agent_stream)
    response = await endpoint(_RouteRequest(mode))
    async for _ in response.body_iterator:
        pass
    got_candidates, statuses = captured[mode]
    assert AIT in got_candidates
    assert {401, 403} <= set(statuses)


def test_total_fallbacks_stay_capped_and_backup_wins(backups):
    many = tuple((f"https://list{i}.example/v1", f"m{i}", {}) for i in range(MAX_FOREGROUND_FALLBACKS))
    routes = build_foreground_routes(*POLZA, policy=_policy(
        fallback_candidates=many, fallback_descriptors=tuple(_desc(f"l{i}") for i in range(len(many)))))
    assert len(routes.candidates) == 1 + MAX_FOREGROUND_FALLBACKS
    assert len(routes.descriptors) == len(routes.candidates)
    assert routes.candidates[1] == AIT


def test_parsed_backup_ids_are_deduped_and_capped(monkeypatch):
    ids = ["ait", "ait"] + [f"b{i}" for i in range(MAX_FOREGROUND_FALLBACKS + 5)]
    monkeypatch.setattr(fmr, "_load_policy_preferences", lambda owner=None: {
        "foreground_fallback_enabled": True, "foreground_route_backups": {"polza": ids}})
    (primary, parsed), = resolve_foreground_model_policy("admin").route_backups
    assert parsed[0] == "ait" and parsed.count("ait") == 1 and len(parsed) == MAX_FOREGROUND_FALLBACKS


def test_polza_in_stream_balance_error_hands_turn_to_aitunnel(backups, monkeypatch):
    """polza's real adapter output for its 200-stream balance event (status 400 → 402) reaches the backup."""
    polza_chunks = _run_provider_stream(
        monkeypatch, POLZA[0], ["data: " + json.dumps({"error": _POLZA_EMPTY_BALANCE, "status": 400})])

    async def fake_stream_llm(url, model, messages, headers=None, **kwargs):
        if url == POLZA[0]:
            for c in polza_chunks:
                yield c
            return
        yield 'data: {"delta": "Париж"}\n\n'
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(llm_core, "stream_llm", fake_stream_llm)
    routes = build_foreground_routes(*POLZA, policy=_policy())
    for statuses in (routes.fallback_statuses, FOREGROUND_AVAILABILITY_STATUSES):
        async def run():
            return [c async for c in llm_core.stream_llm_with_fallback(
                routes.candidates, [{"role": "user", "content": "?"}],
                fallback_statuses=statuses, fallback_on_empty=False,
                candidate_route_descriptors=routes.descriptors,
            )]
        out = "".join(asyncio.run(run()))
        assert out.count("Париж") == 1, statuses
