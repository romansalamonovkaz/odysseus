"""Explicit foreground Chat and Agent model-routing policy."""

from dataclasses import dataclass
from typing import Any, Collection, Dict, FrozenSet, Optional, Tuple

from src.endpoint_resolver import (
    _endpoint_enabled_models,
    endpoint_cost_tracked,
    resolve_fallback_entries,
    resolve_fallback_entries_with_descriptors,
    resolve_route_descriptor,
    resolve_route_descriptor_by_id,
)

_DEFAULT_FALLBACK_ENTRY_RESOLVER = resolve_fallback_entries


FOREGROUND_FALLBACK_ENABLED_KEY = "foreground_fallback_enabled"
FOREGROUND_FALLBACK_LIST_KEY = "foreground_model_fallbacks"
# Fork 2026-10-14: 402 (provider balance empty, e.g. polza.ai) is an availability failure of that
# route, not of the request — the same model through the next paid intermediary should answer.
FOREGROUND_AVAILABILITY_STATUSES: FrozenSet[int] = frozenset({
    402, 408, 425, 429, 500, 502, 503, 504, 507, 508, 529,
})
MAX_FOREGROUND_FALLBACKS = 10

# Fork 2026-10-25: explicit same-model backup routes. ``{"<primary endpoint_id>": ["<backup endpoint_id>", ...]}``
# in the owner's preferences makes the SAME model on a backup endpoint (e.g. polza.ai → AITUNNEL) the first
# fallback, before ``foreground_model_fallbacks``. Only explicitly mapped endpoints are used, never "any endpoint
# that lists the model" (that could route to a marked-up gateway). 401/403 (key blocked, model not allowed for
# this key) are a route problem when the same model has another route, so they become eligible then.
FOREGROUND_ROUTE_BACKUPS_KEY = "foreground_route_backups"
BACKUP_ROUTE_EXTRA_STATUSES: FrozenSet[int] = frozenset({401, 403})


@dataclass(frozen=True)
class ForegroundModelPolicy:
    """Resolved per-user foreground fallback policy."""

    enabled: bool = False
    fallback_candidates: Tuple[tuple, ...] = ()
    fallback_descriptors: Tuple[dict, ...] = ()
    eligible_statuses: FrozenSet[int] = FOREGROUND_AVAILABILITY_STATUSES
    fallback_on_empty: bool = False
    route_backups: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
    allowed_models: Optional[FrozenSet[str]] = None


@dataclass(frozen=True)
class ForegroundRoutes:
    """Ordered foreground candidates, their parallel descriptors, and the statuses that may advance."""

    candidates: list
    descriptors: list
    fallback_statuses: FrozenSet[int]


def _model_identity(model: Optional[str]) -> str:
    """Provider-neutral model id: lowercase, one leading ``vendor/`` dropped (openai/gpt-6-sol ≡ gpt-6-sol)."""

    m = (model or "").strip().lower()
    return m.split("/", 1)[1] if "/" in m else m


def _parse_route_backups(raw) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    if not isinstance(raw, dict):
        return ()
    out = []
    for primary, backups in raw.items():
        if not isinstance(primary, str) or not primary or not isinstance(backups, list):
            continue
        ids = tuple(dict.fromkeys(
            b for b in backups if isinstance(b, str) and b and b != primary
        ))[:MAX_FOREGROUND_FALLBACKS]
        if ids:
            out.append((primary, ids))
    return tuple(out)


def _load_policy_preferences(owner: Optional[str]) -> dict:
    """Load only preferences that explicitly belong to ``owner``.

    The generic preferences loader intentionally treats a legacy flat store as
    the single-user preferences object.  That compatibility must not cross an
    authentication transition: once a named owner is present, foreground
    fallback consent exists only in an actual ``_users[owner]`` dictionary.
    """

    from routes import prefs_routes

    if owner is None:
        prefs = prefs_routes._load_for_user(None)
        return dict(prefs) if isinstance(prefs, dict) else {}

    raw = prefs_routes._load()
    users = raw.get("_users") if isinstance(raw, dict) else None
    if not isinstance(users, dict):
        return {}
    prefs = users.get(owner)
    return dict(prefs) if isinstance(prefs, dict) else {}


def resolve_foreground_model_policy(
    owner: Optional[str] = None,
    allowed_models: Optional[Collection[str]] = None,
) -> ForegroundModelPolicy:
    """Resolve an explicit owner-scoped policy, failing closed to strict mode.

    The policy is stored in user preferences even when authentication is
    disabled. Historical ``default_model_fallbacks`` values are deliberately
    unrelated and are never read or migrated.
    """

    try:
        prefs = _load_policy_preferences(owner)
    except Exception:
        return ForegroundModelPolicy()

    if prefs.get(FOREGROUND_FALLBACK_ENABLED_KEY) is not True:
        return ForegroundModelPolicy()
    # Fork 2026-10-25: same-model backup routes ride on the same explicit consent.
    route_backups = _parse_route_backups(prefs.get(FOREGROUND_ROUTE_BACKUPS_KEY))
    allowed = frozenset(allowed_models) if allowed_models is not None else None

    def _no_list_policy() -> ForegroundModelPolicy:
        if not route_backups:
            return ForegroundModelPolicy()
        return ForegroundModelPolicy(enabled=True, route_backups=route_backups, allowed_models=allowed)

    entries = prefs.get(FOREGROUND_FALLBACK_LIST_KEY)
    if not isinstance(entries, list) or not entries:
        return _no_list_policy()
    if allowed_models is not None:
        entries = [
            entry for entry in entries
            if (
                isinstance(entry, dict)
                and isinstance(entry.get("model"), str)
                and entry.get("model") in allowed
            )
        ]
        if not entries:
            return _no_list_policy()
    entries = entries[:MAX_FOREGROUND_FALLBACKS]

    if resolve_fallback_entries is not _DEFAULT_FALLBACK_ENTRY_RESOLVER:
        # Preserve the long-standing resolver seam used by downstream tests and
        # integrations. Production uses the descriptor-aware resolver below.
        compatibility_candidates = resolve_fallback_entries(
            entries,
            owner=owner,
            require_exact_model=True,
        )
        # Known limitation of this test-only seam: alignment matches on model
        # alone, so when two entries share a model and the resolver skips the
        # first, the surviving candidate inherits the skipped entry's
        # endpoint_id. Production uses the descriptor-aware branch below,
        # which is unaffected.
        resolved_routes = []
        remaining_entries = list(entries)
        for candidate in compatibility_candidates:
            matching_index = next(
                (
                    index for index, entry in enumerate(remaining_entries)
                    if isinstance(entry, dict)
                    and entry.get("model") == candidate[1]
                ),
                None,
            )
            matching_entry = (
                remaining_entries.pop(matching_index)
                if matching_index is not None
                else {}
            )
            descriptor = {
                "endpoint_id": matching_entry.get("endpoint_id"),
                "endpoint_label": matching_entry.get("endpoint_id") or "Fallback route",
                "endpoint_cost_tracked": endpoint_cost_tracked(candidate[0]),
            }
            resolved_routes.append((candidate, descriptor))
    else:
        resolved_routes = resolve_fallback_entries_with_descriptors(
            entries,
            owner=owner,
            require_exact_model=True,
        )
    candidates = [candidate for candidate, _descriptor in resolved_routes]
    if not candidates:
        return _no_list_policy()

    return ForegroundModelPolicy(
        enabled=True,
        fallback_candidates=tuple(candidates),
        fallback_descriptors=tuple(
            dict(descriptor) for _candidate, descriptor in resolved_routes
        ),
        route_backups=route_backups,
        allowed_models=allowed,
    )


def resolve_foreground_fallback_candidates(owner: Optional[str] = None) -> list:
    """Return only candidates explicitly enabled by the current user."""

    return list(resolve_foreground_model_policy(owner).fallback_candidates)


def _backup_model_for(endpoint_id: str, identity: str) -> Optional[str]:
    """The backup endpoint's own id for ``identity`` (enabled, not hidden), or None."""

    from core.database import SessionLocal, ModelEndpoint

    db = SessionLocal()
    try:
        ep = db.query(ModelEndpoint).filter(
            ModelEndpoint.id == endpoint_id,
            ModelEndpoint.is_enabled == True,
        ).first()
        if not ep:
            return None
        for candidate in _endpoint_enabled_models(ep):
            if _model_identity(candidate) == identity:
                return candidate
        return None
    finally:
        db.close()


def _same_model_backup_routes(policy, primary_endpoint_id, model, owner) -> list:
    """(candidate, descriptor) pairs for the same model on the owner's mapped backup endpoints."""

    if not policy.route_backups or not primary_endpoint_id:
        return []
    backup_ids = dict(policy.route_backups).get(primary_endpoint_id, ())
    identity = _model_identity(model)
    if policy.allowed_models is not None and identity not in {
        _model_identity(m) for m in policy.allowed_models
    }:
        return []
    routes = []
    for backup_id in backup_ids:
        backup_model = _backup_model_for(backup_id, identity)
        if not backup_model:
            continue
        routes.extend(resolve_fallback_entries_with_descriptors(
            [{"endpoint_id": backup_id, "model": backup_model}],
            owner=owner,
            require_exact_model=True,
        ))
    return routes


def _selected_descriptor(endpoint_url, model, headers, owner, selected_endpoint_id):
    selected = None
    if selected_endpoint_id:
        selected = resolve_route_descriptor_by_id(
            selected_endpoint_id,
            endpoint_url,
            model,
            headers or {},
            owner=owner,
        )
    if selected is None:
        selected = resolve_route_descriptor(endpoint_url, model, headers or {}, owner=owner)
    return selected


def _build_routes(
    endpoint_url, model, headers, owner, policy, selected_endpoint_id, *, with_descriptors
) -> ForegroundRoutes:
    policy = policy or resolve_foreground_model_policy(owner)
    primary = (endpoint_url, model, headers or {})
    selected = None
    if with_descriptors or (policy.enabled and policy.route_backups):
        selected = _selected_descriptor(endpoint_url, model, headers, owner, selected_endpoint_id)
    candidates = [primary]
    descriptors = [selected]
    statuses = policy.eligible_statuses
    if policy.enabled:
        primary_id = (selected or {}).get("endpoint_id") or selected_endpoint_id
        for candidate, descriptor in _same_model_backup_routes(policy, primary_id, model, owner):
            if candidate in candidates:
                continue
            if len(candidates) > MAX_FOREGROUND_FALLBACKS:
                break
            candidates.append(candidate)
            descriptors.append(dict(descriptor))
            statuses = statuses | BACKUP_ROUTE_EXTRA_STATUSES
    for index, candidate in enumerate(policy.fallback_candidates):
        if candidate in candidates:
            continue
        if len(candidates) > MAX_FOREGROUND_FALLBACKS:
            break  # one cap for backups + list together; backups come first and win
        candidates.append(candidate)
        if with_descriptors:
            descriptors.append(
                dict(policy.fallback_descriptors[index])
                if index < len(policy.fallback_descriptors)
                else {
                    "endpoint_id": None,
                    "endpoint_label": "Fallback route",
                    "endpoint_cost_tracked": endpoint_cost_tracked(candidate[0]),
                }
            )
    return ForegroundRoutes(
        candidates=candidates,
        descriptors=descriptors if with_descriptors else [],
        fallback_statuses=frozenset(statuses),
    )


def build_foreground_routes(
    endpoint_url: str,
    model: str,
    headers: Optional[Dict[str, Any]] = None,
    owner: Optional[str] = None,
    policy: Optional[ForegroundModelPolicy] = None,
    selected_endpoint_id: Optional[str] = None,
) -> ForegroundRoutes:
    """Candidates, parallel descriptors and eligible statuses for one foreground request (one source of truth)."""

    return _build_routes(
        endpoint_url, model, headers, owner, policy, selected_endpoint_id, with_descriptors=True
    )


def build_foreground_model_candidates(
    endpoint_url: str,
    model: str,
    headers: Optional[Dict[str, Any]] = None,
    owner: Optional[str] = None,
    policy: Optional[ForegroundModelPolicy] = None,
    selected_endpoint_id: Optional[str] = None,
) -> list:
    """Build the ordered candidate list for a foreground request."""

    return _build_routes(
        endpoint_url, model, headers, owner, policy, selected_endpoint_id, with_descriptors=False
    ).candidates


def build_foreground_route_descriptors(
    endpoint_url: str,
    model: str,
    headers: Optional[Dict[str, Any]] = None,
    owner: Optional[str] = None,
    policy: Optional[ForegroundModelPolicy] = None,
    selected_endpoint_id: Optional[str] = None,
) -> list:
    """Build safe route metadata parallel to foreground candidates."""

    return build_foreground_routes(
        endpoint_url, model, headers, owner=owner, policy=policy, selected_endpoint_id=selected_endpoint_id
    ).descriptors
