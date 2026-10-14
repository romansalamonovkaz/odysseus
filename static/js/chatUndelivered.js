// A send whose POST never produced response headers may never have reached the
// server (2026-10-14: Waterfox froze a background tab; the client showed
// "Streaming was interrupted" although no POST /api/chat_stream ever arrived,
// and refreshing could not help because nothing had been saved). These helpers
// decide when to offer "send again" and check the server before resending, so a
// question that did arrive is never sent twice.

const UNDELIVERED_ABORT_REASONS = new Set(['recovery', 'timeout', 'offline', 'stale-local']);

/**
 * True when the send failed before the server answered with headers, nothing
 * was streamed, and the failure was not a user Stop.
 *
 * abortReason: the AbortController reason when the fetch was aborted, or ''
 * when it failed on its own (network error).
 */
export function isUndeliveredSend({ headersSeen, accumulated, text, isBackground, aborted, abortReason, recoverableError }) {
  if (headersSeen || isBackground) return false;
  if (accumulated) return false;
  if (!text || !String(text).trim()) return false;
  if (aborted) return UNDELIVERED_ABORT_REASONS.has(abortReason || '');
  return !!recoverableError;
}

function _norm(s) {
  return String(s || '').replace(/\s+/g, ' ').trim();
}

/**
 * Ask the server whether this question already arrived: a run is still active
 * for the session, or the last saved user message is this text.
 * Returns false only when both probes answered cleanly and found nothing;
 * throws when the server could not be asked — the caller must not resend then
 * (it cannot rule out a duplicate).
 */
export async function questionReachedServer(fetchFn, apiBase, sessionId, text) {
  const sid = encodeURIComponent(sessionId);
  const status = await fetchFn(`${apiBase}/api/chat/stream_status/${sid}`, {
    credentials: 'same-origin', cache: 'no-store',
  });
  if (status.ok) return true;
  if (status.status !== 404) throw new Error(`stream_status ${status.status}`);

  const res = await fetchFn(`${apiBase}/api/history/${sid}?limit=6`, {
    credentials: 'same-origin', cache: 'no-store',
  });
  if (!res.ok) throw new Error(`history ${res.status}`);
  const data = await res.json();
  const history = Array.isArray(data && data.history) ? data.history : [];
  const lastUser = [...history].reverse().find(m => m && m.role === 'user');
  return !!lastUser && _norm(lastUser.content) === _norm(text);
}
