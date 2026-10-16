"""Association of consumer HTTP calls with provider API operations.

A call's URL template (e.g. ``("http://user-service/users/", None)`` for
``f"{BASE}/users/{user_id}"`` with a known BASE) is reduced to path segments and
compared with OpenAPI path templates segment by segment. Hosts are ignored, so a
consumer calling an identically shaped path on a different service would also match.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from api_drift_guard.source_analysis import HttpCall, UrlTemplate

_DYNAMIC = "\0"
_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")

# Per-segment match scores: an exact literal beats a path parameter, which beats a
# literal template segment matched only by an unknown (dynamic) call segment.
_EXACT, _PARAM, _WILDCARD = 2, 1, 0

PathSegment = str | None  # None: statically unknown segment


@dataclass(frozen=True)
class Operation:
    method: str
    path: str

    @property
    def key(self) -> str:
        return f"{self.method} {self.path}"


@dataclass(frozen=True)
class CallMatch:
    call: HttpCall
    # Keys of the best-scoring operations; more than one means the match is ambiguous.
    operations: tuple[str, ...]
    # True when the URL could not be resolved, so the call might target any operation.
    unresolved: bool = False

    @property
    def ambiguous(self) -> bool:
        return self.unresolved or len(self.operations) > 1


def url_path_segments(url: UrlTemplate) -> tuple[PathSegment, ...] | None:
    """Reduce a URL template to path segments, or None if the path cannot be located."""
    text = "".join(_DYNAMIC if part is None else part for part in url)
    if text.startswith(_DYNAMIC):
        # A leading unknown part followed by "/" is treated as a base-URL variable.
        text = text.lstrip(_DYNAMIC)
        if not text.startswith("/"):
            return None
    elif match := _SCHEME_RE.match(text):
        after_scheme = text[match.end():]
        slash = after_scheme.find("/")
        text = after_scheme[slash:] if slash >= 0 else "/"
    text = re.split(r"[?#]", text, maxsplit=1)[0]
    return tuple(None if _DYNAMIC in seg else seg for seg in text.split("/") if seg)


def _score(segments: tuple[PathSegment, ...], template: str) -> int | None:
    template_segments = [s for s in template.split("/") if s]
    if len(template_segments) != len(segments):
        return None
    score = 0
    for call_segment, template_segment in zip(segments, template_segments):
        if template_segment.startswith("{") and template_segment.endswith("}"):
            score += _PARAM
        elif call_segment is None:
            score += _WILDCARD
        elif call_segment == template_segment:
            score += _EXACT
        else:
            return None
    return score


def is_resolved(call: HttpCall) -> bool:
    return call.url is not None and url_path_segments(call.url) is not None


def match_call(call: HttpCall, operations: list[Operation]) -> CallMatch | None:
    """Match a call to its most specific operation(s); None if nothing matches."""
    candidates = [op for op in operations if call.method is None or call.method == op.method]
    if not is_resolved(call):
        keys = tuple(op.key for op in candidates)
        return CallMatch(call, keys, unresolved=True) if keys else None

    segments = url_path_segments(call.url)  # type: ignore[arg-type]
    scored = [(s, op) for op in candidates if (s := _score(segments, op.path)) is not None]  # type: ignore[arg-type]
    if not scored:
        return None
    best = max(score for score, _ in scored)
    return CallMatch(call, tuple(op.key for score, op in scored if score == best))


def match_calls(calls: list[HttpCall], operations: list[Operation]) -> list[CallMatch]:
    return [m for call in calls if (m := match_call(call, operations)) is not None]
