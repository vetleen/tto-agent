"""Shared transcription-language definitions.

Single source of truth for the transcription language codes, their labels, the
dropdown option order, validation, and the resolution helpers used by the
preference cascade and every meeting transcription read point.

Value semantics (one scheme across org prefs, user prefs, and the meeting field):

* ``""`` / key absent — *not set at this layer* → inherit the layer below
  (user → org → system).
* ``"auto"`` — *explicit auto-detect* (lets a user override an org language default
  back to per-utterance detection).
* ``"en"`` / ``"no"`` / … — that specific language.

``resolve_api_language`` maps ``""``/``"auto"`` to ``None`` so no language hint is
sent to the transcription API in the auto-detect case.

Deliberately import-free to avoid cycles (imported by ``core.preferences``,
``meetings`` views/tasks/services, and ``accounts`` settings views).
"""
from __future__ import annotations

# Ordered ``(code, label)`` pairs. Order defines dropdown order; the first entry
# ("auto") is the system default and first-class auto-detect option.
TRANSCRIPTION_LANGUAGE_CHOICES: list[tuple[str, str]] = [
    ("auto", "Auto-detect"),
    ("en", "English"),
    ("no", "Norwegian"),
    ("sv", "Swedish"),
    ("da", "Danish"),
    ("de", "German"),
    ("fr", "French"),
    ("es", "Spanish"),
]

LANGUAGE_LABELS: dict[str, str] = dict(TRANSCRIPTION_LANGUAGE_CHOICES)

# System default when nothing is set at any layer — auto-detect preserves the
# historical behavior for anyone who never touches the setting.
DEFAULT_TRANSCRIPTION_LANGUAGE = "auto"

# Accepted stored values. Includes ``""`` (unset/inherit) and the legacy ``nb``/
# ``nn`` codes for back-compat with the old inline whitelist in meetings.views,
# even though they aren't offered in the dropdowns.
VALID_TRANSCRIPTION_LANGUAGE_VALUES: frozenset[str] = frozenset(
    {"", "auto", "en", "no", "nb", "nn", "sv", "da", "de", "fr", "es"}
)


def is_valid_transcription_language(code: str | None) -> bool:
    """True if ``code`` is an accepted stored value (incl. ``""`` for unset)."""
    return isinstance(code, str) and code in VALID_TRANSCRIPTION_LANGUAGE_VALUES


def language_label(code: str | None) -> str:
    """Human label for a stored code (``"no"`` → ``"Norwegian"``).

    Falls back to ``"Auto-detect"`` for the unset/``"auto"`` states and echoes any
    unknown code unchanged.
    """
    if not code or code == "auto":
        return LANGUAGE_LABELS["auto"]
    return LANGUAGE_LABELS.get(code, code)


def resolve_api_language(code: str | None) -> str | None:
    """Map a stored value to the API language hint.

    ``""`` (unset) and ``"auto"`` (explicit auto-detect) both mean "send no
    language hint" → ``None``. Any other value is passed through verbatim.
    """
    if not code or code == "auto":
        return None
    return code


def effective_meeting_language(
    meeting_forced: str | None, default: str | None
) -> str | None:
    """Resolve the language hint for a meeting: its own value, else the default.

    ``meeting_forced`` is ``Meeting.forced_language`` (``""`` = inherit the
    preference default). ``default`` is the resolved preference default
    (``ResolvedPreferences.transcription_language``, always concrete, e.g.
    ``"auto"``/``"no"``). Returns the API language hint (``None`` for auto-detect).
    """
    code = (meeting_forced or "").strip() or (default or "")
    return resolve_api_language(code)
