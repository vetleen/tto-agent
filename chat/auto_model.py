"""Automatic model choice for chat.

A thread on "Automatic" runs each turn on a routing *category* (cheap, mid,
writing, complex analysis, …). Each org maps every category to a model and a
reasoning level in its settings (feature keys ``auto_<category>`` in
core.preferences.FEATURE_DEFAULTS), so the router never names a model.

This module holds the category table, the category → (model, level)
resolution and the router (route_turn): cache boundaries, attached skills, the
decisions call and the no-downgrade floor. The router prompt lives in
chat/auto_model_prompt.py.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# The pseudo model id the chat client sends (and the org chat default may hold)
# for "Automatic".
AUTO_MODEL_ID = "auto"


@dataclass(frozen=True)
class AutoCategory:
    key: str
    label: str
    # Fixed quality rank: 0 cheap, 1 mid, 2 high. Drives the no-downgrade floor
    # and the skill tie-break; deliberately independent of which model an org
    # maps the category to.
    rank: int
    # Settings-page description (the router's definitions live in
    # chat/auto_model_prompt.py).
    desc: str

    @property
    def feature_key(self) -> str:
        return f"auto_{self.key}"


AUTO_CATEGORIES: tuple[AutoCategory, ...] = (
    AutoCategory("cheap", "Cheap", 0, "Greetings, thanks, one-step app operations (attach a skill, move text to the canvas) and questions about the app itself."),
    AutoCategory("mid", "Default", 1, "Ordinary questions and everyday tasks — most turns land here. Also the fallback when the automatic choice fails."),
    AutoCategory("top", "Top", 2, "Hard, open-ended problems that don't fit a more specific category."),
    AutoCategory("short_writing", "Short writing", 1, "Short texts: a post, a short email, a paragraph, a reply."),
    AutoCategory("writing", "Writing", 2, "Drafting or revising longer texts: reports, applications, agreements, skills and templates."),
    AutoCategory("light_reading", "Light reading", 1, "Finding, summarising or extracting from documents when the answer is mostly on the page."),
    AutoCategory("difficult_reading", "Difficult reading", 2, "Close reading of long, dense or technical documents: contracts, patents, applications."),
    AutoCategory("light_analysis", "Light analysis", 1, "Straightforward analysis: comparisons, simple calculations, tables and plans from given facts."),
    AutoCategory("complex_analysis", "Complex analysis", 2, "Multi-step reasoning and judgement: legal and IP assessments, gap analyses, evaluations, strategy."),
    AutoCategory("research", "Research", 2, "Searching the web, patents or literature and synthesising what is found, e.g. prior art."),
    AutoCategory("slide_design", "Slide design", 2, "Creating or redesigning slides and decks."),
)

CATEGORIES_BY_KEY: dict[str, AutoCategory] = {c.key: c for c in AUTO_CATEGORIES}
AUTO_FEATURE_KEYS: tuple[str, ...] = tuple(c.feature_key for c in AUTO_CATEGORIES)
DEFAULT_CATEGORY = "mid"


def category_rank(key: str | None) -> int:
    cat = CATEGORIES_BY_KEY.get(key or "")
    return cat.rank if cat else CATEGORIES_BY_KEY[DEFAULT_CATEGORY].rank


def valid_level(model: str, level: str | None) -> str | None:
    """*level* if *model* offers it, else None (the model's registry default)."""
    if not level or not model:
        return None
    from llm.model_registry import get_model_info

    info = get_model_info(model)
    return level if info and level in info.reasoning_levels else None


def resolve_category(prefs, key: str, level_override: str | None = None) -> tuple[str, str | None]:
    """The (model, reasoning level) an org runs *key* on.

    *prefs* is a ResolvedPreferences. The level is the org's pin for the
    category (only while it still runs on the pinned model), overridden by
    *level_override* (a skill's choice) when the model offers it. None = the
    model's registry default.
    """
    from core.preferences import resolve_org_feature_reasoning

    cat = CATEGORIES_BY_KEY.get(key) or CATEGORIES_BY_KEY[DEFAULT_CATEGORY]
    model = prefs.feature_models.get(cat.feature_key) or prefs.mid_model
    level = valid_level(model, level_override) or resolve_org_feature_reasoning(
        {"feature_reasoning": prefs.feature_reasoning}, cat.feature_key, model
    )
    return model, level


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

ROUTE_FEATURE_KEY = "auto_model_route"
QUESTION_NAME = "category"

# Router input budget, in characters (~4 per token): the first user message
# frames the task, the latest is what we route, the assistant head disambiguates
# short replies. Keeps a decision well under 10k tokens.
FIRST_USER_CHARS = 6_000
LATEST_USER_CHARS = 16_000
EARLIER_USER_CHARS = 4_000
EARLIER_USER_COUNT = 3
ASSISTANT_HEAD_CHARS = 1_600


@dataclass(frozen=True)
class RouteDecision:
    model: str
    thinking_level: str | None
    category: str | None  # None: a cached turn kept a manually chosen model
    source: str  # "cached" | "skill" | "router" | "fallback"
    raw_category: str | None = None  # the router's or skill's pick before the floor
    floor_applied: bool = False

    def log_metadata(self) -> dict:
        return {
            "category": self.category,
            "raw_category": self.raw_category,
            "source": self.source,
            "floor_applied": self.floor_applied,
        }


def cache_ttl_seconds(model: str) -> int:
    """How long *model*'s provider keeps a prompt cache warm (0 = treat as cold)."""
    from django.conf import settings

    from llm.model_registry import get_model_info

    info = get_model_info(model)
    if not info:
        return 0
    ttls = getattr(settings, "AUTO_MODEL_CACHE_TTL_SECONDS", None) or {}
    return int(ttls.get(info.provider, 0))


def _parse_ts(value):
    from django.utils.dateparse import parse_datetime

    if not isinstance(value, str):
        return None
    try:
        return parse_datetime(value)
    except ValueError:
        return None


def is_cache_warm(state: dict | None, *, thread_model: str, summary_up_to, allowed_models, now) -> bool:
    """Whether the thread's next turn would still hit its provider prompt cache.

    Warm = the last turn ran on *thread_model* (still allowed), within the
    provider's TTL, with no history summarise since. A thread on the cheap
    category is never "warm": it re-routes every turn, since starting cheap and
    moving on to real work is common and the cache it gives up is small.
    """
    if not state or not thread_model or state.get("model") != thread_model:
        return False
    if thread_model not in (allowed_models or ()):
        return False
    if state.get("enabled") and state.get("category") == "cheap":
        return False
    if state.get("summary_up_to") != (str(summary_up_to) if summary_up_to else None):
        return False
    last = _parse_ts(state.get("last_turn_at"))
    ttl = cache_ttl_seconds(thread_model)
    return bool(last and ttl > 0 and (now - last).total_seconds() < ttl)


def _blended_price_or_inf(model: str) -> float:
    from llm.display import _blended_price

    price = _blended_price(model) if model else None
    return float(price) if price is not None else float("inf")


def skill_category(skills, prefs) -> tuple[str, str | None] | None:
    """The category the attached skills ask for, deterministically.

    Each skill may name a preferred category (and a reasoning level). The
    highest-ranked category wins; among several of that rank, the one whose
    org model is cheapest. None when no attached skill names a category.
    """
    picks: list[tuple[str, str | None]] = []
    for skill in skills or ():
        key = (skill.get("preferred_model_category") or "").strip()
        if key in CATEGORIES_BY_KEY:
            picks.append((key, skill.get("preferred_reasoning_level") or None))
    if not picks:
        return None
    top_rank = max(category_rank(k) for k, _ in picks)
    top = [p for p in picks if category_rank(p[0]) == top_rank]
    return min(top, key=lambda p: _blended_price_or_inf(resolve_category(prefs, p[0])[0]))


def apply_floor(chosen: str, current: str | None) -> tuple[str, bool]:
    """No downgrades: keep *current* when *chosen* ranks below it. A thread on
    the cheap category (or none yet) has no floor."""
    if not current or current == "cheap" or current not in CATEGORIES_BY_KEY:
        return chosen, False
    if category_rank(chosen) < category_rank(current):
        return current, True
    return chosen, False


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " …[truncated]"


def build_router_input(thread_id, *, current_category: str | None, facts: dict | None) -> list:
    """The decision input: the router prompt (developer) + one JSON user part.

    Reads ChatMessage rows directly — never the turn's history budget — so
    routing doesn't wait on history loading or checkpoints. Sync (DB).
    """
    from chat.auto_model_prompt import ROUTER_PROMPT
    from chat.models import ChatMessage
    from llm.types.decisions import InputMessage, JsonPart, TextPart

    visible = ChatMessage.objects.filter(
        thread_id=thread_id, is_hidden_from_user=False, is_redacted=False,
    )
    users = visible.filter(role="user").exclude(content="")
    recent = list(users.order_by("-created_at").values_list("id", "content")[: EARLIER_USER_COUNT + 1])
    first = users.order_by("created_at").values_list("id", "content").first()
    assistant = (
        visible.filter(role="assistant").exclude(content="").exclude(metadata__has_key="ui_only")
        .order_by("-created_at").values_list("content", flat=True).first()
    )

    data: dict = {"latest_user_message": ""}
    if recent:
        data["latest_user_message"] = _clip(recent[0][1], LATEST_USER_CHARS)
        earlier = [content for _id, content in reversed(recent[1:])]
        if earlier:
            data["earlier_user_messages"] = [_clip(c, EARLIER_USER_CHARS) for c in earlier]
        if first and first[0] not in {i for i, _c in recent}:
            data["first_user_message"] = _clip(first[1], FIRST_USER_CHARS)
    if assistant:
        data["last_assistant_message_head"] = _clip(assistant, ASSISTANT_HEAD_CHARS)
    data["current_category"] = current_category
    data["facts"] = facts or {}

    return [
        InputMessage(role="developer", content=[TextPart(text=ROUTER_PROMPT)]),
        InputMessage(role="user", content=[JsonPart(data=data)]),
    ]


def build_question():
    from chat.auto_model_prompt import CATEGORY_DEFINITIONS, QUESTION_INSTRUCTIONS
    from llm.types.decisions import Choice, Question

    return Question(
        type="choice",
        name=QUESTION_NAME,
        instructions=QUESTION_INSTRUCTIONS,
        choices=[Choice(value=c.key, description=CATEGORY_DEFINITIONS[c.key]) for c in AUTO_CATEGORIES],
    )


class RouterNoAnswer(Exception):
    """The decision came back without a usable category (refusal, unknown value)."""


async def decide_category(thread_id, *, org_id, current_category, facts) -> str:
    from channels.db import database_sync_to_async

    from llm.service.decisions_service import get_decisions_service

    messages = await database_sync_to_async(build_router_input)(
        thread_id, current_category=current_category, facts=facts,
    )
    result = await get_decisions_service().adecide(
        [build_question()], messages,
        feature_key=ROUTE_FEATURE_KEY, org_id=org_id,
        prompt_cache_key=f"decisions:{ROUTE_FEATURE_KEY}",
    )
    answer = result[QUESTION_NAME]
    if answer.type != "choice" or answer.choice not in CATEGORIES_BY_KEY:
        raise RouterNoAnswer(answer.type)
    return answer.choice


async def route_turn(
    *, thread_id, org_id, prefs, state: dict | None, thread_model: str,
    summary_up_to, skills, facts: dict | None, now=None,
) -> RouteDecision:
    """Pick the model + reasoning level for an Automatic turn. Never raises.

    1. Warm prompt cache → keep the thread's model and level.
    2. Attached skills naming a category → that category (skill_category).
    3. The router's decision.
    4. Router failure or timeout → the default category.
    2–4 are clamped by the no-downgrade floor (apply_floor).
    """
    from django.conf import settings
    from django.utils import timezone

    now = now or timezone.now()
    state = state or {}
    current = state.get("category") if state.get("enabled") else None

    if is_cache_warm(
        state, thread_model=thread_model, summary_up_to=summary_up_to,
        allowed_models=prefs.allowed_models, now=now,
    ):
        return RouteDecision(
            model=thread_model,
            thinking_level=valid_level(thread_model, state.get("thinking_level")),
            category=current,
            source="cached",
            raw_category=current,
        )

    level_override = None
    picked = skill_category(skills, prefs)
    if picked:
        raw, level_override = picked
        source = "skill"
    else:
        try:
            raw = await asyncio.wait_for(
                decide_category(thread_id, org_id=org_id, current_category=current, facts=facts),
                timeout=float(getattr(settings, "AUTO_MODEL_ROUTE_TIMEOUT_SECONDS", 5)),
            )
            source = "router"
        except Exception as exc:  # timeout, provider error, refusal: never block the turn
            logger.warning(
                "auto_model: routing failed for thread=%s (%s); using %s",
                thread_id, type(exc).__name__, DEFAULT_CATEGORY,
            )
            raw = None
            source = "fallback"

    category, floored = apply_floor(raw or DEFAULT_CATEGORY, current)
    if floored:
        level_override = None  # a skill's level belongs to the skill's category
    model, level = resolve_category(prefs, category, level_override)
    return RouteDecision(
        model=model, thinking_level=level, category=category, source=source,
        raw_category=raw, floor_applied=floored,
    )
