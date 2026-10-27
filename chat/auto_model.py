"""Automatic model choice for chat.

A thread on "Automatic" runs each turn on a routing *category* (cheap, mid,
writing, complex analysis, …). Each org maps every category to a model and a
reasoning level in its settings (feature keys ``auto_<category>`` in
core.preferences.FEATURE_DEFAULTS), so the router never names a model.

This module holds the category table and the category → (model, level)
resolution. The router itself (cache boundaries, skills, the decisions call)
builds on it.
"""
from __future__ import annotations

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
