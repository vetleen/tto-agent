"""Question, input and answer types for the decisions service
(``llm.service.decisions_service``).

The types double as the wire format of OpenAI's Decisions endpoint
(``POST /v1/decisions``): ``Question.to_api()`` and ``InputMessage.model_dump()``
are the request body, and the answer models mirror its response. The structured
pathway (``llm.service.decisions_structured``) produces the same answer models,
so a caller never sees which pathway answered.

Three question types, all answered with probabilities:

* ``predicate`` — how likely a statement about the input is true → ``probability``.
* ``choice`` — one of the given ``choices`` → the argmax ``choice``, a probability
  per option and ``confidence`` (the chosen option's probability).
* ``score`` — a rating against ordered ``levels`` (lowest first) → ``score`` (the
  probability-weighted mean level index, 0-based), a probability per level and
  ``confidence``.

Any question may come back as a ``refusal``.
"""

from __future__ import annotations

import json
import math
import re
from typing import Annotated, Any, Iterable, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from llm.types.responses import Usage

QuestionType = Literal["predicate", "choice", "score"]
Pathway = Literal["quick", "structured"]

# Question names are the join key between questions and answers (and field
# names in log summaries), so keep them simple and unique per call.
_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------


class Choice(BaseModel):
    """One option of a ``choice`` question."""

    model_config = ConfigDict(extra="forbid")

    value: str = Field(min_length=1)
    description: str | None = None


class Level(BaseModel):
    """One level of a ``score`` question; levels are listed lowest → highest."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1)
    description: str | None = None


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: QuestionType
    name: str
    instructions: str = Field(min_length=1)
    choices: list[Choice] | None = None  # choice questions only
    levels: list[Level] | None = None  # score questions only

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not _NAME_RE.match(value):
            raise ValueError(
                "question name must be 1–64 characters of letters, digits, '_', '.' or '-'"
            )
        return value

    @model_validator(mode="after")
    def _shape_matches_type(self) -> "Question":
        if self.type == "predicate":
            if self.choices is not None or self.levels is not None:
                raise ValueError("a predicate question takes neither choices nor levels")
        elif self.type == "choice":
            if self.levels is not None:
                raise ValueError("a choice question takes choices, not levels")
            if not self.choices or len(self.choices) < 2:
                raise ValueError("a choice question needs at least 2 choices")
            values = [c.value for c in self.choices]
            if len(set(values)) != len(values):
                raise ValueError("choice values must be unique")
        else:
            if self.choices is not None:
                raise ValueError("a score question takes levels, not choices")
            count = len(self.levels or [])
            if not MIN_SCORE_LEVELS <= count <= MAX_SCORE_LEVELS:
                raise ValueError(
                    f"a score question needs {MIN_SCORE_LEVELS}–{MAX_SCORE_LEVELS} levels, lowest first"
                )
            labels = [lv.label for lv in self.levels]
            if len(set(labels)) != len(labels):
                raise ValueError("level labels must be unique")
        return self

    @property
    def option_count(self) -> int:
        """Number of options (choice) or levels (score); 0 for a predicate."""
        return len(self.choices or self.levels or ())

    def to_api(self) -> dict:
        """The question as the Decisions endpoint expects it."""
        return self.model_dump(exclude_none=True)


def validate_questions(questions: Sequence[Question | dict]) -> list[Question]:
    """Coerce and validate a caller's questions: at least one, names unique."""
    out = [q if isinstance(q, Question) else Question.model_validate(q) for q in questions]
    if not out:
        raise ValueError("at least one question is required")
    names = [q.name for q in out]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"question names must be unique; duplicated: {', '.join(duplicates)}")
    return out


def chunk_questions(questions: Sequence[Question], size: int) -> list[list[Question]]:
    """Split questions into batches of at most ``size``, order preserved."""
    if size < 1:
        raise ValueError("batch size must be positive")
    return [list(questions[i:i + size]) for i in range(0, len(questions), size)]


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


class TextPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["input_text"] = "input_text"
    text: str = Field(min_length=1)


# The Decisions endpoint rejects an input_json part whose serialised data exceeds
# this (400 "input_json data exceeds the maximum serialized size of 262144
# bytes", observed 2026-10-23). Larger structured input goes in as text parts.
MAX_JSON_PART_BYTES = 262_144


class JsonPart(BaseModel):
    """Structured data as input; sent as ``input_json`` on the quick pathway and
    as a fenced JSON block on the structured one. Serialised size is capped at
    ``MAX_JSON_PART_BYTES`` (the endpoint's limit); split or use ``TextPart``."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["input_json"] = "input_json"
    data: Any

    @field_validator("data")
    @classmethod
    def _json_serialisable(cls, value: Any) -> Any:
        try:
            size = len(json.dumps(value).encode("utf-8"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"input_json data must be JSON-serialisable: {exc}") from None
        if size > MAX_JSON_PART_BYTES:
            raise ValueError(
                f"input_json data is {size} bytes serialised; the endpoint's cap is "
                f"{MAX_JSON_PART_BYTES} — split it or pass it as text"
            )
        return value


# Room for an image part later; v1 is text + JSON.
InputPart = Annotated[TextPart | JsonPart, Field(discriminator="type")]


class InputMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["developer", "user", "system", "assistant"] = "user"
    content: list[InputPart] = Field(min_length=1)


DecisionInput = str | list[InputMessage]


def normalize_input(input: DecisionInput) -> list[InputMessage]:
    """A caller's input as a non-empty message list (a string = one user message)."""
    if isinstance(input, str):
        if not input.strip():
            raise ValueError("input must not be empty")
        return [InputMessage(role="user", content=[TextPart(text=input)])]
    messages = [m if isinstance(m, InputMessage) else InputMessage.model_validate(m) for m in input]
    if not messages:
        raise ValueError("input must not be empty")
    return messages


def estimate_input_tokens(messages: Iterable[InputMessage]) -> int:
    """cl100k estimate of the shared input (text + JSON parts)."""
    from core.tokens import count_tokens

    total = 0
    for message in messages:
        for part in message.content:
            if isinstance(part, TextPart):
                total += count_tokens(part.text)
            else:
                total += count_tokens(json.dumps(part.data, ensure_ascii=False))
    return total


# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------


class PredicateAnswer(BaseModel):
    type: Literal["predicate"] = "predicate"
    name: str
    probability: float


class ChoiceProbability(BaseModel):
    value: str
    probability: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    name: str
    choice: str
    probabilities: list[ChoiceProbability]
    confidence: float


class ScoreProbability(BaseModel):
    value: int  # level index, 0-based (lowest level = 0)
    label: str
    probability: float


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    name: str
    score: float  # probability-weighted mean of the level indices
    probabilities: list[ScoreProbability]
    confidence: float


class RefusalAnswer(BaseModel):
    type: Literal["refusal"] = "refusal"
    name: str


Answer = Annotated[
    PredicateAnswer | ChoiceAnswer | ScoreAnswer | RefusalAnswer,
    Field(discriminator="type"),
]


def clamp_probability(value: float) -> float:
    """A model-produced probability as a finite number in [0, 1]."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return min(1.0, max(0.0, value))


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


class CallRecord(BaseModel):
    """One provider call made for a ``decide()`` — observability only; nothing
    about the pathway is attached to the answers themselves."""

    pathway: Pathway
    model: str
    questions: int
    duration_ms: int
    usage: Usage | None = None
    # Set on the failed quick attempt that triggered the fallback.
    error_code: str = ""


class DecisionsResult(BaseModel):
    answers: list[Answer]  # same order as the questions passed in
    usage: Usage  # summed over successful calls
    calls: list[CallRecord]
    fell_back: bool = False

    @property
    def by_name(self) -> dict[str, Answer]:
        return {a.name: a for a in self.answers}

    def __getitem__(self, name: str) -> Answer:
        for answer in self.answers:
            if answer.name == name:
                return answer
        raise KeyError(name)

    def refused(self, name: str) -> bool:
        return self[name].type == "refusal"

    @property
    def pathway(self) -> Literal["quick", "structured", "mixed"]:
        """Which pathway answered: a single one, or "mixed" when a later quick
        batch failed over to structured within the same call."""
        used = {c.pathway for c in self.calls if not c.error_code}
        if len(used) == 1:
            return next(iter(used))
        return "mixed" if used else "structured"


def sum_usage(usages: Iterable[Usage | None]) -> Usage:
    """Field-wise sum of the given usages (None fields stay None when no call
    reported them)."""
    total = Usage()
    for usage in usages:
        if usage is None:
            continue
        for name in Usage.model_fields:
            value = getattr(usage, name)
            if value is None:
                continue
            setattr(total, name, (getattr(total, name) or 0) + value)
    return total


__all__ = [
    "Answer",
    "CallRecord",
    "Choice",
    "ChoiceAnswer",
    "ChoiceProbability",
    "DecisionInput",
    "DecisionsResult",
    "InputMessage",
    "InputPart",
    "JsonPart",
    "Level",
    "MAX_JSON_PART_BYTES",
    "MAX_SCORE_LEVELS",
    "MIN_SCORE_LEVELS",
    "Pathway",
    "PredicateAnswer",
    "Question",
    "QuestionType",
    "RefusalAnswer",
    "ScoreAnswer",
    "ScoreProbability",
    "TextPart",
    "chunk_questions",
    "clamp_probability",
    "estimate_input_tokens",
    "normalize_input",
    "sum_usage",
    "validate_questions",
]
