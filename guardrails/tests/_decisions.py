"""Test helpers: hand-made Decisions answers for the guardrail and PII jobs."""

from llm.types.decisions import (
    CallRecord,
    ChoiceAnswer,
    ChoiceProbability,
    DecisionsResult,
    PredicateAnswer,
    RefusalAnswer,
)
from llm.types.responses import Usage

CONCERNS = (
    "prompt_injection", "jailbreak", "data_extraction",
    "social_engineering", "encoding_bypass", "delimiter_injection", "none",
)


def predicate(name, probability):
    return PredicateAnswer(name=name, probability=probability)


def refusal(name):
    return RefusalAnswer(name=name)


def concern(name, choice="none", probabilities=None):
    probabilities = probabilities or {choice: 1.0}
    return ChoiceAnswer(
        name=name, choice=choice, confidence=max(probabilities.values()),
        probabilities=[ChoiceProbability(value=v, probability=probabilities.get(v, 0.0)) for v in CONCERNS],
    )


def result(*answers, pathway="quick"):
    return DecisionsResult(
        answers=list(answers), usage=Usage(),
        calls=[CallRecord(pathway=pathway, model="openai/gpt-6-luna", questions=len(answers), duration_ms=5)],
    )
