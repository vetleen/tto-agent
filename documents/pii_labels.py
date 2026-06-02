"""Human-readable labels and groupings for GDPR PII category tags.

Single source of truth for turning ``pii_*`` tag keys (produced by
``documents.services.pii_scan``) into display text. Used by the document-list
UI, the data room view, and the ``/pii`` chat command. Kept free of heavy
imports (no LLM service) so any layer can import it cheaply.
"""
from __future__ import annotations

from documents.services.pii_scan import PII_CATEGORIES

# GDPR groupings.
ORDINARY = "ordinary"
SPECIAL = "special"
CRIMINAL = "criminal"

PII_BUCKET: dict[str, str] = {
    "pii_ordinary_identity": ORDINARY,
    "pii_ordinary_professional": ORDINARY,
    "pii_ordinary_communication": ORDINARY,
    "pii_ordinary_contact": ORDINARY,
    "pii_ordinary_security": ORDINARY,
    "pii_ordinary_preferences": ORDINARY,
    "pii_ordinary_financial": ORDINARY,
    "pii_ordinary_social": ORDINARY,
    "pii_special_category": SPECIAL,
    "pii_criminal_offence": CRIMINAL,
}

# Full label per bucket — used in the /pii report and the pill hover tooltips.
BUCKET_LABEL: dict[str, str] = {
    ORDINARY: "Ordinary Personal Data",
    SPECIAL: "Special Category Data (Art. 9)",
    CRIMINAL: "Criminal Offence Data (Art. 10)",
}

# Compact label per bucket — used as the document-list pill text so the row does
# not overflow when several pills are shown. The full label + article reference
# live in the pill's hover tooltip.
PILL_LABEL: dict[str, str] = {
    ORDINARY: "Personal data",
    SPECIAL: "Special category",
    CRIMINAL: "Criminal offence",
}

# Noun phrase per ordinary category — assembled into the pill hover tooltip.
PII_TOOLTIP_PHRASE: dict[str, str] = {
    "pii_ordinary_identity": "personal identity",
    "pii_ordinary_professional": "professional and employment details",
    "pii_ordinary_communication": "the content of communications",
    "pii_ordinary_contact": "contact and location data",
    "pii_ordinary_security": "account security data",
    "pii_ordinary_preferences": "user preferences",
    "pii_ordinary_financial": "financial details",
    "pii_ordinary_social": "social and family details",
}

# Richer fragment per category — used as bullets in the /pii command output.
PII_DESCRIPTION: dict[str, str] = {
    "pii_ordinary_identity": "Identity data, such as names, emails, phone numbers, and addresses",
    "pii_ordinary_professional": "Professional data, such as job titles, employment history, and qualifications",
    "pii_ordinary_communication": "Communication content, such as meeting minutes, transcripts, and email bodies",
    "pii_ordinary_contact": "Contact and location data, such as IP addresses and device identifiers",
    "pii_ordinary_security": "Security data, such as password hashes, tokens, and login history",
    "pii_ordinary_preferences": "Preference data, such as settings and workflow choices",
    "pii_ordinary_financial": "Financial data, such as payment details and ownership stakes tied to named people",
    "pii_ordinary_social": "Social and family data, such as relationships and personal history",
    "pii_special_category": "Special category data, such as health, ethnicity, religion, political opinions, or biometrics",
    "pii_criminal_offence": "Criminal offence data, such as convictions, charges, or offences",
}

# Fixed tooltip text for the single-category buckets (same for every tagged doc).
SPECIAL_TOOLTIP = (
    "This document contains special category data (GDPR Art. 9), such as health, "
    "racial or ethnic origin, religious or political beliefs, or biometric data."
)
CRIMINAL_TOOLTIP = (
    "This document contains criminal offence data (GDPR Art. 10), such as criminal "
    "convictions, charges, or offences."
)

# Canonical display order (mirrors pii_scan.PII_CATEGORIES).
_ORDER = list(PII_CATEGORIES)


def summarize_pii_keys(keys) -> dict:
    """Group a document's ``pii_*`` tag keys into display-ready buckets.

    Returns:
        {
            "has_ordinary": bool,
            "ordinary_tooltip": str,          # full hover sentence, "" if none
            "ordinary_descriptions": [str],   # /pii bullet fragments, ordered
            "special": bool,
            "criminal": bool,
        }
    """
    present = {k for k in keys if k in PII_BUCKET}
    ordinary_keys = [k for k in _ORDER if k in present and PII_BUCKET.get(k) == ORDINARY]
    phrases = [PII_TOOLTIP_PHRASE[k] for k in ordinary_keys if k in PII_TOOLTIP_PHRASE]

    return {
        "has_ordinary": bool(ordinary_keys),
        "ordinary_tooltip": (
            "This document contains personal data relating to " + ", ".join(phrases) + "."
            if phrases else ""
        ),
        "ordinary_descriptions": [PII_DESCRIPTION[k] for k in ordinary_keys if k in PII_DESCRIPTION],
        "special": "pii_special_category" in present,
        "criminal": "pii_criminal_offence" in present,
    }


def format_thread_pii_report(keys) -> str:
    """Build a Markdown report of the PII categories present across a set of tag keys.

    Used by the ``/pii`` chat command to summarize what categories of personal data
    the documents a thread has used contain. Returns a friendly empty-state message
    when no categories are present.
    """
    summary = summarize_pii_keys(keys)
    sections = []

    if summary["has_ordinary"]:
        bullets = "\n".join(f"- {d}" for d in summary["ordinary_descriptions"])
        sections.append(f"**{BUCKET_LABEL[ORDINARY]}**\n{bullets}")
    if summary["special"]:
        sections.append(f"**{BUCKET_LABEL[SPECIAL]}**\n- {PII_DESCRIPTION['pii_special_category']}")
    if summary["criminal"]:
        sections.append(f"**{BUCKET_LABEL[CRIMINAL]}**\n- {PII_DESCRIPTION['pii_criminal_offence']}")

    if not sections:
        return "No personal data categories were detected in the documents this thread has used."

    intro = "This thread has used documents containing the following categories of personal data:"
    return intro + "\n\n" + "\n\n".join(sections)
