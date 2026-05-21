"""Classify document text by GDPR personal data categories."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PII_SYSTEM_PROMPT = """\
You are a GDPR personal data classifier. Your task is to determine which \
categories of personal data are clearly present in a document.

## Background

Under the EU General Data Protection Regulation (GDPR), personal data is any \
information relating to an identified or identifiable natural person \
(Article 4(1)). Certain categories receive heightened protection:

- **Ordinary personal data** (Article 6): processed under a lawful basis.
- **Special category data** (Article 9): processing is prohibited unless an \
  explicit exception applies. Includes health data, racial/ethnic origin, \
  political opinions, religious beliefs, trade union membership, genetic data, \
  biometric data used for identification, and data on sex life or orientation.
- **Criminal offence data** (Article 10): requires specific legal authority.

## Categories to assess

For each category below, return `true` if the document **clearly contains** \
that type of personal data relating to identifiable individuals. Return \
`false` if the category is absent or only mentioned abstractly (e.g. a policy \
*about* health data does not constitute health data itself).

### pii_ordinary_identity
Personal identity information that directly or indirectly identifies a \
natural person. Examples: full names, email addresses, telephone numbers, \
physical/postal addresses, official identifiers (national ID numbers, \
passport numbers, organisation numbers linked to a natural person), \
photographs or portraits, biometric data that is NOT used for identification \
purposes (probably not relevant for most documents).

### pii_ordinary_professional
Information related to a person's education, employment, and professional \
life. Examples: job titles or roles, organisational affiliations, education \
and qualifications, work history or CV information, professional evaluations \
or performance reviews, salary or compensation details, professional \
relationships (co-authors, supervisors, collaborators), group or committee \
memberships, career history.

### pii_ordinary_communication
Content of communications between or about persons. Examples: meeting minutes \
or transcripts, email body content (not just headers), chat or conversation \
content, voice recordings. Note: this covers the *content*, not metadata like \
sender addresses (which fall under identity).

### pii_ordinary_contact
Digital contact and location data used to reach or locate a person. \
Examples: IP addresses, geolocation data, device identifiers or fingerprints.

### pii_ordinary_security
Authentication and account security data. Examples: password hashes, session \
tokens, authentication logs or login history.

### pii_ordinary_preferences
User preferences and configuration choices. Examples: system settings or \
display preferences, work-related tool or workflow preferences.

### pii_ordinary_financial
Financial and business data linked to identifiable natural persons. Examples: \
business information tied to sole proprietorships or identifiable founders, \
account or payment information, ownership stakes or intellectual property \
rights linked to named inventors or authors.

### pii_ordinary_social
Social and family information. Examples: family relationships (spouse, \
children), personal life history or biographical details beyond professional \
career.

### pii_special_category
GDPR Article 9 special categories — processing is generally prohibited. \
Examples: biometric data processed for the purpose of uniquely identifying a \
person, trade union membership, health data (medical conditions, diagnoses, \
treatments, disability status), racial or ethnic origin, political opinions, \
religious or philosophical beliefs, genetic data, data concerning sex life or \
sexual orientation.

### pii_criminal_offence
GDPR Article 10 data — requires specific legal authority. Examples: criminal \
convictions, charges, or offences.

## Decision guidance

- Mark a category `true` only when the document clearly contains personal \
  data of that type relating to identifiable natural persons.
- Generic data does not count: "A patient was treated" is not personal data alone. 
- Anonymized data counts: "Patient 42 received treatment" is personal health data.
- A document *about* a data category (e.g. a privacy policy discussing health \
  data) does not itself contain that category of personal data.
- When in doubt, err on the side of `false`.
"""

# All category field names, in schema order
PII_CATEGORIES = [
    "pii_ordinary_identity",
    "pii_ordinary_professional",
    "pii_ordinary_communication",
    "pii_ordinary_contact",
    "pii_ordinary_security",
    "pii_ordinary_preferences",
    "pii_ordinary_financial",
    "pii_ordinary_social",
    "pii_special_category",
    "pii_criminal_offence",
]


def scan_pii_categories(
    text: str,
    user_id: int | None = None,
    data_room_id: int | None = None,
    org_id: int | None = None,
) -> dict[str, bool]:
    """Classify document text into GDPR PII categories.

    Returns a dict of only the categories detected as ``True``.
    """
    from core.preferences import resolve_org_feature_model
    from llm import get_llm_service
    from llm.types import ChatRequest, Message, RunContext
    from llm.types.structured import PIICategoryOutput

    if not text.strip():
        return {}

    from documents.services.description import _prepare_document_text

    document_text = _prepare_document_text(text)
    model = resolve_org_feature_model(org_id, "pii_scan")

    context = RunContext.create(user_id=user_id)
    request = ChatRequest(
        messages=[
            Message(role="system", content=_PII_SYSTEM_PROMPT),
            Message(role="user", content=document_text),
        ],
        model=model,
        stream=False,
        tools=[],
        context=context,
    )

    service = get_llm_service()
    parsed, usage = service.run_structured(request, PIICategoryOutput)

    result = {}
    for category in PII_CATEGORIES:
        if getattr(parsed, category, False):
            result[category] = True

    logger.info(
        "scan_pii_categories: user_id=%s detected=%s",
        user_id,
        list(result.keys()),
    )
    return result
