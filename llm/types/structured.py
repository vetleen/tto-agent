"""Structured output schemas for LLM responses."""

from pydantic import BaseModel, Field


class DocumentDescriptionOutput(BaseModel):
    """Structured output for document description and type classification."""

    description: str = Field(description="A relevance-signal paragraph (~100 tokens)")
    document_type: str = Field(
        description="Document type classification (e.g. Agreement, Patent, License, Report)"
    )
    document_date: str | None = Field(
        default=None,
        description=(
            "The primary date of this document in YYYY-MM-DD format, if clearly "
            "identifiable (e.g. signing date, publication date, effective date, "
            "date of correspondence). Return null if no clear date is present."
        ),
    )


class PIICategoryOutput(BaseModel):
    """Structured output for GDPR personal data category classification."""

    pii_ordinary_identity: bool = Field(
        default=False,
        description="Personal identity data: names, email addresses, phone numbers, physical addresses, official identifiers (national ID, passport), photographs, or non-identifying biometric data.",
    )
    pii_ordinary_professional: bool = Field(
        default=False,
        description="Professional data: job titles, organisational affiliations, education, qualifications, work history, professional evaluations, salary/compensation, professional relationships, group memberships, or career history.",
    )
    pii_ordinary_communication: bool = Field(
        default=False,
        description="Communication content: meeting content or transcripts, email body content, chat or conversation content, or voice recordings.",
    )
    pii_ordinary_contact: bool = Field(
        default=False,
        description="Digital contact and location data: IP addresses, geolocation data, or device identifiers.",
    )
    pii_ordinary_security: bool = Field(
        default=False,
        description="Security and authentication data: password hashes, session tokens, or authentication history.",
    )
    pii_ordinary_preferences: bool = Field(
        default=False,
        description="User preferences: system preferences or work-related system settings.",
    )
    pii_ordinary_financial: bool = Field(
        default=False,
        description="Financial and business data: business information, account or payment information, or ownership and intellectual property rights.",
    )
    pii_ordinary_social: bool = Field(
        default=False,
        description="Social and family data: family relationships or personal life history.",
    )
    pii_special_category: bool = Field(
        default=False,
        description="GDPR Article 9 special categories: biometric data used for identification, trade union membership, health data, racial or ethnic origin, political opinions, religious or philosophical beliefs, genetic data, sex life or sexual orientation.",
    )
    pii_criminal_offence: bool = Field(
        default=False,
        description="GDPR Article 10 data: criminal convictions and offences.",
    )
