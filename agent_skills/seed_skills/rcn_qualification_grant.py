"""RCN Qualification Grant Application Drafter seed skill definition."""

RCN_QUALIFICATION_GRANT = {
    "slug": "rcn-qualification-grant",
    "name": "RCN Qualification Grant Application Drafter",
    "description": """\
Draft Research Council of Norway (Forskningsrådet) "Kvalifiseringsprosjekt" \
(Qualification Project) grant applications. Combines gap analysis of existing \
source materials with structured drafting of the 5-page project description.

Use when the user mentions "Kvalifiseringsprosjekt", "qualification project", \
"RCN grant", "Forskningsrådet", "Forskningsradet", "commercialisation grant", \
"TRL", "technology transfer grant", or "innovation project funding" in the \
context of a Norwegian research-council application.

Do NOT use for other RCN instruments (IPN, KSP, SFI), EU Horizon proposals, \
or general grant-writing tasks unrelated to the Kvalifiseringsprosjekt call.""",
    "instructions": """\
# RCN Qualification Grant Application Drafter

Guide TTO professionals through drafting a Research Council of Norway \
"Kvalifiseringsprosjekt" (Qualification Project) application. Work in four \
phases: information gathering, gap analysis, interactive gap filling, and \
project description drafting.

## Phase 1 — Information gathering

Search the attached data room for evidence covering the full checklist below. \
For every item, note what you found and where (document + page/section).

### Checklist

**A. Technology & IP (items 1-10)**
1. Problem statement — what unmet need does the technology address?
2. Technical solution description (mechanism, architecture, principle)
3. Current TRL level and evidence supporting the assessment
4. Target TRL at project end
5. Key technical risks and mitigation strategies
6. IP status (patents filed/granted, freedom-to-operate, licensing)
7. Competing / alternative solutions and differentiation
8. Regulatory or standards requirements
9. Prototype, proof-of-concept, or pilot data
10. Publications or prior project results

**B. Market & Commercialisation (items 11-20)**
11. Target market segment(s) and size (TAM / SAM / SOM)
12. Customer pain points — evidence of demand (interviews, LOIs, surveys)
13. Value proposition — quantified benefit vs. status quo
14. Business model outline (licensing, spin-off, sale, partnership)
15. Revenue or savings projections (order of magnitude)
16. Go-to-market timeline and key milestones
17. Identified industry partner(s) and their role
18. Market dialogue evidence (meetings, pilot agreements, letters of support)
19. Competitor landscape and positioning
20. Green transition or sustainability impact

**C. Project Plan & Team (items 21-28)**
21. Project objectives (SMART)
22. Work packages with deliverables and go/no-go decision points
23. Project timeline (Gantt-level)
24. Budget breakdown by cost category and partner
25. Team competence and roles
26. Need for external expertise or sub-contractors
27. Collaboration agreements or consortium setup
28. Triggering effect — why public funding is necessary

Present a brief summary of findings to the user before moving to Phase 2.

## Phase 2 — Gap analysis disposition

Load the "Disposition" template to the canvas using `load_template_to_canvas`. \
Fill in each row:

- **Status:** ✅ Covered, ⚠️ Partial, ❌ Missing
- **Source:** document name + section, or "—" if missing
- **Summary:** one-line description of what was found or what is needed

Present the completed disposition to the user.

## Phase 3 — Interactive gap filling

Walk the user through every ⚠️ and ❌ item, starting with the most critical \
gaps (items that RCN evaluators weight heavily: triggering effect, market \
dialogue, commercialisation plan, go/no-go milestones).

For each gap:
1. Explain why this item matters for the application score.
2. Suggest what good evidence looks like.
3. Ask the user to provide the information or confirm an assumption.
4. Update the disposition in the canvas as gaps are resolved.

Continue until the user is satisfied or explicitly moves to drafting.

## Phase 4 — Project description drafting

Load the "Project Description" template to the canvas using \
`load_template_to_canvas`. Draft the full 5-page project description following \
the template structure.

### RCN-specific writing principles

- **Frame everything as commercialisation, not research.** RCN \
Kvalifiseringsprosjekt funds the path from research result to market, not \
more research. Every sentence should point toward commercial value.
- **Quantify benefits.** Replace vague claims ("significant market potential") \
with numbers ("NOK 2.4B addressable market by 2028, targeting 5% share").
- **Show market dialogue.** Reference specific conversations, meetings, or \
agreements with industry partners. Evaluators look for evidence that the \
market has been consulted, not just desk research.
- **Mirror RCN language.** Use terminology from the call text: \
"kvalifisering", "kommersialiseringsløp", "utløsende effekt", \
"verdiskaping", "grønt skifte".
- **Emphasise green transition where relevant.** RCN prioritises projects \
contributing to sustainability. If there is a green angle, make it explicit.
- **Demonstrate triggering effect.** Explain clearly why the project cannot \
happen without public funding — what specific barriers does the grant remove?
- **Include go/no-go milestones.** Show that the project has built-in \
decision points to cut losses early if results are negative.
- **Align budget with activities.** Every cost should trace to a work package. \
Flag any mismatch.

Draft section by section, pausing after each major section to let the user \
review and provide feedback before continuing.""",
    "tool_names": ["view_template", "load_template_to_canvas"],
    "templates": {
        "Disposition": """\
# Qualification Project — Gap Analysis Disposition

## Status Key
- ✅ Covered — sufficient information found in data room
- ⚠️ Partial — some information found but incomplete
- ❌ Missing — no information found; must be provided

---

## A. Technology & IP

| # | Item | Status | Source | Summary |
|---|------|--------|--------|---------|
| 1 | Problem statement | | | |
| 2 | Technical solution description | | | |
| 3 | Current TRL level + evidence | | | |
| 4 | Target TRL at project end | | | |
| 5 | Key technical risks + mitigation | | | |
| 6 | IP status | | | |
| 7 | Competing solutions + differentiation | | | |
| 8 | Regulatory / standards requirements | | | |
| 9 | Prototype / PoC / pilot data | | | |
| 10 | Publications / prior project results | | | |

## B. Market & Commercialisation

| # | Item | Status | Source | Summary |
|---|------|--------|--------|---------|
| 11 | Target market segment(s) + size | | | |
| 12 | Customer pain points / demand evidence | | | |
| 13 | Value proposition (quantified) | | | |
| 14 | Business model outline | | | |
| 15 | Revenue / savings projections | | | |
| 16 | Go-to-market timeline + milestones | | | |
| 17 | Industry partner(s) + role | | | |
| 18 | Market dialogue evidence | | | |
| 19 | Competitor landscape + positioning | | | |
| 20 | Green transition / sustainability | | | |

## C. Project Plan & Team

| # | Item | Status | Source | Summary |
|---|------|--------|--------|---------|
| 21 | Project objectives (SMART) | | | |
| 22 | Work packages + go/no-go points | | | |
| 23 | Project timeline | | | |
| 24 | Budget breakdown | | | |
| 25 | Team competence + roles | | | |
| 26 | External expertise needs | | | |
| 27 | Collaboration agreements | | | |
| 28 | Triggering effect | | | |

---

*Updated: [date]*""",
        "Project Description": """\
# [Project Title]

**Qualification Project — Research Council of Norway (Forskningsrådet)**

*Target length: ~5 pages (12,000 characters excl. references)*

---

## 1. Background and Need (~0.5 page)

[Describe the unmet need or market failure the technology addresses. \
Quantify the problem. Reference relevant trends, policy drivers, or \
industry pain points.]

## 2. Technology and Innovation (~1 page)

[Describe the technical solution: mechanism, architecture, or principle. \
State current TRL and target TRL at project end. Highlight what is novel \
compared to existing solutions. Reference IP status.]

### 2.1 Current Status and Prior Results

[Summarise prototype data, publications, or prior project results that \
demonstrate feasibility.]

### 2.2 Technical Risks and Mitigation

[Identify key technical risks. For each, describe the mitigation strategy \
and how go/no-go milestones address the risk.]

## 3. Market and Commercialisation (~1.5 pages)

### 3.1 Target Market

[Define market segments. Quantify TAM / SAM / SOM with sources. Describe \
target customers.]

### 3.2 Value Proposition

[Quantify the benefit vs. status quo. Use concrete numbers: cost savings, \
time reduction, performance improvement.]

### 3.3 Business Model and Go-to-Market

[Describe the commercialisation path: licensing, spin-off, partnership, or \
direct sales. Include timeline and key milestones.]

### 3.4 Market Dialogue and Industry Engagement

[Reference specific meetings, LOIs, pilot agreements, or letters of support. \
Name partners where possible. Demonstrate that market demand is validated \
beyond desk research.]

### 3.5 Competitive Landscape

[Compare against alternatives. Use a positioning table if helpful. Explain \
sustainable differentiation.]

## 4. Green Transition and Societal Impact (~0.5 page)

[Describe contribution to sustainability / green transition if relevant. \
Quantify environmental benefits. Reference UN SDGs or national green \
transition priorities.]

## 5. Project Plan (~1 page)

### 5.1 Objectives

[List SMART project objectives.]

### 5.2 Work Packages and Milestones

| WP | Title | Months | Deliverable | Go/No-Go |
|----|-------|--------|-------------|----------|
| WP1 | [Title] | M1-M6 | [Deliverable] | [Decision criterion] |
| WP2 | [Title] | M4-M12 | [Deliverable] | [Decision criterion] |
| WP3 | [Title] | M10-M18 | [Deliverable] | — |

### 5.3 Budget Overview

| Cost Category | Amount (kNOK) | Notes |
|---------------|--------------|-------|
| Personnel | | |
| Equipment | | |
| External services | | |
| Other operating costs | | |
| **Total** | | |

## 6. Project Team and Triggering Effect (~0.5 page)

### 6.1 Team Competence

[Describe key personnel, their roles, and relevant expertise.]

### 6.2 Triggering Effect (Utløsende effekt)

[Explain why public funding is necessary. What specific barriers does the \
grant remove? What happens if funding is not received?]

---

## References

[Numbered reference list]
""",
    },
}
