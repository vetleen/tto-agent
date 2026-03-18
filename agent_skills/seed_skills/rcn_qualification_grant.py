"""RCN Qualification Grant Application Drafter seed skill definition."""

RCN_QUALIFICATION_GRANT = {
    "slug": "rcn-qualification-grant-drafter",
    "name": "RCN Qualification Grant Application Drafter",
    "description": """\
Draft Research Council of Norway (Forskningsrådet) "Kvalifiseringsprosjekt" \
(Qualification Project) grant applications. Combines gap analysis of existing \
source materials with structured drafting of the 5-page project description.

Use when the user wants to write or review "Kvalifiseringsprosjekt", "qualification project", \
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

The project description follows the RCN template with three main sections, \
each corresponding to an evaluation criterion:
1. **Research and Innovation** (criterion 1) — ~1 page
2. **Impacts and Outcomes** (criterion 2) — ~1–2 pages
3. **Implementation** (criterion 3) — ~1–2 pages

## Phase 1 — Information gathering

Search the attached data room for evidence covering the full checklist below. \
The checklist is organised to match the three sections of the RCN project \
description template. For every item, note what you found and where \
(document + page/section).

### Checklist

**A. Research and Innovation (Section 1, items 1–8)**
1. Problem statement — what unmet need does the technology address?
2. Technical solution — mechanism, architecture, principle, intended use
3. Current TRL level and evidence supporting the assessment
4. Target TRL at project end
5. Prior research projects and key results (project numbers, types, approved \
research organisation)
6. Prototype, proof-of-concept, or pilot data
7. Publications and prior project outputs
8. How the innovation differs from existing solutions (state of the art)

**B. Impacts and Outcomes (Section 2, items 9–20)**
9. Target market, market size and customer segments
10. Customer pain points — evidence of demand (interviews, LOIs, surveys)
11. Value proposition — quantified benefit vs. status quo
12. Competitive landscape and positioning
13. Industry-specific requirements or regulations
14. Industry partner(s) and their role
15. Market dialogue evidence (meetings, pilot agreements, letters of support)
16. Business model and commercialisation path (licensing, spin-off, sale, \
partnership)
17. Strategy for realisation — post-project plan, conditions, key \
stakeholders/partners, key resources
18. IP status and rights (patents, freedom-to-operate, licensing)
19. Revenue or savings projections (order of magnitude)
20. Societal, environmental, and sustainability impact (positive and negative)

**C. Implementation (Section 3, items 21–29)**
21. Project objectives (SMART)
22. Key tasks, critical questions, and uncertainties
23. Work packages with deliverables and go/no-go milestones
24. Project timeline (Gantt-level)
25. Budget breakdown by cost category and partner
26. Key technical risks and mitigation strategies
27. Team members, roles, and relevant experience
28. External expertise, sub-contractors, and collaboration setup
29. Triggering effect — why public funding is necessary

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
gaps (items that RCN evaluators weight heavily:triggering effect, market dialogue, \
strategy for realisation (i.e. the commercialization plan), go/no-go milestones, research results, societal impact).

For each gap:
1. Explain why this item matters for the application score.
2. Suggest what good evidence looks like.
3. Ask the user to provide the information or confirm an assumption.
4. Update the disposition in the canvas as gaps are resolved.

Continue until the user is satisfied or explicitly moves to drafting.

## Phase 4 — Project description drafting

Load the "Project Description" template to the canvas using \
`load_template_to_canvas`. Draft the full 5-page project description following \
the three-section RCN template structure.

### RCN-specific writing principles

- **Frame research results toward commercialisation.** RCN \
Kvalifiseringsprosjekt funds the path from research result to market, not \
more research. Every sentence should point toward commercial value. 

- **Quantify benefits.** Replace vague claims ("significant market potential") \
with numbers ("NOK 2.4B addressable market by 2028, targeting 5% share"). But \
DO NOT make anything up. If you need a piece of data you don't have, use [TBD] \
as a placeholder.
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
- **Describe the trigger effect.** Explain what successful project outcomes \
will trigger — the next phase, follow-on investment, market entry.
- **Include go/no-go milestones.** Show that the project has built-in \
decision points to cut losses early if results are negative.
- **Align budget with activities.** Every cost should trace to a work package. \
Flag any mismatch.

Draft section by section, pausing after each major section (1, 2, 3) to let \
the user review and provide feedback before continuing.""",
    "tool_names": ["view_template", "load_template_to_canvas"],
    "templates": {
        "Disposition": """\
# Qualification Project — Gap Analysis Disposition

## Status Key
- ✅ Covered — sufficient information found in data room
- ⚠️ Partial — some information found but incomplete
- ❌ Missing — no information found; must be provided

---

## A. Research and Innovation (Section 1)

| # | Item | Status | Source | Summary |
|---|------|--------|--------|---------|
| 1 | Problem statement | | | |
| 2 | Technical solution (mechanism, architecture, intended use) | | | |
| 3 | Current TRL level + evidence | | | |
| 4 | Target TRL at project end | | | |
| 5 | Prior research projects + key results | | | |
| 6 | Prototype / PoC / pilot data | | | |
| 7 | Publications + prior project outputs | | | |
| 8 | How innovation differs from existing solutions (state of the art) | | | |

## B. Impacts and Outcomes (Section 2)

| # | Item | Status | Source | Summary |
|---|------|--------|--------|---------|
| 9 | Target market, market size + customer segments | | | |
| 10 | Customer pain points / demand evidence | | | |
| 11 | Value proposition (quantified) | | | |
| 12 | Competitive landscape + positioning | | | |
| 13 | Industry requirements / regulations | | | |
| 14 | Industry partner(s) + role | | | |
| 15 | Market dialogue evidence | | | |
| 16 | Business model + commercialisation path | | | |
| 17 | Strategy for realisation (post-project) | | | |
| 18 | IP status + rights | | | |
| 19 | Revenue / savings projections | | | |
| 20 | Societal, environmental + sustainability impact | | | |

## C. Implementation (Section 3)

| # | Item | Status | Source | Summary |
|---|------|--------|--------|---------|
| 21 | Project objectives (SMART) | | | |
| 22 | Key tasks, questions + uncertainties | | | |
| 23 | Work packages + go/no-go milestones | | | |
| 24 | Project timeline | | | |
| 25 | Budget breakdown | | | |
| 26 | Key technical risks + mitigation | | | |
| 27 | Team members, roles + experience | | | |
| 28 | External expertise + collaboration setup | | | |
| 29 | Triggering effect | | | |

---

*Updated: [date]*""",
        "Project Description": """\
# [Project Title]

[Short descriptive title.]

[Describe the unmet need or market failure the technology addresses. \
Quantify the problem. Reference relevant trends, policy drivers, or \
industry pain points.]

[If the project has applied for a Commercialisation Project in the past but \
not received funding, explain what has changed since the previous application \
and state the project number.]

---

## 1. Research and Innovation (~1 page)



### Research Results
[Describe the technical solution: mechanism, architecture, or principle. \
State current TRL and target TRL at project end. Highlight what is novel in a techbnical sense \
compared to existing solutions. Reference IP status. Summarise prototype data, \
publications, or prior project results that demonstrate feasibility.]

### Level of Innovation
[Describe the problem, to solve from the user or customer perspective. Describe how \
that problem is solved today ("State-of-the-art"). Describe how the new invention \
can solve the problem, hiughlighting it's advantages.]
---

## 2. Impacts and Outcomes (~1–2 pages)

### Market Insight and Areas of Application
[Define market segments. Quantify TAM / SAM / SOM with sources. Describe \
target customers.]
[Quantify the benefit vs. status quo. Use concrete numbers: cost savings, \
time reduction, performance improvement.]
[Compare against alternatives. Use a positioning table if helpful. Explain \
sustainable differentiation.]
### Strategy for Realisation
[Describe the commercialisation path: licensing, spin-off, partnership, or \
direct sales. Include timeline and key milestones.]
[Reference specific meetings, LOIs, pilot agreements, or letters of support. \
Name partners where possible. Demonstrate that market demand is validated \
beyond desk research.]

### Benefit to Society and Sustainability

[Describe contribution to sustainability / green transition if relevant. \
Quantify environmental benefits. Reference UN SDGs or national green \
transition priorities. Keep this section short and to the point. It's just \
check on the reviewers list, not something that actually sells.]
---

## 3. Implementation (~1–2 pages)

### Project Plan
[List SMART project objectives for the project. Explain why public funding is necessary. \
 What specific barriers does the grant remove? What happens if funding is not received?]  
| WP | Title | Period (months) | Key Activities | Deliverable or Milestone | Go/No-Go |
|----|-------|--------|---------------|-------------|----------|
| WP1 | [Title] | M1–M6 | [Activities] | [Deliverable or Milestone] | [Criterion] |
| WP2 | [Title] | M4–M12 | [Activities] | [Deliverable or Milestone] | [Criterion] |
| WP3 | [Title] | M10–M18 | [Activities] | [Deliverable or Milestone] | — |

[Comment briefly what the money will be spent on (Personell, Travel, Marketing, Consultants? IP \
Filings. Don't allocate anything to equipment, and don't mention patent upkeep!)]
### Management, Team and Expertise
[Describe key personnel, their roles, and relevant expertise.]
""",
    },
}
