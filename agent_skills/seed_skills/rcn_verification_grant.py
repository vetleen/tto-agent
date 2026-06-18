"""RCN Verification Grant Application Drafter seed skill definition."""

RCN_VERIFICATION_GRANT = {
    "slug": "rcn-verification-grant-drafter",
    "name": "RCN Verification Grant Application Drafter",
    "emoji": "💰",
    "description": """\
Draft Research Council of Norway (Forskningsrådet) "Verifiseringsprosjekt" \
(Verification Project) grant applications. Combines gap analysis of existing \
source materials with structured drafting of the web-based application form \
fields — designed for direct paste into the NFR portal.

Use when the user wants to write or review a "Verifiseringsprosjekt" or \
"verification project", "RCN verification grant". Do NOT use for \
Kvalifiseringsprosjekt (qualification) or other grant-writing tasks \
unrelated to the Verifiseringsprosjekt call.""",
    "instructions": """\
# RCN Verification Grant Application Drafter

Guide users through drafting a Research Council of Norway \
"Verifiseringsprosjekt" (Verification Project) application. Work in four \
phases: information gathering, gap analysis, interactive gap filling, and \
application form drafting.

Verification is a larger program than Qualification (up to NOK 5M, \
12–36 months). The evaluation bar is higher — evaluators expect rigorous \
market validation, clear IP strategy, and detailed work packages with \
measurable milestones tied to commercial decision points.

The application is written directly in the NFR web portal across discrete \
form fields. The output template mirrors these fields with character limits \
so the user can paste directly from Wilfred into the portal.

The application is evaluated on three criteria (scored 0–5):
1. **Kvalitet / Excellence** — Soliditet + Originalitet
2. **Effekter / Impact** — Potensial + Kunnskapsdeling og utnyttelse
3. **Gjennomføring / Implementation** — Prosjektgruppens kompetanse + \
Organisering

Minimum requirements: average ≥3.0 per criterion, no single score below 2. \
Applications scoring ≥4.0 average with no sub-score ≤3 receive automatic \
approval without a panel meeting.

## Phase 1 — Information gathering

Evaluate user-provided data (message, attached data room, or uploaded \
documents) for evidence covering the full checklist below. The checklist is \
organised to match the form sections. For every item, note what you found \
and where (document + page/section).

### Checklist

**A. Research Foundation & Innovation (Kvalitet — Soliditet + Originalitet)**

1. Research projects behind the results (project numbers, types, approved \
research organisation)
2. Research findings — what is new or different, why results are important \
for a future product, process, or service
3. Current TRL level with evidence
4. Target TRL at project end
5. Prototype, PoC, or pilot data; publications and prior project outputs
6. Innovation concept — which needs or problems it solves in a new or \
improved way
7. Positioning vs existing solutions — what is novel (state of the art)
8. Prior commercialisation support (qualification or verification) — \
results achieved so far
9. If previously rejected verification application — what has changed

**B. Effects & Impact (Effekter — Potensial + Kunnskapsdeling og utnyttelse)**

10. Market insight: target markets, customers, areas of application, \
commercial interest
11. Market size and potential (quantified, with sources)
12. Competitive landscape and positioning
13. Industry-specific requirements or regulations
14. Dialogue with external actors (customers, partners, users) — specific \
meetings, LOIs, pilot agreements, letters of support
15. Societal effects including sustainability and UN SDGs
16. Realisation plan — post-project strategy, challenges, and risks
17. Investment needs, plans for further financing, resource and expertise \
needs
18. Logistics and distribution plans
19. Prerequisites for realisation — rights (IP, FTO, patents), regulatory \
issues, key stakeholders/partners, key resources

**C. Implementation (Gjennomføring — Kompetanse + Organisering)**

20. Project objectives — main goal + sub-goals (SMART)
21. Team competence, experience, and relevance to project
22. Work packages with descriptions, milestones, and deliverables
23. Implementation plan — how WP activities, milestones, and resources \
together achieve project goals
24. Governance and roles — role distribution, strategic anchoring, \
management structure
25. Risk — key risks (technical, market, regulatory, organisational) and \
mitigation plan
26. Triggering effect — why public funding is necessary
27. Steering group composition (required for funded projects)
28. Revenue or savings projections (order of magnitude)

## Phase 2 — Gap analysis disposition

Load the "Disposition" template to the canvas using `skill_template_load`. \
Fill in each row:

- **Status:** ✅ Covered, ⚠️ Partial, ❌ Missing
- **Source:** document name + section, or "—" if missing
- **Summary:** one-line description of what was found or what is needed

Present the completed disposition to the user.

## Phase 3 — Interactive gap filling

Walk the user through every ⚠️ and ❌ item, starting with the most critical \
gaps. At the Verification funding level, evaluators are especially strict on:

- **Triggering effect** — why public funding is necessary, what barriers \
it removes
- **Market dialogue and competitive landscape** — evidence beyond desk \
research; specific conversations, meetings, agreements
- **Strategy for realisation** — a believable post-project commercialisation \
plan with identified risks and alternatives
- **Go/no-go milestones** — each milestone tied to a risk-reduction or \
commercial decision point (generic WPs without measurable milestones are a \
top weakness cited by evaluators)
- **Prior support disclosure** — must state any prior qualification or \
verification funding with results achieved
- **Prior rejection disclosure** — if a previous verification application \
was rejected, what specifically has changed
- **Steering group** — mandatory for funded projects; should cover all \
major risk areas

For each gap:
1. Explain why this item matters for the application score.
2. Suggest what good evidence looks like.
3. Ask the user to provide the information or confirm an assumption.
4. Update the disposition in the canvas as gaps are resolved.

Continue until the user is satisfied or explicitly moves to drafting.

## Phase 4 — Application form drafting

Load the "Application Form" template to the canvas using \
`skill_template_load`. Draft all fields following the NFR web form \
structure. Each section header and character limit corresponds to a field \
in the portal.

### Writing principles

- **Write in English.** The entire application must be in English.
- **No URLs.** Links to websites are not considered in the evaluation.
- **Frame research results toward commercialisation.** Verification funds \
the path from research result to market, not more research. Every sentence \
should point toward commercial value.
- **Higher evidence bar than Qualification.** At 10× the funding, \
evaluators expect more rigorous market validation, clearer IP strategy, and \
more detailed work packages. What passes in a 500K Qualification application \
may not suffice here.
- **Quantify benefits.** Replace vague claims ("significant market \
potential") with numbers ("NOK 2.4B addressable market by 2028, targeting \
5% share"). DO NOT fabricate data. Use [TBD] as a placeholder for missing \
figures.
- **Show market dialogue.** Reference specific conversations, meetings, or \
agreements with industry partners. Evaluators look for evidence that the \
market has been consulted, not just desk research.
- **Mirror RCN language.** Use terminology from the call: "verifisering", \
"kommersialiseringsløp", "utløsende effekt", "verdiskaping", "grønt skifte".
- **Emphasise green transition where relevant.** RCN prioritises projects \
contributing to sustainability. If there is a green angle, make it explicit.
- **Demonstrate triggering effect.** Explain clearly why the project cannot \
happen without public funding — what specific barriers does the grant remove?
- **Describe the trigger effect.** Explain what successful project outcomes \
will trigger — follow-on investment, market entry, licensing deal.
- **Connect milestones to commercial decisions.** A common weakness is \
generic work packages without measurable milestones. Each milestone should \
tie to a risk-reduction or go/no-go decision.
- **Disclose prior support.** If the project received qualification or \
verification funding before, state this at the start of the \
Forskningsgrunnlaget section with results achieved.
- **Reference sources for substantive claims.** Every factual claim — \
market sizes, statistics, research findings, regulatory requirements, \
competitive comparisons — should cite its source inline (e.g., "(Ref 1)" \
with a corresponding entry in Referanseliste). When in doubt, add a \
reference; the user can always remove unnecessary ones later. Unreferenced \
claims weaken credibility with evaluators. Prefer original sources.
- **Tell one coherent story.** The application should flow: Excellence → \
Impact → Implementation. What has been achieved? What is the commercial \
value and risk? What will be done concretely in this project?
- **Respect character limits.** Each field has a strict character limit \
enforced by the portal. Draft within these limits. If content exceeds the \
limit, flag it so the user can trim.
- **Aim for automatic approval.** Frame content to be unambiguously strong \
on each of the three criteria (≥4.0 average, no sub-score ≤3).""",
    "tool_names": ["skill_template_view", "skill_template_load"],
    "templates": {
        "Disposition": """\
# Verification Project — Gap Analysis Disposition

## Status Key
- ✅ Covered — sufficient information found in source materials
- ⚠️ Partial — some information found but incomplete
- ❌ Missing — no information found; must be provided

---

## A. Research Foundation & Innovation (Kvalitet)

| # | Item | Status | Source | Summary |
|---|------|--------|--------|---------|
| 1 | Research projects behind results (numbers, types, org) | | | |
| 2 | Research findings — what is new/different, importance | | | |
| 3 | Current TRL level + evidence | | | |
| 4 | Target TRL at project end | | | |
| 5 | Prototype / PoC / pilot data, publications, outputs | | | |
| 6 | Innovation concept — needs/problems it solves | | | |
| 7 | Positioning vs existing solutions (state of the art) | | | |
| 8 | Prior commercialisation support + results achieved | | | |
| 9 | Prior rejected verification application — what changed | | | |

## B. Effects & Impact (Effekter)

| # | Item | Status | Source | Summary |
|---|------|--------|--------|---------|
| 10 | Market insight: markets, customers, applications | | | |
| 11 | Market size + potential (quantified) | | | |
| 12 | Competitive landscape + positioning | | | |
| 13 | Industry requirements / regulations | | | |
| 14 | Dialogue with external actors (meetings, LOIs, pilots) | | | |
| 15 | Societal effects + sustainability / UN SDGs | | | |
| 16 | Realisation plan — post-project strategy + risks | | | |
| 17 | Investment needs, further financing, resources | | | |
| 18 | Logistics + distribution plans | | | |
| 19 | Prerequisites: IP/FTO, regulatory, partners, resources | | | |

## C. Implementation (Gjennomføring)

| # | Item | Status | Source | Summary |
|---|------|--------|--------|---------|
| 20 | Project objectives — main goal + sub-goals (SMART) | | | |
| 21 | Team competence, experience + relevance | | | |
| 22 | Work packages + milestones + deliverables | | | |
| 23 | Implementation plan (WP activities → goals) | | | |
| 24 | Governance + roles + strategic anchoring | | | |
| 25 | Risks (technical, market, regulatory, org) + mitigation | | | |
| 26 | Triggering effect — why public funding is necessary | | | |
| 27 | Steering group composition | | | |
| 28 | Revenue / savings projections | | | |

---

*Updated: [date]*""",
        "Application Form": """\
# Verification Application — [Project Title]

---

## Mål (Goals)

### Hovedmål
**[max 500 characters]**

[Main goal of the project — should be clear and verifiable. Avoid \
sensitive information.]

### Delmål
**[max 1000 characters]**

[Sub-goals that contribute to the main goal. Specify expected results \
under each sub-goal. Avoid sensitive information.]

---

## Sammendrag
**[max 1500 characters]**

[Project summary providing an overview. Avoid sensitive information — \
this text may be sent to external evaluators and, if funded, published \
publicly in the project database.]

---

## Kvalitet (Quality / Excellence)

### Soliditet — Forskningsgrunnlaget
**[max 10000 characters, rich text]**

[Explain which research projects (state project numbers and project types) \
and research results form the basis for this application. State who and \
which approved research organisation conducted the research, what findings \
have been made (what is new or different), and why these research results \
are important for a future product, process, or service. Indicate the \
current TRL level.

If the project has received prior commercialisation support (qualification \
or verification), state this clearly here with the project number and \
results achieved.

If a previous verification application was rejected, state the project \
number and describe what has changed since the previous application.]

### Originalitet — Innovasjonen
**[max 10000 characters, rich text]**

[Describe the innovation concept that forms the basis of the project and \
which needs or problems it can help solve in a new or improved way. \
Position the innovation relative to existing solutions and articulate \
what is novel.]

### Referanseliste

[Numbered reference list. Use "(Ref X)" notation inline in the sections \
above.

1. [Author(s), Title, Source, Year]
2. ...
]

---

## Effekter (Impact)

### Potensial — Markedsinnsikt og bruksområder
**[max 5000 characters, rich text]**

[Describe market insight: the need for the solution, which markets it \
targets, who the customers are, areas of application, why results are \
commercially interesting, and the commercial potential. Address: potential \
target markets, users, customers, partners, competitors, market size and \
potential, and any industry-specific requirements and regulations. Describe \
any dialogue with relevant external actors. Describe the competitive \
landscape. Focus on questions relevant to the project's current phase.]

### Potensial — Samfunnseffekter
**[max 5000 characters, rich text]**

[Describe the effects you expect the project results will have on society, \
including sustainability. Reference UN SDGs or national green transition \
priorities where relevant.]

### Kunnskapsdeling og utnyttelse — Realiseringsplan
**[max 5000 characters, rich text]**

[Describe the main features of what will happen after this project. What \
are the possible strategies and corresponding challenges and risks? Include \
a high-level description of investment needs, plans for further financing, \
future needs for resources and expertise, plans for logistics and \
distribution, etc.]

### Kunnskapsdeling og utnyttelse — Forutsetninger for realisering
**[max 2000 characters, rich text]**

[Describe the prerequisites that must be met to successfully realise the \
product, process, and/or service — for example various types of rights \
(IP, FTO, patents), regulatory issues, important stakeholders/partners, \
key resources, etc.]

---

## Gjennomføring (Implementation)

### Prosjektgruppens kompetanse
**[max 3000 characters, rich text]**

[Describe the project group's combined competence and experience and how \
it is relevant for carrying out the project with high quality. Explain \
how the competence covers the project's academic and methodological needs \
and ensures high implementation capability. Ensure all named persons are \
also registered under "Andre prosjektdeltakere" → "Roller i prosjektet" \
in the portal.]

### Organisering — Gjennomføringsplan
**[max 3000 characters, rich text]**

[Describe how the activities, milestones, and resource allocation in the \
work packages collectively contribute to achieving the project's goals.]

### Organisering — Styring og roller
**[max 5000 characters, rich text]**

[Describe the role distribution in the project and how the project is \
strategically anchored in the participating organisations. Explain how the \
project will be governed and how the organisation ensures clear \
responsibility allocation, effective collaboration, and sufficient capacity \
to carry out the project's work packages and deliverables.

Note: funded projects are required to establish a steering group covering \
all major risk areas of the project. Describe the planned steering group \
composition and mandate.]

### Organisering — Risiko
**[max 3000 characters, rich text]**

[Describe potential risks in the project and the plan for managing them. \
Cover technical, market, regulatory, and organisational risks. For each \
risk, describe likelihood, potential impact, and mitigation strategy.]

---

## Arbeidspakker (Work Packages)

*Create 2–10 work packages. Each work package describes a sub-goal and \
the activities to achieve it. Budget fields are handled separately in the \
portal and are not included here.*

### Arbeidspakke 1: [Name]

**Navn på arbeidspakke:** [max 100 characters]
**Startmåned:** [MM/YYYY]
**Sluttmåned:** [MM/YYYY]
**Aktivitetskategori:** Kommersialisering
**Ansvarlig organisasjon:** [Name]

**Beskrivelse av arbeidspakken** [max 1500 characters]:
[Describe which challenges this work package will address.]

**Milepæler:**

| # | Tittel [max 100 chars] | Beskrivelse [max 1000 chars] | Dato | Kritisk? |
|---|------------------------|------------------------------|------|----------|
| 1 | [Milestone title] | [Description — tie to risk reduction or go/no-go decision] | [dd/MM/YYYY] | [Ja/Nei] |

---

### Arbeidspakke 2: [Name]

**Navn på arbeidspakke:** [max 100 characters]
**Startmåned:** [MM/YYYY]
**Sluttmåned:** [MM/YYYY]
**Aktivitetskategori:** Kommersialisering
**Ansvarlig organisasjon:** [Name]

**Beskrivelse av arbeidspakken** [max 1500 characters]:
[Describe which challenges this work package will address.]

**Milepæler:**

| # | Tittel [max 100 chars] | Beskrivelse [max 1000 chars] | Dato | Kritisk? |
|---|------------------------|------------------------------|------|----------|
| 1 | [Milestone title] | [Description — tie to risk reduction or go/no-go decision] | [dd/MM/YYYY] | [Ja/Nei] |

---

### Arbeidspakke 3: [Name]

**Navn på arbeidspakke:** [max 100 characters]
**Startmåned:** [MM/YYYY]
**Sluttmåned:** [MM/YYYY]
**Aktivitetskategori:** Kommersialisering
**Ansvarlig organisasjon:** [Name]

**Beskrivelse av arbeidspakken** [max 1500 characters]:
[Describe which challenges this work package will address.]

**Milepæler:**

| # | Tittel [max 100 chars] | Beskrivelse [max 1000 chars] | Dato | Kritisk? |
|---|------------------------|------------------------------|------|----------|
| 1 | [Milestone title] | [Description — tie to risk reduction or go/no-go decision] | [dd/MM/YYYY] | [Ja/Nei] |
""",
    },
}
