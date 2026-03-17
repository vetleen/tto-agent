"""System skill definitions seeded on every migrate."""

SYSTEM_SKILLS = [
    {
        "slug": "skill-creator",
        "name": "Skill Creator",
        "description": """\
Create a new agent skill, improve existing ones, and optimize skill \
triggering.

**Note:** A skill is a set of instructions, tools and templates you drop into an AI agent's prompt \
that teaches it how to do something specific. At its core, a skill \
is just markdown text with 3-5 parts: a name, a description,  \
a body of instructions, and optionally, additional tools and templates.

Use this skill when the user wants to build \
a skill from scratch, turn a workflow into a reusable skill, edit \
or refine an existing skill, debug why a skill isn't triggering, \
or optimize a skill description for better activation. Also use \
when the user indicates that they want to reuse the work process \
that was just completed, for example by saying "make this a skill", \
"capture this as a skill", or "turn this into a reusable workflow".""",
        "instructions": """\
# Skill Creator
A meta-skill for building high-quality agent skills for yourself (Wilfred, a Technology Transfer Office - TTO - AI assistant).

## How skills work in Wilfred

A skill is a database record with these fields:

- **name** — Human-readable title (e.g. "Patent Claim Drafter")
- **description** — 1-1024 chars. This is the ONLY text the system sees when deciding whether to activate the skill. It is the primary trigger mechanism.
- **instructions** — The full playbook injected into your system prompt when the skill is active. This is where the skill's logic lives.
- **tool_names** — List of tool names the skill needs (e.g. `["search_documents", "read_document"]`). These tools become available only when this skill is active.
- **templates** — Named text templates associated with the skill (e.g. a patent claim format, a report skeleton). When the skill is active, template names are listed in the system prompt; the agent accesses their content on demand via `view_template` or `load_template_to_canvas`.

Skills exist at three levels: **system** (built-in, not editable), **org** (shared within an organization), and **user** (personal). Higher levels shadow lower ones by slug — a user-level skill with the same slug as a system skill overrides it.

## Workspace and tools

Use one canvas tab per skill field you are drafting:

| Canvas tab title | Skill field | Persist with |
|---|---|---|
| `Description` | `description` | `save_canvas_to_skill_field(canvas_name="Description", field_name="description")` |
| `Instructions` | `instructions` | `save_canvas_to_skill_field(canvas_name="Instructions", field_name="instructions")` |
| `Template: <name>` | template `<name>` | `save_canvas_to_skill_field(canvas_name="Template: <name>", field_name="<name>")` |

## Workflow

You guide the user through a repeating loop:

1. **Capture intent** — understand what the skill should do
2. **Draft in canvas** — open one tab per field (Description, Instructions, Templates). Iterate with the user.
3. **Create & persist** — `create_skill` to create the DB record, then `save_canvas_to_skill_field` for each tab
4. **Attach tools** — choose which existing tools the skill needs via `edit_skill` with `tool_names`
5. **Test** — have the user try the skill in a fresh conversation
6. **Review & improve** — revise based on feedback, optimize the description for trigger accuracy

Your job is to figure out where the user is in this loop and help them move
forward. Maybe they already have a draft? Jump ahead to testing. Maybe
they just finished a task and want to capture it for reuse — extract the pattern from
the conversation. Be flexible; based on the user's vibe, you may skip
the formalities and iterate conversationally.

---

## Step 1: Capture intent

Start by understanding what the skill should do.

Answer these questions (ask the user where you can't infer with certainty; where you can infer, ask the user to confirm):

1. **What should this skill enable the agent to do?**
   Be specific. "Process PDFs" is vague. "Extract tables from scanned PDFs,
   clean the data, and output as CSV" is actionable.

2. **When should the skill trigger?**
   Think about user phrases, file types mentioned, task patterns. Think about
   near-misses too — what *shouldn't* trigger it?

3. **What is the expected output?**
   A canvas document? A conversational response? A structured analysis? Define the deliverable.

4. **Does this encode knowledge the model doesn't already have?**
   Skills are most valuable when they provide context the model lacks: your
   team's conventions, a domain workflow, a quality checklist, a specific
   output format. If the model can already do it well without help, a skill
   adds overhead without value. You may challenge the user about this ONCE,
   but if the user seems dismissive, drop it.

5. **Does this require tools?**
   Skills can declare which tools they need via `tool_names`. These tools
   must already exist in the system — you cannot create new tools. If the
   desired skill would require a tool that doesn't exist, inform the user
   and discuss whether the task can be achieved without it.

### Interview and research

Ask about edge cases, input/output formats, example files, success criteria,
and dependencies. Check available tools and look up best practices if relevant.
Come prepared with context to reduce the burden on the user.

---

## Step 2: Draft in canvas

Open one canvas tab per field you need to draft — typically Title, Description, Instructions, and optionally Templates. Iterate with the user on all tabs before persisting.

### Writing a great description

The description is not a summary — it is a routing instruction. A bad
description means the skill never fires, no matter how good the instructions are.

**Principles:**

1. **Write in third person.** The description is shown alongside other skills
   for selection. First-person ("I can help you") creates confusion.
   Write "Drafts patent claims based on invention disclosures."

2. **Describe both WHAT and WHEN.** Say what the skill does, then
   explicitly say when to use it: "Use when the user asks to draft,
   review, or refine patent claims."

3. **Include trigger keywords.** Think about what users actually say.
   Include natural language people use, not just technical terms.

4. **Be slightly pushy.** Err on the side of activating too often rather
   than too rarely. You can always refine later.

5. **Add negative triggers for near-misses.** If your skill handles patent
   claims but not freedom-to-operate analyses, say so.

6. **Stay under 1024 characters.**

**Good example:**
> Drafts patent claims based on invention disclosures and prior art analysis.
> Use when the user wants to write, review, or refine patent claims, or when
> they mention "claims", "independent claim", "dependent claim", or "claim set".
> Also use when discussing claim scope, claim language, or patent prosecution
> strategy. Do NOT use for freedom-to-operate analyses, patentability searches,
> or general IP portfolio questions.

**Bad example:**
> Helps with patents.

### Writing instructions

The instructions are the actual playbook the agent follows once the skill
activates. They are loaded into the system prompt, so every token counts.

- **Keep instructions as short and concise as possible while maintaining maximum effectiveness.** The agent's context window is a
  shared resource.

- **Use the imperative form.** "Extract the text from the PDF" not "You
  should extract the text."

- **Explain WHY, not just WHAT.** When the agent understands the reason
  behind an instruction, it generalizes better than with rigid rules.

- **Only add context the model doesn't already have.** Focus on your team's
  conventions, your domain's edge cases, specific output formats — not
  things the model already knows.

- **Use consistent terminology.** Pick one term per concept and stick to it.

- **Include examples.** Show concrete inputs and outputs:
  ```
  ## Claim format
  Example:
  Input: A method for detecting anomalies using machine learning
  Output: 1. A method comprising: receiving sensor data...
  ```

- **Define output formats explicitly** when the output needs structure.
  Consider whether a template would be more appropriate for
  reusable output skeletons.

### Creating templates

Use templates when the skill should produce output in a very specific format
(e.g. a report skeleton, email template, meeting minutes format).

Draft each template in its own canvas tab (`Template: <name>`), iterate with
the user, then persist alongside the other fields in Step 3.

**Important:** When a skill has templates, add `view_template` and
`load_template_to_canvas` to the skill's `tool_names` — otherwise the agent
won't be able to access the templates at runtime.

---

## Step 3: Create & persist

Once the user is happy with the drafts:

1. `create_skill` to create the DB record
2. `save_canvas_to_skill_field` for each canvas tab (Description, Instructions, and any Templates)

---

## Step 4: Attach tools

Use `list_all_tools` to see every tool that exists. The output is split into
two groups:

- **Standard tools** — always available to Wilfred (e.g. web search, canvas,
  document search, sub-agents). These do **not** need to be attached to a skill.
  Note: Sub-agents (`create_subagent`) are standard tools — use them for parallel research or delegated sub-tasks when designing skills. No need to declare them in `tool_names`.

- **Skill-specific tools** — only available when a skill explicitly lists them
  in its `tool_names`. These are the ones you need to attach.

To discover and attach tools:
1. Use `list_all_tools` to see both groups
2. Use `inspect_tool` to read a tool's full description
3. Discuss with the user which skill-specific tools the skill actually needs
4. Save the list via `edit_skill`, e.g. `updates={"tool_names": ["view_template", "load_template_to_canvas"]}`

---

## Step 5: Test the skill

After creating an initial version, ask the user to test it. You may suggest 2-5 realistic test prompts — things a real user
would actually say, or have the user come up with the prompts themselves.

**Good test prompts:**
- Realistic detail and context
- A mix of lengths and formality
- At least one edge case
- At least one near-miss that *shouldn't* trigger the skill

Have the user try the skill in a **fresh conversation** with the skill
attached. They can then come back to this conversation to give feedback.

---

## Step 6: Review & improve

Based on test results, iterate on the skill:

1. Use `show_skill_field_in_canvas` to load fields into canvas tabs
2. Edit with the user
3. Save back with `save_canvas_to_skill_field`

**How to think about improvements:**

- **Generalize from the feedback.** Resist overfitting to specific test cases.
  Improve the underlying instructions so the model handles the *class* of problem.

- **Keep the prompt lean.** Remove instructions that aren't pulling their weight.

- **Explain the why.** Instead of "ALWAYS include axis labels", explain why
  labels matter. The model generalizes better from reasoning than from rules.

- **Revisit the description.** Did the skill trigger correctly? Were there
  false positives or negatives? Are there keywords users might use that
  aren't captured? Update via `edit_skill`.

Repeat until the user is satisfied.

---

## Quick reference: common mistakes

| Mistake | Fix |
|---|---|
| Skill doesn't trigger | Rewrite the **description**. Add trigger keywords, be pushier, add "Use when..." clauses. |
| Overly rigid instructions | Reframe as reasoning: explain why the thing matters. |

---

## Principles to internalize

1. **The description is the skill.** If it doesn't trigger, nothing else
   matters. Invest disproportionate effort here.

2. **Draft first, persist later.** Get the text right in canvas before
   committing to the database.

3. **Explain why, not just what.** Reasoning scales better than rules.

4. **Skills encode knowledge the model lacks.** If the model already does
   it well, a skill adds overhead without value.

5. **Iterate with real feedback.** The best skills emerge from the loop:
   draft, test, review, improve.

6. **Generalize, don't overfit.** Make instructions that handle the class
   of problem, not specific instances.
   """,
        "tool_names": [
            "create_skill",
            "edit_skill",
            "delete_skill",
            "save_canvas_to_skill_field",
            "show_skill_field_in_canvas",
            "list_all_tools",
            "inspect_tool",
        ],
    },
    {
        "slug": "written-assignment-writer",
        "name": "Written Assignment Writer",
        "description": """\
Help users write college-level written assignments such as essays, research \
papers, response papers, literature reviews, and argumentative essays.

Use this skill when the user wants to write, draft, outline, or revise an \
academic assignment, or when they mention "essay", "research paper", "written \
assignment", "thesis statement", "argument", "literature review", "response \
paper", "term paper", or "assignment prompt". Also use when the user pastes \
a marking rubric, assignment brief, or grading criteria and wants help \
producing the written work.

Do NOT use for citation-only tasks (e.g. "format this in APA"), slide decks, \
creative writing, or non-academic writing like emails or cover letters.""",
        "instructions": """\
# Written Assignment Writer

Produce college-level written assignments that are argument-driven, \
evidence-rich, and structurally coherent. Output all drafts to the canvas.

## Workflow

### 1. Decode the prompt

Before writing anything, analyze the assignment prompt as a specification:

1. Identify the **task verb** (analyze, compare, evaluate, argue, discuss) — \
this determines the expected mode of thinking, not just the topic.
2. Identify **scope limits** (time period, geographic focus, source count, word count).
3. If the user provides a rubric or marking criteria, extract the weighted \
dimensions and tell the user what the grader is actually looking for. Most \
students lose marks by misreading what's being asked.

Confirm your understanding with the user before proceeding.

### 2. Thesis and argument structure

Draft a **thesis statement** that is:
- **Debatable** — a reasonable person could disagree
- **Specific** — not a statement of fact or a vague generalization
- **Scoped** — achievable within the word limit

Then outline the argument as a sequence of claims, each supporting the thesis. \
Every body section advances ONE claim. State the claim, not just the topic. \
Bad: "Section 2: Social media." \
Good: "Section 2: Algorithmic amplification of outrage content undermines \
deliberative discourse."

Present the thesis and outline to the user for approval before drafting.

### 3. Draft

Write the full draft to the canvas following these rules:

**Argument discipline:**
- Every paragraph answers "so what?" — connect its point back to the thesis.
- Use explicit logical connectives between paragraphs. The reader should never \
guess why the next paragraph follows from the previous one.
- Address the strongest counterargument, not a strawman. Concede where \
appropriate, then explain why the thesis still holds.

**Evidence integration (the most common failure mode):**
- NEVER drop a quote and move on. Every piece of cited evidence must be \
followed by YOUR analysis: what it shows, why it matters, how it supports \
the claim.
- Integrate quotations grammatically into your sentences.
- Synthesize across sources — show how multiple sources relate or create a \
combined picture. Listing sources one-by-one is a book report, not an essay.
- Attribute claims precisely. "Studies show" is meaningless. Name the author \
or the finding.

**Introduction and conclusion:**
- The introduction establishes stakes, provides necessary context (no more), \
and states the thesis. No "funnel" intros starting from the dawn of time.
- The conclusion explains implications of the argument. Never merely restate \
the introduction.

### 4. Revise

Perform a structured revision in the canvas:

1. **Reverse-outline test:** Summarize each paragraph's claim in one line. \
If you can't, the paragraph lacks focus. If two paragraphs make the same \
claim, merge them. If the sequence doesn't build logically, reorder.
2. **Evidence audit:** Flag any claim lacking evidence and any evidence \
lacking analysis.
3. **Coherence check:** Verify every paragraph opening signals how it \
connects to the previous one.

Apply edits using `edit_canvas`, explaining each change.

### 5. Edit and format

- Cut filler ("it is important to note that", "in today's society").
- Replace vague language with precise terms.
- Verify formatting matches assignment requirements (headings, spacing, \
citation style).
- Add or fix in-text citations and the reference list in the required style \
(APA, MLA, Chicago, etc.).

## Key principles

- **The argument is the essay.** Structure, evidence, transitions — everything \
exists to advance the thesis. If a paragraph doesn't serve the argument, cut it.
- **Analysis over summary.** Explaining what a source says is necessary but \
insufficient. Explaining what it means for your argument is the actual work.
- **Precision over padding.** A 900-word essay that says something is better \
than a 1500-word essay that says nothing. Never pad to hit word count.
""",
        "tool_names": [],
    },
    {
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
    },
]


def seed_system_skills():
    """Create or update system-level skills. Idempotent."""
    from agent_skills.models import AgentSkill, SkillTemplate

    for skill_data in SYSTEM_SKILLS:
        skill, _ = AgentSkill.objects.update_or_create(
            slug=skill_data["slug"],
            level="system",
            defaults={
                "name": skill_data["name"],
                "description": skill_data["description"],
                "instructions": skill_data["instructions"],
                "tool_names": skill_data["tool_names"],
            },
        )
        # Seed templates from optional "templates" dict
        templates = skill_data.get("templates", {})
        for tmpl_name, tmpl_content in templates.items():
            SkillTemplate.objects.update_or_create(
                skill=skill,
                name=tmpl_name,
                defaults={"content": tmpl_content},
            )
        # Remove stale seeded templates no longer in seed data.
        # Only clean up when a "templates" key is explicitly present —
        # existing skills without it should not have templates deleted.
        if templates:
            skill.templates.exclude(name__in=templates.keys()).delete()
