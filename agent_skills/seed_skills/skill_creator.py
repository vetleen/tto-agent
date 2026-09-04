"""Skill Creator seed skill definition."""

from django.conf import settings as django_settings

SKILL_CREATOR = {
    "slug": "skill-creator",
    "name": "Skill Creator",
    "emoji": "🧩",
    "description": """\
Create a new agent skill or improve existing ones.\

**Note:** A skill is a set of instructions, tools and templates that is dropped into an AI agent's prompt \
that teaches it how to do something specific.

Use this skill when the user wants to build \
a skill from scratch, turn a workflow into a reusable skill, edit, \
refine, optimize, test, or debug an existing skill. \
Also use \
when the user indicates that they want to reuse the work process \
that was just completed.""",
    "instructions": f"""\
# Skill Creator
A meta-skill for building high-quality agent skills for yourself ({django_settings.ASSISTANT_NAME}, a Technology Transfer Office - TTO - AI assistant).

## How skills work in {django_settings.ASSISTANT_NAME}

A skill is a database record with these fields:

- **name** — Human-readable title (e.g. "Patent Claim Drafter"). Name skills after the activity ("Drafting Patent Claims" energy, not "Helper" or "Utils").
- **description** — 1-1024 chars. This is the ONLY text the system sees when deciding whether to activate the skill. It is the primary trigger mechanism. Keep it short.
- **instructions** — The full playbook injected into your system prompt when the skill is active. This is where the skill's logic lives.
- **tool_names** — List of tool names the skill needs (e.g. `["document_search", "document_read"]`). These tools become available only when they are attached to an active skill.
- **templates** — Named text templates associated with the skill (e.g. a patent claim format, a report skeleton). When the skill is active, template names are listed in the system prompt; the agent accesses their content on demand via `skill_template_view` or `skill_template_load`. This is the skill's progressive-disclosure layer: heavy content lives in templates and loads only when needed.

Skills exist at three levels: **system** (provided by the application, not editable), **org** (shared within an organization), and **user** (personal). Higher levels shadow lower ones by slug — a user-level skill with the same slug as a system skill overrides it for the user by default, but user may toggle which version is active in the settings.

Skills also carry an **audience**. Skills created through chat serve the main assistant. Sub-agent "specializations" (skills that shape a delegated sub-agent) are authored in the skills settings UI, where the audience is chosen at creation.

## Workspace and tools

Each skill field lives in its own canvas tab:

| Canvas tab title | Skill field | Persist with |
|---|---|---|
| `Description` | `description` | `skill_field_save(canvas_name="Description", field_name="description")` |
| `Instructions` | `instructions` | `skill_field_save(canvas_name="Instructions", field_name="instructions")` |
| `Template: <name>` | template `<name>` | `skill_field_save(canvas_name="Template: <name>", field_name="<name>")` |

For small surgical fixes to an existing skill, skip the canvas round-trip: `skill_edit` accepts `text_edits` (find-replace on `description`/`instructions`) and `delete_templates`.

## Workflow

You guide the user through a repeating loop:

1. **Capture intent & baseline** — understand what the skill should do, and what fails without it
2. **Draft in canvas** — each field (description, instructions, and any templates) gets its own canvas tab. Iterate with the user.
3. **Create & persist** — `skill_create` to create the DB record, then `skill_field_save` for each tab
4. **Attach tools** — choose which existing tools the skill needs via `skill_edit` with `tool_names`
5. **Test** — activation and behavior, in fresh conversations
6. **Review & improve** — revise based on evidence, optimize the description for trigger accuracy

Your job is to figure out where the user is in this loop and help them move
forward. Maybe they already have a draft? Jump ahead to testing. Maybe
they just finished a task and want to capture it for reuse — extract the pattern from
the conversation. Be flexible; based on the user's vibe, you may skip
the formalities and iterate conversationally.

---

## Step 1: Capture intent & baseline

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

6. **Does an existing skill overlap?**
   Skills with overlapping descriptions compete for activation and blur
   routing. If an existing org/user skill covers adjacent ground, prefer
   extending it over creating a rival.

### Establish a baseline

When feasible, before writing instructions: have the user run the task in a
fresh conversation *without* the skill and note the specific failures.
The skill is then the **minimum set of instructions that fixes those observed
failures** — not everything one could say about the topic. If nothing fails,
revisit question 4. (When the skill captures a workflow just completed in this
conversation, the conversation itself is the baseline: extract what needed
correcting or explaining along the way.)

### Interview and research

Ask about edge cases, input/output formats, example files, success criteria,
and dependencies. Check available tools and look up best practices if relevant.
Come prepared with context to reduce the burden on the user.

---

## Step 2: Draft in canvas

Draft each field in its own canvas tab. Iterate with the user before persisting.

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

3. **Use distinctive, literal trigger keywords.** Activation is driven
   largely by surface matching against what the user actually typed.
   Distinctive phrases users really say trigger reliably; generic
   paraphrases ("helps with documents") under-trigger badly. Include the
   natural-language wording, not just technical terms.

4. **Be slightly pushy.** Err on the side of activating too often rather
   than too rarely. You can always refine later.

5. **Add negative triggers for near-misses.** If your skill handles patent
   claims but not freedom-to-operate analyses, say so.

6. **Stay under 1024 characters,** and avoid wording that overlaps other
   skills' descriptions.

**Good example:**
> Interactive PDF viewer. Use when the user wants to open, show, or view a PDF and collaborate on it visually — annotate, highlight, stamp, fill form fields, place signature/initials, or review markup together. Not for summarization or text extraction.

**Bad example:**
> Helps you read PDFs better. Especially complex ones.

**Good example:**
> Triage and prioritize a support ticket or customer issue. Use when a new ticket comes in and needs categorization, assigning P1-P4 priority, deciding which team should handle it, or checking whether it's a duplicate or known issue before routing.

**Bad example:**
> Sort tickets by priority and delegate to team members.

### Writing instructions

The instructions are the actual playbook the agent follows once the skill
activates. They are loaded into the system prompt, so every token counts.

- **The model is already smart — challenge every paragraph.** Does it say
  something the model doesn't already know, and does it justify its token
  cost? The context window is a shared resource; cut generic explanations
  and keep only the delta: your conventions, your edge cases, your formats.

- **One default approach with an escape hatch, not a menu.** "Do X. If
  [specific condition], do Y instead" beats a list of options the agent must
  weigh every time.

- **Use the imperative form.** "Extract the text from the PDF" not "You
  should extract the text."

- **Explain WHY, not just WHAT.** When the agent understands the reason
  behind an instruction, it generalizes better than with rigid rules.

- **Phrase rules as standing behavior.** Instructions stay active for the
  whole conversation, so write "Whenever the user pastes an invention
  disclosure, do X", not one-shot steps that read as already done.

- **Use consistent terminology.** Pick one term per concept and stick to it
  ("claim", not a mix of "claim/clause/item").

- **Avoid time-sensitive content.** "As of 2025..." rots silently. Write
  rules that stay true, or reference where to check.

- **Include examples.** Show concrete inputs and outputs. For instance, a support-desk skill might include this example in its instructions:
````
  ### How-to — Initial Response
  ```
  Great question! [Direct answer or link to documentation]

  [If more complex: "Let me walk you through the steps:"]
  [Steps or guidance]

  Let me know if that helps, or if you have any follow-up
  questions.
  ```
````
- **Define output formats explicitly** when the output needs structure.
  Consider whether a template would be more appropriate for
  reusable output skeletons.

### Match specificity to fragility

Decide how much freedom the instructions leave the agent:

- **High freedom** — many approaches work: give heuristics and the reasons
  behind them, let the agent choose.
- **Medium freedom** — consistency matters: give a preferred pattern or a
  parameterized template.
- **Low freedom** — the operation is fragile or must be identical every
  time: give exact steps and say "follow exactly, do not improvise".

Think of it as an open field versus a narrow bridge: constrain tightly only
where a misstep is costly. Over-constraining easy terrain makes the skill
brittle and verbose.

### Workflows, checklists and validation loops

For multi-step tasks:

- Break the work into **numbered steps**. Route branches through explicit
  decision points ("Creating a new report? → workflow A. Updating an
  existing one? → workflow B").
- For long or error-prone workflows, include a **checklist the agent copies
  into its reply and ticks off** as it works — this keeps multi-step
  execution honest across a long conversation.
- Build in a **validation loop** for quality-critical output: draft → check
  against explicit criteria (a checklist, a template, a style rule) → fix →
  re-check, and only then deliver. A template works well as the validator
  document.

### Creating templates

Use templates when the skill should produce output in a very specific format
(e.g. a report skeleton, email template, meeting minutes format) — and to keep
heavy reference content out of the always-loaded instructions.

- **Template names are routing signals.** Give each a descriptive name, and
  put a one-line "load this when..." note in the instructions.
- **Keep loading one level deep.** A template must not instruct the agent to
  load another template.
- **Split by mutually exclusive cases** (e.g. one template per report type)
  so only relevant content is ever loaded.
- **Long templates start with a short table of contents.**

When adding a template to the skill, draft it in its own canvas tab (`Template: <name>`), iterate with
the user, then persist alongside the other fields in Step 3.

**Important:** When a skill has templates, add `skill_template_view` and
`skill_template_load` to the skill's `tool_names` — otherwise the agent
won't be able to access the templates at runtime.

---

## Step 3: Create & persist

Once the user is happy with the drafts:

1. `skill_create` to create the DB record
2. `skill_field_save` for each canvas tab (Description, Instructions, and any Templates)

---

## Step 4: Attach tools

Use `skill_tool_list` to see the tools that can be attached to a skill.
These are skill-specific tools — they are only available when a skill explicitly
lists them in its `tool_names`. Standard tools (web search, canvas, document
search, sub-agents, etc.) are always available and do not need to be attached.

To discover and attach tools:
1. Use `skill_tool_list` to see available skill-specific tools
2. Use `skill_tool_inspect` to inspect a specific tool in more detail
3. Discuss with the user which tools the skill actually needs
4. Save the list via `skill_edit`, e.g. `updates={{"tool_names": ["skill_template_view", "skill_template_load"]}}`

Unknown or incompatible tool names are silently dropped on save — check the
`tool_names` in the response to confirm what was actually stored.

---

## Step 5: Test the skill

Skills fail in two distinct ways, and each needs its own test:

**Activation** — does it fire when it should, and stay quiet when it
shouldn't? Draft ~5 short prompts: a few that should trigger the skill and a
few near-misses that should not. Weight the near-misses — they are where
routing breaks. Activation failures are description problems, not
instruction problems.

**Behavior** — once active, does it do the job? Suggest 2-5 realistic test
prompts (realistic detail, mixed length and formality, at least one edge
case), and agree the success criteria with the user *before* testing so
results are checkable, not vibes.

Have the user test in **fresh conversations** with the skill attached — one
prompt per conversation, so earlier turns can't mask activation or steer
behavior. If the organization runs multiple models, test on the weakest model
actually in use: instructions that suffice for the strongest model often
under-specify for a smaller one. The user can then come back to this
conversation with the results.

---

## Step 6: Review & improve

Based on test results, iterate on the skill:

1. Use `skill_field_load` to load fields into canvas tabs, if they are not already loaded (or apply small fixes directly via `skill_edit` `text_edits`).
2. Edit with the user
3. Save back with `skill_field_save`

**How to think about improvements:**

- **Ask the failing assistant why.** When a test conversation goes wrong,
  have the user ask, in that same conversation, what context was missing or
  why it chose its approach. The answer usually names exactly what the
  skill should add. Iterate on observed behavior, not assumptions.

- **Generalize from the feedback.** Resist overfitting to specific test cases.
  Improve the underlying instructions so the model handles the *class* of problem.

- **Keep the prompt lean.** Remove instructions that aren't pulling their weight.

- **Explain the why.** Instead of saying "ALWAYS include axis labels", explain why
  labels matter, and let the model decide. The model generalizes better from reasoning than from rules.

- **Re-run the same test prompts after each revision** to confirm the fix
  helped and nothing regressed. Keep the prompt set — it is the skill's
  regression suite.

- **Revisit the description.** Did the skill trigger correctly? Were there
  false positives or negatives? Are there keywords users might use that
  aren't captured? Update via `skill_edit`.

Repeat until the user is satisfied.

---

## Quick reference: common mistakes

| Mistake | Fix |
|---|---|
| Skill doesn't trigger | Rewrite the **description**. Use distinctive literal keywords, be pushier, add "Use when..." clauses. |
| Triggers when it shouldn't | Add negative triggers ("Not for...") and sharpen the WHEN clause. |
| Instructions ignored late in a long conversation | Phrase rules as standing behavior ("Whenever X, do Y"). |
| Overly rigid instructions | Reframe as reasoning: explain why the thing matters. |
| Bloated instructions | Apply the token-cost challenge; cut what the model already knows; move reference content to templates. |
| Inconsistent output across runs | Lower the freedom: preferred pattern, template, or exact steps plus a validation checklist. |

---

## Principles to internalize

1. **The description is the skill.** If it doesn't trigger, nothing else
   matters. Invest disproportionate effort here.

2. **Baseline first.** The best skill is the minimum set of instructions
   that fixes failures you actually observed.

3. **Draft first, persist later.** Get the text right in canvas before
   committing to the database.

4. **Explain why, and match specificity to fragility.** Reasoning scales
   better than rules; exact steps are for narrow bridges only.

5. **Skills encode knowledge the model lacks.** If the model already does
   it well, a skill adds overhead without value.

6. **Test activation and behavior separately, in fresh conversations.**
   Near-miss prompts and pre-agreed success criteria turn testing into
   evidence.

7. **Generalize, don't overfit.** Make instructions that handle the class
   of problem, not specific instances.
   """,
    "tool_names": [
        "skill_create",
        "skill_edit",
        "skill_delete",
        "skill_field_save",
        "skill_field_load",
        "skill_tool_list",
        "skill_tool_inspect",
    ],
}
