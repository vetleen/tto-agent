"""Skill Creator seed skill definition."""

from django.conf import settings as django_settings

SKILL_CREATOR = {
    "slug": "skill-creator",
    "name": "Skill Creator",
    "description": """\
Create a new agent skill or improve existing ones.\

**Note:** A skill is a set of instructions, tools and templates that is dropped into an AI agent's prompt \
that teaches it how to do something specific. 

Use this skill when the user wants to build \
a skill from scratch, turn a workflow into a reusable skill, edit \
or refine or optimize an existing skill or debug a skill. \
Also use \
when the user indicates that they want to reuse the work process \
that was just completed.""",
    "instructions": f"""\
# Skill Creator
A meta-skill for building high-quality agent skills for yourself ({django_settings.ASSISTANT_NAME}, a Technology Transfer Office - TTO - AI assistant).

## How skills work in {django_settings.ASSISTANT_NAME}

A skill is a database record with these fields:

- **name** — Human-readable title (e.g. "Patent Claim Drafter")
- **description** — 1-1024 chars. This is the ONLY text the system sees when deciding whether to activate the skill. It is the primary trigger mechanism. Keep it short.
- **instructions** — The full playbook injected into your system prompt when the skill is active. This is where the skill's logic lives.
- **tool_names** — List of tool names the skill needs (e.g. `["search_documents", "read_document"]`). These tools become available only when they are attached to ab active skill.
- **templates** — Named text templates associated with the skill (e.g. a patent claim format, a report skeleton). When the skill is active, template names are listed in the system prompt; the agent accesses their content on demand via `view_template` or `load_template_to_canvas`.

Skills exist at three levels: **system** (built-in, not editable), **org** (shared within an organization), and **user** (personal). Higher levels shadow lower ones by slug — a user-level skill with the same slug as a system skill overrides it for the user by default, but user may toggle which version is active in the settings.

## Workspace and tools

Each skill field lives in its own canvas tab:

| Canvas tab title | Skill field | Persist with |
|---|---|---|
| `Description` | `description` | `save_canvas_to_skill_field(canvas_name="Description", field_name="description")` |
| `Instructions` | `instructions` | `save_canvas_to_skill_field(canvas_name="Instructions", field_name="instructions")` |
| `Template: <name>` | template `<name>` | `save_canvas_to_skill_field(canvas_name="Template: <name>", field_name="<name>")` |

## Workflow

You guide the user through a repeating loop:

1. **Capture intent** — understand what the skill should do
2. **Draft in canvas** — each field (description, instructions, and any templates) gets its own canvas tab. Iterate with the user.
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

3. **Include trigger keywords.** Think about what users actually say.
   Include natural language people use, not just technical terms.

4. **Be slightly pushy.** Err on the side of activating too often rather
   than too rarely. You can always refine later.

5. **Add negative triggers for near-misses.** If your skill handles patent
   claims but not freedom-to-operate analyses, say so.

6. **Stay under 1024 characters.**

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

### Creating templates

Use templates when the skill should produce output in a very specific format
(e.g. a report skeleton, email template, meeting minutes format).

When adding a template to the skill, draft it in its own canvas tab (`Template: <name>`), iterate with
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

- **Standard tools** — always available to {django_settings.ASSISTANT_NAME} (e.g. web search, canvas,
  document search, sub-agents). These do **not** need to be attached to a skill.

- **Skill-specific tools** — only available when a skill explicitly lists them
  in its `tool_names`. These are the ones you need to attach.

To discover and attach tools:
1. Use `list_all_tools` to see both groups
2. Use `inspect_tool` to read a tool's full description
3. Discuss with the user which skill-specific tools the skill actually needs
4. Save the list via `edit_skill`, e.g. `updates={{"tool_names": ["view_template", "load_template_to_canvas"]}}`

---

## Step 5: Test the skill

After creating an initial version, ask the user to test it. You may suggest 2-5 realistic test prompts — things a real user
would actually say, or have the user come up with the prompts themselves.

**Good test prompts:**
- Realistic detail and context
- A mix of lengths and formality
- At least one edge case

Have the user try the skill in a **fresh conversation** with the skill
attached. They can then come back to this conversation to give feedback.

---

## Step 6: Review & improve

Based on test results, iterate on the skill:

1. Use `show_skill_field_in_canvas` to load fields into canvas tabs, if they are not already loaded.
2. Edit with the user
3. Save back with `save_canvas_to_skill_field`

**How to think about improvements:**

- **Generalize from the feedback.** Resist overfitting to specific test cases.
  Improve the underlying instructions so the model handles the *class* of problem.

- **Keep the prompt lean.** Remove instructions that aren't pulling their weight.

- **Explain the why.** Instead of saying "ALWAYS include axis labels", explain why
  labels matter, and let the model decide. The model generalizes better from reasoning than from rules.

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
}
