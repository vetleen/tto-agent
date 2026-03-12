"""System skill definitions seeded on every migrate."""

SYSTEM_SKILLS = [
    {
        "slug": "skill-creator",
        "name": "Skill Creator",
        "description": """\
Create a new agent skill, improve existing ones, and optimize skill \
triggering. A skill is a set of instructions, tools and templates you drop into an AI agent's prompt \
that teaches it how to do something specific. At its core, a skill \
is just markdown text with 3-5 parts: a name, a description,  \
a body of instructions, and optionally, additional tools and templates. \
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
- **slug** — URL-safe identifier, auto-generated from name (e.g. `patent-claim-drafter`). 1-64 chars.
- **description** — 1-1024 chars. This is the ONLY text the system sees when deciding whether to activate the skill. It is the primary trigger mechanism.
- **instructions** — The full playbook injected into your system prompt when the skill is active. This is where the skill's logic lives.
- **tool_names** — List of tool names the skill needs (e.g. `["search_documents", "read_document"]`). These tools become available only when this skill is active.
- **templates** — Named text templates associated with the skill (e.g. a patent claim format, a report skeleton). When the skill is active, template names are listed in the system prompt; the agent accesses their content on demand via `view_template` or `load_template_to_canvas`.

Skills exist at three levels: **system** (built-in, not editable), **org** (shared within an organization), and **user** (personal). Higher levels shadow lower ones by slug — a user-level skill with the same slug as a system skill overrides it.

## Key workflow concept

**The canvas is your workspace for long-form text.** Use `write_canvas` / `edit_canvas` to draft instructions or templates, iterate with the user, then `save_canvas_to_skill_field` to persist the final version to the skill. Use `show_skill_field_in_canvas` to load existing content back into the canvas for editing.

## Workflow

You guide the user through a repeating loop:

1. **Capture intent** — understand what the skill should do
2. **Create the skill** — use `create_skill` to create the DB record
3. **Write the description** — craft the trigger description, save via `edit_skill`
4. **Draft instructions** — write the instructions in the canvas
5. **Iterate** — spar with the user, refine formulations in the canvas
6. **Save** — use `save_canvas_to_skill_field` to persist instructions
7. **Create templates** — when relevant, add templates the skill should use
8. **Attach tools** — choose which existing tools the skill needs via `edit_skill` with `tool_names`
9. **Test** — have the user try the skill in a fresh conversation
10. **Review & improve** — revise based on feedback
11. **Optimize description** — tune trigger accuracy

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

## Step 2: Create the skill and write the description

Use `create_skill` to create a new user-level skill. Then craft the description.

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

Save the description via `edit_skill` with a text edit on the description field.

---

## Step 3: Draft the instructions

The instructions are the actual playbook the agent follows once the skill
activates. They are loaded into the system prompt, so every token counts.

**Use the canvas as your workspace:**
1. Write the instructions using `write_canvas`
2. Iterate with the user using `edit_canvas`
3. When the user is satisfied, persist with `save_canvas_to_skill_field`

**Writing rules:**

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
  Consider whether a template (see Step 4) would be more appropriate for
  reusable output skeletons.

---

## Step 4: Create templates

Templates are named text blocks associated with a skill. When the skill is
active, template names are listed in the system prompt. The agent accesses
their full content on demand via `view_template` or `load_template_to_canvas`.

Use templates when the skill should produce output in a very specific format:
- A report skeleton
- A very specific email template
- A meeting minutes format
- A funding application template

**Workflow:**
1. Draft the template content in the canvas using `write_canvas`
2. Iterate with the user
3. Save to the skill with `save_canvas_to_skill_field` using the template name as the field_name
   — this creates or updates a template with that name

Or for short templates, use `add_skill_template` directly with content.

To view an existing template, use `show_skill_field_in_canvas` to load it into the canvas.

**Important:** When a skill has templates, add `view_template` and
`load_template_to_canvas` to the skill's `tool_names` — otherwise the agent
won't be able to access the templates at runtime.

---

## Step 5: Attach tools

Standard tools (canvas, web search, document search, etc.) are always
available to Wilfred and don't need to be added to a skill. But some
specialized tools are only available when a skill explicitly declares them
in its `tool_names`.

To discover and attach specialized tools:
1. Use `list_all_tools` to see which skill-specific tools exist
2. Use `inspect_tool` to read a tool's description and understand better what it does
3. You may discuss with the user which tools the skill actually needs
4. Save the list via `edit_skill`, passing the tool names in the `updates` parameter, e.g. `updates={"tool_names": ["search_documents", "read_document"]}`

The user's organization admin can disable specific tools per-skill, so the
effective tool set may be narrower than what you declare.

---

## Step 6: Test the skill

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

## Step 7: Review and improve

Based on test results, iterate on the skill:

1. Use `show_skill_field_in_canvas` to load the instructions into the canvas
2. Edit with the user
3. Save back with `save_canvas_to_skill_field`

**How to think about improvements:**

- **Generalize from the feedback.** Resist overfitting to specific test cases.
  Improve the underlying instructions so the model handles the *class* of problem.

- **Keep the prompt lean.** Remove instructions that aren't pulling their weight.

- **Explain the why.** Instead of "ALWAYS include axis labels", explain why
  labels matter. The model generalizes better from reasoning than from rules.

Repeat until the user is satisfied.

---

## Step 8: Optimize the description

After the instructions are solid, revisit the description. Think about:
- Did the skill trigger correctly during testing?
- Were there false positives or false negatives?
- Are there keywords users might use that aren't captured?

Update the description via `edit_skill` with text edits.

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

2. **The canvas is your workspace.** Draft, iterate, then save. Don't try
   to write perfect instructions in one shot.

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
            "add_skill_template",
            "edit_skill_template",
            "delete_skill_template",
            "list_all_tools",
            "inspect_tool",
        ],
    },
]


def seed_system_skills():
    """Create or update system-level skills. Idempotent."""
    from agent_skills.models import AgentSkill

    for skill_data in SYSTEM_SKILLS:
        AgentSkill.objects.update_or_create(
            slug=skill_data["slug"],
            level="system",
            defaults={
                "name": skill_data["name"],
                "description": skill_data["description"],
                "instructions": skill_data["instructions"],
                "tool_names": skill_data["tool_names"],
            },
        )
