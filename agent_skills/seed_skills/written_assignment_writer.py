"""Written Assignment Writer seed skill definition."""

WRITTEN_ASSIGNMENT_WRITER = {
    "slug": "written-assignment-writer",
    "name": "Written Assignment Writer",
    "description": """\
Help users write college-level written assignments such as essays, research \
papers, response papers, literature reviews, and argumentative essays.

Use this skill when the user wants to write, draft, outline, or revise an \
academic assignment.

Do NOT use for non-academic writing.""",
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
}
