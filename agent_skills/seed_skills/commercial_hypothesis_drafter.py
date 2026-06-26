"""Commercial Hypothesis Drafter seed skill definition."""

COMMERCIAL_HYPOTHESIS_DRAFTER = {
    "slug": "commercial-hypothesis-drafter",
    "name": "Commercial Hypothesis Drafter",
    "emoji": "🎯",
    "description": """\
Forge a single, sharp, commercial hypothesis for an early-stage \
technology idea, invention disclosure (DOFI), or other business idea: The skill takes you through defining the (named) user, \
the paying customer, the current workaround, the one-sentence solution, and the \
"10x better than what" claim, distilled into one Jobs-to-be-done paragraph.

Use this when an idea needs a crisp commercial hypothesis, typically before deeper work such \
as planing the commercial assesment, doing a market analysis or thinking about who to partner with.""",
    "instructions": """\
# Commercial Hypothesis Drafter

Use this skill to forge a single, sharp commercial hypothesis for an early-stage \
technology idea, invention disclosure (DOFI), or research result. A commercial \
hypothesis is the concrete bet about who has a problem (or undercovered need), what they do about it today, \
and why a proposed solution would make them switch. It is the foundational artifact \
that commercialization fo technology pivots around: if the \
hypothesis is absent or vague, every piece of downstream work is wasted.

State one limitation plainly: a commercial hypothesis, per this skill, is an artifact of judgment and prior knowledge\
 — a sharp, *falsifiable* starting bet, not proven demand. The fastest way to \
test it is to talk to the market. This skill produces well formulated hypothesis  \
possibly without interviews or desktop research; so treat the result as a thing to test next, not as definitive a conclusion.

## What a usable commercial hypothesis names

A usable hypothesis names at least:
- **User** — the human who actually uses the thing.
- **Customer** — the entity or role that pays or signs off (may differ from the user).
- **Current behavior / workaround** — what they do today, even poorlyly.
- **Proposed solution** — one sentence, no tech jargon.
- **The "10x better than what" claim** — why they'd switch.

Distil it into a Jobs-to-be-done synthesis: **"When [situation], I want to \
[motivation], so I can [outcome]."**

A good hypothesis reads like a story about a real person, not a filled-in form.

## Non-negotiable principles

These apply from intake to output.

**Start from a real person's problem, not the technology.** A hypothesis derived from \
what the invention *does* — rather than from a real person who feels a problem — is \
technology-push, and it is the most common failure mode for deep tech from research \
labs. Anchor on the person.

**Specificity is the only currency.** Vague answers get pushed. "Enterprises in \
healthcare" is not a customer. "Everyone needs this" means you can't find anyone who \
needs it. A useful answer has a name, a role, an organization, and a reason.

**The status quo is your real competitor.** Not the other startup, not the big \
company — the cobbled-together workaround the user is already living with. If \
"nothing" is the current solution, that's usually a sign the problem isn't painful \
enough to act on.

**Narrow beats wide, early.** The smallest version someone would pay real money for is \
more valuable than the full platform vision. Wedge first. Expand from strength. We \
don't want "the AI market" or "the healthcare market" — we want a tight statement like \
"infection-surveillance software for tertiary hospitals."

**Separate the user, the buyer, and the approver.** In many technology markets the \
person who benefits is not the person who pays, and the person who pays is not the \
person who approves technical, clinical, legal, or procurement acceptance. Name all \
three when they differ.

### Pushback patterns

These show the difference between soft exploration and useful diagnosis:

**Vague market → force specificity.**
- User: "This is for the healthcare sector."
- Weak: "That's a big market! What part of healthcare?"
- Strong: "Healthcare is not a market — it's a continent. What specific workflow, in \
what department, at what type of hospital, breaks down today? Name the person whose \
day gets worse."

**Platform vision → wedge challenge.**
- User: "The technology can be used across three industries."
- Weak: "Which industry should we focus on first?"
- Strong: "Three industries means zero customers. Which single use case, in which \
single segment, would someone pay for right now — even in a rough version?"

**Growth stats → vision test.**
- User: "The market report says 20 % CAGR."
- Weak: "That's a strong tailwind."
- Strong: "Growth rate is not a thesis. Every competitor cites the \
same stat. What's YOUR read on how this market changes in a way \
that makes THIS product more essential?"

**Undefined terms → precision demand.**
- User: "The solution makes the process more efficient."
- Weak: "What does the current process look like?"
- Strong: "'More efficient' is not a product feature — it's a feeling. What specific \
step takes too long or fails? How often? What does that cost?"

## Operating model

This is mostly a conversation, not a research project.
Gather the info you need from the user, then propose a commercial hypothesis, then discuss and iterate.

You probably don't need to use the canvas. If you need to look up facts online, use a subagent \
to keep the orchestrator context as clean as possbile. 

## Phase 1 — Build a clean fact base

Read or search throigh the provided materials for relevant information. From it, extract:

- what the invention does
- what problem it claims to solve
- who has this problem, how often, and how intensly?
- what they do to solve it today (even poorly)
- what other use cases exist (then repeat the above items for each use case)
- how mature the evidence is
- whether a prototype or reduction to practice exists

*If you don't find these answers in the provided materials, pause and challenge the user \
to produce the answers.*

Using the info you should be able to rewrite the invention into one sentence that a \
business person can understand, one sentence that a technical buyer can understand, and \
one sentence that captures the economic improvement.

Don't make the user repeat what's already in the materials. Note briefly whether you \
found good information, then challenge the user over the remaining gaps.

If the basic facts aren't in the materials and the user can't supply them, pause and \
challenge the user further. Maybe they need to go talk to the inventors to \
understand the idea better? The inventor, having invented the invention and sent an \
invention disclosure to the TTO must surely have had a hypothesis that this was useful \
in some way for someone?

**Failure modes**

When encountered, name these failure modes directly:

- **Invented need** — the "need" was derived from what the technology does, not from a \
real person saying it's a problem.
- **Technology-push** — fascination with the solution, no user anchor. Common for deep \
tech from research labs.
- **Boiling the ocean** — problem defined too broadly ("sustainability in logistics"). \
Narrow until you can name real organizations or users that feel it.
- **Sitcom customer** — plausible-sounding (e.g. "a social network for pet owners") \
but no proof that anyone actually needs it. Ask instead: "Who wants this so much they'd \
use a crappy, inconvenient v1 with bad UX?"
- **Conflating user and customer** — especially in B2B, B2B2C, regulated, or \
procurement-heavy markets.

## Phase 2 — Formulate the commercial hypothesis with the user

A usable hypothesis names at least:
- **User** — the human who actually uses the thing.
- **Customer** — the entity or role that pays or signs off (may differ \
from user).
- **Current behavior / workaround** — what they do today (even poorly).
- **Proposed solution** — one sentence, no tech jargon.
- **The "10x better than what" claim** — why they'd switch.

Aim for a Jobs-to-be-done synthesis: **"When [situation], I want to \
[motivation], so I can [outcome]."**

By now you have context from Phase 1. Use it. **Draft a hypothesis \
yourself and propose it to the user.** Don't ask the user to write \
it from scratch — synthesize what you've learned into a concrete \
statement and put it to them for judgment. The user's job is to \
refine it.

This is a critical moment for human judgment: the commercial hypothesis defines \
what the entire market analysis will revolve around. Make sure the user \
actively engages with it rather than rubber-stamping your proposal. \
Push back if they accept too quickly without engaging. Point out that \
this specifically requires human judgement.

After the user responds, iterate. Push for specificity — category- \
level answers don't count. "Healthcare enterprises" is not a \
customer. Push until there is a named role at a named type of \
organization, and a problem described as the user or customer would \
say it, not as the inventor would pitch it.

**GOOD hypothesis example**:

> **Proposed hypothesis:** Cardiac surgeons at Norwegian university \
hospitals currently rely on pre-operative CT scans and mental \
reconstruction to plan complex valve repairs — a process that takes \
30–60 minutes per case and depends heavily on individual experience. \
The proposed solution is a 3D-printed patient-specific heart model \
generated from existing CT data, letting the surgeon physically \
rehearse the procedure beforehand. The claim is that this cuts \
planning time by 50%+ and reduces intra-operative surprises, \
particularly for less experienced surgeons.
>
> **Customer job to be done (Cardiac surgeons):** "When I'm planning a complex valve repair, I want to \
physically see and handle the patient's anatomy beforehand, so I \
can anticipate complications instead of discovering them on the \
table."
>
> Does this capture it, or would you frame it differently?

**BAD hypothesis example:**

> **Proposed hypothesis:** 3D-printed heart models for surgeons to practice on. Hospitals could benefit from better surgical planning \
tools. The 3D printing technology is superior to current methods.
>
> Shall we proceed with planning?

This is bad because: no specific user role, no specific problem \
described from the user's perspective, no current behavior named, \
"superior" is undefined, and it asks to proceed without engaging \
the user's judgment.

**Another BAD hypothesis example:**

> Based on my analysis, the key user is cardiac surgeons, the \
customer is hospital procurement, the current behavior is CT-based \
planning, the solution is 3D-printed models, and the 10x claim is \
faster planning. The JTBD is "When planning surgery, I want better \
tools, so I can do better surgery." Moving on to Phase 3.

This is bad because: it reads like a filled-in form, not a story. \
The JTBD is generic — "better tools" and "better surgery" could \
apply to anything. It doesn't engage the user at all.

Name these failure modes directly when you see them:

- **Invented need** — the "need" was derived from what the technology \
does, not from a real person saying it's a problem.
- **Technology-push** — fascination with the solution, no user anchor. \
Common for deep tech from research labs.
- **Boiling the ocean** — problem defined too broadly ("sustainability \
in logistics"). Narrow until you can name real organizations or users \
that feel it.
- **Sitcom customer** — plausible-sounding (e.g. "a social network for \
pet owners") but no proof is established that anyone actually needs it. \
Ask instead: "Who wants this so much they want to use a crappy v1 that \
is super inconvenient and has a super bad UX"?
- **Conflating user and customer** — especially in B2B, B2B2C, \
regulated or procurement-heavy markets.

**Sharpening a vague hypothesis**

When technology features and user needs are confused and intermingled, multiple \
possible users are implied, or there's no clear target — the hypothesis isn't ready. \
Before narrowing, have the user map the full user × need space: list every plausible \
user or customer group and the need each might have. Only then narrow — this prevents \
premature commitment to the loudest-imagined user. When multiple needs are plausible, \
have the user rank them by importance × how poorly served they are today, and target \
the high-importance / underserved cell.

Once the hypothesis is confirmed, write it as one specific paragraph for the user to review.
""",
    "tool_names": [],
}
