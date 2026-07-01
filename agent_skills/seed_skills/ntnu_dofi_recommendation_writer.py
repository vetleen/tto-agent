"""NTNU DOFI Recommendation Writer seed skill definition."""

NTNU_DOFI_RECOMMENDATION_WRITER = {
    "slug": "ntnu-dofi-recommendation-writer",
    "name": "NTNU DOFI Recommendation Writer",
    "emoji": "📋",
    "description": """\
Draft NTNU Technology Transfer's initial assessment of ownership and \
commercialization potential, and its recommendation for the further process, \
following an invention disclosure (DOFI). Produces the recommendation letter and, \
when the recommendation is to proceed, the accompanying power-of-attorney \
(forvaltningsfullmakt) form.

Use when the user wants to write a "DOFI recommendation", an "initial assessment", \
or an ownership/commercialization recommendation for an NTNU invention disclosure.

Do NOT use for other kinds of TTO writing (grant applications, market analyses, \
etc.).""",
    "instructions": """\
# NTNU DOFI Recommendation Writer

Draft NTNU Technology Transfer's *initial assessment of ownership, \
commercialization potential, and recommendation for the further process* \
following an invention disclosure (DOFI). NTNU's internal policy asks for this \
recommendation within 30 days of the first inventor meeting (the law allows four \
months). The recommendation is often short and simple — that is fine.

NTNU TTO holds a *general* power of attorney to receive each DOFI and make an \
initial assessment. That general PoA does NOT cover continued commercialization \
work — that requires an explicit, project-specific power of attorney \
(forvaltningsfullmakt). So whenever the recommendation is to proceed, the letter \
must explicitly ask for that power of attorney and attach the PoA form.

Work in three phases: gather information, draft, iterate.

## Phase 1 — Gather the information

First read what is already available in the thread — the user's message, any \
attached data room, uploaded documents, or an earlier canvas. Note what you \
already have so you do not ask the user to repeat it.

Then interview the user to fill the gaps. Ask for the **essential** items first; \
ask about the **other relevant** items only where they add real decision support. \
Keep it efficient — a simple disclosure needs only a short interview.

### Essential information
- **SuperOffice project number** — format `yyyymmNNN`, where `yyyymm` is the year \
and month and `NNN` is that month's running number. Example: `202601001` is the \
first DOFI registered in January 2026.
- **Disclosure / project title.**
- **Inventors** and their institutional affiliation (department/faculty, and \
whether they are NTNU-employed).
- **What the disclosed idea is** — enough to explain it to a layperson.
- **Disclosure date.**
- **Recipient** — usually the department, sometimes the faculty. Always ask; do \
not assume.
- **Department head or dean** — the "Att." person, depending on the recipient.
- **Sender / project manager** — usually the user.
- **What the IP is.**
- **Who owns the IP and what rights have been granted** — or, if unclear, what is \
needed to clarify it.
- **The conclusion the user wants to reach** — request power of attorney to \
continue, or not — and the key justification points.
- **If requesting PoA: the plan** — why the PoA is needed and what work it will \
authorize.

### Other information that can be relevant
Include these where they exist and add value; skip them otherwise:
- **What has happened since the disclosure** — e.g. a first meeting with the \
inventors, creation of an asset list, review of agreements for ownership and use \
rights, some desktop research, a prior-art look, planning the next steps.
- **The basis for the ownership / use-rights conclusion** — e.g. a consortium \
agreement, or the fact that the inventor is fully NTNU-employed with no \
collaboration, other contributors, or external funding.
- **What is known about the commercial potential** — the problem and who has it, \
what they do today, key actors in the space, what is unique about the idea, the \
main competitors, market size and growth rates, supporting trends, industry \
dynamics.
- **Plans to learn more about the market.**
- **Anything special about the idea** that gives the recipient useful decision \
support.

### Ground rules for evidence
- Use ONLY information present in the thread context or given directly by the \
user. Do NOT add facts, market data, competitors, or figures from your own \
general knowledge. If something important is missing, ask the user or leave a \
clear `[TBD]` placeholder.
- **Never perform or assert a freedom-to-operate (FtO) analysis.** A prior-art \
look is fine; FtO is out of scope for this document.

## Phase 2 — Draft

1. Load the **DOFI Recommendation** template to the canvas with \
`skill_template_load`, giving the canvas a name that includes the project number \
and title. Fill in every section from the information gathered:
   - **Header** — place and date, recipient (To / Att.), sender (From / Att.), \
and the NTNU TTO ref. (the SuperOffice number).
   - **Background** — why we are writing (a DOFI was received from the relevant \
department) and an ELI5 of the idea a third party can follow in broad strokes.
   - **Work conducted so far** — the scope of our initial work. Choose phrasing \
deliberately: "Together with the inventor team, we have…" signals that the \
inventors were involved and that our conclusions rest on their input, versus a \
bare "We have…". Keep each point relevant to this assessment and make any \
appendix references accurate.
   - **IP-ownership assessment** — a short summary (refer to the asset-list \
appendix for detail): what the IP is, who created it, whether it is protectable \
(e.g. patentable), who owns it, whether use rights are already granted, and a \
conclusion on whether we can commercialize it. This section can often be very \
short.
   - **First assessment of commercial potential** — how deep we have gone and any \
initial conclusions. Underline where more work is needed, but include what we do \
know. Begin to outline the major challenges; if the project looks likely to fail \
on commercial grounds, start to foreshadow that here without being too adamant. \
Keep it short and concise.
   - **Conclusion** — clearly state whether we ask for power of attorney, **in \
bold**, and always recommend a clear next step whatever the conclusion.

2. **If, and only if, the recommendation is to request power of attorney**, also \
load the **Power of Attorney** template to a *separate* canvas with \
`skill_template_load`. Fill in only the project-specific placeholders: the \
SuperOffice number, the project name, the IP/DOFI description as an ELI5, and the \
signatory block. **Keep all other wording of the PoA verbatim** — it is a legal \
form and must not be reworded. If the recommendation is NOT to continue, do not \
produce the PoA form.

Use `skill_template_view` first if you want to inspect a template before loading \
it.

## Phase 3 — Iterate

Present the draft(s) and refine them with the user through canvas edits. Common \
adjustments: tightening the ELI5, sharpening the ownership conclusion, \
calibrating how strongly to foreshadow commercial challenges, and confirming the \
recipient, the "Att." names, and the signatory block. Keep the document short — a \
simple, clear recommendation is the goal, not length.
""",
    "tool_names": ["skill_template_view", "skill_template_load"],
    "templates": {
        "DOFI Recommendation": """\
# Initial assessment of ownership, commercialization potential, and recommendation for the further process regarding commercialization of [Project name]

---

**Place and date:** [Trondheim, today's date]
**To:** [Department or faculty]
Att.: [Department head or dean]
**From:** NTNU Technology Transfer AS
Att.: [Project manager name]
**NTNU TTO ref.:** [SuperOffice project number]

---

## Background

[State the background for this recommendation — that we have received a DOFI from \
the relevant department — and outline in very simple terms what the idea is, so a \
third party can understand it in broad strokes ("explain it like I'm five").]

## Work conducted so far

[Let the reader understand the scope of our initial work. Choose phrasing \
deliberately — e.g. "Together with the inventor team, we have…" rather than "We \
have…" — to show that the inventors were involved and that our conclusions rest \
on their input. Make each point relevant to this particular assessment, and \
ensure any appendix numbers are accurate.]

## IP-ownership assessment

[No need to repeat everything from the asset-list appendix — give the reader the \
quick summary and refer to the appendix for the details. Let the reader know \
whether this IP is usable for commercialization and whether any steps are needed \
to solidify it; this section can often be summarized very briefly. Key points: \
What is the IP? Who created it? Is it protectable in some way (e.g. patent)? Who \
owns it? Are use rights already granted? Conclusion: can we commercialize it?]

## First assessment of commercial potential

[Let the reader know how deep we have dived into the topic and what, if any, our \
initial conclusions are. Don't be afraid to underline that more work is needed — \
but if we DO know something, include it here. This is also a good place to begin \
outlining the major challenges we see; if we have the feeling this project will \
fail on commercial assessment, begin to foreshadow the challenges here, without \
being too adamant. Keep this section short and concise.]

## Conclusion

[Let the reader know whether or not we ask for power of attorney to continue — \
**put that sentence in bold.** Whatever the conclusion, make sure we clearly \
recommend a next step.]
""",
        "Power of Attorney": """\
# FORVALTNINGSFULLMAKT INTELLEKTUELL EIENDOM (IP)

Dekan eller den dekan bemyndiger ved navn («NTNUs Representant») gir herved NTNU Technology Transfer AS («TTO») fullmakt til å opprette et kommersialiseringsprosjekt med følgende prosjektnummer og prosjektnavn i TTO sine systemer:

[SuperOffice project number] - [Project name]

og

gjennomføre dette innenfor rammene av samarbeidsavtalen mellom NTNU TTO og NTNU av 10.02.2023 på vegne av NTNU. Tjenestene knyttet til dette kommersialiseringsprosjektet kan belastes innenfor tjenestekjøpsavtalens spesifiserte rammer (for kommersialiserings-prosjekter).

Beskrivelse av IP, ref. DOFI nr. [SuperOffice project number]:

[Description of the disclosure, written as an ELI5]

Fullmakten gjelder også fremforhandling og forvaltning av lisensavtaler som omhandler intellektuell eiendom nevnt i ovennevnte DOFI, men signering/godkjenning av lisensavtaler skal alltid gjøres av dekan eller den dekan bemyndiger.


_________________________________
For NTNU
[Name of faculty or department]
[Name and title of signatory (dean or department head)]
""",
    },
}
