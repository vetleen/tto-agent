"""IRL Project Assessor seed skill definition."""

IRL_PROJECT_ASSESSOR = {
    "slug": "irl-project-assessor",
    "name": "IRL Project Assessor",
    "emoji": "📊",
    "description": """\
Assess innovation projects using the \
KTH Innovation Readiness Level (IRL) framework. Scores projects across 6 \
dimensions — Customer Readiness (CRL), Technology Readiness (TRL), Business \
Readiness (BRL), IP Readiness (IPRL), Team Readiness (TeRL), and Funding \
Readiness (FRL) — on 1–9 scales using objective criteria. 

Use when the user mentions scoring a project using "IRL", "innovation readiness" \
or "readiness level", or if the users wants your help to to assess any of the \
other concrete explicitly dimensions mentioned.""",
    "instructions": """\
# IRL Project Assessor

Assess innovation projects using the KTH Innovation Readiness Level (IRL) \
framework. Produce a structured assessment report in the canvas.

## Scoring Methodology

- **Strict scoring:** ALL criteria at a level must be met to achieve that \
level. Meeting 3 of 4 criteria at level 5 means the score is 4 (given that all levl 4, and before, critera is met), not 5.
- **Cumulative:** a score of N means all criteria at levels 1 through N are \
fully met.
- **No benefit of the doubt.** If evidence is ambiguous or missing, default \
to the lowest defensible score. Uncertainty = not met.
- **Evidence-based.** Every criterion scored as "met" must cite a specific \
document, section, or data point from the data room.
- **Score = highest level where ALL criteria (including all lower levels) \
are fully and demonstrably met.**

## Workflow

### Phase 1 — Evidence Gathering

Systematically search the attached data room and other provided docs for evidence against each \
dimension's criteria. Use `search_documents` and `read_document` for data rooms.

For each dimension (CRL, TRL, BRL, IPRL, TeRL, FRL):
1. Search for evidence relevant to that dimension's criteria.
2. Note what was found, where (document name + section), and which \
criterion it maps to.
3. Note gaps — criteria with no supporting evidence.

If the data room and any provided docs are sparse, ask the user whether they have provided \
all relevant information or whether to proceed with what is available.

### Phase 2 — Scoring & Gap Analysis

Load the "IRL Assessment Report" template to the canvas using \
`load_template_to_canvas`. Then fill in the report:

For each dimension:
1. Walk through levels 1–9 and determine the highest level where ALL \
criteria are met.
2. Record the score.
3. For the current level: list which criteria are met and the evidence.
4. For the next level (current + 1): list which criteria are met and which \
are NOT met — these are the gaps.
5. Write a one-line justification for the score.

Fill the score overview table and each per-dimension section in the canvas.

### Phase 3 — Next Steps & Recommendations

For each dimension, write actionable recommendations:
1. **To reach the next level (current + 1):** What specific evidence, \
actions, or milestones are needed to close the identified gaps?
2. **To reach the level after that (current + 2):** What should be planned \
or initiated now to prepare for this level?

Be specific — reference the exact criteria not yet met and suggest concrete \
actions (e.g., "conduct 5 customer discovery interviews" not "talk to \
customers").

## IRL Criteria Reference

### CRL — Customer Readiness Level

**Level 1 — Unvalidated need hypothesis**
- A potential customer need or problem has been identified
- The need is based on the inventor's assumption, not customer evidence

**Level 2 — Need described and initial desktop research done**
- The customer need is articulated in writing
- Desktop research (literature, reports, market data) supports the \
existence of the need
- Initial target customer segments are hypothesized

**Level 3 — Need discussed with potential customers**
- At least 3 potential customers/users have been interviewed or surveyed
- Feedback confirms the need exists (not just polite interest)
- Pain points are described in the customers' own words

**Level 4 — Customer segments defined and value proposition drafted**
- Distinct customer segments are identified with clear characteristics
- A value proposition is drafted for each primary segment
- Willingness to engage further (meetings, pilots) is expressed by at \
least one potential customer

**Level 5 — Value proposition validated with customers**
- Customers confirm the value proposition addresses their need
- Evidence of demand: LOIs, signed pilot agreements, or survey data \
showing purchase intent
- Customer requirements and priorities are documented

**Level 6 — Customer committed to testing or piloting**
- At least one customer has signed a pilot/test agreement
- Customer is contributing resources (time, data, access) to the pilot
- Success criteria for the pilot are jointly defined

**Level 7 — Customer validated solution in relevant setting**
- Pilot/test completed with positive results
- Customer confirms the solution meets their requirements
- Customer feedback incorporated into product/solution refinement

**Level 8 — Customer committed to purchase or adoption**
- At least one customer has placed an order or signed a purchase agreement
- Commercial terms (price, delivery, support) are agreed
- Reference customer is available

**Level 9 — Paying customers and scalable acquisition**
- Multiple paying customers across segments
- Repeatable customer acquisition process in place
- Customer satisfaction data collected and positive

### TRL — Technology Readiness Level

**Level 1 — Basic principles observed**
- Scientific literature or research identifies underlying principles
- No experimental work yet

**Level 2 — Technology concept formulated**
- Potential application of principles identified
- Concept or invention described (paper, invention disclosure)
- Initial feasibility considerations documented

**Level 3 — Experimental proof of concept**
- Laboratory experiments validate critical function(s)
- Proof of concept demonstrates key technical parameters
- Results documented and reproducible

**Level 4 — Technology validated in laboratory**
- Integrated prototype or model tested in lab environment
- Key performance parameters measured against targets
- Technical risks identified and mitigation approaches outlined

**Level 5 — Technology validated in relevant environment**
- Prototype tested in conditions resembling the real use environment
- Performance meets minimum requirements for intended application
- Integration with other subsystems or components demonstrated

**Level 6 — Technology demonstrated in relevant environment**
- Representative prototype demonstrated in relevant environment
- Performance validated against user requirements
- Scale-up or manufacturing feasibility assessed

**Level 7 — System prototype demonstrated in operational environment**
- Full-scale or near-full-scale prototype tested in operational setting
- Performance verified under realistic conditions
- Remaining technical issues identified and resolution plan in place

**Level 8 — System complete and qualified**
- Final system tested and meets all specifications
- Manufacturing/production process established
- Quality assurance and regulatory requirements addressed

**Level 9 — System proven in operational environment**
- System deployed and operating successfully in the field
- Performance data from real-world operation collected
- Reliability and maintainability demonstrated

### BRL — Business Readiness Level

**Level 1 — Commercial opportunity identified**
- A potential commercial application of the technology is recognized
- No structured business analysis yet

**Level 2 — Business model hypotheses formulated**
- Revenue model, cost structure, and value chain position hypothesized
- Target market size estimated (order of magnitude)
- Key assumptions listed

**Level 3 — Key business model components tested**
- Pricing hypothesis tested with potential customers
- Cost structure estimated with reasonable basis
- Competitive landscape analysed

**Level 4 — Business model partially validated**
- Revenue model supported by customer feedback or pilot data
- Key partnerships or channel relationships identified
- Unit economics estimated

**Level 5 — Business model validated with real data**
- Revenue generated or firm commitments received
- Customer acquisition cost and lifetime value estimated from real data
- Supply chain or delivery model tested

**Level 6 — Business model refined and scalable**
- Repeatable sales or delivery process established
- Margins validated at current scale
- Growth strategy defined with specific milestones

**Level 7 — Business operations established**
- Legal entity operational (if spin-off) or licensing deal signed
- Core business processes in place (sales, delivery, support)
- Financial tracking and reporting established

**Level 8 — Business model proven at scale**
- Revenue growing consistently
- Operations scaled to serve multiple customers/markets
- Key business metrics tracked and trending positively

**Level 9 — Sustainable profitable business**
- Profitable or clear path to profitability demonstrated
- Market position established and defensible
- Organizational maturity supports continued growth

### IPRL — IP Readiness Level

**Level 1 — IP awareness**
- Inventive knowledge or know-how exists
- No IP landscape analysis performed
- Disclosure of invention to TTO (if applicable)

**Level 2 — IP landscape scan performed**
- Prior art search conducted (patent databases, literature)
- Key existing patents and publications identified
- Initial assessment of novelty and inventive step

**Level 3 — IP strategy outlined**
- Decision on protection approach (patent, trade secret, copyright, \
open source)
- Geographic scope of protection considered
- Freedom-to-operate risks identified at high level

**Level 4 — IP protection initiated**
- Provisional patent application filed, or trade secret measures \
implemented
- Inventor agreements and assignment of rights in place
- Publication strategy aligned with IP protection

**Level 5 — IP protection filed**
- Full patent application filed (national or PCT)
- Claims drafted to cover key commercial embodiments
- IP budget allocated for prosecution

**Level 6 — IP position strengthened**
- Freedom-to-operate analysis completed
- Additional filings (continuations, divisionals, design patents) where \
appropriate
- IP strategy aligned with business model and commercialisation path

**Level 7 — IP granted or registered**
- At least one key patent granted in a target market
- IP portfolio reviewed and maintained
- Enforcement strategy considered

**Level 8 — IP commercialised**
- IP licensed, assigned, or used in a commercial product/service
- License terms or valuation established
- IP contributing to competitive advantage

**Level 9 — IP portfolio managed strategically**
- Ongoing IP management integrated with business strategy
- Portfolio optimised (filings, abandonments, licensing)
- IP generates measurable value (revenue, market exclusivity, \
partnerships)

### TeRL — Team Readiness Level

**Level 1 — Inventor or researcher with idea**
- One or few individuals with technical knowledge
- No dedicated team for commercialisation
- Idea not yet shared beyond immediate research group

**Level 2 — Core technical competence identified**
- Key technical skills needed for development are identified
- At least one person is actively driving the project forward
- Time allocation is informal and limited

**Level 3 — Complementary skills identified, recruitment started**
- Gaps in team competence identified (business, regulatory, technical)
- Recruitment or advisory contacts initiated
- At least one business-oriented advisor or mentor engaged

**Level 4 — Core team formed with key competences**
- Team has both technical and business/commercial competence
- Roles and responsibilities defined
- Team members have dedicated (though possibly part-time) commitment

**Level 5 — Extended team with business competence**
- Business lead or CEO identified (if spin-off path)
- Domain experts or industry advisors on board
- Team covers technology, business development, and market understanding

**Level 6 — Full team with management capacity**
- Management team in place with relevant experience
- Board of directors or advisory board established (if spin-off)
- Team has capacity to execute current project plan

**Level 7 — Team proven in relevant setting**
- Team has demonstrated ability to deliver milestones
- Decision-making processes are effective
- Team has weathered at least one significant challenge or pivot

**Level 8 — Team operating effectively with governance**
- Governance structure in place and functional
- Performance management and accountability established
- Team can attract and retain additional talent

**Level 9 — Scalable organization**
- Organizational structure supports growth
- Key person dependencies mitigated
- Culture and processes enable scaling beyond founding team

### FRL — Funding Readiness Level

**Level 1 — Self-funded or university resources only**
- Project runs on existing research group budget or personal time
- No external funding specifically for commercialisation

**Level 2 — Funding need identified**
- Estimated budget for next development phase exists
- Gap between available and needed funding quantified
- No funding applications submitted

**Level 3 — Funding sources mapped**
- Relevant grant programs, investors, or funding instruments identified
- Eligibility confirmed for at least one source
- Funding timeline aligned with project milestones

**Level 4 — Initial grant or pre-seed funding obtained**
- At least one external grant or proof-of-concept fund awarded
- Funding covers current development phase
- Reporting and compliance requirements understood

**Level 5 — Milestone-based funding secured**
- Funding tied to specific milestones and deliverables
- Go/no-go decision points define continued funding
- Budget sufficient for next 6–12 months of activity

**Level 6 — Mixed funding model**
- Combination of grants and early revenue or in-kind contributions
- Funding diversified across at least two sources
- Financial runway extends 12+ months

**Level 7 — Commercial funding attracted**
- Angel investment, VC seed round, or corporate investment received \
(if spin-off path), or commercial licensing revenue (if licensing path)
- Investors or licensees validated business potential
- Valuation or deal terms established

**Level 8 — Growth funding secured**
- Series A or equivalent growth capital raised (if spin-off), or \
multiple licensing deals signed (if licensing path)
- Funding supports market expansion and scaling
- Financial controls and reporting meet investor/partner standards

**Level 9 — Sustainable funding model**
- Revenue-funded growth or profitable operations
- Access to capital markets or follow-on funding established
- Financial sustainability demonstrated or clearly projected""",
    "tool_names": ["view_template", "load_template_to_canvas"],
    "templates": {
        "IRL Assessment Report": """\
# IRL Assessment Report — [Project Name]

*Innovation Readiness Level assessment based on the KTH IRL framework*
*Date: [date]*

---

## Score Overview

| Dimension | Score (1–9) | Current Level | Key Gap to Next Level |
|-----------|:-----------:|---------------|----------------------|
| CRL — Customer Readiness | /9 | | |
| TRL — Technology Readiness | /9 | | |
| BRL — Business Readiness | /9 | | |
| IPRL — IP Readiness | /9 | | |
| TeRL — Team Readiness | /9 | | |
| FRL — Funding Readiness | /9 | | |

**Overall IRL Profile:** [1-2 sentence summary of the project's maturity \
profile — e.g., "Strong technology base (TRL 5) but early-stage customer \
validation (CRL 2) and no IP protection (IPRL 1). Primary bottleneck is \
customer and business development."]

---

## CRL — Customer Readiness Level

**Score: [X] / 9** — [One-line justification]

### Criteria Assessment

| Level | Criterion | Status | Evidence / Notes |
|:-----:|-----------|:------:|------------------|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| [current] | | | |
| [next] | | | |

### Gap to Next Level (Level [X+1])

[What specific criteria at the next level are NOT met? What evidence is \
missing?]

### Recommended Next Steps

**To reach Level [X+1]:**
- [Specific action 1]
- [Specific action 2]

**To prepare for Level [X+2]:**
- [Specific action 1]
- [Specific action 2]

---

## TRL — Technology Readiness Level

**Score: [X] / 9** — [One-line justification]

### Criteria Assessment

| Level | Criterion | Status | Evidence / Notes |
|:-----:|-----------|:------:|------------------|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| [current] | | | |
| [next] | | | |

### Gap to Next Level (Level [X+1])

[What specific criteria at the next level are NOT met? What evidence is \
missing?]

### Recommended Next Steps

**To reach Level [X+1]:**
- [Specific action 1]
- [Specific action 2]

**To prepare for Level [X+2]:**
- [Specific action 1]
- [Specific action 2]

---

## BRL — Business Readiness Level

**Score: [X] / 9** — [One-line justification]

### Criteria Assessment

| Level | Criterion | Status | Evidence / Notes |
|:-----:|-----------|:------:|------------------|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| [current] | | | |
| [next] | | | |

### Gap to Next Level (Level [X+1])

[What specific criteria at the next level are NOT met? What evidence is \
missing?]

### Recommended Next Steps

**To reach Level [X+1]:**
- [Specific action 1]
- [Specific action 2]

**To prepare for Level [X+2]:**
- [Specific action 1]
- [Specific action 2]

---

## IPRL — IP Readiness Level

**Score: [X] / 9** — [One-line justification]

### Criteria Assessment

| Level | Criterion | Status | Evidence / Notes |
|:-----:|-----------|:------:|------------------|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| [current] | | | |
| [next] | | | |

### Gap to Next Level (Level [X+1])

[What specific criteria at the next level are NOT met? What evidence is \
missing?]

### Recommended Next Steps

**To reach Level [X+1]:**
- [Specific action 1]
- [Specific action 2]

**To prepare for Level [X+2]:**
- [Specific action 1]
- [Specific action 2]

---

## TeRL — Team Readiness Level

**Score: [X] / 9** — [One-line justification]

### Criteria Assessment

| Level | Criterion | Status | Evidence / Notes |
|:-----:|-----------|:------:|------------------|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| [current] | | | |
| [next] | | | |

### Gap to Next Level (Level [X+1])

[What specific criteria at the next level are NOT met? What evidence is \
missing?]

### Recommended Next Steps

**To reach Level [X+1]:**
- [Specific action 1]
- [Specific action 2]

**To prepare for Level [X+2]:**
- [Specific action 1]
- [Specific action 2]

---

## FRL — Funding Readiness Level

**Score: [X] / 9** — [One-line justification]

### Criteria Assessment

| Level | Criterion | Status | Evidence / Notes |
|:-----:|-----------|:------:|------------------|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| [current] | | | |
| [next] | | | |

### Gap to Next Level (Level [X+1])

[What specific criteria at the next level are NOT met? What evidence is \
missing?]

### Recommended Next Steps

**To reach Level [X+1]:**
- [Specific action 1]
- [Specific action 2]

**To prepare for Level [X+2]:**
- [Specific action 1]
- [Specific action 2]

---

## Priority Actions

Based on the assessment above, the top priorities for advancing this \
project's innovation readiness are:

1. **[Highest priority action]** — [Which dimension(s) it advances and why \
it is the bottleneck]
2. **[Second priority]** — [Rationale]
3. **[Third priority]** — [Rationale]

---

*Assessment performed using the KTH Innovation Readiness Level (IRL) \
framework. Scores reflect the highest level where ALL criteria are fully \
and demonstrably met based on evidence in the data room.*
""",
    },
}
