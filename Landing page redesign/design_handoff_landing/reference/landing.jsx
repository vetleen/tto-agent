/* Wilfred — commercial landing page. Composed with the Wilfred Design System. */
const { Button } = window.WilfredDesignSystem_894737;
const I = (n, props = {}) => <i data-lucide={n} {...props}></i>;
const LOGIN = "login.html";

function useLucide() {
  const { useEffect } = React;
  useEffect(() => { if (window.lucide) window.lucide.createIcons(); });
}

/* ------------------------------------------------------------------ Nav */
function Nav() {
  return (
    <header className="ld-nav">
      <nav className="ld-container ld-nav__inner">
        <a className="ld-brand" href="#top"><img src="assets/wilfred-mark.svg" alt="" /><b>Wilfred</b></a>
        <div className="ld-nav__links">
          <a href="#capabilities">Capabilities</a>
          <a href="#security">Security</a>
          <a href="#workspace">The workspace</a>
        </div>
        <div className="ld-nav__cta">
          <a className="ld-nav__login" href={LOGIN}>Log in</a>
          <Button variant="primary" size="sm" href={LOGIN}>Log in</Button>
        </div>
      </nav>
    </header>
  );
}

/* ------------------------------------------------------------------ Hero */
function Hero() {
  return (
    <section className="ld-hero" id="top">
      <div className="ld-hero__bg"><img src="uploads/landing_page_illustration.png" alt="" /></div>
      <div className="ld-hero__scrim"></div>
      <div className="ld-container ld-hero__inner">
        <div className="ld-eyebrow ld-eyebrow--onfilm ld-rise">For technology transfer offices</div>
        <h1 className="ld-rise" style={{ animationDelay: '60ms' }}>
          The office that turns research into <em>revenue</em>.
        </h1>
        <p className="ld-hero__lede ld-rise" style={{ animationDelay: '120ms' }}>
          Wilfred is an AI colleague for the technology-transfer office. It handles intake and
          disclosures, reads your data rooms, takes meeting minutes, and drafts the memos — so
          your team can spend its hours on the deals, not the documents.
        </p>
        <div className="ld-hero__cta ld-rise" style={{ animationDelay: '180ms' }}>
          <Button variant="accent" size="lg" href={LOGIN}>Log in</Button>
          <Button variant="secondary" size="lg" href="#capabilities">See what Wilfred does</Button>
        </div>
        <div className="ld-hero__meta ld-rise" style={{ animationDelay: '260ms' }}>
          <span>{I('shield-check')} Confidential by design</span>
          <i className="dot"></i>
          <span>{I('quote')} Every answer cited</span>
          <i className="dot"></i>
          <span>{I('file-search')} Built on your documents</span>
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ Trust */
function Trust() {
  const names = ["Northgate University", "Meridian Institute", "Calder Research", "Ashford Tech", "Lindholm Labs"];
  return (
    <section className="ld-trust">
      <div className="ld-container ld-trust__inner">
        <div className="ld-trust__label">Working inside research institutions</div>
        <div className="ld-trust__names">
          {names.map((n) => <span key={n}>{n}</span>)}
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ Statement */
function Statement() {
  return (
    <section className="ld-section ld-statement">
      <div className="ld-container ld-statement__grid">
        <div>
          <div className="ld-seceye">The problem</div>
          <h2 style={{ marginTop: 18 }}>
            The work that creates value is buried under the work that records it.
          </h2>
        </div>
        <div className="ld-statement__col">
          <p>
            A disclosure arrives. Somewhere across a dozen PDFs, an old data room, and a meeting no
            one transcribed sit the prior art, the inventor's claims, and the three precedents that
            decide whether this is a patent or a pass. Finding them is a day's work. Doing it well is a week's.
          </p>
          <p>
            Wilfred reads all of it — the disclosures, the diligence packs, the minutes — and answers
            with citations you can open. It drafts the summary, flags the confidential material, and
            keeps the record straight. The judgment stays yours. The legwork doesn't.
          </p>
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ Capabilities */
function Feature({ flip, eyebrow, title, body, points, media }) {
  return (
    <article className={"ld-feature" + (flip ? " ld-feature--flip" : "")}>
      <div className="ld-feature__body">
        <div className="ld-feature__eyebrow">{eyebrow}</div>
        <h3>{title}</h3>
        <p>{body}</p>
        <ul className="ld-feature__list">
          {points.map((p) => <li key={p}>{I('check')}<span>{p}</span></li>)}
        </ul>
      </div>
      {media}
    </article>
  );
}

function ChatMock() {
  return (
    <div className="ld-mock" id="workspace">
      <div className="ld-mock__bar">
        <i className="ld-mock__dot"></i><i className="ld-mock__dot"></i><i className="ld-mock__dot"></i>
        <span className="ld-mock__title">Thread · Patent Portfolio Q1</span>
      </div>
      <div className="ld-mock__body">
        <div className="ld-mock__rooms">
          <span className="ld-chip">{I('folder-lock')} Disclosure 2024-0421</span>
          <span className="ld-chip">{I('folder-lock')} Diligence pack</span>
        </div>
        <div className="ld-msg ld-msg--u">
          <div className="ld-msg__av ld-msg__av--u">DO</div>
          <div className="ld-msg__bubble">Is there blocking prior art for the microfluidic claim in this disclosure?</div>
        </div>
        <div className="ld-msg ld-msg--w">
          <div className="ld-msg__av ld-msg__av--w">W</div>
          <div className="ld-msg__bubble">
            I searched 2 data rooms and found one close precedent. Claim 1 overlaps with WO-2021-0188
            on the channel geometry, but the surface-treatment step appears novel.
            <span className="ld-msg__cite">{I('file-text')} diligence-pack.pdf · p.14</span>
            <span className="ld-mock__caret"></span>
          </div>
        </div>
      </div>
    </div>
  );
}

function Capabilities() {
  return (
    <section className="ld-section ld-caps" id="capabilities">
      <div className="ld-container">
        <div className="ld-caps__head">
          <div className="ld-seceye">Capabilities</div>
          <h2>Everything the office does, with a colleague on it.</h2>
          <p>Four surfaces, one trusted assistant. Wilfred works across them the way your best analyst would — reading widely, citing carefully, and writing it down.</p>
        </div>

        <Feature
          eyebrow="Data rooms"
          title="Secure rooms that read themselves"
          body="Upload disclosures, patents, and diligence packs as PDF, DOCX, or text. Wilfred chunks and indexes every file for hybrid semantic and full-text search, and surfaces PII and guardrail status per document."
          points={[
            "Hybrid retrieval across thousands of pages in seconds",
            "Per-document confidentiality and PII status",
            "Answers link straight back to the source page",
          ]}
          media={
            <div className="ld-feature__media ld-feature__media--tint">
              <img src="uploads/landing_page_hero.png" alt="A researcher's desk with a notebook, disclosure documents, and a microfluidic chip" />
              <span className="ld-feature__tag">{I('folder-lock')} 12 documents · 412 chunks indexed</span>
            </div>
          }
        />

        <Feature
          flip
          eyebrow="Chat that drafts"
          title="Ask across the room. Get a memo back."
          body="Attach one or more data rooms to a thread and ask in plain language. Wilfred drafts summaries, term sheets, and prior-art memos into an editable canvas — every claim traceable to its source."
          points={[
            "Drafts into an editable, multi-canvas document",
            "Citations on every factual claim",
            "Delegates research to sub-agents when a question is broad",
          ]}
          media={<ChatMock />}
        />

        <Feature
          eyebrow="Meetings, on the record"
          title="Inventor intakes that minute themselves"
          body="Start a meeting and Wilfred transcribes live, or upload the audio after. It writes the minutes, extracts the action items, and files everything back to the right data room — searchable from the next thread."
          points={[
            "Live transcription and post-hoc audio upload",
            "Minutes and action items, written for you",
            "Filed straight back into the linked data room",
          ]}
          media={
            <div className="ld-feature__media">
              <img src="uploads/landing_page_illustration.png" alt="A quiet research library corridor" />
              <span className="ld-feature__tag">{I('mic')} Minutes with Wilfred · saved</span>
            </div>
          }
        />
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ Security */
function Security() {
  const cells = [
    ['shield-check', 'Confidential by design', 'Pre-publication research and sensitive IP stay inside their data room. Wilfred never trains on your documents and treats confidentiality as a default, not a setting.'],
    ['quote', 'Cited, not improvised', 'Every factual claim links to the document and page it came from. If Wilfred can\u2019t find a source, it tells you — rather than guessing.'],
    ['scan-eye', 'PII and guardrails surfaced', 'Each document carries its PII and guardrail status, so you know what\u2019s sensitive before it ever reaches a thread.'],
  ];
  return (
    <section className="ld-section ld-secure" id="security">
      <div className="ld-container">
        <div className="ld-secure__top">
          <img className="ld-secure__seal" src="assets/wilfred-seal.svg" alt="" />
          <div>
            <div className="ld-eyebrow ld-secure__eye">Trust</div>
            <h2 style={{ marginTop: 16 }}>Built for the most confidential work in the building.</h2>
          </div>
        </div>
        <div className="ld-secure__grid">
          {cells.map(([ic, t, p]) => (
            <div className="ld-secure__cell" key={t}>
              {I(ic)}
              <h4>{t}</h4>
              <p>{p}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ Testimonial */
function Testimonial() {
  return (
    <section className="ld-section ld-quote">
      <div className="ld-container ld-quote__inner">
        <div className="ld-seceye" style={{ marginBottom: 28 }}>From the office</div>
        <blockquote>
          <span className="mk">&ldquo;</span>Wilfred reads the data room before I do, and it always shows
          its work. It has given our officers back the part of the week that used to disappear into PDFs.<span className="mk">&rdquo;</span>
        </blockquote>
        <div className="ld-quote__by">
          <div className="ld-quote__avatar">DO</div>
          <div className="ld-quote__who">
            <b>Dr. Daniel Okafor</b>
            <span>Director, Technology Transfer Office</span>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ Final CTA */
function FinalCTA() {
  return (
    <section className="ld-final">
      <div className="ld-container ld-final__inner">
        <h2>Meet your office&rsquo;s newest colleague.</h2>
        <p>Log in to your workspace and put Wilfred to work on the next disclosure.</p>
        <div className="ld-final__cta">
          <Button variant="accent" size="lg" href={LOGIN}>Log in</Button>
        </div>
        <div className="ld-final__note">An AI colleague for technology transfer</div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ Footer */
function Footer() {
  return (
    <footer className="ld-foot">
      <div className="ld-container ld-foot__inner">
        <div className="ld-foot__brand"><img src="assets/wilfred-mark.svg" alt="" /><b>Wilfred</b></div>
        <div className="ld-foot__meta">Wilfred · AI workflows for technology transfer</div>
      </div>
    </footer>
  );
}

/* ------------------------------------------------------------------ Page */
function WilfredLanding() {
  useLucide();
  return (
    <div className="ld-wrap">
      <Nav />
      <main>
        <Hero />
        <Trust />
        <Statement />
        <Capabilities />
        <Security />
        <Testimonial />
        <FinalCTA />
      </main>
      <Footer />
    </div>
  );
}

Object.assign(window, { WilfredLanding });
