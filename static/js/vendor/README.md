# Vendored third-party JS

These libraries were previously loaded from `cdn.jsdelivr.net`. They are vendored
(self-hosted) so the Content-Security-Policy can use `script-src 'self'` — no CDN
trust, no SRI upkeep, no third-party outage risk. The CSP barrier (DOMPurify) is
itself in this list, so it must not be delivered by a CDN we don't control.

Pinned versions (update by re-downloading the exact version, then re-running
`python manage.py collectstatic`):

| File | npm package @ version |
|------|-----------------------|
| `flowbite.min.js` | flowbite@4.0.1 |
| `html2canvas-pro.min.js` | html2canvas-pro@1.6.7 |
| `marked.min.js` | marked@12.0.0 |
| `marked-footnote.umd.min.js` | marked-footnote@1.2.4 (dist/index.umd.min.js) |
| `purify.min.js` | dompurify@3.0.8 |
| `diff.min.js` | diff@7.0.0 |
| `mermaid.min.js` | mermaid@11.15.0 |
| `dom-to-image-more.min.js` | dom-to-image-more@3.9.0 |

Example re-download:

    curl -fsSL -o mermaid.min.js "https://cdn.jsdelivr.net/npm/mermaid@11.15.0/dist/mermaid.min.js"
