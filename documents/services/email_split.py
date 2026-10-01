"""Parse an .eml/.msg into headers, a Markdown body and its attachment parts.

The one place that enumerates an email's attachments, shared by the data-room
loader (``documents.services.chunking``) and the chat/meeting splitter
(``chat.email_attachments``) so both see the same parts in the same order —
``EmailPart.ordinal`` is the stable identity of an attachment within its email.

Inline images (cid-referenced pictures in the HTML body: signature logos,
banners) are dropped here; they are decoration, not attachments. A real image
*attachment* is kept and described inline by the callers. A forwarded email is
an ordinary part whose bytes are a standalone .eml/.msg (``message/rfc822``
parts are serialized; an embedded Outlook message is exported).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_RE_BAD_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]+')


@dataclass
class EmailPart:
    """One attachment of an email."""

    ordinal: int  # 1-based position among the email's (non-inline) attachments
    filename: str
    data: bytes | None  # None when the bytes couldn't be read
    size: int

    @property
    def ext(self) -> str:
        return self.filename.rsplit(".", 1)[-1].lower() if "." in self.filename else ""


@dataclass
class ParsedEmail:
    subject: str | None
    from_addr: str | None
    to_addr: str | None
    date: str | None
    cc: str | None
    body_markdown: str
    parts: list[EmailPart] = field(default_factory=list)


def _clean_filename(name: str, fallback: str) -> str:
    name = _RE_BAD_FILENAME_CHARS.sub(" ", name or "").strip().strip(".")
    return name or fallback


def _email_filename(subject, ext: str) -> str:
    """A filename for a forwarded email that carries none of its own."""
    stem = _clean_filename(str(subject or ""), "Forwarded message")[:120]
    return f"{stem}.{ext}"


def _is_image_name(filename: str) -> bool:
    from core.file_types import is_image_extension

    return is_image_extension(filename.rsplit(".", 1)[-1] if "." in filename else "")


def parse_email(source, ext: str) -> ParsedEmail:
    """Parse *source* (a path, or the raw bytes) as an ``eml`` or ``msg`` email.

    Raises ``ValueError`` when the email has no body at all (matching the loader's
    long-standing behaviour).
    """
    ext = (ext or "").lower().lstrip(".")
    if ext == "msg":
        return _parse_msg(source)
    if ext == "eml":
        return _parse_eml(source)
    raise ValueError(f"Not an email file type: {ext}")


# ---------------------------------------------------------------------------
# .eml
# ---------------------------------------------------------------------------
def _parse_eml(source) -> ParsedEmail:
    import email
    import email.policy

    from markdownify import markdownify as md

    if isinstance(source, (bytes, bytearray)):
        msg = email.message_from_bytes(bytes(source), policy=email.policy.default)
    else:
        with open(Path(source), "rb") as f:
            msg = email.message_from_binary_file(f, policy=email.policy.default)

    body_part = msg.get_body(preferencelist=("html", "plain"))
    if body_part is None:
        raise ValueError("Email has no body content (no HTML, no plain text)")
    body_content = body_part.get_content()
    if body_part.get_content_type() == "text/html":
        body_md = md(body_content, heading_style="ATX")
    else:
        body_md = body_content

    parts: list[EmailPart] = []
    for att in msg.iter_attachments():
        filename = att.get_filename()
        if att.get_content_type() == "message/rfc822":
            # A forwarded email attached as a message part: its payload is a
            # parsed message, not bytes — serialize it back into a standalone .eml.
            try:
                inner = att.get_content()
                data = inner.as_bytes()
                if not filename:
                    filename = _email_filename(inner.get("subject"), "eml")
            except Exception:
                logger.debug("parse_email: could not serialize a message/rfc822 part", exc_info=True)
                data = None
            filename = _clean_filename(filename or "", "Forwarded message.eml")
            if not filename.lower().endswith((".eml", ".msg")):
                filename += ".eml"
        else:
            try:
                data = att.get_payload(decode=True)
            except Exception:
                data = None
            filename = _clean_filename(filename or "", "unnamed")
            # Inline images (cid-referenced from the HTML body) are decoration.
            if _is_image_name(filename) and (
                att.get_content_disposition() == "inline" or att.get("Content-ID")
            ):
                continue
        parts.append(EmailPart(
            ordinal=len(parts) + 1, filename=filename, data=data or None, size=len(data) if data else 0,
        ))

    date = msg["date"]
    return ParsedEmail(
        subject=msg["subject"], from_addr=msg["from"], to_addr=msg["to"],
        date=str(date) if date else None, cc=msg["cc"], body_markdown=body_md, parts=parts,
    )


# ---------------------------------------------------------------------------
# .msg
# ---------------------------------------------------------------------------
def _parse_msg(source) -> ParsedEmail:
    import extract_msg
    from markdownify import markdownify as md

    msg = extract_msg.openMsg(bytes(source)) if isinstance(source, (bytes, bytearray)) else extract_msg.Message(str(source))
    try:
        html_body = msg.htmlBody
        plain_body = msg.body
        if html_body:
            if isinstance(html_body, bytes):
                html_body = html_body.decode("utf-8", errors="replace")
            body_md = md(html_body, heading_style="ATX")
        elif plain_body:
            body_md = plain_body
        else:
            raise ValueError("Email has no body content (no HTML, no plain text)")

        parts: list[EmailPart] = []
        for att in msg.attachments:
            data, filename = _msg_attachment_bytes(att)
            # Hidden attachments are the inline pictures of the HTML body.
            if _is_image_name(filename) and (getattr(att, "hidden", False) or getattr(att, "cid", None)):
                continue
            parts.append(EmailPart(
                ordinal=len(parts) + 1, filename=filename, data=data, size=len(data) if data else 0,
            ))

        return ParsedEmail(
            subject=msg.subject, from_addr=msg.sender, to_addr=msg.to,
            date=str(msg.date) if msg.date else None, cc=msg.cc, body_markdown=body_md, parts=parts,
        )
    finally:
        msg.close()


def _msg_attachment_bytes(att) -> tuple[bytes | None, str]:
    """``(bytes|None, filename)`` for one extract_msg attachment.

    An embedded Outlook message's ``data`` is an ``MSGFile``, not bytes: export it
    as a standalone .msg.
    """
    name = getattr(att, "longFilename", None) or getattr(att, "shortFilename", None) or ""
    try:
        data = getattr(att, "data", None)
    except Exception:
        logger.debug("parse_email: could not read a .msg attachment", exc_info=True)
        data = None
    if data is not None and not isinstance(data, (bytes, bytearray)):
        if hasattr(data, "exportBytes"):
            try:
                inner = data
                data = inner.exportBytes()
                if not name:
                    name = _email_filename(getattr(inner, "subject", None), "msg")
            except Exception:
                logger.debug("parse_email: could not export an embedded .msg", exc_info=True)
                data = None
            name = _clean_filename(name, "Forwarded message.msg")
            if not name.lower().endswith((".msg", ".eml")):
                name += ".msg"
            return (bytes(data) if data else None), name
        data = None
    return (bytes(data) if data else None), _clean_filename(name, "unnamed")
