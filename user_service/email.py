"""Email utility — welcome, invite, and meeting-MOM emails.

Provider-agnostic SMTP. Configured for Brevo by default (free 300/day, commercial-OK):
    SMTP_HOST=smtp-relay.brevo.com   SMTP_PORT=587
    SMTP_USER=<your Brevo SMTP login>   SMTP_PASS=<your Brevo SMTP key>
    FROM_EMAIL=<a verified sender on your domain>   APP_URL=https://app.your-domain.com
Any other SMTP provider (MailerSend, Resend, Amazon SES, Gmail) works by overriding
these env vars. If SMTP_USER/PASS are unset, sends are logged (no-op) instead.

To send AS a Gmail address instead of relaying through a third party (no domain
of your own to verify with a relay):
    SMTP_HOST=smtp.gmail.com   SMTP_PORT=587
    SMTP_USER=<the gmail address>   SMTP_PASS=<a Google App Password, NOT the account password>
    FROM_EMAIL=<the same gmail address>
This is a genuinely different situation from relaying a gmail.com From address
through Brevo/SES/etc: Google is authenticating and delivering its own domain's
mail here, so SPF/DKIM alignment holds (see sender_domain_warning below). An App
Password requires 2-Step Verification enabled on the Google account, and Gmail
caps outbound mail around 500 recipients/day for a personal account — both are
outside this module's control.
"""
from __future__ import annotations

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)

SMTP_HOST = os.getenv("SMTP_HOST", "smtp-relay.brevo.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", SMTP_USER) or "noreply@jarvis.app"
APP_URL = os.getenv("APP_URL", "http://localhost:3000").rstrip("/")


# Free-webmail domains cannot be used as a relay From address and have mail
# actually delivered: a THIRD-PARTY relay can neither sign DKIM for the domain
# nor appear in its SPF record, so DMARC alignment fails and receivers reject or
# spam-folder the message. This does NOT apply when the "relay" is the mailbox
# provider's own server authenticating as that exact account (see
# _OWN_DOMAIN_SMTP_HOSTS below) — there the sender and the server are the same
# party, so alignment holds.
_FREE_WEBMAIL_DOMAINS = {
    "gmail.com", "googlemail.com", "yahoo.com", "outlook.com",
    "hotmail.com", "live.com", "aol.com", "icloud.com", "proton.me",
}

# Maps a mailbox provider's own SMTP host to the From domain(s) it can actually
# send aligned mail for. Sending FROM_EMAIL=x@gmail.com THROUGH smtp.gmail.com
# is Google delivering its own domain's mail; sending the same address through
# Brevo/SES/anything else is a third party impersonating gmail.com.
_OWN_DOMAIN_SMTP_HOSTS: dict[str, set[str]] = {
    "smtp.gmail.com": {"gmail.com", "googlemail.com"},
}

# Set once the sender misconfiguration has been logged, to keep it out of every
# subsequent send.
_sender_warning_logged = False


def sender_domain_warning() -> str | None:
    """Return a warning when FROM_EMAIL can't pass DMARC through SMTP_HOST."""
    domain = FROM_EMAIL.rsplit("@", 1)[-1].strip().lower()
    if domain not in _FREE_WEBMAIL_DOMAINS:
        return None
    if domain in _OWN_DOMAIN_SMTP_HOSTS.get(SMTP_HOST.strip().lower(), set()):
        return None  # authenticated directly to the provider that owns this domain
    return (
        f"FROM_EMAIL is {FROM_EMAIL}, sent via {SMTP_HOST}. A third-party relay "
        f"cannot be DKIM-signed for {domain} and is not in its SPF record, so it "
        f"fails DMARC alignment and is usually spam-filtered even when the relay "
        f"accepts it. Either use an address on a domain verified with this relay, "
        f"or send through {domain}'s own SMTP server authenticated as that exact "
        f"account (see the module docstring for the Gmail case)."
    )


class _RecordingSMTP(smtplib.SMTP):
    """SMTP client that keeps the server's reply to DATA.

    Relays return their queue id there (Brevo: "250 ... queued as <id>"), which
    is the only handle for finding a specific message in the provider's own
    delivery log. Without it, a message the relay accepted and then dropped is
    untraceable from our side.
    """

    last_data_response: str = ""

    def data(self, msg):  # type: ignore[override]
        code, resp = super().data(msg)
        self.last_data_response = (
            resp.decode(errors="replace") if isinstance(resp, bytes) else str(resp)
        )
        return code, resp


def _send(to_email: str, subject: str, plain: str, html: str) -> bool:
    """Deliver one message.

    Returns True when it was actually handed to the relay, False when sending is
    switched off (no SMTP credentials). Raises RuntimeError when delivery was
    attempted and failed — callers must not treat that as success.
    """
    if not SMTP_USER or not SMTP_PASS:
        logger.warning("[email] SMTP not configured — would send to <%s>: %s", to_email, subject)
        return False
    # Static misconfiguration — warn once per process rather than on every send.
    # It is also returned to the caller via sender_domain_warning(), so quieting
    # the log here doesn't hide it.
    global _sender_warning_logged
    warning = sender_domain_warning()
    if warning and not _sender_warning_logged:
        logger.warning("[email] %s", warning)
        _sender_warning_logged = True
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"Jarvis <{FROM_EMAIL}>"
    msg["To"] = to_email
    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html, "html"))
    try:
        with _RecordingSMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            refused = server.send_message(msg)
            queued = server.last_data_response
        if refused:
            # Partial acceptance: some recipients were rejected outright.
            logger.error("[email] Recipients refused for '%s': %s", subject, refused)
            raise RuntimeError(f"Recipients refused: {refused}")
        # Log the relay's queue id — the handle for looking this exact message up
        # in the provider's delivery log when it is accepted here but never lands.
        logger.info(
            "[email] Accepted by relay: to=%s from=%s subject=%r relay_response=%r",
            to_email, FROM_EMAIL, subject, queued,
        )
        return True
    except Exception as exc:
        logger.error("[email] Delivery failed for %s: %s", to_email, exc)
        raise RuntimeError(f"Email delivery failed: {exc}") from exc


def send_welcome_email(to_email: str, name: str) -> None:
    """Send a welcome / confirmation email after successful registration."""
    plain = (
        f"Hi {name},\n\n"
        f"Welcome to Jarvis! Your account has been created successfully.\n\n"
        f"Sign in at: {APP_URL}/login\n\n"
        f"— The Jarvis Team"
    )
    html = f"""<!doctype html>
<html lang="en">
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:520px;margin:40px auto;padding:0 16px;color:#111">
  <div style="text-align:center;margin-bottom:32px">
    <div style="display:inline-flex;align-items:center;gap:8px">
      <div style="width:12px;height:12px;border-radius:50%;background:linear-gradient(135deg,#6366f1,#8b5cf6)"></div>
      <span style="font-weight:600;font-size:18px">Jarvis</span>
    </div>
  </div>
  <div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:32px">
    <h2 style="font-size:20px;margin:0 0 8px">Welcome, {name}!</h2>
    <p style="color:#6b7280;margin:0 0 24px">
      Your Jarvis account has been created. You're all set to get started.
    </p>
    <a href="{APP_URL}/login"
       style="display:block;text-align:center;background:#6366f1;color:#fff;padding:14px 24px;border-radius:8px;text-decoration:none;font-weight:600;font-size:15px">
      Sign In &rarr;
    </a>
  </div>
  <p style="color:#9ca3af;font-size:12px;text-align:center;margin-top:24px">
    If you didn't create this account, please ignore this email.
  </p>
</body>
</html>"""
    _send(to_email, "Welcome to Jarvis", plain, html)


def send_invite_email(to_email: str, team_name: str, inviter_name: str, code: str, role: str) -> bool:
    """Send a team invite email.

    Returns True if it was handed to the relay, False if sending is switched off
    (the code is logged instead). Raises RuntimeError if delivery was attempted
    and failed — the caller surfaces that so nobody is told "invite sent" when
    no mail left the building.
    """
    if not SMTP_USER or not SMTP_PASS:
        logger.warning(
            "[invite] SMTP not configured — invite code for <%s>: %s  accept at: %s/invite/%s",
            to_email, code, APP_URL, code,
        )
        return False

    plain = (
        f"Hi,\n\n"
        f"{inviter_name} has invited you to join \"{team_name}\" as {role} on Jarvis.\n\n"
        f"Accept here: {APP_URL}/invite/{code}\n\n"
        f"Or enter invite code manually: {code}\n\n"
        f"This invite expires in 1 hour.\n\n— Jarvis"
    )

    html = f"""<!doctype html>
<html lang="en">
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:520px;margin:40px auto;padding:0 16px;color:#111">
  <div style="text-align:center;margin-bottom:32px">
    <div style="display:inline-flex;align-items:center;gap:8px">
      <div style="width:12px;height:12px;border-radius:50%;background:linear-gradient(135deg,#6366f1,#8b5cf6)"></div>
      <span style="font-weight:600;font-size:18px">Jarvis</span>
    </div>
  </div>
  <div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:32px">
    <h2 style="font-size:20px;margin:0 0 8px">You&rsquo;re invited to join <em>{team_name}</em></h2>
    <p style="color:#6b7280;margin:0 0 24px">
      <strong>{inviter_name}</strong> has invited you as <strong>{role}</strong>.
    </p>
    <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;padding:24px;text-align:center;margin-bottom:24px">
      <p style="margin:0 0 8px;color:#6b7280;font-size:13px">Your invite code</p>
      <p style="margin:0;font-size:17px;font-weight:700;letter-spacing:2px;font-family:monospace;color:#111;word-break:break-all">{code}</p>
    </div>
    <a href="{APP_URL}/invite/{code}"
       style="display:block;text-align:center;background:#6366f1;color:#fff;padding:14px 24px;border-radius:8px;text-decoration:none;font-weight:600;font-size:15px">
      Accept Invitation &rarr;
    </a>
  </div>
  <p style="color:#9ca3af;font-size:12px;text-align:center;margin-top:24px">
    This invite expires in 1 hour. If you weren&rsquo;t expecting this, you can safely ignore it.
  </p>
</body>
</html>"""

    return _send(to_email, f"You're invited to join {team_name} on Jarvis", plain, html)


def send_mom_email(to_email: str, summary: dict) -> None:
    """Email the meeting Minutes of Meeting (MOM) to a recipient.

    `summary` is the jarvis_sessions.summary object (MeetingSummary shape):
    title, date, summary, key_topics[], decisions[], action_items[{description,owner}],
    mom[{topic,summary}], participants[].
    """
    title = summary.get("title") or "Meeting Summary"
    date = summary.get("date") or ""
    overview = summary.get("summary") or ""
    key_topics = [t for t in (summary.get("key_topics") or []) if isinstance(t, str)]
    decisions = [d for d in (summary.get("decisions") or []) if isinstance(d, str)]
    action_items = summary.get("action_items") or []
    mom = summary.get("mom") or []
    participants = [p for p in (summary.get("participants") or []) if isinstance(p, str)]

    # ── Plain text ──
    lines = [title]
    if date:
        lines.append(date)
    lines.append("")
    if overview:
        lines += ["SUMMARY", overview, ""]
    if key_topics:
        lines += ["KEY TOPICS"] + [f"- {t}" for t in key_topics] + [""]
    if decisions:
        lines += ["DECISIONS"] + [f"- {d}" for d in decisions] + [""]
    if action_items:
        lines.append("ACTION ITEMS")
        for a in action_items:
            desc = a.get("description", "") if isinstance(a, dict) else str(a)
            owner = a.get("owner") if isinstance(a, dict) else None
            lines.append(f"- {desc}" + (f" (owner: {owner})" if owner else ""))
        lines.append("")
    if mom:
        lines.append("MINUTES")
        for m in mom:
            if isinstance(m, dict):
                lines.append(f"- {m.get('topic', '')}: {m.get('summary', '')}")
        lines.append("")
    if participants:
        lines += ["PARTICIPANTS", ", ".join(participants), ""]
    lines.append("— Jarvis")
    plain = "\n".join(lines)

    # ── HTML ──
    def _ul(items: list[str]) -> str:
        return "<ul style='margin:0 0 16px;padding-left:20px;color:#374151'>" + "".join(
            f"<li style='margin:4px 0'>{i}</li>" for i in items
        ) + "</ul>"

    sections = ""
    if overview:
        sections += f"<h3 style='font-size:15px;margin:20px 0 6px'>Summary</h3><p style='color:#374151;margin:0 0 16px'>{overview}</p>"
    if key_topics:
        sections += "<h3 style='font-size:15px;margin:20px 0 6px'>Key Topics</h3>" + _ul(key_topics)
    if decisions:
        sections += "<h3 style='font-size:15px;margin:20px 0 6px'>Decisions</h3>" + _ul(decisions)
    if action_items:
        ai = []
        for a in action_items:
            if isinstance(a, dict):
                desc = a.get("description", "")
                owner = a.get("owner")
                ai.append(f"{desc}" + (f" <em style='color:#6b7280'>— {owner}</em>" if owner else ""))
        sections += "<h3 style='font-size:15px;margin:20px 0 6px'>Action Items</h3>" + _ul(ai)
    if mom:
        mm = [f"<strong>{m.get('topic', '')}</strong>: {m.get('summary', '')}" for m in mom if isinstance(m, dict)]
        sections += "<h3 style='font-size:15px;margin:20px 0 6px'>Minutes</h3>" + _ul(mm)
    if participants:
        sections += f"<h3 style='font-size:15px;margin:20px 0 6px'>Participants</h3><p style='color:#374151;margin:0'>{', '.join(participants)}</p>"

    html = f"""<!doctype html>
<html lang="en">
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:640px;margin:40px auto;padding:0 16px;color:#111">
  <div style="text-align:center;margin-bottom:24px">
    <div style="display:inline-flex;align-items:center;gap:8px">
      <div style="width:12px;height:12px;border-radius:50%;background:linear-gradient(135deg,#6366f1,#8b5cf6)"></div>
      <span style="font-weight:600;font-size:18px">Jarvis</span>
    </div>
  </div>
  <div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:32px">
    <h2 style="font-size:20px;margin:0 0 4px">{title}</h2>
    {f"<p style='color:#9ca3af;font-size:13px;margin:0 0 16px'>{date}</p>" if date else ""}
    {sections}
  </div>
  <p style="color:#9ca3af;font-size:12px;text-align:center;margin-top:24px">
    Minutes of Meeting generated by Jarvis.
  </p>
</body>
</html>"""

    _send(to_email, f"Minutes of Meeting — {title}", plain, html)


def send_recap_email(to_email: str, summary: dict) -> None:
    """Send a concise post-meeting recap (summary + decisions + action items)."""
    title = summary.get("title") or "Meeting Recap"
    overview = summary.get("summary") or ""
    decisions = [d for d in (summary.get("decisions") or []) if isinstance(d, str)]
    action_items = summary.get("action_items") or []

    lines = [f"Recap — {title}", ""]
    if overview:
        lines += [overview, ""]
    if decisions:
        lines += ["DECISIONS"] + [f"- {d}" for d in decisions] + [""]
    if action_items:
        lines.append("ACTION ITEMS")
        for a in action_items:
            desc = a.get("description", "") if isinstance(a, dict) else str(a)
            owner = a.get("owner") if isinstance(a, dict) else None
            lines.append(f"- {desc}" + (f" ({owner})" if owner else ""))
    plain = "\n".join(lines + ["", "— Jarvis"])

    def _ul(items: list[str]) -> str:
        return "<ul style='margin:0 0 16px;padding-left:20px;color:#374151'>" + "".join(
            f"<li style='margin:4px 0'>{i}</li>" for i in items
        ) + "</ul>"

    body = ""
    if overview:
        body += f"<p style='color:#374151;margin:0 0 16px'>{overview}</p>"
    if decisions:
        body += "<h3 style='font-size:15px;margin:20px 0 6px'>Decisions</h3>" + _ul(decisions)
    if action_items:
        ai = []
        for a in action_items:
            if isinstance(a, dict):
                ai.append(a.get("description", "") + (f" <em style='color:#6b7280'>— {a.get('owner')}</em>" if a.get("owner") else ""))
        body += "<h3 style='font-size:15px;margin:20px 0 6px'>Action Items</h3>" + _ul(ai)

    html = f"""<!doctype html>
<html lang="en"><body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:600px;margin:40px auto;padding:0 16px;color:#111">
  <div style="text-align:center;margin-bottom:20px"><span style="font-weight:600;font-size:18px">Jarvis</span></div>
  <div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:28px">
    <h2 style="font-size:20px;margin:0 0 12px">Recap — {title}</h2>
    {body or "<p style='color:#6b7280'>No recap content available.</p>"}
  </div>
</body></html>"""
    _send(to_email, f"Meeting recap — {title}", plain, html)


def send_action_items_email(to_email: str, name: str, items: list[dict], meeting_title: str | None = None) -> None:
    """Send a person their assigned action items for a meeting."""
    where = f" from {meeting_title}" if meeting_title else ""
    rows = [f"- {i.get('description','')}" + (f" (due {i['due']})" if i.get("due") else "") for i in items]
    plain = f"Hi {name},\n\nYour action items{where}:\n\n" + "\n".join(rows) + "\n\n— Jarvis"

    li = "".join(
        f"<li style='margin:6px 0'>{i.get('description','')}"
        + (f" <span style='color:#6b7280'>(due {i['due']})</span>" if i.get("due") else "")
        + "</li>"
        for i in items
    )
    html = f"""<!doctype html>
<html lang="en"><body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:560px;margin:40px auto;padding:0 16px;color:#111">
  <div style="text-align:center;margin-bottom:20px"><span style="font-weight:600;font-size:18px">Jarvis</span></div>
  <div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:28px">
    <h2 style="font-size:18px;margin:0 0 8px">Your action items{(' from ' + meeting_title) if meeting_title else ''}</h2>
    <p style="color:#6b7280;margin:0 0 14px">Hi {name}, please complete and mark these done in Jarvis.</p>
    <ul style="margin:0;padding-left:20px;color:#374151">{li}</ul>
  </div>
</body></html>"""
    _send(to_email, f"Your action items{(' — ' + meeting_title) if meeting_title else ''}", plain, html)
