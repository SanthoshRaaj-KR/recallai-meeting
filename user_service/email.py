"""Email utility — welcome, invite, and meeting-MOM emails.

Provider-agnostic SMTP. Configured for Brevo by default (free 300/day, commercial-OK):
    SMTP_HOST=smtp-relay.brevo.com   SMTP_PORT=587
    SMTP_USER=<your Brevo SMTP login>   SMTP_PASS=<your Brevo SMTP key>
    FROM_EMAIL=<a verified sender on your domain>   APP_URL=https://app.your-domain.com
Any other SMTP provider (MailerSend, Resend, Amazon SES, Gmail) works by overriding
these env vars. If SMTP_USER/PASS are unset, sends are logged (no-op) instead.
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


def _send(to_email: str, subject: str, plain: str, html: str) -> None:
    if not SMTP_USER or not SMTP_PASS:
        logger.warning("[email] SMTP not configured — would send to <%s>: %s", to_email, subject)
        return
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"Jarvis <{FROM_EMAIL}>"
    msg["To"] = to_email
    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html, "html"))
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        logger.info("[email] Sent '%s' to %s", subject, to_email)
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


def send_invite_email(to_email: str, team_name: str, inviter_name: str, code: str, role: str) -> None:
    """Send a team invite email. Logs the code to console if SMTP is not configured."""
    if not SMTP_USER or not SMTP_PASS:
        logger.warning(
            "[invite] SMTP not configured — invite code for <%s>: %s  accept at: %s/invite/%s",
            to_email, code, APP_URL, code,
        )
        return

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
      <p style="margin:0;font-size:36px;font-weight:700;letter-spacing:10px;font-family:monospace;color:#111">{code}</p>
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

    _send(to_email, f"You're invited to join {team_name} on Jarvis", plain, html)


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
