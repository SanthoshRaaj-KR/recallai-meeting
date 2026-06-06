"""Email utility — sends team invite emails, falls back to console log if SMTP not configured."""
from __future__ import annotations

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", SMTP_USER) or "noreply@jarvis.app"
APP_URL = os.getenv("APP_URL", "http://localhost:3000").rstrip("/")


def send_invite_email(to_email: str, team_name: str, inviter_name: str, code: str, role: str) -> None:
    """Send a team invite email. Logs the code to console if SMTP is not configured."""
    if not SMTP_USER or not SMTP_PASS:
        logger.warning(
            "[invite] SMTP not configured — invite code for <%s>: %s  accept at: %s/invite/%s",
            to_email, code, APP_URL, code,
        )
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"You're invited to join {team_name} on Jarvis"
    msg["From"] = f"Jarvis <{FROM_EMAIL}>"
    msg["To"] = to_email

    plain = (
        f"Hi,\n\n"
        f"{inviter_name} has invited you to join \"{team_name}\" as {role} on Jarvis.\n\n"
        f"Accept here: {APP_URL}/invite/{code}\n\n"
        f"Or enter invite code manually: {code}\n\n"
        f"This invite expires in 48 hours.\n\n— Jarvis"
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
    This invite expires in 48 hours. If you weren&rsquo;t expecting this, you can safely ignore it.
  </p>
</body>
</html>"""

    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        logger.info("[invite] Email sent to %s", to_email)
    except Exception as exc:
        logger.error("[invite] Email delivery failed for %s: %s", to_email, exc)
        raise RuntimeError(f"Email delivery failed: {exc}") from exc
