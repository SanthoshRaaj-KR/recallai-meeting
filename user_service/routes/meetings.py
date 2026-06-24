"""Meeting MOM email route — any authenticated org user can email themselves the
Minutes of Meeting for a session in their organisation."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from ..auth import get_current_user
from ..database import select, select_one
from ..email import send_mom_email, send_recap_email

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/meetings", tags=["meetings"])


@router.post("/{session_id}/email-mom")
def email_mom(session_id: str, claims: dict = Depends(get_current_user)):
    """Email the meeting's MOM to the signed-in user's own email address."""
    user = select_one("org_users", {"id": f"eq.{claims['sub']}"})
    if not user or not user.get("email"):
        raise HTTPException(404, "Your account email was not found")

    s = select_one("jarvis_sessions", {"session_id": f"eq.{session_id}"})
    if not s:
        raise HTTPException(404, "Meeting not found")

    # Ownership guard: don't let one org email out another org's MOM.
    team_id = s.get("team_id")
    if team_id:
        team = select_one("org_teams", {"id": f"eq.{team_id}"})
        if not team or team.get("org_id") != claims.get("org_id"):
            raise HTTPException(403, "This meeting is not in your organisation")

    summary = s.get("summary") or {}
    if not summary:
        raise HTTPException(409, "This meeting has no summary/MOM yet")

    try:
        send_mom_email(user["email"], summary)
    except Exception as exc:
        logger.warning("email-mom failed for %s: %s", session_id, exc)
        raise HTTPException(502, f"Could not send the email: {exc}")

    return {"ok": True, "sent_to": user["email"]}


@router.post("/{session_id}/share")
def share_meeting(session_id: str, body: dict, claims: dict = Depends(get_current_user)):
    """ADMIN/MANAGER: share the MOM/summary with a specific email address."""
    if claims.get("role") not in ("CEO", "ADMIN", "MANAGER"):
        raise HTTPException(403, "Only a manager, ADMIN or CEO can share")
    to_email = (body.get("email") or "").strip()
    if not to_email:
        raise HTTPException(400, "email is required")
    s = select_one("jarvis_sessions", {"session_id": f"eq.{session_id}"})
    if not s:
        raise HTTPException(404, "Meeting not found")
    team_id = s.get("team_id")
    if team_id:
        team = select_one("org_teams", {"id": f"eq.{team_id}"})
        if not team or team.get("org_id") != claims.get("org_id"):
            raise HTTPException(403, "This meeting is not in your organisation")
    summary = s.get("summary") or {}
    if not summary:
        raise HTTPException(409, "This meeting has no summary yet")
    try:
        send_recap_email(to_email, summary)
    except Exception as exc:
        logger.warning("share failed for %s: %s", to_email, exc)
        raise HTTPException(502, f"Could not send the email: {exc}")
    return {"ok": True, "sent_to": to_email}


@router.post("/{session_id}/recap-email")
def recap_email(session_id: str, claims: dict = Depends(get_current_user)):
    """Email a meeting recap to the session team's roster.

    Participants in the summary are display names without emails, so the recap is
    sent to the team-roster emails (the people who'd act on it)."""
    s = select_one("jarvis_sessions", {"session_id": f"eq.{session_id}"})
    if not s:
        raise HTTPException(404, "Meeting not found")
    team_id = s.get("team_id")
    if team_id:
        team = select_one("org_teams", {"id": f"eq.{team_id}"})
        if not team or team.get("org_id") != claims.get("org_id"):
            raise HTTPException(403, "This meeting is not in your organisation")
    summary = s.get("summary") or {}
    if not summary:
        raise HTTPException(409, "This meeting has no summary yet")

    emails: list[str] = []
    if team_id:
        mids = [m["user_id"] for m in select("org_team_members", {"team_id": f"eq.{team_id}", "select": "user_id"})]
        if mids:
            for u in select("org_users", {"id": f"in.({','.join(mids)})", "select": "email"}):
                if u.get("email"):
                    emails.append(u["email"])
    if not emails:  # fall back to the caller
        me = select_one("org_users", {"id": f"eq.{claims['sub']}"})
        if me and me.get("email"):
            emails.append(me["email"])

    sent = 0
    for addr in emails:
        try:
            send_recap_email(addr, summary)
            sent += 1
        except Exception as exc:
            logger.warning("recap-email failed for %s: %s", addr, exc)
    return {"ok": True, "emailed": sent}
