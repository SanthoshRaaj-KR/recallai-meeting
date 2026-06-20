"""Meeting MOM email route — any authenticated org user can email themselves the
Minutes of Meeting for a session in their organisation."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from ..auth import get_current_user
from ..database import select_one
from ..email import send_mom_email

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
