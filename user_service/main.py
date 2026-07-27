"""
Org / User Service — port 8003.

Manages users, teams, org hierarchy, RBAC, and bot assignment.
JWT-protected; all data stored in Supabase.

Run:
    uvicorn user_service.main:app --host 0.0.0.0 --port 8003

Required env vars:
    SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
    JWT_SECRET (strong random string in production)

Optional:
    JWT_ACCESS_TTL_MINUTES   (default: 60)
    JWT_REFRESH_TTL_DAYS     (default: 7)
    CORS_ORIGINS             (comma-separated, default: *)

API surface:
    POST   /auth/register                  create user
    POST   /auth/login                     obtain JWT
    POST   /auth/refresh                   rotate token
    POST   /auth/logout                    (stateless — client discards token)

    GET    /users                          list all users (CEO)
    GET    /users/me                       own profile
    GET    /users/{id}                     user profile
    PATCH  /users/{id}                     update user
    GET    /users/{id}/reports             direct reports
    GET    /users/{id}/manager             reporting manager

    GET    /teams                          list teams (own teams or all for CEO)
    POST   /teams                          create team (CEO)
    GET    /teams/{id}                     team details
    PATCH  /teams/{id}                     update team (CEO or team manager)
    DELETE /teams/{id}                     delete team (CEO)
    GET    /teams/{id}/members             list members
    POST   /teams/{id}/members             add member (CEO or team manager)
    DELETE /teams/{id}/members/{uid}       remove member (CEO or team manager)

    GET    /org/hierarchy                  full org tree (CEO)

    GET    /teams/{id}/bot                 assigned bot
    POST   /teams/{id}/bot                 assign bot (CEO)
    GET    /teams/{id}/meetings            team meeting history (any team member)
    GET    /teams/{id}/meetings/{sid}/participants  per-person in-call time (team manager/ADMIN)
    POST   /teams/{id}/meetings/{sid}/kick  kick bot from a team meeting (team manager/ADMIN)

    GET    /org/hierarchy                  full org tree (ADMIN/CEO)
    GET    /org/hierarchy/me               personal reporting tree (any user)

    GET    /admin/meetings/live           live meetings org-wide (ADMIN/CEO)
    GET    /admin/meetings                meeting history org-wide (ADMIN/CEO)
    GET    /admin/meetings/{id}           meeting detail + summary/MOM (ADMIN/CEO)
    POST   /admin/meetings/{id}/kick      remove bot from a meeting (ADMIN/CEO)

    GET    /health
"""

import logging
import os
from dotenv import load_dotenv

load_dotenv()

# Uvicorn configures only its OWN loggers, leaving the root logger at WARNING —
# so every application logger.info() was silently discarded. That hid the
# "[email] Sent ... to ..." confirmations while letting the warnings through,
# which made a delivery problem impossible to diagnose from the logs: you could
# see neither that a mail had been sent nor that it hadn't. bot_service has
# always done this; org-service was the odd one out.
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .database import DBError, delete_all

from .routes.auth import router as auth_router
from .routes.users import router as users_router
from .routes.teams import router as teams_router
from .routes.hierarchy import router as hierarchy_router
from .routes.bots import router as bots_router
from .routes.invites import router as invites_router
from .routes.analytics import router as analytics_router
from .routes.admin_meetings import router as admin_meetings_router
from .routes.meetings import router as meetings_router
from .routes.action_items import router as action_items_router

_CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
_ALLOW_DB_RESET = os.getenv("ALLOW_DB_RESET", "false").lower() == "true"

app = FastAPI(
    title="Jarvis Org / User Service",
    version="1.0",
    description="Manages organisation hierarchy, teams, RBAC, and bot assignments.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(DBError)
async def db_error_handler(request: Request, exc: DBError):
    return JSONResponse(status_code=500, content={"detail": str(exc)})

app.include_router(auth_router)
app.include_router(users_router)
app.include_router(teams_router)
app.include_router(hierarchy_router)
app.include_router(bots_router)
app.include_router(invites_router)
app.include_router(analytics_router)
app.include_router(admin_meetings_router)
app.include_router(meetings_router)
app.include_router(action_items_router)


@app.get("/health")
def health():
    return {"status": "ok", "service": "org-user-service"}


# Dev-only: wipes all rows from every org table (schema stays intact).
# Called automatically by the frontend after Google sign-in.
_RESET_ORDER = [
    ("org_team_invitations", "id"),
    ("org_team_bots", "id"),
    ("org_team_members", "team_id"),
    ("org_reporting_hierarchy", "ancestor_id"),
    ("jarvis_sessions", "session_id"),
    ("org_users", "id"),
    ("org_teams", "id"),
    ("organizations", "id"),
]

@app.post("/admin/reset", status_code=200)
def reset_db():
    if not _ALLOW_DB_RESET:
        raise HTTPException(status_code=403, detail="DB reset is disabled. Set ALLOW_DB_RESET=true in .env to enable.")
    errors = []
    for table, col in _RESET_ORDER:
        try:
            delete_all(table, col)
        except Exception as exc:
            errors.append(f"{table}: {exc}")
    if errors:
        return {"ok": False, "errors": errors}
    return {"ok": True, "cleared": [t for t, _ in _RESET_ORDER]}
