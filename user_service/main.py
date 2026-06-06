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
    GET    /teams/{id}/meetings            team-scoped meetings

    GET    /health
"""

import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .database import DBError

from .routes.auth import router as auth_router
from .routes.users import router as users_router
from .routes.teams import router as teams_router
from .routes.hierarchy import router as hierarchy_router
from .routes.bots import router as bots_router
from .routes.invites import router as invites_router

_CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]

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


@app.get("/health")
def health():
    return {"status": "ok", "service": "org-user-service"}
