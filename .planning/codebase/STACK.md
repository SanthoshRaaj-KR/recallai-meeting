# Technology Stack

**Analysis Date:** 2026-05-11

## Runtime & Language

**Backend:**
- Python 3 (asyncio-native) — all server-side logic in `confluence_logic/` and `local_office_logic/`

**Frontend (review-ui):**
- TypeScript 5 — `review-ui/` Next.js app

**Frontend (sync-sage-bot):**
- TypeScript 5 — `sync-sage-bot/` Vite/React SPA

## Frameworks & Libraries

**Backend Web Framework:**
- FastAPI — HTTP + WebSocket server; main app at `confluence_logic/jarvis_agentic.py`, review router at `confluence_logic/review/api.py`
- Uvicorn — ASGI server; launched with `uvicorn confluence_logic.jarvis_agentic:app`

**AI / Agents:**
- `openai` — LLM chat completions, streaming, embeddings, TTS (gpt-4o-mini, gpt-5-mini defaults)
- `openai-agents` (import name `agents`) — OpenAI Agents SDK; used for `Agent`, `Runner`, `function_tool` in `confluence_logic/agents/` and `local_office_logic/agents/`

**Frontend (review-ui):**
- Next.js 16 (^16.2.4) with App Router — `review-ui/src/app/`
- React 18 — UI rendering
- Tailwind CSS 3 — styling
- TypeScript 5
- No state management library — local `useState`/`useEffect` only
- `next.config.ts` rewrites `/api/*` to `http://localhost:8000/*` (FastAPI proxy)

**Frontend (sync-sage-bot):**
- Vite 7 + `@vitejs/plugin-react-swc` — build tooling
- React 18.3 — UI rendering
- React Router DOM 6 — client-side routing
- TanStack React Query 5 — server state / data fetching
- Radix UI (full component suite, ^1.x–^2.x per component) — headless UI primitives
- shadcn/ui conventions (Radix + Tailwind + CVA)
- Zod 3 — schema validation
- React Hook Form 7 + `@hookform/resolvers` — form management
- `@supabase/supabase-js` 2 — Supabase client
- Recharts 2 — charts
- Tailwind CSS 3 + `tailwindcss-animate` + `@tailwindcss/typography`
- `date-fns` 3, `lucide-react`, `sonner`, `cmdk`, `vaul`, `embla-carousel-react`

## Package Management

**Backend:**
- pip — `requirements.txt` at repo root (no lockfile committed)

**Frontend:**
- npm — `package-lock.json` present in both `review-ui/` and `sync-sage-bot/`

## Build & Tooling

**Backend:**
- `python-dotenv` — environment loading via `.env` files
- `pytest` + `pytest-asyncio` — test runner

**Frontend (review-ui):**
- `next build` / `next dev` — standard Next.js build
- ESLint 8 + `eslint-config-next`
- PostCSS + Autoprefixer

**Frontend (sync-sage-bot):**
- Vite 7 — dev server and production bundler
- `vitest` 3 + `@testing-library/react` 16 + `jsdom` — unit/component tests
- ESLint 9 + `typescript-eslint` 8 + `eslint-plugin-react-hooks`
- `lovable-tagger` — Lovable platform integration tag

**Document Conversion:**
- `docling` — HTML-to-Markdown document conversion in `confluence_logic/ingestion/doc_pipeline.py`
- LibreOffice headless (`soffice`) — office format conversion in `local_office_logic/utils/office_runtime.py`

## Key Dependencies

| Package | Version (requirements.txt) | Purpose |
|---------|---------------------------|---------|
| fastapi | latest | HTTP + WebSocket API server |
| uvicorn | latest | ASGI production server |
| openai | latest | LLM completions, embeddings, TTS |
| openai-agents | latest | Agentic tool orchestration (Agent/Runner) |
| requests | latest | HTTP calls to Recall.ai, Confluence, Supabase REST, Tavily |
| python-dotenv | latest | `.env` config loading |
| pinecone | latest | Vector store client (`confluence_logic/db/vector_store.py`) |
| neo4j | 6.1.0 | AuraDB graph database driver (Graph RAG) — **NOTE: this version does not exist on PyPI** |
| gTTS | latest | Google Text-to-Speech (fallback TTS provider) |
| edge-tts | latest | Microsoft Edge TTS (low-latency TTS provider) |
| websockets | latest | WebSocket protocol support |
| pyaudio | latest | Audio I/O |
| pydantic | latest | Data models and schema validation |
| docling | latest | HTML document conversion to Markdown |
| beautifulsoup4 | latest | HTML parsing (`confluence_logic/confluence_page_graph.py`) |
| ngrok | latest | Tunnel for local webhook exposure |
| python-docx | latest | DOCX file manipulation |
| openpyxl | latest | XLSX file manipulation |
| odfpy | latest | ODF (ODT/ODS) file manipulation |
| lxml | latest | XML/HTML parsing |
| pytest | latest | Test framework |
| pytest-asyncio | latest | Async test support |

## Environment Variables Reference

Key variables loaded at module level (not exhaustive — see `.env`):

| Variable | Default | Used by |
|----------|---------|---------|
| `RECALL_API_KEY` | — | `jarvis_agentic.py` |
| `OPENAI_API_KEY` | — | all LLM/TTS calls |
| `SUPABASE_URL` | — | `review/supabase_store.py` |
| `SUPABASE_ANON_KEY` | — | `review/supabase_store.py` |
| `SUPABASE_SERVICE_ROLE_KEY` | — | `review/supabase_store.py` |
| `PINECONE_API_KEY` | — | `db/vector_store.py` |
| `NEO4J_URI` | — | `graph_rag.py`, `confluence_page_graph.py` |
| `TAVILY_API_KEY` | `""` | `confluence_logic/general_responder.py` (optional web search) |
| `JARVIS_AGENT_MODEL` | `gpt-5-mini` | agent constructors (**invalid model name**) |
| `JARVIS_GENERAL_MODEL` | `gpt-4o-mini` | `meeting_responder.py`, `general_responder.py` |
| `JARVIS_REVIEW_MODEL` | `gpt-5-mini` | `review/api.py` (**invalid model name**) |
| `JARVIS_TTS_PROVIDER` | `edge_tts` | `jarvis_agentic.py` |
| `APP_HOST` | `0.0.0.0` | `jarvis_agentic.py` |
| `APP_PORT` | `8000` | `jarvis_agentic.py` |

---

*Stack analysis: 2026-05-11*
