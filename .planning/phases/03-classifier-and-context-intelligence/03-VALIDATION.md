---
phase: 03
slug: classifier-and-context-intelligence
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-12
---

# Phase 03 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (installed) |
| **Config file** | None detected — pyproject.toml [tool.pytest.ini_options] may need creation |
| **Quick run command** | `pytest confluence_logic/tests/ -x -q` |
| **Full suite command** | `pytest confluence_logic/tests/ -v` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest confluence_logic/tests/ -x -q`
- **After every plan wave:** Run `pytest confluence_logic/tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 03-sliding-window | 01 | 1 | TOPIC-01 | unit | `pytest confluence_logic/tests/test_jarvis_agentic.py -x -k "test_format_general_history or test_sliding_window"` | ❌ W0 | ⬜ pending |
| 03-web-search-router | 01 | 1 | WEBSEARCH-01 | unit | `pytest confluence_logic/tests/test_general_responder.py -x -k "test_web_search_router"` | ❌ W0 | ⬜ pending |
| 03-graph-rag-ingest | 02 | 1 | GRAPHRAG-01 | unit (mock) | `pytest confluence_logic/tests/test_graph_rag.py -x -k "test_ingest"` | ❌ W0 | ⬜ pending |
| 03-graph-rag-query | 02 | 1 | GRAPHRAG-01 | unit (mock) | `pytest confluence_logic/tests/test_graph_rag.py -x -k "test_query_context"` | ❌ W0 | ⬜ pending |
| 03-graph-injection | 02 | 2 | CLASSIFY-03 | unit (mock) | `pytest confluence_logic/tests/test_jarvis_agentic.py -x -k "test_general_question_graph_injection"` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `confluence_logic/tests/test_graph_rag.py` — stubs for GRAPHRAG-01 (mock AsyncGraphDatabase, mock OpenAI)
- [ ] `confluence_logic/tests/test_general_responder.py` — stubs for WEBSEARCH-01 (mock openai for `_needs_web_search`)
- [ ] Add sliding window test stubs to `confluence_logic/tests/test_jarvis_agentic.py` — stubs for TOPIC-01
- [ ] `pip install pytest-asyncio neo4j==6.1.0` — needed for async tests and Neo4j driver

**Neo4j mocking strategy (no live DB required):**
```python
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_ingest_transcript_entry_calls_merge():
    mock_driver = AsyncMock()
    mock_driver.execute_query = AsyncMock(return_value=([], None, []))
    with patch("confluence_logic.graph_rag._get_driver", return_value=mock_driver):
        with patch("confluence_logic.graph_rag._extract_entities", return_value={"topics": ["React"], "people": [], "decisions": []}):
            await ingest_transcript_entry({"participant": "Alice", "text": "Let's use React"})
    mock_driver.execute_query.assert_called()
```

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Graph query returns meeting context for "what model did colleague suggest" | GRAPHRAG-01 + CLASSIFY-03 | Requires live Neo4j AuraDB + populated transcript | In active meeting, have someone mention a model name, then ask Jarvis "what model did X suggest?" — verify Jarvis names the correct model |
| Topic drift clears after 3 turns | TOPIC-01 | Requires live meeting session with real Q&A sequence | Ask 3+ questions on solar panels, then switch to F1 — after 3 F1 questions, Jarvis should not mention solar panels |
| Weather query triggers web search | WEBSEARCH-01 | LLM response varies | Say "what's the weather in Paris?" — verify Jarvis searches and returns an actual weather result |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
