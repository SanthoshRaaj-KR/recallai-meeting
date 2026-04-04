# Feature Landscape: Meeting Memory RAG Bot

**Domain:** Meeting intelligence with hybrid RAG retrieval
**Researched:** 2026-04-04
**Confidence:** MEDIUM-HIGH — competitive landscape verified against live platforms; RAG patterns corroborated by multiple current sources

---

## Table Stakes

Features users expect in a meeting memory system. Missing any of these means the product feels broken, not just incomplete.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| End-of-meeting summary generation | Every meeting note product does this; users expect it automatically | Medium | Must trigger on meeting end AND support `/summarize` manual fallback. Auto-detect via meeting_state.is_active going false. |
| Action item extraction | Users cite this as #1 use case across all competitive products | Medium | Requires structured output format: `{owner, task, due_date, source_meeting_id}`. LLM prompt engineering outperforms fine-tuning for this. |
| Exact-match lookup by date | "What was decided last Tuesday?" — the most natural query type | Medium | Requires natural language date parsing AND metadata filtering. Failing this query type destroys trust immediately. |
| Natural language query via Slack | Bot lives in Slack; query interface must be in-channel | Low | Already have Slack bot framework. Core query path is wake-word or slash command. |
| Persist meeting summaries | Data must survive restart | Low | Dual persistence: JSON file on disk + Pinecone vector record as specified in requirements. |
| Speaker attribution in summaries | "Who said what" matters for accountability | Medium | Recall.ai provides participant + sentence chunks; must propagate to summary metadata. |
| Retrieve by channel/series | "What did #eng-standup discuss this month?" | Medium | Slack channel_id as series key is the right approach — no external calendar dependency. |

---

## Differentiators

Features that create genuine competitive advantage over Notion AI, Fireflies, Otter, and Fathom for this specific embedded-in-meeting use case.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Hybrid RAG (semantic + BM25 + metadata) | Catches both exact-term queries ("decision about vendor X") and semantic queries ("what did we decide about tooling") that pure-semantic-only or pure-keyword-only systems miss | High | Pinecone sparse-dense index handles this natively. Alpha parameter controls the blend; default 0.5. Tune per query type. |
| Cross-meeting trend detection | "Has topic X come up before?" — surfaces recurring concerns across meeting history | High | Topic clustering across embeddings; requires aggregation query type distinct from single-meeting lookup. High research flag (see PITFALLS). |
| Recurring series awareness | Tracks meeting series via Slack channel ID, auto-groups recurring standup/sprint/etc. meetings together | Medium | Enables "this recurring thing" context. series_name and recurrence_pattern in metadata schema supports this. |
| Action item status tracking across series | "What action items from the sprint planning last week are still open?" | High | Requires action items to have persistent identity across meetings. Complex: items need to be linked, not just extracted. Dependency on action item extraction. |
| Agentic multi-step query handling | Complex queries ("compare what was decided about deployment in the last three standups") require multi-hop retrieval + synthesis | High | OpenAI Agents SDK with handoffs enables this. Single-step RAG cannot handle comparative or multi-period queries. |
| Disambiguation clarification | When "last Wednesday" could mean multiple meetings, or a topic appears in 5 meetings, proactively surfaces options instead of silently picking one | Medium | IEEE research confirms interactive disambiguation dramatically improves retrieval accuracy. Present candidate titles; let user narrow. |
| In-meeting real-time context | Can query meeting history mid-meeting ("have we discussed this before?") — Jarvis is live in the meeting, not post-hoc | Low | Unique to this embedded architecture. Competitors are bots that process after. Already have wake-word detection. |

---

## Anti-Features

Things to deliberately NOT build for this milestone. Each is a trap that adds scope without adding proportional value.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Calendar integration (Google Cal, Outlook) | Adds OAuth complexity, per-user credential management, and calendar sync edge cases. Slack channel is already the series identity. | Use Slack channel_id as meeting series key. It's simpler, already available, and works without external permissions. |
| Action item push to task managers (Asana, Jira, Linear) | Integration surface is large, authentication per-user, and mapping to task schemas is fragile. Fireflies does this; it's a distinct product. | Surface action items via Slack messages or query results. External push is Phase N+1. |
| Real-time query streaming / incremental retrieval | Adds WebSocket complexity at the retrieval layer. Meetings move slowly; latency of 2-5s for a memory query is acceptable. | Standard request-response query handling. The Jarvis speaking latency (gTTS + audio) already dominates. |
| Multi-workspace Slack support | Token management, routing complexity, data isolation requirements multiply scope dramatically | Single workspace as specified. Clearly documented in PROJECT.md. |
| Fine-tuning or custom model training | Prompt engineering outperforms fine-tuning for structured extraction (per 2025 research). Training requires labeled data we don't have. | GPT-4o-mini with structured output prompts. Meeting-specific system prompts are sufficient. |
| Full-text search index (Elasticsearch/OpenSearch) | BM25 via Pinecone sparse vectors covers keyword retrieval without operating a separate search cluster. | Pinecone's built-in sparse vectors for BM25. No separate search infrastructure. |
| User preference profiles or personalization | Not warranted at this scale. Single workspace, small team. Research shows personalization matters at scale; it's noise at MVP. | Consistent summaries for everyone. Personalization is Phase N+2. |
| Video/screen recording | Recall.ai provides transcript; video adds storage, processing, and privacy surface for zero query value. | Audio transcript only, as specified. |

---

## Feature Dependencies

Understanding build order matters. These dependencies constrain the implementation sequence.

```
Meeting transcript (existing Recall.ai integration)
  └── Summary generation (requires transcript)
        └── Pinecone storage (requires summary + metadata)
              ├── Exact-date lookup (requires Pinecone + metadata filter)
              │     └── Natural language date resolution (requires date parser)
              ├── Semantic query (requires Pinecone + embeddings)
              ├── Hybrid RAG (requires Pinecone sparse-dense + BM25)
              │     └── Cross-meeting trends (requires hybrid RAG + aggregation)
              └── Action item extraction (requires summary)
                    └── Action item tracking across series (requires extraction + item identity)

Multi-agent handoffs (requires OpenAI Agents SDK)
  ├── Query routing (requires handoffs + agent definitions)
  ├── Disambiguation clarification (requires query routing)
  └── Agentic cross-meeting queries (requires query routing + hybrid RAG)
```

Key constraint: **Pinecone storage with metadata is the critical path**. Everything retrieval-related blocks on it. Build it first.

Second constraint: **Action item tracking across meetings** depends on action item extraction AND on establishing persistent item identity — items need IDs and linkage logic, not just extraction. This is the most complex feature and should be scoped carefully.

---

## Query Type Taxonomy

The system must handle four structurally distinct query types. Each has different retrieval logic:

| Query Type | Example | Retrieval Strategy | Complexity |
|------------|---------|-------------------|------------|
| Exact event lookup | "What was decided last Tuesday?" | Date filter → metadata lookup → single meeting fetch | Medium — requires date parsing + filter |
| General summary | "What did the team discuss this month?" | Date range filter → multi-doc retrieval → synthesis | Medium — requires aggregation |
| Cross-meeting trends | "Has deployment latency come up repeatedly?" | Semantic search across all history → topic grouping → trend synthesis | High — no natural date anchor; must score frequency + recency |
| Action item tracking | "What action items from sprint planning are still open?" | Meeting series filter → action item extraction → status inference | High — requires item identity across meetings |

The query router (multi-agent handoff entrypoint) must classify incoming queries into one of these types before selecting a retrieval strategy. Misclassification is a primary failure mode.

---

## Summarization Quality: What Actually Matters

Research on LLM-powered meeting recap systems (ACM CHI 2024/2025) identified these as the features users actually care about, ranked by impact:

1. **Context access** — Users consistently want to drill into source context ("show me the original transcript chunk"). Summaries without links to source feel untrustworthy. Surface source meeting_id and timestamp in every answer.

2. **Attribution accuracy** — Speaker attribution errors ("Sarah said X" when Bob said it) destroy trust faster than any other error type. Pronoun assignment for non-Western names was specifically cited as a failure mode. Test with diverse participant names.

3. **Hierarchical depth** — Both flat highlights (quick scan) and topic-organized detail (deep read) are needed for different use cases. Generate structured summaries with topics_covered array for navigation.

4. **Action item completeness** — Users were far more tolerant of summary omissions than of missing action items. Action items are held to a higher recall standard than general content.

**What does NOT matter much at this scale:** sentiment analysis, talk-time ratios, speaker coaching. These are enterprise add-ons for large sales teams (Fireflies' differentiator). Not relevant to internal team meetings.

---

## Disambiguation Pattern: When Multiple Meetings Match

This is the most under-specified feature in the requirements and the most common user-facing failure mode in RAG systems. Based on research:

**When to disambiguate (trigger conditions):**
- Date expression is ambiguous (e.g., "last sprint" could span multiple meetings, or the user's timezone makes "last Wednesday" ambiguous at day boundaries)
- Query matches 3+ meetings with similar semantic scores (score variance < threshold)
- Query has no temporal anchor and recurring series has many matches

**Disambiguation response pattern (from IEEE interactive disambiguation research):**
Present candidate meeting titles with dates, not just "which meeting?" — users cannot disambiguate without context. Format:

```
Found 3 matching meetings:
1. Eng Standup — Mar 31 (3 days ago)
2. Eng Standup — Mar 28 (6 days ago)
3. Eng Standup — Mar 24 (10 days ago)
Which one? (reply 1, 2, 3, or "all")
```

**When NOT to disambiguate (don't over-ask):**
- Single clear match regardless of phrasing
- Cross-meeting trend queries explicitly ask for "all" / "ever" / "across meetings"
- User already specified a specific channel — trust that as series identifier

**Anti-pattern:** Asking "did you mean X or Y?" without showing enough context for the user to answer. Forces a second round trip.

---

## Recurring Meeting Series: Feature Scope

Using Slack channel_id as meeting series identity (already decided) enables these specific feature behaviors:

**Include:**
- Group meetings by series in query results
- "Latest meeting in this series" shortcut query
- Show meeting N-of-series context in summaries ("this was the 4th sprint planning")
- Carry forward unresolved action items in series-scoped action item queries

**Exclude from this milestone:**
- Recurrence pattern inference (daily/weekly/biweekly detection) — adds complexity for marginal value; series grouping is sufficient
- Cross-series correlation ("how does eng standup relate to product planning?") — requires graph-like query structure; out of scope

---

## MVP Recommendation

For the first shippable iteration of the memory milestone, prioritize in this order:

**Must ship:**
1. Summary generation (auto on meeting end + `/summarize`)
2. Pinecone storage with full metadata schema
3. Exact date lookup ("what was decided last Tuesday?")
4. Action item extraction per meeting
5. Natural language date resolution

**Ship in second iteration:**
6. Hybrid RAG (semantic + BM25) — table stakes for production accuracy but adds setup complexity
7. Multi-agent query routing (query type classification + handoffs)
8. Recurring series grouping

**Defer to explicit later scope:**
9. Cross-meeting trend detection — high complexity, requires enough historical data to be useful
10. Action item status tracking across meetings — requires item identity system, not just extraction
11. Disambiguation clarification flow — can ship with "pick most recent match" heuristic initially

**Reasoning:** Date lookup + action item extraction on single meetings is immediately useful. Cross-meeting features require history to exist first — you can't detect trends from one meeting.

---

## Sources

- [Summaries, Highlights, and Action Items: LLM-powered Meeting Recap (ACM 2024)](https://arxiv.org/html/2307.15793v3) — MEDIUM confidence; peer-reviewed research on what features users value
- [Contextual RAG for Meeting Notes and Slack (Luna.ai)](https://withluna.ai/blog/contextual-rag-product-meeting-notes-slack) — MEDIUM confidence; practitioner experience with meeting RAG
- [Mitigating Retrieval Errors via Ambiguity Detection (IEEE 2025)](https://ieeexplore.ieee.org/document/11225289/) — MEDIUM confidence; specific to disambiguation patterns
- [Optimizing RAG with Hybrid Search and Reranking (Superlinked/VectorHub)](https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking) — MEDIUM confidence; hybrid search best practices
- [Pinecone Hybrid Search Documentation](https://docs.pinecone.io/guides/search/hybrid-search) — HIGH confidence; official Pinecone docs
- [Top Meeting Intelligence Platforms 2026 (AssemblyAI)](https://www.assemblyai.com/blog/meeting-intelligence-platforms) — MEDIUM confidence; competitive landscape
- [RAG Retrieval Performance Enhancement (DEV Community)](https://dev.to/jamesli/rag-retrieval-performance-enhancement-practices-detailed-explanation-of-hybrid-retrieval-and-self-query-techniques-59ja) — LOW confidence; single source, practical guide
- [How to Build Contextual RAG Pipeline (Luna.ai)](https://withluna.ai/blog/contextual-rag-product-meeting-notes-slack) — MEDIUM confidence; practitioner blog with specifics on segment enrichment
