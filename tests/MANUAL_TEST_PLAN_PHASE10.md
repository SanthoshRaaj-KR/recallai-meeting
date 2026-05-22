# Manual Test Plan — Phase 10 (auto-propose-pipeline-quality-redesign-v2)

> **Scope:** Five manual UAT scenarios that exercise behaviors not covered by
> automated tests. Every scenario maps to a requirement and a manual-only
> verification row in `.planning/phases/10-auto-propose-pipeline-quality-redesign-v2/10-VALIDATION.md`.
>
> **T-10-02 mitigation:** All Confluence URLs, page IDs, space keys, and
> workspace IDs in this document use placeholder tokens
> (`<WORKSPACE_ID>`, `<SPACE_KEY>`, `<PAGE_ID>`, `<CONFLUENCE_BASE_URL>`).
> Never embed real customer page URLs when filling in steps locally.

## §1. ProposalCard renders against real Confluence URLs (PROP-V2-04)

**Preconditions**

- A test Confluence workspace with at least one page that has a section
  heading you can target (e.g., "Setup" on `<CONFLUENCE_BASE_URL>/wiki/spaces/<SPACE_KEY>/pages/<PAGE_ID>`).
- The Phase 10 backend running locally via
  `conda activate ml && uvicorn confluence_logic.jarvis_agentic:app`.
- sync-sage-bot running locally via `cd sync-sage-bot && npm run dev`.
- A recorded meeting session whose transcript proposes one
  `replace` op against the target page.

**Steps**

1. Sign in to sync-sage-bot with a real Supabase account.
2. Trigger the post-meeting proposal pipeline for the recorded session.
3. When the proposal card lands, inspect the card without expanding any
   collapsible section.
4. Click the page-title link and confirm it opens
   `<CONFLUENCE_BASE_URL>/wiki/spaces/<SPACE_KEY>/pages/<PAGE_ID>` in a new tab.

**Expected result**

- The card shows: page title (linked, `target="_blank"`), breadcrumb
  `<SPACE_KEY> › Parent › Page`, change-type pill, location line, plain-English
  `change_summary` (≤120 chars), word-level inline diff, why-line, and
  the three buttons [Accept] [Reject] [Regenerate] — all default-visible
  with no clicks required.

**Fail criteria**

- Any of the nine D-07 default-visible elements is missing or hidden behind
  a collapse/expand control.
- The page link does not resolve to a real Confluence URL, or opens in
  the same tab.
- The breadcrumb shows a stale space key.

---

## §2. Visual quality of long replacement diff (PROP-V2-04)

**Preconditions**

- A test Confluence page with a paragraph longer than 500 characters in
  a single section.
- A transcript that proposes a replace of one phrase inside that long
  paragraph.

**Steps**

1. Trigger the post-meeting pipeline.
2. When the replace card lands, scroll the diff region.
3. Visually inspect the word-level diff: removed words should appear
   red with strikethrough; added words green with underline.
4. Confirm the diff fits inside the card without horizontal scroll.

**Expected result**

- Diff is readable end-to-end.
- Removed and added words are clearly distinguishable by color and
  decoration.
- No horizontal scroll bar.
- Unchanged words around the edit form recognizable context (at least 3
  unchanged words to either side of the edit).

**Fail criteria**

- The diff is dense to the point of illegibility (>50% changed words
  with no visual grouping).
- Horizontal scroll is required to view the diff.
- Removed/added are visually indistinguishable.

---

## §3. End-to-end real Confluence commit via EditorAgent through dispatcher (PROP-V2-06)

**Preconditions**

- A test Confluence space where the test user has write permission.
- The Phase 10 EditorDispatcher (`confluence_logic/agents/editor_dispatcher.py`)
  is shipped (Wave 1+).
- `editor_agent.py` is byte-identical to its Phase 10 start state
  (`git diff confluence_logic/agents/editor_agent.py` returns empty).

**Steps**

1. Trigger a meeting that produces one card per D-02 instruction shape
   (replace / insert_after / reorder / delete_section / create_section
   / create_page).
2. For each card, click [Accept].
3. Open the target Confluence page in a new tab and visually inspect.

**Expected result**

- Each of the six operations applies correctly:
  - replace — old text gone, new text in place.
  - insert_after — new block inserted directly after the anchor.
  - reorder — moved item at the new index; unmoved items byte-identical
    in source order.
  - delete_section — section heading and its body removed.
  - create_section — new heading and content inserted under the parent.
  - create_page — new page exists under the parent page.

**Fail criteria**

- Any operation results in HTML corruption, duplicate sections, or
  lost data.
- Any sibling content unrelated to the operation is modified.
- The dispatcher imports `editor_agent` directly (static check via
  `grep -L "from confluence_logic.agents.editor_agent" confluence_logic/agents/editor_dispatcher.py`).

---

## §4. Regenerate against page changed mid-session (PROP-V2-05)

**Preconditions**

- A test Confluence page with a section a card targets.
- A live proposal card waiting in the review queue.
- A second browser tab or human collaborator who will edit the same
  page while the card is sitting in the queue.

**Steps**

1. Generate a proposal card; do not Accept yet.
2. In a second tab (or via a collaborator), edit the targeted section
   so the proposal's `old_text` no longer appears on the page.
3. Click [Regenerate from current page] on the original card.
4. Wait for the new card to land.

**Expected result**

- The regenerate request triggers a forced REST fetch (bypassing the
  RAG cache for this page only).
- The new card reflects the post-edit page state.
- The new card passes the full Phase 10 grounding gate.
- If the new card fails the grounding gate, the UI keeps the original
  card and shows the failure reason.

**Fail criteria**

- Regenerate returns a card whose `old_text` still references the
  pre-edit page state.
- Regenerate succeeds even when the grounding gate fails.
- The full graph cache is invalidated (regenerate should affect only
  the one page).

---

## §5. LiveKit voice path unaffected by Phase 10 changes (PROP-V2-06 scope-lock)

**Preconditions**

- A real Recall.ai meeting in progress with the bot joined.
- LiveKit audio device active.
- Phase 9 LiveKit path expected baseline: "Hey Jarvis, summarize the
  meeting" returns a spoken summary within ~3 seconds.

**Steps**

1. Join the meeting room via the bot.
2. Speak the wake phrase "Hey Jarvis".
3. Ask a read-only question, e.g., "Hey Jarvis, summarize the
   discussion so far."
4. Time the response from end-of-question to first audible word.

**Expected result**

- Wake-word loop in `jarvis_agentic.py` triggers as before.
- The summary is spoken back within 3 seconds.
- The Phase 9 ConfluenceQAAgent latency benchmark still holds for
  read-only Confluence queries.

**Fail criteria**

- Wake-word loop does not trigger.
- Response latency exceeds 5 seconds.
- Phase 10 code paths leak into the live-meeting voice loop (any
  unexpected log lines from `page_router`, `grounding_gate`,
  `structure_aware_drafter`, or `editor_dispatcher` during a live
  meeting indicate a scope-lock violation).

---

## Sign-off Checklist

- [ ] §1 ProposalCard against real Confluence URLs passes
- [ ] §2 Long replacement diff visually acceptable
- [ ] §3 End-to-end Confluence commit through dispatcher passes for all six D-02 shapes
- [ ] §4 Regenerate against mid-session edit passes (and fails gracefully when grounding gate rejects)
- [ ] §5 LiveKit voice path latency unchanged from Phase 9 baseline
- [ ] `editor_agent.py` byte-identical (`git diff confluence_logic/agents/editor_agent.py` is empty)
- [ ] No customer Confluence URLs embedded in this document (T-10-02 mitigation)
