# Feature Landscape: Post-Meeting Confluence Change Proposal Review

**Domain:** AI-generated knowledge-base change review system
**Project:** Jarvis — Meeting Intelligence Platform (sync-sage-bot UI)
**Researched:** 2026-05-11
**Overall confidence:** HIGH (grounded in codebase analysis + established patterns from GitHub PR review, Google Docs suggesting mode, diff viewer tools, and async pipeline UX)

---

## Table Stakes

Features users expect when reviewing AI-generated document changes. Missing any of these makes the system feel broken or untrustworthy, regardless of pipeline quality.

| Feature | Why Expected | Complexity | Current State |
|---------|--------------|------------|---------------|
| Before/after side-by-side diff | Without "what was there before", users cannot evaluate the change at all | Low–Med | Partially present: `before_content` + `after_content` rendered as two colored `<pre>` blocks. Not a true word-level diff — all-or-nothing color. |
| Change type badge | Users need to instantly know whether a card will create, edit, delete, or rename | Low | Present: `change_type` badge with colour per type |
| Target page + section display | Users must know exactly where in Confluence the change lands | Low | Present: `page_title` + `section_heading` shown on card |
| Rationale / why this change | Without a reason, "accept" is a leap of faith | Low | Present: `rationale` field rendered as small text under page title |
| Per-card accept/reject | Granularity to accept good proposals and reject bad ones from the same run | Low | Partially present: checkbox per card + bulk "Approve N" button. No explicit per-card reject action — unchecking is the implicit reject. |
| Pipeline progress feedback | Generation can take up to 20 minutes — silence = user thinks it broke | Med | Present as spinner inside "Generate proposals" button while `proposingChanges` is true. No step-level progress beyond the spinner. |
| Empty state with clear CTA | If no changes exist, user must know what to do next | Low | Present: dashed-border empty state with text prompt |
| Post-apply status per card | After clicking "Approve", each card needs to show applied/failed — not a global toast | Low | Present: `result` badge per card ("Applied" / "Failed") + error text on failure |
| Idempotent re-generation | User should be able to click "Generate" again without manually clearing stale proposals | Low–Med | Present: `_replace_agent_generated_changes` removes prior agent proposals before inserting new ones |
| Error surfacing on apply failure | If a card fails to apply, the user needs the error, not just a red badge | Low | Present: `result.error` text displayed under failed card |

---

## Differentiators

Features not universally expected but that meaningfully increase trust, speed of review, and willingness to approve.

| Feature | Value Proposition | Complexity | Priority |
|---------|-------------------|------------|----------|
| Transcript evidence snippets per card | The single highest-trust signal: show the exact quote(s) that caused this proposal. Maps the AI claim back to human speech. Borrowed from code review — GitHub shows the diff hunk that triggered a comment. | Med | Critical |
| Confidence score with visual encoding | A numeric or band score (e.g., HIGH / MEDIUM / LOW) lets users fast-track high-confidence edits and scrutinise low-confidence ones. Must be rendered visually (colour band, not just a number) to be scannable. | Med | High |
| Risk level badge | Distinct from confidence: risk reflects blast radius (deleting a root section = HIGH risk, adding a bullet = LOW risk). Lets users pre-filter by "show me risky changes first". | Med | High |
| Inline "Edit before applying" | Users who want 80% of a proposal but disagree with 20% of the wording need to edit the `after_content` before approving. Without this, the only options are accept-as-is or reject, which reduces acceptance rate. | Med–High | High |
| Source fact linking | Each change card lists the structured facts it was derived from (e.g., "Derived from Decision: 'We are moving to Postgres'"). Gives users a cross-reference between the proposal card and the meeting summary section above. | Med | Medium |
| Grouping by target page | If 5 proposals touch the same page, rendering them as a collapsible page group rather than 5 flat cards makes it easier to mentally commit "approve all changes to this page". | Low–Med | Medium |
| Bulk accept/reject with filter | "Accept all HIGH confidence" or "Accept all LOW risk" reduces review burden for trust-established users. Should not be the default — first-time users need per-card review to calibrate trust. | Med | Medium |
| Card dismissal (soft reject) | A soft "Not now" dismiss — separate from hard reject — signals "maybe later" rather than "definitely wrong". This preserves rejected proposals so users can reconsider later in the session without re-running the pipeline. | Low | Medium |
| "Why not X?" explainability | A secondary expandable section on each card showing what the agent considered but did not include. Borrowed from code review tools' "rule not triggered" reasoning. Requires backend support. | High | Low |
| Diff granularity selector | Toggle between full-section diff and sentence-level diff. Full-section diff is default; sentence diff helps for large documents. Requires word-diff computation client-side (e.g., `diff-match-patch`). | Med | Low |
| Page preview link | A direct link to open the current Confluence page in a new tab. Lets users verify the "before" state against live Confluence without trusting the cached `before_content`. | Low | Low |

---

## Anti-Features

Patterns to explicitly avoid. Several appear in AI change-proposal products and consistently reduce trust or increase error rate.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Auto-approve on high confidence | Even 95%-confidence AI changes must be human-approved. The system's explicit safety requirement is "no Confluence edits without user approval". Auto-approve erodes the guarantee and creates surprise edits users didn't see. | Keep the approve button as a mandatory gate. Surface confidence visually to speed the user's decision, not to bypass it. |
| Bulk "Accept all" as the primary CTA | A prominent "Accept all N changes" button trains users to rubber-stamp the AI. The first-run experience should encourage per-card review to build calibration. | Make "Approve N selected" the primary action. Bulk-by-filter can be an advanced option, not the default. |
| Confidence number without context | Showing "0.87" without anchoring to a scale or meaning causes users to either ignore it or over-trust it. A raw decimal is not meaningful without comparison. | Use labelled bands: HIGH / MEDIUM / LOW with colour coding, plus a tooltip explaining what the band means. |
| Generating proposals silently in the background | If the pipeline starts automatically and surfaces cards mid-reading, it's disorienting. | The "Generate Confluence Changes" button must be an explicit user-initiated trigger, not an automatic post-meeting hook. |
| Hiding the before state | Showing only `after_content` (the "new" state) makes it impossible to evaluate whether the change is an improvement. Some tools show only the proposed output. | Always show before/after side-by-side for edit-type changes. For create-type, show a clear "new page" indicator since there is no before. |
| Treating `delete` changes the same visually as `edit` | Deletion is irreversible and high-risk. Treating it the same as an edit visually desensitises users. | Give delete-type cards a distinct visual treatment: red border, explicit warning text ("This will remove content permanently"), and require an extra confirmation step before applying. |
| One global progress bar for the whole pipeline | A single "Generating..." bar gives no information about where in the pipeline the system is. Users don't know if it's stuck at RAG retrieval, agent drafting, or verification. | Show step-level progress: "Extracting facts from transcript" → "Retrieving relevant Confluence pages" → "Drafting proposals" → "Verifying against evidence" → "Ready". Even rough stage names dramatically reduce abandon rate on long-running tasks. |
| Mixing pipeline errors with proposal cards | Surfacing a 500-error or timeout as an empty card list with no explanation trains users to dismiss errors as "no proposals found". | Distinguish clearly between "no proposals found" (pipeline ran, nothing applicable) and "pipeline failed" (show the specific stage and a retry button). |
| Infinite loading with no timeout feedback | If the pipeline takes >20 minutes, the UI must tell the user what is happening or offer them an escape. | Add a stale-pipeline message after N minutes: "This is taking longer than usual — the pipeline is still running. You can leave this page and return." |
| Editable proposals that silently change backend state | If the user edits `after_content` in the card and those edits are not persisted to the backend, approving will still apply the original unedited version. | Either persist edits to the backend before applying, or make the edit flow explicit (a "Save edit" step that updates the stored proposal). |

---

## Proposal Card Design Patterns

### What metadata makes accept/reject decisions easy

Drawn from code review tool (GitHub PRs, Gerrit, GitLab MR) and document review tool (Google Docs suggesting mode, tracked changes in Word) patterns applied to AI-generated changes.

**The core decision a reviewer makes on each card:**
"Is this change factually grounded in the meeting and safe to apply to the document?"

To answer that question confidently, the card must expose seven things:

1. **Target location** — `page_title` + `section_heading`. The reviewer needs to know the blast radius before reading the content. If the target is unclear, the card cannot be approved.

2. **Change type with semantics** — `change_type` (create / edit / delete / title). Each type has a different risk profile. Create is generally safe (additive). Edit is moderate (overwrites existing content). Delete and title-rename are high-risk (destructive or disorienting for other page readers).

3. **Before/after diff** — Side-by-side for edits. For creates, the "after" pane shows the proposed page content with a "New page" label replacing the empty left pane. For deletes, the "before" pane is highlighted in red with strikethrough and the "after" pane is blank or shows a tombstone.

4. **Transcript evidence** — The verbatim quote(s) from the meeting that justify the change. This is the single most important trust-building element. Without it, the user is approving on faith. With it, the user is validating an AI interpretation. Format: a blockquote with speaker name and timestamp offset (already available from `transcript_highlights`).

5. **Rationale** — A sentence explaining why the AI believes this change is needed. Different from evidence: rationale is the AI's reasoning; evidence is the raw meeting content. Both are needed.

6. **Confidence + Risk** — Two independent signals on the card header. Confidence = how certain the AI is that the change is correct. Risk = how much damage an incorrect apply would cause. A high-confidence + high-risk card (e.g., "definitely deletes this section") still deserves careful review; a low-confidence + low-risk card (e.g., "probably adds a bullet point") can be approved quickly.

7. **Status indicator** — Pending / Accepted / Rejected / Applying / Applied / Failed. The status must be visible at all times. During execution, show per-card progress, not just a global spinner.

### Current gap analysis

The current `ChangeItem` type in `types.ts` is missing `confidence_score`, `risk_level`, and `transcript_evidence`. These are required for users to make informed accept/reject decisions and are the most important additions to the card schema.

The `ProposedChangesAgent` system prompt currently generates `change_type`, `page_id`, `page_title`, `section_heading`, `before_content`, `after_content`, `rationale` — but not `confidence_score`, `risk_level`, or `transcript_evidence`. These fields must be added to the agent output schema.

---

## Granularity: Per-Card vs. Section-Level vs. Page-Level

**Recommendation: Per-card is correct as the primary granularity. Add grouping, not coarser granularity.**

Rationale:

- Per-card accept/reject is the right default because the AI may correctly identify that page X needs updating but wrongly draft the exact change. The user should be able to accept the intent and edit the content, or reject one card on a page while accepting another.
- Section-level accept/reject (approving all changes within a section at once) is a useful shortcut but should appear as an affordance within a page group, not replace per-card control.
- Page-level bulk-approve is appropriate only once the user has reviewed all cards for that page and wants a convenience action. It should be a secondary affordance.
- Never offer "approve all across all pages" as a top-level action — this bypasses the review entirely.

The current implementation (checkbox per card + "Approve N selected" button) is correct. The missing piece is grouping cards by target page so users can scope their attention.

---

## UX Patterns for Long-Running Async Pipelines

The proposal pipeline takes up to 20 minutes. This duration demands specific UX treatment that differs from typical sub-5-second async calls.

### Step-level progress (most important)

Replace the single spinner with a named-step progress indicator. The pipeline has identifiable stages:

```
Stage 1: Extracting facts from transcript        [done]
Stage 2: Retrieving relevant Confluence pages    [active]
Stage 3: Drafting proposals (agent)              [waiting]
Stage 4: Verifying against transcript evidence   [waiting]
Stage 5: Ready for review                        [waiting]
```

The backend should emit stage events over a WebSocket or SSE stream. The UI renders each stage as a progress row. Completed stages are checked. The active stage has an animated indicator. Future stages are greyed out.

The existing `jarvis_agentic.py` FastAPI app has WebSocket infrastructure — this is the right channel for pipeline progress events.

### Abandon-and-return safety

Because 20 minutes is long enough for users to close the tab, the pipeline result must persist to Supabase so the user can return to the summary page and find the proposals already loaded. This is partially implemented (proposals are stored in `pending_changes` within the history snapshot), but the UI should explicitly tell users "You can close this tab — your proposals will be waiting when you return."

### Retry without re-running

If the pipeline fails at stage 4 (verifier) after completing stages 1–3, the user should be able to retry from the failed stage, not restart from scratch. This requires the backend to checkpoint intermediate results. It's a medium-complexity addition but dramatically reduces user frustration on pipeline failures.

### Non-blocking: let user read summary while pipeline runs

The proposal pipeline should run in the background while the user reads the executive summary, action items, and minutes. This is already architecturally supported (the "Propose updates" button is separate from the summary loading). The UX should make this explicit: "Proposals are generating below — read your summary while you wait."

### Timeout and stale-session handling

Add a stale-pipeline UI state triggered after a configurable timeout (e.g., 25 minutes). The stale state should:
- Show how long the pipeline has been running
- Offer a "Retry" button that calls the propose endpoint again
- Not clear existing proposals if any were generated before the stall

### Toast vs. inline feedback

- **Toasts** are correct for transient completion events ("5 proposals generated", "3 changes applied").
- **Inline card state** is correct for persistent per-card status (applied / failed badge on the card itself).
- Do not use toasts for per-card apply results — toasts disappear and the user loses the information. The current implementation correctly shows the badge on the card.
- Do not show a toast for pipeline stage transitions — they are frequent and low-signal. Show them only in the progress indicator.

---

## MVP Recommendation

Given the current state of the codebase (proposal cards exist, before/after diff renders, accept flow works end-to-end), the MVP additions that unlock the most trust and usability:

**Must-have for MVP:**
1. **Transcript evidence snippets** on each card — highest-trust signal, agents already have the transcript, just needs to be included in output schema
2. **Step-level pipeline progress** — 20-minute waits are only tolerable with stage visibility
3. **Confidence + risk fields** in the card schema — backend agent must generate them; UI renders as coloured bands on card header
4. **Explicit per-card reject action** — current UX has no reject affordance; unchecking a checkbox is ambiguous (user might think they'll check it later)
5. **Delete-change safety treatment** — delete cards need distinct visual treatment and a confirmation step separate from edit/create cards

**Defer to later phase:**
- Inline edit before applying — valuable but requires backend write-path for edited proposals
- Page-level grouping — UX polish, not blocking
- Abandon-and-return explicit messaging — partially works already via Supabase persistence
- Diff granularity toggle — low impact for MVP card counts (typically 3–8 cards per meeting)

---

## Sources and Confidence Assessment

| Claim | Confidence | Source |
|-------|------------|--------|
| Before/after diff is table stakes | HIGH | Universal pattern across GitHub PR diffs, Google Docs track changes, Word review mode |
| Transcript evidence is the highest-trust signal | HIGH | Directly analogous to "PR comment with diff hunk context" in GitHub — the citation is what makes the claim checkable |
| Per-card granularity is correct | HIGH | Established pattern from code review: you can approve individual commits/files without approving the whole PR |
| Step-level progress for 20-min tasks | HIGH | Standard practice in CI/CD pipelines (GitHub Actions step view), data pipeline tools |
| Confidence bands vs raw score | HIGH | Widely documented in human-AI interaction: ordinal categories (HIGH/MED/LOW) outperform raw decimals for non-expert decision-making |
| Auto-approve is an anti-pattern | HIGH | Consistent finding across AI writing assistant products where bypass of human review leads to trust collapse after first error |
| Delete type needs separate visual treatment | HIGH | UI safety principle: irreversible actions require distinct affordances; established in iOS destructive action sheets, confirmation dialogs |
| Page-level grouping is a differentiator | MEDIUM | Pattern observed in multi-file PR review tools (Gerrit, Review Board) but not universal |
| "Edit before applying" increases acceptance rate | MEDIUM | Inferred from document editing UX (track changes "accept with modifications") — no direct data for this domain |
| Confidence + risk as two independent signals | MEDIUM | Reasonable product design decision; risk and confidence are genuinely orthogonal but this exact pairing is not established practice in directly comparable products |
