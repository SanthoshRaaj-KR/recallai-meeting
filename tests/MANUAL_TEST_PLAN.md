# Phase 8 — Manual UAT Plan

These four scenarios exercise the real Recall.ai → Confluence wire that automated
tests can't cover. Run them with a real Confluence space and a real meeting.

The in-meeting "Hey Jarvis" voice path is OUT OF SCOPE for Phase 8 and is not exercised here.

---

## Scenario 1 — Auto-generated create with verbatim bullets

**Setup:** Confluence space with no page titled "bot pros and cons".

**Steps:**
1. Start a meeting; bot joins.
2. Have a participant say: "We should document our bot platform tradeoffs. Create a pros and cons page for bots with these four points: A is fast, B is cheap, C is expensive, D is slow."
3. End the meeting.
4. Click "Generate Confluence Changes" on the meeting summary page.
5. Wait for the pipeline to complete (~30-60s).

**Expected:**
- Exactly one card appears, with the change-type badge "Create" and page title "bot pros and cons" (or close).
- The card's change_summary line reads something like "Create new page 'bot pros and cons'".
- The default-visible After block contains the literal text "A is fast", "B is cheap", "C is expensive", "D is slow" — no paraphrase, no extra invented bullets.
- Click Accept → page is created in Confluence with the four bullets intact.

---

## Scenario 2 — Multi-topic meeting

**Setup:** Confluence space with the pages: "Dev Runbook" (containing "Python 2"), "Payments On-Call Rotation" (containing "Marcus").

**Steps:**
1. Start a meeting.
2. Discuss three things in sequence:
   a. "We're upgrading Python from 2 to 3 in the dev environment."
   b. "Priya is taking over payments on-call from Marcus."
   c. "We need a new page for the Inventory Service we just launched."
3. End the meeting → click Generate.

**Expected:**
- Three cards.
- Card A: change_type=edit, page_title="Dev Runbook", change_summary mentions "Python 2 → Python 3", diff shows Python 2 → Python 3.
- Card B: change_type=edit, page_title="Payments On-Call Rotation", change_summary mentions "Marcus → Priya".
- Card C: change_type=create, page_title="Inventory Service" (or close), change_summary "Create new page 'Inventory Service'".
- All three cards have non-empty rationale + visible before/after where applicable.

---

## Scenario 3 — Stale-page Accept failure + Regenerate recovery

**Setup:** Run Scenario 2 to get Card A (the Python edit).

**Steps:**
1. BEFORE clicking Accept on Card A, open the Dev Runbook page directly in Confluence and delete the line "We use Python 2".
2. Return to the review UI and click Accept on Card A.
3. Observe the toast.
4. Click the "Regenerate from current page" button that appears on the card.
5. Wait ~3-5s.
6. Re-click Accept.

**Expected:**
- Step 3: toast shows the real backend message — something like "Text to replace was not found on 'Dev Runbook'. The page may have been edited since the proposal was generated." (NOT a generic "Failed to apply change.")
- Step 4: card shows a "Regenerate from current page" button.
- Step 5: card refreshes; its before/after may now show different content reflecting the current page state (or downgrade to append with a `[REGENERATED-FALLBACK]` note).
- Step 6: Accept now succeeds; the Dev Runbook page is updated.

---

## Scenario 4 — Card scannability

**Setup:** Run Scenario 2 so the review page has 3 cards across 2 pages.

**Steps:**
1. Open the review page in a fresh browser tab.
2. Look at each card WITHOUT clicking anything.
3. Try to answer for each: (a) which Confluence page is this for? (b) what type of change? (c) what is being changed in one line?

**Expected:**
- All three answers come from on-card text in <5 seconds per card.
- No "View full changes" click required to see what's changing.
- The page title is visible on every card (not only in the group header).
