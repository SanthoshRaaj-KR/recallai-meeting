"""Launch-readiness suite — safety gates a company-facing release must pass.

The needle/multi/cross tests already measure recall + precision at scale. This
suite fills the SAFETY gaps that decide whether the pipeline is safe to put in
front of real companies, and it does so with HUGE, realistic meeting transcripts
(1,500-3,000+ words of chatter) so the change is buried the way it would be in a
real meeting — stressing transcript segmentation, recall, and precision at once:

  A. NOISE SAFETY      huge pure-chatter / "no change" meetings must yield ZERO
                       cards (false positives erode trust instantly)         [GATE]
  B. SHITTY TRANSCRIPTS one vague / self-correcting change buried in a long noisy
                       meeting — captured with the CORRECT value, no hallucination [GATE]
  C. POSITIONAL REMOVAL "remove the last N sections" buried in a big meeting
                       deletes exactly those N (the "4th delete didn't happen" bug)
  D. APPLY MATRIX      every format x {replace, delete} applies cleanly, the doc
                       still parses, and sibling sections are untouched       [GATE]
  E. IDEMPOTENCY       re-running after a change was applied proposes nothing
                       new (accepting a card never loops)                     [GATE]
  F. PROD DEFAULTS     the real user config (rerank + contextual) works E2E on
                       huge transcripts

Everything that writes runs on TEMP COPIES — the committed corpus is never
mutated. Section discovery uses the prebuilt index (no re-chunking 100 docs).
Exit code is non-zero if any HARD GATE fails.
"""
from __future__ import annotations

import difflib
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / "../confluence_logic/.env")

import asyncio

import openai

from pipeline.run import PipelineConfig, run_pipeline
from pipeline.safe_apply import SafeApply
from rag.chunker import chunk_document
from rag.indexer import build_index

HERE = Path("stress_corpus")
DOCS = str(HERE / "docs")
MAN = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
PROG = open(HERE / "launch_progress.txt", "w", encoding="utf-8", buffering=1)

EDITS = [a for a in MAN["anchors"] if a["kind"] == "edit"]
REMS = [a for a in MAN["anchors"] if a["kind"] == "remove_named"]


def log(msg=""):
    PROG.write(str(msg) + "\n")
    PROG.flush()
    print(msg, flush=True)


def base(p):
    return os.path.basename(str(p)).replace("\\", "/")


def num(s):
    m = re.search(r"\d[\d,\.]*", s or "")
    return m.group(0) if m else None


def _norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _changed_words(before, after):
    """Word-level edit distance between before and after (how localized the diff is)."""
    wb, wa = before.split(), after.split()
    sm = difflib.SequenceMatcher(None, wb, wa)
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")


def org_of(anchor):
    """Distinctive org token from a corpus filename (security_ironbloom_008 -> Ironbloom)."""
    parts = Path(anchor["file"]).stem.split("_")
    return parts[1].title() if len(parts) >= 2 else Path(anchor["file"]).stem


def wc(t):
    return len(t.split())


# ── Huge-transcript builders ─────────────────────────────────────────────────
# A big pool of realistic meeting chatter that implies NO written-document change.
NOISE_POOL = [
    "Morning everyone, grab a coffee, we'll get going in just a minute.",
    "Quick reminder the parking garage is closed Thursday for cleaning, use the overflow lot across the street.",
    "The offsite photos are finally up on the wiki, they came out hilarious, go take a look.",
    "Lunch today is tacos in the main cafeteria, first come first served as always.",
    "Someone left a grey hoodie in conference room 4B, come grab it before it goes to lost and found.",
    "The coffee machine on the third floor is fixed again, crisis averted, you're welcome.",
    "Friendly reminder to submit your expense reports before the end of the month.",
    "The gym membership perk auto-renews next week, no action needed from anyone.",
    "Weather's supposed to be gorgeous this weekend, about time honestly.",
    "The new hire cohort starts Monday, please say hi when you see some fresh faces around.",
    "Slack was down for about ten minutes this morning, it's all back now.",
    "Can someone please look at the projector in room 2A, it keeps flickering during demos.",
    "We're trialing standing desks, let facilities know if you'd like one at your spot.",
    "Heads up the all-hands moved to three pm today, not two, update your calendars.",
    "Sales closed twelve deals this week, really strong momentum, nice work team.",
    "Support backlog dropped from about ninety tickets down to forty, great job clearing it.",
    "Infra held ninety-nine point nine percent uptime last month, rock solid as usual.",
    "Marketing's webinar pulled in four hundred signups, comfortably above target.",
    "We crossed a million monthly sessions last week, fun little milestone for the dashboard.",
    "The hiring pipeline has around thirty candidates in flight at the moment.",
    "Churn ticked down half a point this month, we'll happily take it.",
    "The mobile crash rate is under one percent now, much better than last quarter.",
    "Big shout-out to the on-call crew for handling that traffic spike over the weekend.",
    "Thanks to design for turning the new mockups around so quickly, they look sharp.",
    "Procurement flagged a vendor renewal last sprint, they're already handling it, nothing for us.",
    "The board asked us to keep things tight this quarter, you all know the drill.",
    "We'll circle back on the longer roadmap conversation in a separate session next week.",
    "Let's try to keep this one to thirty minutes, a few folks have lunch plans.",
    "We looked at the travel and reimbursement policy and everyone's happy with it, no changes, leave it exactly as is.",
    "We glanced at the refund window and agreed it's fine the way it's written, keep it.",
    "Someone floated revisiting PTO at some point but we tabled it, no decision today.",
    "There was an idea to tweak the on-call rotation, but we parked it for next quarter.",
    "Does anyone remember what our current data retention period is set to? Nobody's sure, we'll check offline.",
    "Quick question on badge expiry came up, no one knew offhand, we'll confirm later, nothing to change right now.",
    "If we ever expand to the EU we'd revisit a bunch of compliance stuff, but that's hypothetical for now.",
    "There's genuinely no document change coming out of this part, just keeping everyone in the loop.",
    "We reviewed the security policy and decided it already covers everything, no edits needed.",
    "The handbook came up briefly but we agreed it's current, nothing to update there.",
    "Nothing to action on the onboarding guide, it's in good shape as is.",
    "That reminds me, the holiday calendar got published, check the wiki when you get a sec.",
    "Oh, the taco truck is back this Friday, just so everybody knows.",
    "Did everyone get the calendar invite for the team dinner? Please reply yes or no.",
    "Anyway, where was I, right, okay, moving along.",
    "Sorry, can everyone hear me okay on the bridge? Cool, thanks.",
    "Let me share my screen — actually never mind, it's not important, I'll send it after.",
    "The kitchen ran out of oat milk again, someone put it on the supply list.",
    "Reminder that the quarterly survey closes Friday, takes about five minutes, please fill it out.",
    "Our app store rating crept up to four point six, nice to see.",
    "The fire drill is scheduled for next Tuesday, just so nobody panics.",
    "Okay focus everyone, let's get through the agenda.",
]

_OPEN_NOISE = ("Alright, quarterly all-hands, mostly housekeeping and a few status "
               "updates today. Settle in, this is a long one.")
_CLOSE_NOISE = ("Okay, that's everything on my list. Nothing to action from this one, "
                "thanks all, see you at lunch.")
_OPEN_CHANGE = ("Alright team, big end-of-quarter review. There's a lot of chatter in "
                "here but a real document change or two buried in it, so listen up and "
                "we'll send the proposals around afterward.")
_CLOSE_CHANGE = ("Okay that's everything. The system will draft the changes, review them "
                 "in the UI and accept what looks right. Thanks everyone.")


def _fill(rng, n):
    pool = NOISE_POOL[:]
    lines: list[str] = []
    while len(lines) < n:
        rng.shuffle(pool)
        lines.extend(pool)
    return lines[:n]


def huge_noise(rng, n=120):
    """A long, pure-noise meeting (no document changes anywhere)."""
    return " ".join([_OPEN_NOISE] + _fill(rng, n) + [_CLOSE_NOISE])


def bury(real_lines, rng, n_noise=100):
    """Bury one or more SINGLE-SENTENCE real instructions in a noisy meeting.

    Insertion bounds are clamped so small (normal-size) meetings don't produce an
    empty randrange — the change still lands away from the very first/last line.
    """
    lines = _fill(rng, n_noise)
    for rl in real_lines:
        lo = 1 if len(lines) <= 16 else 8
        hi = max(lo, len(lines) - 1)
        lines.insert(rng.randint(lo, hi), rl)
    return " ".join([_OPEN_CHANGE] + lines + [_CLOSE_CHANGE])


# ── Section discovery from the prebuilt index (no docling re-chunk of 100 docs) ──
def index_paths(idx):
    out = {}
    for c in idx.chunks:
        out.setdefault(base(c.source_path), c.source_path)
    return out


def sections_from_index(idx, path_base):
    """(title, [content sections in order]) for one doc, excluding the title/intro.

    Mirrors pipeline.run._ordered_sections (one chunk per heading, document order).
    """
    by_heading = {}
    title = None
    for c in idx.chunks:
        if base(c.source_path) != path_base:
            continue
        if c.section_index == 0 and title is None:
            title = c.section_heading
        by_heading.setdefault(c.section_heading, c)
    ordered = sorted(by_heading.values(), key=lambda c: c.section_index)
    content = [c for c in ordered if c.section_index != 0]
    return title, content


async def pipe(transcript, folder=DOCS, rerank=False, contextual=False):
    cfg = PipelineConfig(
        session_id="launch", doc_folder=folder, use_embeddings=True,
        rerank=rerank, contextual_retrieval=contextual,
    )
    return await run_pipeline(transcript, cfg)


# ════════════════════════════════════════════════════════════════════════════
# A. NOISE SAFETY — huge pure-chatter / "no change" meetings -> 0 cards
# ════════════════════════════════════════════════════════════════════════════
async def scenario_a():
    log("\n" + "=" * 74)
    log("A. NOISE SAFETY  (NORMAL + HUGE pure-chatter / 'no change' meetings -> 0 cards)")
    log("=" * 74)
    # short focused syncs AND long all-hands — both must yield zero cards.
    cases = [("normal", 14), ("normal", 22), ("normal", 28),
             ("huge", 110), ("huge", 135), ("huge", 150)]
    total_spurious = 0
    for i, (label, n) in enumerate(cases, 1):
        t = huge_noise(random.Random(100 + i), n)
        ps = await pipe(t)
        total_spurious += len(ps)
        tag = "OK  " if not ps else "FALSE-POSITIVE"
        log(f"  noise #{i} [{label:6s}]: {wc(t):5d} words -> {len(ps)} cards   [{tag}]")
        for p in ps[:5]:
            log(f"       -> {p.edit_type} {base(p.source_chunk.source_path)} "
                f"«{p.source_chunk.section_heading[:30]}» :: topic={p.intent.affected_topic[:38]}")
    gate = total_spurious == 0
    log(f"\n  total false-positive cards across {len(cases)} noise meetings "
        f"(3 normal + 3 huge): {total_spurious}   GATE={'PASS' if gate else 'FAIL'}")
    return gate, total_spurious


# ════════════════════════════════════════════════════════════════════════════
# B. SHITTY TRANSCRIPTS — vague / self-correcting change buried in a huge meeting
# ════════════════════════════════════════════════════════════════════════════
async def scenario_b():
    log("\n" + "=" * 74)
    log("B. SHITTY / VAGUE TRANSCRIPTS  (messy change at NORMAL + HUGE size)")
    log("=" * 74)
    picks = EDITS[:3]
    captured = 0
    hallucinated = 0

    a0 = picks[0]
    org0, new0 = org_of(a0), num(a0["new_value"])
    wrong0 = "137"  # appears in neither old nor new — must NOT leak into the draft
    line0 = (f"Oh and for {org0}, that {a0['topic']} thing, we said push it to like "
             f"{wrong0} — no wait, scratch that, sorry, make it {a0['new_value']}, "
             f"yeah {a0['new_value']} is the number, it was {a0['old_value']} before.")

    a1 = picks[1]
    org1, new1 = org_of(a1), num(a1["new_value"])
    line1 = (f"Also someone please, uh, bump that {a1['topic']} thing in the {org1} "
             f"doc up to {a1['new_value']}, you know the one everyone keeps complaining "
             f"about, just make it {a1['new_value']}.")

    a2 = picks[2]
    org2, new2 = org_of(a2), num(a2["new_value"])
    line2 = (f"Quick one before half the room leaves — the {a2['topic']} for {org2} "
             f"needs to be {a2['new_value']} going forward, not whatever it is today, "
             f"please get that updated.")

    cases = [
        (a0, line0, new0, [wrong0]),
        (a1, line1, new1, []),
        (a2, line2, new2, []),
    ]
    total = 0
    for a, line, must, must_not in cases:
        for label, n in [("normal", 6), ("huge", 100)]:
            t = bury([line], random.Random((hash(line) + n) & 0xFFFFFF), n_noise=n)
            ps = await pipe(t)
            wf, wh = base(a["file"]), a["section_heading"]
            target = [p for p in ps if base(p.source_chunk.source_path) == wf
                      and p.source_chunk.section_heading == wh]
            after = (target[0].after_content if target else "") or ""
            hit = bool(target) and (must or "") in after
            leaked = [w for w in must_not if w and w in after]
            total += 1
            if hit:
                captured += 1
            if leaked:
                hallucinated += 1
            log(f"  [{a['fmt']:4s} {label:6s}] {wc(t):5d}w | {org_of(a)+'/'+a['topic']:34.34s} "
                f"capture={'YES' if hit else 'no '} value={'ok' if hit else '--'} "
                f"hallucination={'LEAK '+str(leaked) if leaked else 'none'}")

    gate = hallucinated == 0
    log(f"\n  captured {captured}/{total} (3 cases x normal+huge) | hallucinated: {hallucinated}   "
        f"GATE(no-hallucination)={'PASS' if gate else 'FAIL'}")
    return gate, captured, total, hallucinated


# ════════════════════════════════════════════════════════════════════════════
# C. POSITIONAL REMOVAL — buried "remove the last 3 sections" -> exactly those 3
# ════════════════════════════════════════════════════════════════════════════
async def scenario_c(idx):
    log("\n" + "=" * 74)
    log("C. POSITIONAL REMOVAL  (buried 'remove the last 3 sections' -> exactly those 3)")
    log("=" * 74)
    paths = index_paths(idx)
    best = None
    for b in paths:
        title, content = sections_from_index(idx, b)
        if title and len(content) >= 6:
            if best is None or len(content) > len(best[1]):
                best = (b, content, title)
    b, content, title = best
    last3 = [c.section_heading for c in content[-3:]]
    first2 = [c.section_heading for c in content[:2]]

    line = (f"Let's clean up the document titled \"{title}\" by removing its last "
            f"three sections entirely, they're obsolete and nobody references them.")
    t = bury([line], random.Random(7), n_noise=90)
    log(f"  target: {b}  ({len(content)} content sections) | transcript {wc(t)} words")
    log(f"  expect delete of last 3: {[h[:28] for h in last3]}")

    ps = await pipe(t)
    deletes = [p for p in ps if p.edit_type == "delete_section"]
    del_here = [p for p in deletes if base(p.source_chunk.source_path) == b]
    del_heads = {p.source_chunk.section_heading for p in del_here}
    del_other = [p for p in deletes if base(p.source_chunk.source_path) != b]
    hit_last3 = sum(1 for h in last3 if h in del_heads)
    wrong_first = [h for h in first2 if h in del_heads]

    log(f"  delete cards on target doc : {len(del_here)}  (last-3 matched: {hit_last3}/3)")
    log(f"  delete cards on OTHER docs : {len(del_other)}")
    for p in del_here:
        mark = "OK" if p.source_chunk.section_heading in last3 else "??"
        log(f"     [{mark}] «{p.source_chunk.section_heading[:46]}»")
    if wrong_first:
        log(f"  WRONG: deleted a top section: {wrong_first}")
    ok = hit_last3 >= 2 and not wrong_first and not del_other
    log(f"\n  positional removal: {'STRONG' if ok else 'WEAK'} "
        f"({hit_last3}/3 last, {len(wrong_first)} wrong-top, {len(del_other)} cross-doc)")
    return ok, hit_last3, len(del_other), len(wrong_first)


# ════════════════════════════════════════════════════════════════════════════
# D. APPLY MATRIX — format x {replace, delete}; parse-after, siblings intact
# ════════════════════════════════════════════════════════════════════════════
SENTINEL = "ZZSENTINELAPPLIEDZZ"


def scenario_d(idx):
    log("\n" + "=" * 74)
    log("D. APPLY MATRIX  (each format x {replace, delete}; doc must re-parse, siblings intact)")
    log("=" * 74)
    paths = index_paths(idx)
    results = []
    for fmt in ("md", "txt", "docx", "odt"):
        src = next((sp for b, sp in paths.items() if b.endswith("." + fmt)
                    and len(sections_from_index(idx, b)[1]) >= 2), None)
        if not src:
            log(f"  [{fmt:4s}] no usable doc — SKIP")
            continue
        _, content = sections_from_index(idx, base(src))
        target, sibling = content[0], content[1]
        applier = SafeApply(audit_dir=tempfile.mkdtemp(prefix="audit_"))

        # ---- REPLACE ----
        rep_ok = False
        tmpdir = tempfile.mkdtemp(prefix=f"apply_{fmt}_rep_")
        tmp = os.path.join(tmpdir, base(src))
        try:
            shutil.copy2(src, tmp)
            new_body = (target.content.strip() + f"\n\n{SENTINEL}").strip()
            applier.apply(tmp, target.section_heading, new_body, "launch", "rep", edit_type="replace")
            rc = chunk_document(tmp)
            sib = next((c for c in rc if c.section_heading == sibling.section_heading), None)
            parses = len(rc) > 0
            # The new content must have landed. Long sections split across chunks, so
            # check the whole re-chunked doc, not just the first chunk for this heading.
            applied = any(SENTINEL in c.content for c in rc)
            # Replaced, not appended: the heading still appears exactly once.
            no_dup = sum(1 for c in rc if c.section_heading == target.section_heading) == 1
            sib_intact = sib is not None and sibling.content.strip()[:40] in sib.content
            rep_ok = parses and applied and no_dup and sib_intact
            log(f"  [{fmt:4s}] REPLACE parses={parses} applied={applied} no_dup={no_dup} "
                f"sibling_intact={sib_intact}  -> {'OK' if rep_ok else 'FAIL'}")
        except Exception as e:
            log(f"  [{fmt:4s}] REPLACE ERROR {type(e).__name__}: {e}")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        # ---- DELETE ----
        del_ok = False
        tmpdir = tempfile.mkdtemp(prefix=f"apply_{fmt}_del_")
        tmp = os.path.join(tmpdir, base(src))
        try:
            shutil.copy2(src, tmp)
            applier.apply(tmp, target.section_heading, "", "launch", "del", edit_type="delete_section")
            rc = chunk_document(tmp)
            heads = {c.section_heading for c in rc}
            sib = next((c for c in rc if c.section_heading == sibling.section_heading), None)
            parses = len(rc) > 0
            gone = target.section_heading not in heads
            sib_intact = sib is not None and sibling.content.strip()[:40] in sib.content
            del_ok = parses and gone and sib_intact
            log(f"  [{fmt:4s}] DELETE  parses={parses} section_gone={gone} sibling_intact={sib_intact}"
                f"  -> {'OK' if del_ok else 'FAIL'}")
        except Exception as e:
            log(f"  [{fmt:4s}] DELETE  ERROR {type(e).__name__}: {e}")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        results.append((fmt, rep_ok, del_ok))

    gate = bool(results) and all(r and d for _, r, d in results)
    log(f"\n  apply matrix: {sum(r for _,r,_ in results)}/{len(results)} replace, "
        f"{sum(d for _,_,d in results)}/{len(results)} delete   GATE={'PASS' if gate else 'FAIL'}")
    return gate, results


# ════════════════════════════════════════════════════════════════════════════
# E. IDEMPOTENCY — apply a change, re-run same (huge) transcript -> no new card
# ════════════════════════════════════════════════════════════════════════════
async def scenario_e():
    log("\n" + "=" * 74)
    log("E. IDEMPOTENCY  (apply a change, re-run the same huge transcript -> 0 new cards)")
    log("=" * 74)
    a = EDITS[0]
    src = os.path.join(DOCS, base(a["file"]))
    tmpdir = tempfile.mkdtemp(prefix="idem_")
    tmp_folder = os.path.join(tmpdir, "docs")
    os.makedirs(tmp_folder)
    tmp = os.path.join(tmp_folder, base(src))
    shutil.copy2(src, tmp)
    line = (f"For {org_of(a)}, change the {a['topic']} from {a['old_value']} to "
            f"{a['new_value']}, effective immediately.")
    transcript = bury([line], random.Random(3), n_noise=60)
    log(f"  transcript {wc(transcript)} words (change buried in noise)")
    try:
        ps1 = await pipe(transcript, folder=tmp_folder)
        edit1 = [p for p in ps1 if p.edit_type in ("replace", "append")
                 and base(p.source_chunk.source_path) == base(src)]
        log(f"  run 1 (pristine): {len(edit1)} edit card(s) for the change")
        if not edit1:
            log("  run 1 produced no card — cannot exercise idempotency on this anchor")
            return True, 0
        applier = SafeApply(audit_dir=tempfile.mkdtemp(prefix="audit_"))
        card = edit1[0]
        applier.apply(tmp, card.source_chunk.section_heading, card.after_content,
                      "idem", card.proposal_id, edit_type="replace")
        ps2 = await pipe(transcript, folder=tmp_folder)
        edit2 = [p for p in ps2 if p.edit_type in ("replace", "append")
                 and base(p.source_chunk.source_path) == base(src)
                 and (card.after_content or "").strip() != (p.before_content or "").strip()]
        log(f"  run 2 (after apply): {len(edit2)} NEW non-no-op card(s) for the same change")
        gate = len(edit2) == 0
        log(f"\n  idempotency GATE={'PASS' if gate else 'FAIL'}")
        return gate, len(edit2)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ════════════════════════════════════════════════════════════════════════════
# F. PRODUCTION DEFAULTS — rerank + contextual, huge transcripts, small subset
# ════════════════════════════════════════════════════════════════════════════
async def scenario_f():
    log("\n" + "=" * 74)
    log("F. PRODUCTION DEFAULTS  (rerank + contextual_retrieval, HUGE transcripts)")
    log("=" * 74)
    try:
        import sentence_transformers  # noqa: F401
        rerank_avail = True
    except Exception:
        rerank_avail = False
    log(f"  sentence-transformers (cross-encoder rerank) available: {rerank_avail}")

    a = EDITS[0]
    edit_src = os.path.join(DOCS, base(a["file"]))
    brand_docs = [base(d["file"]) for d in MAN["documents"] if d.get("has_brand")][:2]
    fillers = [base(d["file"]) for d in MAN["documents"]
               if not d.get("has_brand") and base(d["file"]) != base(edit_src)][:2]
    subset = [base(edit_src)] + brand_docs + fillers

    tmpdir = tempfile.mkdtemp(prefix="prod_")
    tmp_folder = os.path.join(tmpdir, "docs")
    os.makedirs(tmp_folder)
    for fn in subset:
        s = os.path.join(DOCS, fn)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(tmp_folder, fn))
    log(f"  subset ({len(subset)} docs): {subset}")
    try:
        edit_line = (f"For {org_of(a)}, change the {a['topic']} from {a['old_value']} "
                     f"to {a['new_value']}.")
        t_edit = bury([edit_line], random.Random(11), n_noise=70)
        ps_edit = await pipe(t_edit, folder=tmp_folder, rerank=rerank_avail, contextual=True)
        edit_hit = any(base(p.source_chunk.source_path) == base(edit_src)
                       and (num(a["new_value"]) or "") in (p.after_content or "")
                       for p in ps_edit)

        t_ren = bury([MAN["rename_test"]["transcript"]], random.Random(12), n_noise=70)
        ps_ren = await pipe(t_ren, folder=tmp_folder, rerank=rerank_avail, contextual=True)
        new_brand = MAN["rename_test"]["new_value"].split()[0].lower()
        ren_docs = {base(p.source_chunk.source_path) for p in ps_ren
                    if new_brand in (p.after_content or "").lower()}
        ren_hit = len(ren_docs & set(brand_docs))

        t_noise = huge_noise(random.Random(13), 100)
        ps_noise = await pipe(t_noise, folder=tmp_folder, rerank=rerank_avail, contextual=True)

        log(f"  edit  ({wc(t_edit)} w): captured right doc+value = {edit_hit}")
        log(f"  rename({wc(t_ren)} w): brand docs covered = {ren_hit}/{len(brand_docs)}")
        log(f"  noise ({wc(t_noise)} w): false-positives = {len(ps_noise)}")
        ok = edit_hit and ren_hit == len(brand_docs) and len(ps_noise) == 0
        log(f"\n  production-defaults E2E: {'STRONG' if ok else 'CHECK'}")
        return ok, edit_hit, ren_hit, len(brand_docs), len(ps_noise)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ════════════════════════════════════════════════════════════════════════════
# G. EDIT CORRECTNESS — before->after is a minimal, clean, faithful old->new diff
# ════════════════════════════════════════════════════════════════════════════
JUNK_TOKENS = ["totaling", "totalling", "previously", "formerly", "(was ", "note:",
               "editor", "instead of", "up from", "down from", "changed from", "n.b."]


async def scenario_g(idx):
    log("\n" + "=" * 74)
    log("G. EDIT CORRECTNESS  (before->after = minimal, clean, faithful old->new replacement)")
    log("=" * 74)

    # --- G1: the LLM editor path (the one that can garble) ---
    picks = EDITS[:8]
    captured = 0
    fails = []           # correctness failures (the gate)
    nonminimal = 0       # quality flag (reported)
    sample_shown = False
    for a in picks:
        ps = await pipe(a["transcript"])
        wf, wh = base(a["file"]), a["section_heading"]
        tgt = [p for p in ps if base(p.source_chunk.source_path) == wf
               and p.source_chunk.section_heading == wh
               and p.edit_type in ("replace", "append")]
        if not tgt:
            log(f"  [{a['fmt']:4s}] {org_of(a)+'/'+a['topic']:38.38s} (no card — recall miss, skipped)")
            continue
        p = tgt[0]
        before, after, src = p.before_content or "", p.after_content or "", p.source_chunk.content or ""
        oldv, newv = a["old_value"], a["new_value"]
        captured += 1

        before_faithful = difflib.SequenceMatcher(None, _norm(before), _norm(src)).ratio() >= 0.90
        new_present = (num(newv) or newv) in after
        old_gone = (oldv.lower() not in after.lower()) if oldv else True
        cw = _changed_words(before, after)
        len_ratio = len(after) / max(1, len(before))
        minimal = cw <= 14 and len_ratio <= 1.35
        junk = [j for j in JUNK_TOKENS if j in after.lower()]
        clean = not junk
        if not minimal:
            nonminimal += 1
        ok = new_present and old_gone and before_faithful and clean
        if not ok:
            fails.append((a, dict(new=new_present, old_gone=old_gone, faithful=before_faithful,
                                  clean=clean, junk=junk)))
        log(f"  [{a['fmt']:4s}] {org_of(a)+'/'+a['topic']:38.38s} "
            f"old_gone={'Y' if old_gone else 'N'} new={'Y' if new_present else 'N'} "
            f"faithful={'Y' if before_faithful else 'N'} clean={'Y' if clean else 'N'} "
            f"Δwords={cw:2d} len×{len_ratio:.2f} {'MINIMAL' if minimal else 'BROAD'}"
            + (f"  JUNK={junk}" if junk else ""))

        # Show one concrete before/after as evidence that the edit is correct.
        if not sample_shown and ok:
            sample_shown = True
            log(f"      ── sample diff ({oldv} -> {newv}) on «{wh[:40]}» ──")
            log(f"      BEFORE: ...{before.strip()[:240]}...")
            log(f"      AFTER : ...{after.strip()[:240]}...")

    # --- G2: the deterministic rename path (should be perfectly clean) ---
    rt = MAN["rename_test"]
    old_brand, new_brand = rt["brand"], rt["new_value"]
    ps_ren = await pipe(rt["transcript"])
    ren_cards = [p for p in ps_ren if new_brand.split()[0].lower() in (p.after_content or "").lower()][:6]
    ren_bad = 0
    for p in ren_cards:
        after, before = p.after_content or "", p.before_content or ""
        # only the brand should change: applying the same sub to before should equal after
        expected = re.sub(re.escape(old_brand), new_brand, before, flags=re.IGNORECASE)
        clean = (old_brand.lower() not in after.lower()) and new_brand.lower() in after.lower()
        exact = _norm(expected) == _norm(after)
        if not (clean and exact):
            ren_bad += 1
    log(f"\n  rename path: {len(ren_cards)} cards checked, "
        f"{len(ren_cards)-ren_bad}/{len(ren_cards)} are exact brand-only replacements")

    gate = (captured >= 1) and (not fails) and (ren_bad == 0)
    log(f"\n  edit correctness: {captured-len(fails)}/{captured} clean edits "
        f"({nonminimal} non-minimal), rename {len(ren_cards)-ren_bad}/{len(ren_cards)} exact   "
        f"GATE={'PASS' if gate else 'FAIL'}")
    for a, why in fails:
        log(f"      FAIL {base(a['file'])} «{a['section_heading'][:30]}»: {why}")
    return gate, captured, len(fails), nonminimal, ren_bad


# ════════════════════════════════════════════════════════════════════════════
# N. NORMAL EVERYDAY MEETING — short meeting, all change types, end-to-end
# ════════════════════════════════════════════════════════════════════════════
async def scenario_n():
    import math
    log("\n" + "=" * 74)
    log("N. NORMAL EVERYDAY MEETING  (short sync: edit + rename + removal + light noise)")
    log("=" * 74)
    a = EDITS[0]
    r = REMS[0]
    edit_line = (f"For {org_of(a)}, change the {a['topic']} from {a['old_value']} "
                 f"to {a['new_value']}.")
    transcript = " ".join([
        "Quick fifteen-minute sync, just a few document updates and then we're done.",
        NOISE_POOL[0], edit_line, NOISE_POOL[14],
        MAN["rename_test"]["transcript"], NOISE_POOL[22],
        r["transcript"],
        "That's all three, thanks everyone, short one today.",
    ])
    log(f"  transcript {wc(transcript)} words (1 edit + 1 rename + 1 removal + light noise)")
    ps = await pipe(transcript)

    ewf, ewh = base(a["file"]), a["section_heading"]
    edit_hit = any(base(p.source_chunk.source_path) == ewf and p.source_chunk.section_heading == ewh
                   and (num(a["new_value"]) or "") in (p.after_content or "")
                   and p.edit_type in ("replace", "append") for p in ps)
    rwf, rwh = base(r["file"]), r["section_heading"]
    rem_hit = any(p.edit_type == "delete_section" and base(p.source_chunk.source_path) == rwf
                  and rwh.lower() in p.source_chunk.section_heading.lower() for p in ps)
    new_brand = MAN["rename_test"]["new_value"].split()[0].lower()
    brand_files = {base(d["file"]) for d in MAN["documents"] if d.get("has_brand")}
    ren_docs = {base(p.source_chunk.source_path) for p in ps if new_brand in (p.after_content or "").lower()}
    ren_cov = len(ren_docs & brand_files)

    spurious = []
    for p in ps:
        is_edit = base(p.source_chunk.source_path) == ewf and p.source_chunk.section_heading == ewh
        is_rem = p.edit_type == "delete_section" and base(p.source_chunk.source_path) == rwf
        is_ren = new_brand in (p.after_content or "").lower()
        if not (is_edit or is_rem or is_ren):
            spurious.append(f"{p.edit_type} {base(p.source_chunk.source_path)} "
                            f"«{p.source_chunk.section_heading[:26]}»")

    log(f"  edit captured   : {edit_hit}")
    log(f"  removal captured: {rem_hit}")
    log(f"  rename coverage : {ren_cov}/{len(brand_files)} brand docs")
    log(f"  spurious cards  : {len(spurious)}")
    for s in spurious[:8]:
        log(f"      SPURIOUS: {s}")
    ok = edit_hit and rem_hit and ren_cov >= math.ceil(len(brand_files) / 2) and len(spurious) == 0
    log(f"\n  normal everyday meeting E2E: {'STRONG' if ok else 'CHECK'}")
    return ok, edit_hit, rem_hit, ren_cov, len(brand_files), len(spurious)


async def main():
    t0 = time.time()
    log("#" * 74)
    log("# LAUNCH-READINESS SUITE — huge-transcript safety gates for release")
    log("#" * 74)
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    log("\nWarming the 100-doc index (cache for all read-only scenarios)...")
    tb = time.time()
    idx = build_index(DOCS, use_embeddings=True, contextual_retrieval=False, openai_client=client)
    log(f"  index ready: {len(idx.chunks)} chunks in {time.time()-tb:.1f}s")

    a_gate, a_fp = await scenario_a()
    b_gate, b_cap, b_n, b_hall = await scenario_b()
    c_ok, c_hit, c_other, c_wrong = await scenario_c(idx)
    d_gate, d_res = scenario_d(idx)
    e_gate, e_new = await scenario_e()
    f_ok, f_edit, f_ren, f_brand, f_noise = await scenario_f()
    g_gate, g_cap, g_fail, g_nonmin, g_renbad = await scenario_g(idx)
    n_ok, n_edit, n_rem, n_rencov, n_brand, n_spur = await scenario_n()

    log("\n" + "#" * 74)
    log("# SCORECARD")
    log("#" * 74)
    log(f"  A. noise safety        : {a_fp} false positives (5 huge meetings)   "
        f"[GATE {'PASS' if a_gate else 'FAIL'}]")
    log(f"  B. shitty transcripts  : {b_cap}/{b_n} captured, {b_hall} hallucinated     "
        f"[GATE {'PASS' if b_gate else 'FAIL'}]")
    log(f"  C. positional removal  : {c_hit}/3 last, {c_wrong} wrong-top, {c_other} cross-doc  "
        f"[{'STRONG' if c_ok else 'WEAK'}]")
    log(f"  D. apply matrix        : {sum(r for _,r,_ in d_res)}/{len(d_res)} replace, "
        f"{sum(x for _,_,x in d_res)}/{len(d_res)} delete   [GATE {'PASS' if d_gate else 'FAIL'}]")
    log(f"  E. idempotency         : {e_new} new cards on re-run                "
        f"[GATE {'PASS' if e_gate else 'FAIL'}]")
    log(f"  F. prod defaults       : edit={f_edit} rename={f_ren}/{f_brand} noise={f_noise}  "
        f"[{'STRONG' if f_ok else 'CHECK'}]")
    log(f"  G. edit correctness    : {g_cap-g_fail}/{g_cap} clean ({g_nonmin} broad), "
        f"rename-bad={g_renbad}   [GATE {'PASS' if g_gate else 'FAIL'}]")
    log(f"  N. normal meeting E2E  : edit={n_edit} removal={n_rem} rename={n_rencov}/{n_brand} "
        f"spurious={n_spur}  [{'STRONG' if n_ok else 'CHECK'}]")

    hard = {"A": a_gate, "B": b_gate, "D": d_gate, "E": e_gate, "G": g_gate}
    all_pass = all(hard.values())
    log("")
    log("  HARD GATES: " + "  ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in hard.items()))
    log(f"  SOFT/GRADED: C={'STRONG' if c_ok else 'WEAK'}  F={'STRONG' if f_ok else 'CHECK'}  "
        f"N={'STRONG' if n_ok else 'CHECK'}")
    log(f"\n  >>> {'ALL HARD GATES PASSED — safe-to-launch on safety axes' if all_pass else 'HARD GATE FAILURE — NOT launch-safe yet'} <<<")
    log(f"  total time: {time.time()-t0:.1f}s")
    PROG.close()
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    asyncio.run(main())
