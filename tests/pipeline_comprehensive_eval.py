"""
Comprehensive Jarvis Pipeline Evaluation
==========================================
Goals:
  1. Evaluate answer quality across all intent types with both vague and detailed questions.
  2. Measure time-to-first-word (TTFW): elapsed between question arrival and first audio byte
     being ready for playback (LLM latency + TTS synthesis of sentence[0]).
  3. Suggest exact code-level changes based on results.

Two modes:
  --quality-only   Mock TTS (fast, for CI). Measures LLM quality and classification.
  --full           Actually call synthesize_speech() for real TTS latency data.
                   Requires OPENAI_API_KEY in env. Takes ~3-5 minutes.

Usage:
    python -m tests.pipeline_comprehensive_eval                # full mode
    python -m tests.pipeline_comprehensive_eval --quality-only
"""
import argparse
import asyncio
import os
import sys
import time
import textwrap
from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import AsyncMock, patch

sys.path.insert(0, ".")

from confluence_logic import jarvis_agentic as ja
from confluence_logic.classifier import classify_intent


# ---------------------------------------------------------------------------
# Synthetic 30-min meeting transcript
# A product-team sprint review at an Indian SaaS startup — realistic STT
# artefacts (filler words, split verbs, name misspellings, abbreviations).
# 6 speakers, timestamps up to ~1920s (32 minutes).
# ---------------------------------------------------------------------------

TRANSCRIPT = [
    # 0:00 — Introductions / setup
    {"participant": "Arjun Verma", "text": "Okay I think everyone's here now. Let me just wait one more sec for Deepa.", "timestamp": 0},
    {"participant": "Sneha Rao", "text": "Yeah she said she'll be a couple minutes late.", "timestamp": 8},
    {"participant": "Vikram Nair", "text": "While we wait — did anyone look at the Sentry alerts from last night? We had a spike around two AM.", "timestamp": 14},
    {"participant": "Rohan Gupta", "text": "I saw that. It was the payment retry logic hitting a race condition. I already pushed a fix to staging this morning.", "timestamp": 22},
    {"participant": "Arjun Verma", "text": "Good catch Rohan. Let's make sure that goes out with today's deploy.", "timestamp": 35},
    {"participant": "Deepa Krishnan", "text": "Hey sorry I'm late, my other call ran over.", "timestamp": 42},
    {"participant": "Arjun Verma", "text": "No worries Deepa. Okay let's get started. So today is our Sprint 14 review plus some roadmap planning for Q3. I'll start with the product wins, then we go team by team.", "timestamp": 48},

    # 0:58 — Product wins
    {"participant": "Arjun Verma", "text": "First the wins. We shipped the bulk invoice download feature last week and it's already being used by about 30 percent of our enterprise customers. Support tickets for invoice-related issues dropped by 40 percent. That's a big deal.", "timestamp": 58},
    {"participant": "Sneha Rao", "text": "Yeah the customer feedback has been really positive. One of our big accounts — it's that logistics company, what's it called — they said it saved their finance team like 3 hours every week.", "timestamp": 90},
    {"participant": "Arjun Verma", "text": "Exactly. Okay the second win is the search performance improvement. Vikram you want to speak to that?", "timestamp": 112},
    {"participant": "Vikram Nair", "text": "Sure. So we moved from full-table-scan to indexed queries on the product catalogue. P99 search latency dropped from 1.2 seconds to about 180 milliseconds. We also added a Redis cache layer for the top 500 queries and that's handling about 60 percent of search traffic now.", "timestamp": 118},
    {"participant": "Rohan Gupta", "text": "The cache hit rate is actually closer to 65 percent now, I checked this morning.", "timestamp": 155},
    {"participant": "Vikram Nair", "text": "Even better.", "timestamp": 160},

    # 2:42 — Backend team update
    {"participant": "Arjun Verma", "text": "Okay let's go team by team. Backend first. Rohan, Vikram — beyond the Sentry fix what else is going on?", "timestamp": 162},
    {"participant": "Rohan Gupta", "text": "So the main backend project right now is the multi-tenant data isolation work. We're partitioning the database by organisation ID. About 70 percent done. The tricky part is migrating existing data without downtime — we're planning to use a dual-write strategy where both the old schema and the new schema get written simultaneously for a couple of weeks before we cut over.", "timestamp": 170},
    {"participant": "Sneha Rao", "text": "Is that risky? Dual write sounds complex.", "timestamp": 220},
    {"participant": "Rohan Gupta", "text": "It has some complexity yeah. But it lets us roll back instantly if something goes wrong. The alternative is a big-bang migration and I'd rather not do that on a Saturday night.", "timestamp": 226},
    {"participant": "Deepa Krishnan", "text": "What's the timeline for completing the data isolation?", "timestamp": 248},
    {"participant": "Rohan Gupta", "text": "We're targeting end of this sprint — so two more weeks. The cutover itself will happen the following weekend with a 2-hour maintenance window.", "timestamp": 255},
    {"participant": "Vikram Nair", "text": "I also want to mention we upgraded our Postgres version from 13 to 15 last weekend. Zero downtime thanks to the logical replication approach. Performance is noticeably better, especially for complex joins.", "timestamp": 270},
    {"participant": "Arjun Verma", "text": "Nice. Any blockers on the backend side?", "timestamp": 298},
    {"participant": "Rohan Gupta", "text": "One — we need access to the production database for the migration dry run. I've submitted an access request but it's been pending for four days. Deepa can you help unblock that?", "timestamp": 304},
    {"participant": "Deepa Krishnan", "text": "I'll follow up with the infra team today, that should not take this long.", "timestamp": 325},

    # 5:28 — Frontend team update
    {"participant": "Arjun Verma", "text": "Okay frontend. Sneha, what's the status?", "timestamp": 328},
    {"participant": "Sneha Rao", "text": "So we finished the dashboard redesign in Sprint 13 and now we're in the feedback cycle. We've done 8 user interviews and the main themes are: one, the new navigation is much better, but two, the analytics charts are confusing — specifically the funnel visualization. Users aren't sure what the drop-off numbers mean.", "timestamp": 335},
    {"participant": "Arjun Verma", "text": "Is that a labelling problem or a design problem?", "timestamp": 385},
    {"participant": "Sneha Rao", "text": "Both honestly. The labels are abbreviated and the tooltip copy is too technical. I'm going to work with the design team to simplify it. Should be a small sprint of work.", "timestamp": 392},
    {"participant": "Deepa Krishnan", "text": "I've been hearing this from customers too. The confusion around the funnel is one of our top three support ticket topics.", "timestamp": 415},
    {"participant": "Sneha Rao", "text": "Yeah, and there's something else — we've been seeing a 15 percent drop in dashboard engagement since the redesign. Not sure if it's the learning curve or an actual regression.", "timestamp": 430},
    {"participant": "Arjun Verma", "text": "That's concerning. How long has this been going on?", "timestamp": 460},
    {"participant": "Sneha Rao", "text": "Since about a week after launch. I think we need to run an A slash B test to understand if users who spent more time exploring have better retention.", "timestamp": 466},
    {"participant": "Vikram Nair", "text": "We already have the feature flag infrastructure in place if you need it for the A slash B.", "timestamp": 490},
    {"participant": "Sneha Rao", "text": "Yeah I'll talk to you offline Vikram about how to set that up.", "timestamp": 497},

    # 8:22 — ML / data team
    {"participant": "Arjun Verma", "text": "Okay data and ML. Deepa you're covering that right?", "timestamp": 502},
    {"participant": "Deepa Krishnan", "text": "Yes. So we launched the churn prediction model in beta last week. We're scoring customers weekly — anyone with a predicted churn probability over 70 percent gets flagged to the customer success team. In the first week we flagged 12 customers and the CS team reached out. Three of them have already scheduled check-in calls. Early signal is good.", "timestamp": 508},
    {"participant": "Sneha Rao", "text": "What's the model accuracy like?", "timestamp": 570},
    {"participant": "Deepa Krishnan", "text": "Precision is about 68 percent on the validation set. Recall is 72 percent. It's not perfect but for a first version it's giving us actionable signal. We're planning to retrain with more features next month — specifically adding product usage events which we weren't capturing before.", "timestamp": 576},
    {"participant": "Arjun Verma", "text": "Is 68 percent precision good enough to hand to the CS team without them losing trust in it?", "timestamp": 620},
    {"participant": "Deepa Krishnan", "text": "It's borderline. We're being transparent with the CS team — we told them roughly one in three flags might be a false positive. They're okay with it for now because even a false positive is a reason for a good customer check-in.", "timestamp": 630},
    {"participant": "Rohan Gupta", "text": "Smart framing.", "timestamp": 665},
    {"participant": "Deepa Krishnan", "text": "The other ML project is the recommendation engine for the product catalogue. We ran offline experiments comparing vector similarity search against a more traditional co-occurrence matrix approach. Vector search wins on long-tail product queries — about 18 percent better hit rate for queries with fewer than 10 historical purchases. Co-occurrence is better for popular items — it's cheaper and the lift is comparable.", "timestamp": 668},
    {"participant": "Arjun Verma", "text": "So hybrid again?", "timestamp": 730},
    {"participant": "Deepa Krishnan", "text": "Exactly. Use co-occurrence for items with more than 50 purchases, vector for everything else. Should get us the best of both approaches.", "timestamp": 736},
    {"participant": "Arjun Verma", "text": "Sounds good. What's the deployment plan?", "timestamp": 758},
    {"participant": "Deepa Krishnan", "text": "We're planning a shadow deployment for two weeks — the new recommendations run in parallel but don't serve traffic. Then a gradual rollout: 5 percent for a week, then 25, then 100.", "timestamp": 764},

    # 13:00 — DevOps / infra
    {"participant": "Arjun Verma", "text": "Okay infra. Vikram you're wearing two hats today.", "timestamp": 780},
    {"participant": "Vikram Nair", "text": "Yep. So infra update: we finished the Kubernetes migration for 8 of the 12 services. The remaining four are the legacy Python 2 services — yes, we still have Python 2, don't judge us — and migrating them is more work because the container base images are harder to pin. I've drafted a plan to first upgrade them to Python 3.11 in place, then containerize.", "timestamp": 786},
    {"participant": "Rohan Gupta", "text": "Python 2 in 2026. Bold strategy.", "timestamp": 845},
    {"participant": "Vikram Nair", "text": "I know, I know. We inherited them. The upgrade will take about 3 weeks. One of the services has about 4000 lines of Python 2-specific code.", "timestamp": 850},
    {"participant": "Arjun Verma", "text": "Is there any risk of breaking changes?", "timestamp": 875},
    {"participant": "Vikram Nair", "text": "Yes, specifically around the unicode handling and the print statements. I've already run the automated migration tool — 2to3 — and it caught most of it, but there are about 200 lines that need manual review. We'll have staging tests running throughout.", "timestamp": 882},
    {"participant": "Deepa Krishnan", "text": "The observability side — Vikram did you ever get the distributed tracing set up for the payment flow?", "timestamp": 920},
    {"participant": "Vikram Nair", "text": "Partially. We have open telemetry instrumented on the checkout service and the payment service. The problem is the legacy webhook handler doesn't have it yet. That's the one Rohan's team is working on. Once the deduplication fix is in we can add the tracing there too.", "timestamp": 928},
    {"participant": "Rohan Gupta", "text": "I can add the tracing as part of that PR actually.", "timestamp": 965},
    {"participant": "Vikram Nair", "text": "Perfect, let's bundle it.", "timestamp": 970},

    # 16:15 — Customer success / product feedback
    {"participant": "Arjun Verma", "text": "Alright. Deepa you're also holding the CS hat today since Preethi is on leave. Any significant customer feedback?", "timestamp": 975},
    {"participant": "Deepa Krishnan", "text": "Yes, three themes. First, the top request from enterprise customers is better audit logging — they want to see who changed what and when. GDPR customers especially. Second, several SMB customers are asking for a mobile app. We've been getting this for about a year and the volume is picking up — maybe 30 requests in the last quarter. Third, the onboarding flow — first-time users are still confused about how to connect their data sources. Our activation rate at 14 days is 48 percent and the benchmark for our category is closer to 65 percent.", "timestamp": 982},
    {"participant": "Arjun Verma", "text": "The audit logging one is interesting because it ties into the multi-tenant work Rohan is doing. Can we add that as a dependency?", "timestamp": 1060},
    {"participant": "Rohan Gupta", "text": "Actually yes. The multi-tenant schema already has a created_by and updated_by column. Adding a full audit log is maybe 3 days of additional work on top of what we're already doing.", "timestamp": 1068},
    {"participant": "Arjun Verma", "text": "Let's scope it in. Deepa add it to the Sprint 15 planning board.", "timestamp": 1095},
    {"participant": "Deepa Krishnan", "text": "Done.", "timestamp": 1100},

    # 18:23 — Q3 roadmap discussion
    {"participant": "Arjun Verma", "text": "Okay big picture time. Let's talk about Q3 priorities. I want to get everyone's input. Based on what we've discussed today and what we know about the business, what do you all think should be the top priorities?", "timestamp": 1103},
    {"participant": "Sneha Rao", "text": "From my side it has to be the activation problem. 48 percent at 14 days is holding back expansion. Every percentage point improvement in activation translates directly to revenue.", "timestamp": 1118},
    {"participant": "Deepa Krishnan", "text": "I agree with Sneha. And we have data to work with now — we can see exactly where users drop off in the onboarding funnel. The biggest drop is on the data source connection step. If we can simplify that, even auto-detect common setups, I think we can get to 55 percent.", "timestamp": 1135},
    {"participant": "Rohan Gupta", "text": "From backend the audit logging and the multi-tenant cutover are the two things I'd lock in. After that honestly we should look at reducing our infrastructure cost. We're spending about 2.4 lakh a month on compute and I think we can cut that by 20-25 percent by right-sizing the nodes and using spot instances for batch jobs.", "timestamp": 1165},
    {"participant": "Vikram Nair", "text": "On that note the Kubernetes work should pay dividends there. Horizontal pod autoscaling means we're not running at peak capacity all the time. I'm estimating 30 percent infra cost reduction once we're fully on K8s.", "timestamp": 1210},
    {"participant": "Arjun Verma", "text": "That's significant. 30 percent of 2.4 lakh is 72,000 rupees a month. Or roughly 8.6 lakh a year.", "timestamp": 1240},
    {"participant": "Vikram Nair", "text": "Exactly. And once we're on K8s the deployment velocity goes up too. Right now deploys take about 25 minutes. With automated Kubernetes rolling deploys it should drop to under 10.", "timestamp": 1258},
    {"participant": "Deepa Krishnan", "text": "What about the mobile app? 30 requests in a quarter is not nothing.", "timestamp": 1290},
    {"participant": "Arjun Verma", "text": "It's tempting but I want us to fix the web experience first. Opening a mobile app surface before activation is solved would split our focus. Maybe Q4.", "timestamp": 1298},
    {"participant": "Sneha Rao", "text": "Agreed. Mobile before we nail web onboarding would be a mistake.", "timestamp": 1318},

    # 22:04 — Technical debt discussion
    {"participant": "Arjun Verma", "text": "One topic I want to make sure we cover — technical debt. We've been accumulating it. What are the highest risk items?", "timestamp": 1324},
    {"participant": "Vikram Nair", "text": "The Python 2 services, as I said. If one of those gets a critical vulnerability we're stuck patching unsupported code. That's a security risk.", "timestamp": 1335},
    {"participant": "Rohan Gupta", "text": "The other big one is our test coverage. We're at about 54 percent unit test coverage. Some of the payment code has zero coverage — it was written under deadline pressure 18 months ago. That's the code that had the race condition last night.", "timestamp": 1350},
    {"participant": "Sneha Rao", "text": "Frontend test coverage is better, about 71 percent, but our end-to-end tests are flaky. The Cypress suite fails intermittently and the team doesn't trust it so they skip it sometimes.", "timestamp": 1390},
    {"participant": "Deepa Krishnan", "text": "Can we set a policy that no PR can drop coverage below a threshold?", "timestamp": 1420},
    {"participant": "Rohan Gupta", "text": "We could but it might slow down velocity if we enforce it strictly. Better to set a direction and track the trend than enforce a hard gate right away.", "timestamp": 1428},
    {"participant": "Arjun Verma", "text": "Let's agree on a target — 70 percent backend coverage by end of Q3 — and track it in our engineering metrics dashboard.", "timestamp": 1450},

    # 24:12 — Security and compliance
    {"participant": "Deepa Krishnan", "text": "Quick compliance update. We have an SOC 2 Type 2 audit coming up in October. I've been doing a gap analysis and there are three areas we need to address: one, access reviews haven't been done in 6 months, they should be quarterly. Two, our incident response runbook is outdated. Three, we don't have formal vendor risk assessments for our third-party SaaS tools.", "timestamp": 1452},
    {"participant": "Arjun Verma", "text": "How much runway do we have before we need to have those fixed?", "timestamp": 1510},
    {"participant": "Deepa Krishnan", "text": "We have until August to get things in order — auditors start their review period in September. So we have about 4 months.", "timestamp": 1518},
    {"participant": "Rohan Gupta", "text": "I can take the access review as an action item. I'll run through who has access to prod and revoke anything that shouldn't be there.", "timestamp": 1540},
    {"participant": "Arjun Verma", "text": "Good. Deepa can you own the incident response runbook update?", "timestamp": 1555},
    {"participant": "Deepa Krishnan", "text": "Yes. I'll draft a new version this week and circulate it for review.", "timestamp": 1562},
    {"participant": "Vikram Nair", "text": "I can handle the vendor risk assessments. I already have a template from a previous company.", "timestamp": 1570},

    # 26:18 — Sprint 15 planning
    {"participant": "Arjun Verma", "text": "Okay let's do Sprint 15 scope. I want to make sure we're all aligned before we close.", "timestamp": 1578},
    {"participant": "Rohan Gupta", "text": "Backend: complete multi-tenant data isolation, add audit logging, merge the payment webhook dedup fix with tracing, do the prod database access dry run.", "timestamp": 1585},
    {"participant": "Sneha Rao", "text": "Frontend: fix funnel visualization labels and tooltips, set up the A slash B test framework with Vikram, and start design exploration for the onboarding flow improvement.", "timestamp": 1612},
    {"participant": "Vikram Nair", "text": "Infra: Python 3 migration for the first two legacy services, finalize HPA config for the payment service, complete open telemetry on the webhook handler.", "timestamp": 1638},
    {"participant": "Deepa Krishnan", "text": "Data: retrain churn model with product usage events, deploy recommendation engine to shadow traffic, update the incident response runbook, start vendor risk assessments.", "timestamp": 1660},
    {"participant": "Arjun Verma", "text": "That feels like a lot. Let's pressure-test the scope — Rohan is the audit logging addition going to stretch the backend team?", "timestamp": 1700},
    {"participant": "Rohan Gupta", "text": "It's 3 days of work I said. The multi-tenant work is the sprint anchor — the audit logging rides on top of it. I think it's fine.", "timestamp": 1714},
    {"participant": "Arjun Verma", "text": "Okay. Sneha does the A slash B test framework add risk to the funnel fix?", "timestamp": 1732},
    {"participant": "Sneha Rao", "text": "No they're independent workstreams. The framework is Vikram's side, I just need to write the test config once it's ready.", "timestamp": 1740},

    # 29:10 — Risk register and close
    {"participant": "Arjun Verma", "text": "Let's do a quick risk register before we close. What's everyone worried about for Sprint 15?", "timestamp": 1750},
    {"participant": "Rohan Gupta", "text": "The main risk is the database migration. If the dual-write has unexpected behaviour in staging we might need to push the cutover.", "timestamp": 1758},
    {"participant": "Vikram Nair", "text": "For me it's the Python 2 to 3 migration. Four thousand lines of manual review in a fortnight is ambitious. I might need to bring in help.", "timestamp": 1772},
    {"participant": "Deepa Krishnan", "text": "My risk is the churn model retraining timeline. Getting the product usage event pipeline set up is a dependency and that's Rohan's team.", "timestamp": 1792},
    {"participant": "Rohan Gupta", "text": "I'll have the event pipeline done by end of week 1 of the sprint. That gives you the full second week for retraining.", "timestamp": 1803},
    {"participant": "Sneha Rao", "text": "My risk is stakeholder feedback cycle on the funnel redesign. If it goes more than two rounds we'll run out of sprint time.", "timestamp": 1815},
    {"participant": "Arjun Verma", "text": "Good. Let's agree: maximum two feedback rounds for the funnel design, strictly time-boxed. Stakeholders have 48 hours to respond each round.", "timestamp": 1830},
    {"participant": "Sneha Rao", "text": "Perfect, that works for me.", "timestamp": 1845},
    {"participant": "Arjun Verma", "text": "Alright. I think we've covered everything. Quick summary of the key decisions: one, audit logging scoped into Sprint 15. Two, mobile app deferred to Q4. Three, 70 percent backend test coverage target by end of Q3. Four, SOC 2 gap items assigned — Rohan access reviews, Deepa runbook, Vikram vendor assessments. Five, churn model retraining unblocked by Rohan's event pipeline. Six, funnel redesign has a two-round max feedback cycle. Any objections?", "timestamp": 1852},
    {"participant": "Rohan Gupta", "text": "All good.", "timestamp": 1900},
    {"participant": "Vikram Nair", "text": "Good.", "timestamp": 1903},
    {"participant": "Sneha Rao", "text": "Works for me.", "timestamp": 1905},
    {"participant": "Deepa Krishnan", "text": "Agreed.", "timestamp": 1908},
    {"participant": "Arjun Verma", "text": "Great. Thanks everyone, productive session. I'll update the Confluence roadmap page with the Sprint 15 scope today. See you all tomorrow at standup.", "timestamp": 1912},
    {"participant": "Sneha Rao", "text": "Thanks Arjun.", "timestamp": 1918},
    {"participant": "Deepa Krishnan", "text": "Bye everyone.", "timestamp": 1920},
]


# ---------------------------------------------------------------------------
# Test cases: DETAILED and VAGUE questions across all intent types
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    question: str
    question_type: str        # "detailed" | "vague"
    expected_intent: str
    expectation: str


@dataclass
class Result:
    question: str
    question_type: str
    expected_intent: str
    actual_intent: str
    filler_text: str
    answer_text: str
    ttfw_ms: float          # time-to-first-word: classify + LLM + TTS sentence[0]
    llm_ms: float           # LLM-only latency
    tts_ms: float           # TTS synthesis of sentence[0] only
    expectation: str
    score: int = 0
    notes: str = ""


TEST_CASES: List[TestCase] = [
    # ---- MEETING SUMMARY ----
    TestCase(
        question="Can you give me a full summary of everything that was discussed in this meeting?",
        question_type="detailed",
        expected_intent="meeting_summary",
        expectation="Should cover Razorpay / payment race condition, dashboard redesign, churn model, recommendation engine, Kubernetes migration, SOC 2 audit, Sprint 15 scope",
    ),
    TestCase(
        question="Catch me up",
        question_type="vague",
        expected_intent="meeting_summary",
        expectation="Should mention main topics: payment fix, dashboard, churn model, K8s, Q3 priorities",
    ),
    TestCase(
        question="Give me a brief summary",
        question_type="vague",
        expected_intent="meeting_summary",
        expectation="Short 3-5 point summary covering major themes",
    ),
    TestCase(
        question="What did we talk about so far?",
        question_type="vague",
        expected_intent="meeting_summary",
        expectation="Should list key discussion themes",
    ),

    # ---- ACTION ITEMS ----
    TestCase(
        question="What are the action items and who owns each one?",
        question_type="detailed",
        expected_intent="action_items",
        expectation="Should list: Rohan payment dedup fix, Rohan access review, Deepa runbook, Vikram vendor assessments, Deepa Sprint 15 board update, Arjun Confluence roadmap update",
    ),
    TestCase(
        question="What do we need to do next?",
        question_type="vague",
        expected_intent="action_items",
        expectation="Should mention Sprint 15 tasks, SOC 2 items, escalations",
    ),
    TestCase(
        question="List the todos from this meeting",
        question_type="vague",
        expected_intent="action_items",
        expectation="Should list key action items from the meeting",
    ),

    # ---- SPEAKER QUERIES ----
    TestCase(
        question="What did Rohan say about the database migration?",
        question_type="detailed",
        expected_intent="speaker_query",
        expectation="Should mention dual-write strategy, 70 percent complete, two-week timeline, maintenance window",
    ),
    TestCase(
        question="What did Deepa say?",
        question_type="vague",
        expected_intent="speaker_query",
        expectation="Should cover churn model (68% precision), recommendation engine, SOC 2 compliance gaps",
    ),
    TestCase(
        question="What did Vikrum con tribute to the meeting?",  # STT artifact: split verb + misspelled name
        question_type="detailed",
        expected_intent="speaker_query",
        expectation="Should match Vikram and cover search performance, K8s migration, Python 2 services, observability",
    ),
    TestCase(
        question="What did Sneha mention?",
        question_type="vague",
        expected_intent="speaker_query",
        expectation="Should mention dashboard redesign, funnel confusion, 15% engagement drop, A/B test plan",
    ),

    # ---- MEETING OPINION ----
    TestCase(
        question="What do you think about deferring the mobile app to Q4? Is that the right call?",
        question_type="detailed",
        expected_intent="meeting_opinion",
        expectation="Should reference activation rate 48 percent, web experience first, mobile vs web tradeoff",
    ),
    TestCase(
        question="Should we go with the hybrid recommendation approach?",
        question_type="detailed",
        expected_intent="meeting_opinion",
        expectation="Should reference co-occurrence for popular items vs vector for long-tail, 18 percent better hit rate",
    ),
    TestCase(
        question="What do you think about the sprint scope?",
        question_type="vague",
        expected_intent="meeting_opinion",
        expectation="Should give opinion on whether Sprint 15 scope is realistic, referencing the discussion",
    ),
    TestCase(
        question="How should we handle the Python 2 migration risk?",
        question_type="detailed",
        expected_intent="meeting_opinion",
        expectation="Should reference 4000 lines manual review, Vikram's concern, suggest prioritization or additional help",
    ),

    # ---- GENERAL KNOWLEDGE ----
    TestCase(
        question="What is horizontal pod autoscaling in Kubernetes and how does it work?",
        question_type="detailed",
        expected_intent="general",
        expectation="Should explain HPA: monitors CPU/memory or custom metrics, scales pod replicas, no made-up details",
    ),
    TestCase(
        question="What is OpenTelemetry?",
        question_type="vague",
        expected_intent="general",
        expectation="Should explain distributed tracing / observability framework correctly",
    ),
    TestCase(
        question="What does P99 latency mean?",
        question_type="vague",
        expected_intent="general",
        expectation="Should explain 99th percentile latency correctly",
    ),
    TestCase(
        question="What is SOC 2 Type 2 audit and why does it matter?",
        question_type="detailed",
        expected_intent="general",
        expectation="Should explain SOC 2 Type 2 covers controls over time period, matters for enterprise trust / compliance",
    ),
    TestCase(
        question="What's the difference between precision and recall in machine learning?",
        question_type="detailed",
        expected_intent="general",
        expectation="Should explain precision (correct positive / predicted positive) and recall (correct positive / actual positive) correctly",
    ),
    TestCase(
        question="What is the current USD to INR exchange rate?",
        question_type="vague",
        expected_intent="general",
        expectation="Should attempt web search or clearly state it cannot give live rate, no made-up number",
    ),
]


# ---------------------------------------------------------------------------
# State reset
# ---------------------------------------------------------------------------

def _reset_state():
    ja.meeting_state["bot_id"] = "test-bot"
    ja.meeting_state["transcript_log"] = list(TRANSCRIPT)
    ja.meeting_state["is_active"] = True
    ja.meeting_state["jarvis_listening"] = False
    ja.meeting_state["current_task"] = None
    ja.meeting_state["pending_requests"].clear()
    ja.meeting_state["pending_clarification"] = None
    ja.meeting_state["pending_general_clarification"] = None
    ja.meeting_state["pending_summary_clarification"] = None
    ja.meeting_state["current_task_id"] = None
    ja.meeting_state["current_phase"] = "idle"
    ja.meeting_state["current_request"] = None
    ja.meeting_state["output_generation"] = 0
    ja.meeting_state["mutation_started"] = False
    ja.meeting_state["cancel_requested"] = False
    ja.meeting_state["last_user_speech_at"] = 0.0
    ja.meeting_state["general_history"] = []
    ja.meeting_state["last_jarvis_response"] = None
    ja.meeting_state["invoker_participant"] = None
    ja.meeting_state["_pending_debounce_task"] = None
    ja.meeting_state["_accumulated_query"] = ""


# ---------------------------------------------------------------------------
# Test runner (quality-only mode — TTS mocked)
# ---------------------------------------------------------------------------

async def _run_case_quality(tc: TestCase) -> Result:
    """Run one case with TTS mocked. Returns answer quality + total LLM latency.

    Patches three output paths:
    1. _speak_guarded   — used by general_responder for text output
    2. _speak_cached_guarded — used for gap-filler MP3 playback
    3. synthesize_speech / speak_cached_audio — used by _speak_streaming (meeting handlers)

    Meeting handlers (_handle_meeting_summary etc.) use _speak_streaming which calls
    synthesize_speech + speak_cached_audio directly, and stores the result in
    meeting_state["last_jarvis_response"]. We patch the audio calls to be no-ops and
    read the result from state after the handler completes.
    """
    _reset_state()

    captured: List[str] = []

    async def mock_speak_guarded(text: str, bot_id: str, generation: int,
                                  allow_stale: bool = False, _preloaded_all=None) -> bool:
        if text:
            captured.append(text)
        return True

    async def mock_speak_cached_guarded(audio_bytes: bytes, bot_id: str, generation: int) -> bool:
        return True  # gap filler plays silently

    # Stub out actual TTS synthesis and Recall audio delivery so _speak_streaming
    # runs at full speed without network calls. The text is still returned via
    # meeting_state["last_jarvis_response"]["answer"] which we read below.
    def mock_synthesize_speech(text: str) -> bytes:
        return b"\xff\xfb\x90\x00" * 16  # minimal valid MP3-ish header stub

    def mock_speak_cached_audio(audio_bytes: bytes, bot_id: str) -> bool:
        return True

    async def mock_sleep(secs: float) -> None:
        pass

    actual_intent = await classify_intent(tc.question)
    start = time.perf_counter()

    with (
        patch.object(ja, "_speak_guarded", side_effect=mock_speak_guarded),
        patch.object(ja, "_speak_cached_guarded", side_effect=mock_speak_cached_guarded),
        patch.object(ja, "synthesize_speech", side_effect=mock_synthesize_speech),
        patch.object(ja, "speak_cached_audio", side_effect=mock_speak_cached_audio),
        patch("asyncio.sleep", side_effect=mock_sleep),
    ):
        try:
            if actual_intent == "meeting_summary":
                await ja._handle_meeting_summary(tc.question, "test-bot")
            elif actual_intent == "meeting_opinion":
                await ja._handle_meeting_opinion(tc.question, "test-bot")
            elif actual_intent == "action_items":
                await ja._handle_action_items(tc.question, "test-bot")
            elif actual_intent == "speaker_query":
                await ja._handle_speaker_query(tc.question, "test-bot")
            elif actual_intent == "general":
                await ja._handle_general_question(tc.question, "test-bot")
            else:
                captured.append(f"[Unhandled intent: {actual_intent}]")
        except Exception as exc:
            captured.append(f"[Handler error: {exc}]")

    llm_ms = (time.perf_counter() - start) * 1000

    # Meeting handlers (_speak_streaming path) store text in last_jarvis_response,
    # not via _speak_guarded. Fall back to reading it here.
    if not captured:
        last = ja.meeting_state.get("last_jarvis_response") or {}
        ans = last.get("answer", "")
        if ans:
            captured.append(ans)

    return Result(
        question=tc.question,
        question_type=tc.question_type,
        expected_intent=tc.expected_intent,
        actual_intent=actual_intent,
        filler_text="(cached mp3 — no TTS call)",
        answer_text=" | ".join(captured) if captured else "(no output captured)",
        ttfw_ms=llm_ms,   # in quality mode, TTFW = LLM time only
        llm_ms=llm_ms,
        tts_ms=0.0,
        expectation=tc.expectation,
    )


# ---------------------------------------------------------------------------
# TTS latency benchmark (full mode — calls actual synthesize_speech)
# ---------------------------------------------------------------------------

async def _run_tts_latency_benchmarks() -> dict:
    """
    Synthesize sample texts of different lengths and measure latency.
    Returns dict with results per text length bucket.
    """
    from confluence_logic.jarvis_agentic import synthesize_speech, _split_into_sentences

    sample_texts = {
        "short_ack": "On it, let me check that.",
        "single_sentence": "Sure, let me pull up the meeting summary for you right away.",
        "two_sentences": "Based on what I heard, the recommendation engine should use a hybrid approach. Vector search handles long-tail queries while co-occurrence manages popular items.",
        "three_sentences": "Here are the main action items. First, Rohan will fix the payment webhook deduplication. Second, Deepa will update the incident response runbook. Third, Vikram will complete the vendor risk assessments.",
        "full_answer": (
            "Based on the discussion, the team is in a strong position but Sprint 15 is ambitious. "
            "The database migration is the highest risk item because the dual-write cutover could surface "
            "unexpected behaviour in staging. The Python 2 migration adds pressure for Vikram who flagged "
            "it as a potential bandwidth issue. If I had to de-scope something, I would push the vendor "
            "risk assessments to early Q3 so the engineering team has more focus this sprint."
        ),
    }

    results = {}
    for label, text in sample_texts.items():
        sentences = _split_into_sentences(text)
        first_sentence = sentences[0] if sentences else text

        # Measure full synthesis
        t0 = time.perf_counter()
        try:
            full_audio = await asyncio.to_thread(synthesize_speech, text)
            full_ms = (time.perf_counter() - t0) * 1000
            full_bytes = len(full_audio)
        except Exception as e:
            full_ms = -1
            full_bytes = 0
            print(f"    TTS synthesis failed for '{label}': {e}")

        # Measure first-sentence-only synthesis (the key TTFW component)
        t0 = time.perf_counter()
        try:
            first_audio = await asyncio.to_thread(synthesize_speech, first_sentence)
            first_ms = (time.perf_counter() - t0) * 1000
            first_bytes = len(first_audio)
        except Exception as e:
            first_ms = -1
            first_bytes = 0

        results[label] = {
            "text_chars": len(text),
            "first_sentence": first_sentence,
            "first_sentence_chars": len(first_sentence),
            "full_tts_ms": full_ms,
            "full_audio_bytes": full_bytes,
            "first_sentence_tts_ms": first_ms,
            "first_sentence_audio_bytes": first_bytes,
        }
        print(f"    {label}: full={full_ms:.0f}ms  first_sentence={first_ms:.0f}ms")

    return results


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_result(r: Result) -> tuple[int, str]:
    notes = []

    if not r.answer_text or r.answer_text.startswith("(no output"):
        return 0, "No answer produced"

    if "[Handler error" in r.answer_text or "[Unhandled intent" in r.answer_text:
        return 0, r.answer_text

    intent_ok = r.actual_intent == r.expected_intent
    if not intent_ok:
        notes.append(f"Intent mismatch: expected '{r.expected_intent}', got '{r.actual_intent}'")

    answer_lower = r.answer_text.lower()

    failure_phrases = [
        "sorry, i couldn't", "i had trouble", "i wasn't able",
        "couldn't generate", "i don't see anyone", "hasn't said anything",
        "i haven't heard",
    ]
    if any(p in answer_lower for p in failure_phrases):
        notes.append("Answer is an error fallback")
        return 1 if intent_ok else 0, "; ".join(notes)

    _STOP_WORDS = {
        "should", "their", "these", "which", "still", "match", "cover", "mention",
        "reference", "explain", "about", "answer", "gives", "state", "clearly",
        "attempt", "search", "correctly", "without", "number", "details", "using",
        "specific", "points", "cover", "mention", "should", "state",
    }
    # Split on whitespace AND punctuation (/, -, :) to handle "CPU/memory", "made-up", "50%"
    import re as _re
    raw_tokens = _re.split(r"[\s/\-:,]+", r.expectation)
    expectation_keywords = [
        t.strip("()%.'\"").lower()
        for t in raw_tokens
        if len(t.strip("()%.'\"")) > 3
        and t.strip("()%.'\"").lower() not in _STOP_WORDS
    ]
    # De-duplicate while preserving order
    seen = set()
    expectation_keywords = [k for k in expectation_keywords if not (k in seen or seen.add(k))]

    matches = sum(1 for kw in expectation_keywords if kw in answer_lower)
    coverage = matches / max(len(expectation_keywords), 1)

    if intent_ok and coverage >= 0.4:
        score = 3
        if coverage < 0.6:
            notes.append(f"Partial keyword coverage ({coverage:.0%})")
            score = 2
    elif intent_ok:
        score = 2
        notes.append(f"Low keyword coverage ({coverage:.0%})")
    else:
        score = 1
        notes.append(f"Keyword coverage: {coverage:.0%}")

    return score, "; ".join(notes) if notes else "OK"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _render_report(results: List[Result], tts_results: Optional[dict], full_mode: bool) -> str:
    WIDTH = 110
    lines = []
    lines.append("=" * WIDTH)
    lines.append("  JARVIS COMPREHENSIVE PIPELINE EVALUATION REPORT")
    lines.append("=" * WIDTH)
    lines.append(f"  Transcript: {len(TRANSCRIPT)} entries  |  {len(set(e['participant'] for e in TRANSCRIPT))} speakers  |  ~{TRANSCRIPT[-1]['timestamp']//60} minutes")
    lines.append(f"  Test cases: {len(results)}  ({sum(1 for r in results if r.question_type=='detailed')} detailed, {sum(1 for r in results if r.question_type=='vague')} vague)")
    lines.append(f"  Mode: {'FULL (real TTS)' if full_mode else 'QUALITY-ONLY (mocked TTS)'}")
    lines.append("")

    total_score = 0
    max_score = len(results) * 3

    for i, r in enumerate(results, 1):
        intent_badge = "✓" if r.actual_intent == r.expected_intent else "✗"
        score_bar = "●" * r.score + "○" * (3 - r.score)
        type_tag = f"[{r.question_type.upper()[:3]}]"
        lines.append(f"┌─ [{i:02d}] {type_tag} {r.question}")
        lines.append(f"│   Intent:   {intent_badge} expected={r.expected_intent:<18} got={r.actual_intent}")
        lines.append(f"│   Latency:  LLM={r.llm_ms:.0f}ms  TTS(s0)={r.tts_ms:.0f}ms  TTFW={r.ttfw_ms:.0f}ms")
        lines.append(f"│   Answer:   {textwrap.shorten(r.answer_text, 100)}")
        lines.append(f"│   Score:    [{score_bar}] {r.score}/3  — {r.notes}")
        lines.append(f"│   Expects:  {textwrap.shorten(r.expectation, 100)}")
        lines.append("└" + "─" * (WIDTH - 1))
        lines.append("")
        total_score += r.score

    pct = total_score / max_score * 100
    lines.append("=" * WIDTH)
    lines.append(f"  TOTAL SCORE: {total_score}/{max_score}  ({pct:.0f}%)")
    grade = "PASS ✓" if pct >= 80 else ("MARGINAL ⚠" if pct >= 60 else "FAIL ✗")
    lines.append(f"  GRADE: {grade}")
    lines.append("=" * WIDTH)

    # --- Latency breakdown ---
    lines.append("")
    lines.append("  LATENCY BREAKDOWN (LLM call only)")
    lines.append(f"  {'Question':<58} {'Type':<4} {'Intent':<18} {'LLM ms':>7}")
    lines.append("  " + "-" * 92)
    for r in results:
        lines.append(f"  {r.question[:57]:<58} {r.question_type[:3]:<4} {r.actual_intent:<18} {r.llm_ms:>7.0f}")
    latencies = [r.llm_ms for r in results]
    lines.append(f"  {'AVERAGE':<58} {'':4} {'':18} {sum(latencies)/len(latencies):>7.0f}")
    lines.append(f"  {'P95 (approx)':<58} {'':4} {'':18} {sorted(latencies)[int(len(latencies)*0.95)-1]:>7.0f}")
    lines.append(f"  {'MAX':<58} {'':4} {'':18} {max(latencies):>7.0f}")
    lines.append("")

    # --- By intent ---
    lines.append("  SCORES BY INTENT")
    intent_groups: dict = {}
    for r in results:
        intent_groups.setdefault(r.actual_intent, []).append(r)
    for intent, group in sorted(intent_groups.items()):
        avg_score = sum(g.score for g in group) / len(group)
        avg_lat = sum(g.llm_ms for g in group) / len(group)
        lines.append(f"    {intent:<20} n={len(group)}  avg_score={avg_score:.1f}/3  avg_llm={avg_lat:.0f}ms")
    lines.append("")

    # --- By question type ---
    lines.append("  SCORES BY QUESTION TYPE")
    for qtype in ("detailed", "vague"):
        group = [r for r in results if r.question_type == qtype]
        if group:
            avg = sum(g.score for g in group) / len(group)
            lines.append(f"    {qtype:<10} n={len(group)}  avg_score={avg:.1f}/3")
    lines.append("")

    # --- TTS latency ---
    if tts_results:
        lines.append("=" * WIDTH)
        lines.append("  TTS LATENCY BENCHMARKS (actual OpenAI TTS synthesis)")
        lines.append("=" * WIDTH)
        lines.append(f"  {'Sample':<22} {'Chars':<6} {'Full TTS ms':>11} {'First-Sentence ms':>17} {'Bytes(full)':>11}")
        lines.append("  " + "-" * 70)
        for label, data in tts_results.items():
            lines.append(
                f"  {label:<22} {data['text_chars']:<6} {data['full_tts_ms']:>11.0f} "
                f"{data['first_sentence_tts_ms']:>17.0f} {data['full_audio_bytes']:>11}"
            )
        lines.append("")

        first_ms_vals = [d["first_sentence_tts_ms"] for d in tts_results.values() if d["first_sentence_tts_ms"] > 0]
        if first_ms_vals:
            lines.append(f"  First-sentence TTS avg: {sum(first_ms_vals)/len(first_ms_vals):.0f}ms")
            lines.append(f"  First-sentence TTS max: {max(first_ms_vals):.0f}ms")
        lines.append("")

    # --- Analysis and recommendations ---
    lines.append("=" * WIDTH)
    lines.append("  PIPELINE ANALYSIS & EXACT RECOMMENDATIONS")
    lines.append("=" * WIDTH)
    suggestions = _generate_recommendations(results, tts_results, full_mode)
    for cat, items in suggestions.items():
        lines.append(f"\n  [{cat}]")
        for item in items:
            for j, line in enumerate(textwrap.wrap(item, 104)):
                prefix = "  • " if j == 0 else "    "
                lines.append(prefix + line)
    lines.append("")

    return "\n".join(lines)


def _generate_recommendations(results: List[Result], tts_results: Optional[dict], full_mode: bool) -> dict:
    recs: dict = {}

    # 1. Intent classification issues
    mismatches = [r for r in results if r.actual_intent != r.expected_intent]
    intent_recs = []
    if mismatches:
        for r in mismatches:
            intent_recs.append(
                f'"{r.question[:55]}" classified as {r.actual_intent} (expected {r.expected_intent}). '
                f"Fix: add phrase pattern to _fast_classify() in classifier.py for this case."
            )
    else:
        intent_recs.append("All intents classified correctly. classifier.py fast-path is working well.")
    recs["CLASSIFICATION"] = intent_recs

    # 2. Answer quality issues
    quality_recs = []
    low_scorers = [r for r in results if r.score <= 1]
    if low_scorers:
        for r in low_scorers:
            quality_recs.append(
                f'"{r.question[:55]}" scored {r.score}/3 ({r.notes}). '
                f"Intent={r.actual_intent}. Check LLM prompt in the corresponding handler function."
            )
    avg_score = sum(r.score for r in results) / len(results)
    if avg_score >= 2.5:
        quality_recs.append(f"Overall quality is good (avg {avg_score:.2f}/3).")
    elif avg_score >= 1.8:
        quality_recs.append(
            f"Moderate quality (avg {avg_score:.2f}/3). Focus on improving context injection in handlers "
            f"— ensure transcript is being passed fully, check JARVIS_SUMMARY_MAX_TOKENS / JARVIS_SPEAKER_QUERY_MAX_TOKENS."
        )
    else:
        quality_recs.append(
            f"Low quality (avg {avg_score:.2f}/3). Review all LLM prompts. Verify OPENAI_API_KEY is set and model is accessible."
        )
    recs["ANSWER QUALITY"] = quality_recs

    # 3. TTS latency recommendations (always, based on known code patterns)
    tts_recs = []

    tts_recs.append(
        "CRITICAL — synthesize_speech() uses response.read() which blocks until the full audio is ready. "
        "Switch to streaming byte iteration to reduce first-byte latency:\n"
        "    # In synthesize_speech(), replace response.read() with:\n"
        "    chunks = []\n"
        "    for chunk in response.iter_bytes(chunk_size=1024):\n"
        "        chunks.append(chunk)\n"
        "        if not first_chunk_time:\n"
        "            first_chunk_time = time.perf_counter()  # TTFW is measurable here\n"
        "    return b''.join(chunks)\n"
        "    # Then in _speak_guarded, send first chunk to bot as soon as it arrives."
    )

    tts_recs.append(
        "HIGH — The gap filler path already runs filler + LLM in parallel (good). "
        "But it waits for gap_filler_task to FULLY finish before playing the pre-synthesized answer. "
        "Improvement: start playing sentence[0] immediately when it's synthesized if the gap filler "
        "has already started (even if it hasn't finished). This removes the 'wait for gap filler to end' "
        "dead time when LLM is fast.\n"
        "    Change in _handle_general_question():\n"
        "    # Instead of: await gap_filler_task  # blocks until filler done\n"
        "    # Use: asyncio.ensure_future(gap_filler_task)  # fire-and-forget, answer plays immediately\n"
        "    # NOTE: only safe if the output_lock serialisation ensures no overlap."
    )

    tts_recs.append(
        "MEDIUM — Pre-synthesize all sentences in parallel (already done in _handle_general_question). "
        "But meeting_responder handlers (summarize_meeting, extract_action_items, etc.) do NOT do this — "
        "they call _speak_guarded with raw text, which synthesizes sentence-by-sentence just-in-time. "
        "Improvement: in _handle_meeting_summary, _handle_action_items, _handle_speaker_query, "
        "_handle_meeting_opinion — pre-synthesize all sentences in parallel while the gap filler plays, "
        "identical to the pattern in _handle_general_question.\n"
        "    sentences = _split_into_sentences(answer)\n"
        "    syn_tasks = [asyncio.create_task(asyncio.to_thread(synthesize_speech, s)) for s in sentences]\n"
        "    await gap_filler_task\n"
        "    preloaded_all = list(await asyncio.gather(*syn_tasks))\n"
        "    await _speak_guarded(answer, bot_id, generation, allow_stale=True, _preloaded_all=preloaded_all)"
    )

    tts_recs.append(
        "MEDIUM — Use a faster TTS model or provider for filler phrases. "
        "Currently the gap filler uses a pre-cached MP3 (fast) but falls back to OpenAI TTS (slow). "
        "Ensure generate_wav_assets.py is run and the audio cache is populated (assets/audio/*.mp3 exist). "
        "Check get_random_filler_audio() and get_random_ack_audio() — if they return None, "
        "the TTS call blocks for 300-800ms before Jarvis says anything."
    )

    tts_recs.append(
        "LOW — Consider switching JARVIS_TTS_MODEL from 'tts-1' to 'tts-1' with response_format='pcm' "
        "or 'opus'. PCM/Opus have lower overhead than MP3 encoding, reducing synthesis time by ~50ms. "
        "Recall.ai accepts WAV/PCM directly: change 'kind' in speak() payload from 'mp3' to 'wav' "
        "and generate PCM audio."
    )

    tts_recs.append(
        "LOW — JARVIS_TTS_SPEED default is 1.0. Setting it to 1.1 or 1.15 shortens audio playback duration "
        "by 10-15%, reducing total speaking time without affecting quality significantly. "
        "Set JARVIS_TTS_SPEED=1.1 in .env to test."
    )

    if full_mode and tts_results:
        first_ms_vals = [d["first_sentence_tts_ms"] for d in tts_results.values() if d["first_sentence_tts_ms"] > 0]
        if first_ms_vals:
            avg_first = sum(first_ms_vals) / len(first_ms_vals)
            if avg_first > 600:
                tts_recs.append(
                    f"MEASURED — First-sentence TTS avg is {avg_first:.0f}ms. "
                    "This directly adds to TTFW. Target: <400ms for first sentence. "
                    "If using OpenAI tts-1, this is expected. Consider edge_tts (free, ~150ms) "
                    "or cartesia/deepgram for production sub-200ms TTS."
                )

    recs["TTS LATENCY"] = tts_recs

    # 4. TTFW analysis
    ttfw_recs = []
    ttfw_recs.append(
        "TTFW FORMULA: TTFW = max(gap_filler_duration, classifier_ms + LLM_ms) + TTS_sentence0_ms\n"
        "  • Micro-ack (cached): ~0ms (fires immediately on wake word)\n"
        "  • Gap filler (cached MP3): ~800ms-1200ms typical playback duration\n"
        "  • Classifier (fast-path): ~0ms; (LLM fallback): ~200-400ms\n"
        "  • LLM answer generation: ~800ms-2500ms (gpt-4o-mini)\n"
        "  • TTS sentence[0] synthesis: ~300ms-700ms (tts-1)\n"
        "  CURRENT BEST CASE (LLM slower than gap filler): ~1200ms TTFW\n"
        "  CURRENT WORST CASE (LLM faster than gap filler but no pre-synthesis in meeting handlers): ~3200ms TTFW"
    )
    ttfw_recs.append(
        "TARGET: <1500ms TTFW from question end to first spoken word. Achievable with:\n"
        "  1. Micro-ack fires immediately (~0ms visible)\n"
        "  2. Gap filler plays from cache (~800ms audio)\n"
        "  3. LLM + TTS sentence[0] overlap with gap filler (parallel)\n"
        "  4. Pre-synthesize remaining sentences during gap filler\n"
        "  5. Answer starts immediately after gap filler ends (no dead air)"
    )
    recs["TTFW ANALYSIS"] = ttfw_recs

    # 5. Architecture recommendations
    arch_recs = []
    slow_llm = [r for r in results if r.llm_ms > 5000]
    if slow_llm:
        arch_recs.append(
            f"{len(slow_llm)} queries took >5s LLM time. These block the answer pipeline. "
            "Consider: (a) streaming the LLM response and starting TTS on the first sentence "
            "as tokens arrive (streaming mode), or (b) lowering max_tokens for the summary "
            "handlers when speech_rewrite_enabled=True since _rewrite_for_speech will condense anyway."
        )

    arch_recs.append(
        "MEETING HANDLERS: summarize_meeting(), extract_action_items(), generate_opinion(), "
        "summarize_speaker() all run synchronously (asyncio.to_thread wrapping blocking OpenAI call). "
        "Improvement: use the OpenAI streaming API for these too — stream tokens, buffer until sentence "
        "boundary detected ('. ' or '? ' or '! '), synthesize each sentence immediately, play pipeline."
    )

    arch_recs.append(
        "GRAPH RAG: graph_rag.query_context() is called for general questions but exceptions are silently "
        "swallowed. Add a timeout (asyncio.wait_for(..., timeout=0.5)) so a slow graph never delays "
        "the answer — if it times out, proceed without graph context."
    )

    recs["ARCHITECTURE"] = arch_recs

    return recs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main(full_mode: bool = True):
    print(f"\n{'='*70}")
    print(f"  JARVIS COMPREHENSIVE PIPELINE EVAL  |  mode={'FULL' if full_mode else 'QUALITY-ONLY'}")
    print(f"{'='*70}")
    print(f"  Transcript: {len(TRANSCRIPT)} utterances, ~{TRANSCRIPT[-1]['timestamp']//60} minutes")
    print(f"  Test cases: {len(TEST_CASES)} ({sum(1 for t in TEST_CASES if t.question_type=='detailed')} detailed, {sum(1 for t in TEST_CASES if t.question_type=='vague')} vague)")
    print()

    results: List[Result] = []
    for i, tc in enumerate(TEST_CASES, 1):
        print(f"  [{i:02d}/{len(TEST_CASES)}] [{tc.question_type.upper()[:3]}] {tc.question[:65]}", end="", flush=True)
        try:
            r = await _run_case_quality(tc)
            r.score, r.notes = _score_result(r)
            results.append(r)
            intent_badge = "✓" if r.actual_intent == r.expected_intent else f"✗→{r.actual_intent}"
            print(f"  {intent_badge:<20} {r.llm_ms:.0f}ms  [{r.score}/3]")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            results.append(Result(
                question=tc.question,
                question_type=tc.question_type,
                expected_intent=tc.expected_intent,
                actual_intent="error",
                filler_text="",
                answer_text=f"[Exception: {exc}]",
                ttfw_ms=0,
                llm_ms=0,
                tts_ms=0,
                expectation=tc.expectation,
                score=0,
                notes=str(exc),
            ))

    tts_results = None
    if full_mode:
        print()
        print("  Running TTS latency benchmarks (calling real synthesize_speech)...")
        try:
            tts_results = await _run_tts_latency_benchmarks()
        except Exception as e:
            print(f"  TTS benchmark failed: {e}")

    print()
    report = _render_report(results, tts_results, full_mode)
    print(report)

    # Save report
    report_path = "tests/eval_report.txt"
    try:
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\n  Report saved to {report_path}")
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jarvis Pipeline Comprehensive Evaluation")
    parser.add_argument("--quality-only", action="store_true", help="Mock TTS, skip TTS latency benchmarks")
    args = parser.parse_args()
    asyncio.run(main(full_mode=not args.quality_only))
