# Q3 Infrastructure & Platform Review — Virtual Meeting Transcript
**Date:** Thursday, 10 October 2024
**Time:** 10:02 AM – 12:14 PM EDT
**Platform:** Google Meet
**Participants:**
- Sarah Chen — VP of Engineering (host)
- Marcus Webb — Head of Infrastructure
- Priya Nair — Finance Controller
- Jordan Blake — Data & Analytics Lead
- Tomas Riera — ML & AI Lead
- Keiko Yamamoto — Security Lead
- Devon Holt — Platform Engineering Lead
- Aisha Okonkwo — Commerce Lead
- Ravi Subramaniam — DBA Lead (joined late)
- Lily Marchetti — Engineering Program Manager

---

[10:02 AM]

**Sarah Chen:** Alright, I can see a few people in the waiting room — let me let everyone in. Give me one sec.

**Devon Holt:** Morning Sarah.

**Sarah Chen:** Hey Devon! How are you doing?

**Devon Holt:** Yeah, good, good. Long week already and it's only Thursday.

**Sarah Chen:** Tell me about it. Okay, I've let in Marcus, Priya, Keiko... Jordan, you're in, Tomas is joining — Aisha, there you are.

**Aisha Okonkwo:** Sorry, sorry. My last call ran over by like ten minutes. Hi everyone.

**Marcus Webb:** Morning Aisha.

**Lily Marchetti:** Hi all! Can everyone hear me okay? I'm on my laptop today, my headset cable died this morning.

**Devon Holt:** Yeah you sound fine Lily.

**Tomas Riera:** Good morning everyone. Quick question — is Ravi joining? I had a question for him about the Commerce DB.

**Sarah Chen:** He said he'd be five minutes late. Let's give him till like ten past and then just start. Keiko, you're on mute by the way.

**Keiko Yamamoto:** Oh, sorry. Yeah, I was just saying good morning. Classic.

[Laughter]

**Priya Nair:** Morning everyone. I'm going to have my camera off for the first bit — I'm waiting for a coffee delivery and my building's buzzer is broken so I have to physically go down. I'll be back in two minutes.

**Sarah Chen:** Ha, very important. Go, go.

**Marcus Webb:** Priya's priorities are correct, honestly.

**Jordan Blake:** One hundred percent.

**Sarah Chen:** Okay so while we wait for Priya and Ravi — I feel like I haven't spoken to half of you properly since the offsite in August. Tomas, how was your trip? You went to Portugal after, right?

**Tomas Riera:** I did yeah. Lisbon for a week. It was incredible. Honestly I came back so relaxed and then the August sixth incident happened literally the day I got back and it just — yeah.

**Devon Holt:** Brutal timing.

**Tomas Riera:** Brutal timing. I landed at like seven AM and by noon we were in incident response. So.

**Sarah Chen:** We'll get to that one today. Keiko, how's the new apartment? You moved, yeah?

**Keiko Yamamoto:** I did, yeah, in September. It's great, finally have a proper home office so I'm not doing security reviews from my kitchen table anymore. Very adult.

**Aisha Okonkwo:** That's a game changer. I did that last year and the difference is just — yeah.

**Marcus Webb:** I'm jealous. I'm still in the kitchen table era.

**Sarah Chen:** Marcus you've been saying you're going to set up a home office for like two years.

**Marcus Webb:** I know, I know. Maybe this is the year.

**Devon Holt:** It's October mate.

**Marcus Webb:** Next year then.

[Laughter]

**Priya Nair:** I'm back! Got the coffee. Crisis averted.

**Sarah Chen:** The most important crisis of the day. Okay let's get started and Ravi can catch up when he joins. Lily, are you capturing notes?

**Lily Marchetti:** Yep, already on it.

**Sarah Chen:** Perfect. So — Q3 review. I want to get through three main sections today: infrastructure costs, platform reliability and SLAs, and then a forward look at Q4 commitments. We have until noon so we should have plenty of time. Marcus, do you want to kick off the cost section?

**Marcus Webb:** Sure. Can everyone see my screen?

**Devon Holt:** Yeah.

**Keiko Yamamoto:** Yep.

**Aisha Okonkwo:** I see the spreadsheet, yes.

**Marcus Webb:** Great. So — headline first. Q3 total spend came in at just under two point eight five million. The approved budget was two point seven million, so we came in over by about a hundred and forty-seven thousand.

**Priya Nair:** One forty-seven four twelve to be exact.

**Marcus Webb:** Thank you Priya. Yes, one forty-seven thousand four hundred and twelve dollars. Not ideal. But I want to give some context before people panic because a meaningful chunk of that was anomaly-related and isn't expected to recur.

**Sarah Chen:** What drove the overage at a high level?

**Marcus Webb:** Three things really. The egress costs from the August migration — that was the biggest one. Then the EC2 auto-scaling during the Commerce launch event on the fourteenth of September. And then Keiko's team needed emergency WAF expansion following that CVE in late August.

**Keiko Yamamoto:** Yeah the CVE-2024-3891. We had no choice on that one, for the record.

**Marcus Webb:** Absolutely, I'm not — no one's disputing that. It was the right call.

**Sarah Chen:** How much did the WAF situation add up to?

**Keiko Yamamoto:** Our total overage in Security was about forty-four thousand on the cloud side. A chunk of that was GuardDuty log ingestion because when you expand the WAF rules you're generating dramatically more logs. We've since adjusted the retention policy and brought that back down.

**Sarah Chen:** Good. Marcus, which provider was the biggest contributor to overage?

**Marcus Webb:** So, AWS was the primary overage — about fifty-nine thousand over. GCP was actually the bigger surprise percentage-wise, came in sixty-one thousand over budget. That's mostly BigQuery.

**Jordan Blake:** Yeah, I was going to flag that. The BigQuery expansion was approved in July but I think the finance team's budget hadn't been updated to reflect the new slot commitment. Priya, is that right?

**Priya Nair:** Partially. The slot expansion was in the July planning doc but the dollar impact wasn't modelled correctly. We assumed a higher discount rate on the committed slots than we actually got. So it's a forecasting issue on our side, I'll own that.

**Jordan Blake:** And to be clear the expansion was necessary — the Analytics Platform migration to real-time was completely blocked without it. We couldn't have shipped that on time otherwise.

**Sarah Chen:** Agreed, that was a deliberate decision. Just want to make sure it's in the budget model going forward.

**Priya Nair:** It will be. I'm updating the Q4 and H1 FY25 models this week.

**Marcus Webb:** One other thing I want to flag on the AWS side — compute. So the total compute line for Q3 was just over a million forty-one thousand.

**Jordan Blake:** Wait actually — Marcus, I think that number might be off. I was looking at the AWS Cost Explorer breakdown last week and I think EC2 alone was closer to one point four million.

**Marcus Webb:** One point four? No, that can't be right. The full compute category including EC2, EKS node groups, spot instances — that's what's in the one-oh-four-one number.

**Jordan Blake:** Hmm. Let me pull up — okay so I'm looking at my export from last Tuesday. AWS EC2 line item. One million, four hundred and twelve thousand.

**Marcus Webb:** That's — hold on, are you looking at the consolidated billing or the Thornvale-specific account?

**Jordan Blake:** Oh. Consolidated. We have the subsidiary accounts in there.

**Marcus Webb:** Yeah, that's it. The two subsidiary accounts add about three hundred and sixty thousand together. The number in the report is Thornvale platform only.

**Jordan Blake:** Right, yes. That makes sense. Sorry, ignore me.

**Tomas Riera:** One point four million would have been alarming.

**Marcus Webb:** Just slightly. Yeah no, we're at a million forty-one on compute for the Thornvale platform. Which is still thirty-six percent of total spend, biggest single category.

**Aisha Okonkwo:** Can I ask about the Commerce-specific allocation? I saw the department breakdown and we came in at four hundred and seventy-nine thousand. That felt high to me — is that including the launch event spike?

**Marcus Webb:** It does, yes. The September fourteenth event added about thirty-two thousand in on-demand EC2 and a bit of Cosmos DB throughput.

**Aisha Okonkwo:** Okay, because my team was expecting closer to four forty for the quarter. The launch cost was always going to be there, I just want to make sure there isn't something else lurking.

**Marcus Webb:** Let me cross-check the tagging. Devon, do you know if the Commerce EKS node group tags were all correct in September?

**Devon Holt:** They should be. We did the tagging audit in August, everything was mapped. But I can double check the September deployment specifically.

**Marcus Webb:** Yeah, if you can confirm that'd be good. Aisha, I'll come back to you on that.

**Aisha Okonkwo:** Thanks Marcus.

[10:19 AM — Ravi Subramaniam joins]

**Ravi Subramaniam:** Hey everyone, so sorry. My dentist appointment ran long and then I hit traffic. What did I miss?

**Sarah Chen:** Hey Ravi! No worries. We're in the cost section — Marcus has about ten more minutes I think, and then we're moving to reliability. You're not in trouble yet.

**Ravi Subramaniam:** Perfect. I'll catch up on the notes.

**Lily Marchetti:** Ravi I'll ping you the summary in the chat.

**Ravi Subramaniam:** You're a hero, Lily.

**Marcus Webb:** Okay, so on reserved instance coverage — this is actually a good news story for Q3. We hit seventy-four percent overall compute reserved coverage, up from sixty-one in Q2. That's the first quarter we've met our seventy percent target.

**Devon Holt:** Yeah, the RI purchasing we did in June really paid off. Took a while to reflect in the numbers but you can see it now.

**Sarah Chen:** Nice. What's still below target?

**Marcus Webb:** BigQuery slots are the main one — we're at fifty-five percent coverage against a sixty percent target. Jordan's team committed to purchasing another five hundred annual slots by the fifteenth.

**Jordan Blake:** Already submitted the order, actually. Should be processed by end of day today.

**Marcus Webb:** Oh brilliant, you jumped ahead of me. That's great. Redshift reserved nodes is the other one, we're at sixty-seven percent against a seventy target. Ravi, that purchase is going to the CapEx committee on the twenty-second — you've reviewed the node spec right?

**Ravi Subramaniam:** I have, yes. Twelve nodes, same instance class as the current fleet. I'm comfortable with it.

**Marcus Webb:** Great. One last thing on costs before I hand over — the anomaly events. We had three detected by the alerting system. Total unplanned spend from those three events was ninety-four thousand, which is sixty-four percent of the total variance. So if you strip those out, we were actually quite close to budget.

**Priya Nair:** Which is a useful way to look at it for the board pack, actually. I'll split the variance into structural and anomaly-driven.

**Sarah Chen:** That's a good framing Priya. Okay, Marcus, anything else?

**Marcus Webb:** I think that's the main points. Happy to take questions.

**Tomas Riera:** Quick one — the NAT gateway cost. It showed up at rank eleven in the top twenty resource list and it was twenty-one thousand a month. Is that going down in Q4?

**Marcus Webb:** That's the plan. Devon's team is working on the egress routing optimisation — Devon, what's the status?

**Devon Holt:** We're about halfway through. We've rerouted the US-to-EU sync traffic to stay within region wherever possible. The tricky piece is the cross-region replication for the data pipeline which has stricter consistency requirements. We're talking to Jordan's team about whether we can tolerate a slightly relaxed consistency model on some of those writes.

**Jordan Blake:** Yeah we've had one conversation about it. I think for the historical data sync we can relax it, for the real-time event stream we probably can't.

**Devon Holt:** Right. So we're expecting to save maybe twenty-five to thirty-five thousand per quarter rather than the full forty-five. Still meaningful.

**Marcus Webb:** I'll update the forecast accordingly.

**Sarah Chen:** Okay good. Let's move to reliability. Devon, you want to take this one?

**Devon Holt:** Sure. So the headline is — Q3 was our best reliability quarter since we consolidated the stack. Weighted platform availability came in at ninety-nine point nine four percent, which beats our contractual SLA of ninety-nine point nine.

**Keiko Yamamoto:** Which I'm going to caveat by saying the August sixth incident is what almost cost us that.

**Tomas Riera:** Yeah, I want to address that directly. So the ML Inference outage — it was forty-seven minutes on the sixth of August. Root cause was a CUDA driver version that got bundled into what looked like a routine node pool upgrade. The upgrade was tested in staging but our staging GPU pool runs a slightly older kernel version and the driver incompatibility only manifested on the newer kernel in prod.

**Sarah Chen:** How many customers were affected?

**Tomas Riera:** About three thousand eight hundred. Specifically anyone using the recommendation engine and the personalisation features, which both route through the ML scoring endpoint. The API returned five-oh-threes for those endpoints for the full forty-seven minutes.

**Aisha Okonkwo:** Commerce customers noticed. We had a spike in support tickets during that window about recommendations not loading.

**Tomas Riera:** Yeah, I saw that. I'm sorry about that Aisha.

**Aisha Okonkwo:** Not your fault, these things happen. What's the fix?

**Tomas Riera:** Two things. One, we've added a GPU driver compatibility matrix to the node pool upgrade runbook — any upgrade that touches the driver version now requires explicit sign-off from the ML team before it goes to prod. Two, we've bumped the staging GPU pool to match the prod kernel version. Should have done that a long time ago honestly.

**Sarah Chen:** Is that documented in the post-mortem?

**Tomas Riera:** It is, yeah. The post-mortem was published on the eleventh of August. All Sev-1 post-mortems for Q3 are now published, by the way — that's a hundred percent completion which is better than Q2 where we had three that slipped past quarter end.

**Lily Marchetti:** That's worth calling out. Q2 was nine out of twelve. Q3 is eight out of eight.

**Devon Holt:** Yeah that was a deliberate effort. We set a rule that the post-mortem has to be submitted within five business days or the incident lead's manager gets a notification. That seems to have focused minds.

**Sarah Chen:** Ha. Yeah, I can see how that would help. Okay, other incidents worth calling out?

**Devon Holt:** The auth incidents are worth mentioning. We had two Sev-1s in auth — one in July, one in September. The July one was a DB connection pool exhaustion, eight minutes, about twelve hundred customers affected. The September one was an OAuth token cache miss, six minutes, about nine hundred customers.

**Ravi Subramaniam:** I can comment on the July one — the connection pool exhaustion happened because we had a batch job that was holding connections open longer than expected after a code change. We've added a connection timeout at the application layer now and we haven't seen a recurrence.

**Devon Holt:** Yep. And the September auth incident was actually shorter than it looks because our circuit breaker kicked in and started serving cached tokens within about ninety seconds. The six-minute window was the time to fully restore fresh token issuance.

**Sarah Chen:** Good. Devon, what about the false positive page rate? I saw that in the report and it jumped out at me.

**Devon Holt:** Yeah, that's not great. False positive pages are up about thirty percent year on year. We fired two hundred and seventy-five pages in Q3, of which sixty-eight were noisy — basically alerts that fired but required no human action.

**Keiko Yamamoto:** I had two engineers raise alert fatigue as a concern in their quarterly one-on-ones. It's a real morale issue.

**Devon Holt:** Agreed. We did an alert audit in September — identified forty-one rules that need either threshold adjustment or consolidation. We're committed to clearing all forty-one by the end of October.

**Marcus Webb:** Is that realistic in three weeks?

**Devon Holt:** It's tight. Some of them are quick — it's literally changing a threshold. A handful need more thought because they're composite alerts. I'd say we'll get thirty-five done by October thirty-first and the rest in early November.

**Sarah Chen:** Let's put that as a hard commitment in the tracker. Lily?

**Lily Marchetti:** On it.

**Devon Holt:** On the positive side — deployment frequency hit Elite DORA classification for Auth, Commerce, and API Gateway. Auth was the best at eight point three deployments per week average.

**Priya Nair:** What does that mean in practice for people who aren't close to the DORA stuff?

**Devon Holt:** It basically means we're shipping changes to production multiple times a week with very low failure rates. The auth change failure rate was under one percent in Q3, which is exceptional.

**Priya Nair:** That's impressive. I didn't realise we were tracking that rigorously.

**Devon Holt:** It's relatively new — we started instrumenting it properly in Q2.

**Jordan Blake:** I do want to flag Data Ingestion as a concern on the change failure rate side. We came in at seven point seven percent which is above the Elite threshold of five.

**Devon Holt:** Yep. Main contributor is the lack of automated integration tests for infrastructure-layer changes. Jordan, is the test coverage work planned for Q4?

**Jordan Blake:** It's in the roadmap, yes. Realistically it's a six-week project. If we start the first week of November we should have meaningful coverage by end of Q4 which would set us up for a better Q1.

**Sarah Chen:** What was the Data Ingestion incident count specifically?

**Devon Holt:** Four rollbacks in Q3. Two were schema-related, one was a Kafka configuration regression, one was a dependency version conflict.

**Jordan Blake:** The dependency one was frustrating because it was actually caught in staging but the staging Kafka version was — and there's a pattern here — not matching prod. So same issue as Tomas had with the GPU pool.

**Tomas Riera:** Welcome to the club.

[Laughter]

**Devon Holt:** We need to do a wider audit of staging environment drift. That's showing up as a theme.

**Sarah Chen:** Can you make that a Q4 action item? I want a staging-to-prod parity audit across all services, not just the ones that have already had incidents.

**Devon Holt:** Absolutely. I'll own that.

**Lily Marchetti:** Adding it. Owner Devon, due by when?

**Devon Holt:** Let's say fifteenth of November to allow time to actually fix things we find.

**Sarah Chen:** Good. Okay, Tomas — you wanted to talk about the ML GPU pool capacity situation?

**Tomas Riera:** Yeah. So this is a bit urgent actually. The prod ML GPU pool — we hit ninety-one percent peak memory utilisation during model benchmarking on September nineteenth. That's in the critical zone. We have essentially no headroom for a traffic spike.

**Marcus Webb:** Ninety-one? That's — yeah, that's not comfortable.

**Tomas Riera:** No. And the model we're planning to deploy in Q4 is bigger than the current one. I submitted a request for eight additional A100 nodes and it's going to the CapEx committee today.

**Sarah Chen:** Today as in this CapEx meeting?

**Tomas Riera:** Today as in the two PM call, yes.

**Sarah Chen:** Okay. I'll make sure I'm on that call. What's the cost impact?

**Marcus Webb:** Eight A100 nodes — that's about eighty-two thousand per month.

**Priya Nair:** That's significant. Is there a staged option?

**Tomas Riera:** We looked at four nodes as a first tranche. It gets us to roughly seventy-five percent headroom which is okay but it doesn't account for the new model, which pushes us back into the danger zone.

**Marcus Webb:** What if we use spot instances for the overflow?

**Tomas Riera:** We've tried spot for ML workloads before. The interruption rate kills long-running training jobs. Inference is more tolerant of it but we'd need to build out the fallback logic which is probably a two-sprint piece of work.

**Sarah Chen:** Okay. Tomas, come to the CapEx call with both scenarios — full eight nodes and the four-plus-spot hybrid. I want options on the table.

**Tomas Riera:** Will do.

[10:54 AM]

**Lily Marchetti:** Sarah, we're about halfway through time. Just flagging.

**Sarah Chen:** Thanks Lily. Let's pick up pace slightly. Ravi — the Commerce DB storage situation. You wanted to address that?

**Ravi Subramaniam:** Yes. So the prod-rds-commerce-primary instance is at four point eight terabytes of eight. Current growth rate is about four hundred gigabytes a month. That means we breach the eight TB limit sometime in Q1 FY2025 if we do nothing.

**Aisha Okonkwo:** That's like February?

**Ravi Subramaniam:** March at the current rate, could be February if Commerce volumes pick up with the holiday season.

**Aisha Okonkwo:** Which they will. Holiday season is our biggest quarter by far.

**Ravi Subramaniam:** Right, so I've drafted a storage auto-scaling policy. Basically we set a threshold at seventy-five percent capacity — so six terabytes — and RDS automatically provisions additional storage. There's no downtime for that operation. I just need DBA review sign-off and then a Marcus-side approval on the cost.

**Marcus Webb:** What's the cost of going from eight to, say, twelve terabytes on that instance?

**Ravi Subramaniam:** About two hundred and forty dollars per month additional. It's negligible.

**Marcus Webb:** I approve it. Let's just do it.

**Ravi Subramaniam:** Can I get that in writing?

**Marcus Webb:** Lily, please note that I verbally approved the Commerce DB storage auto-scaling policy in this meeting.

**Lily Marchetti:** Noted. Ravi, I'll put the action item on you to implement by October twentieth.

**Ravi Subramaniam:** That works.

**Sarah Chen:** Good. That's one of those things we really don't want to hit as an incident. Can you imagine explaining to the board that Commerce went down because a database ran out of disk space?

**Aisha Okonkwo:** I would rather not have to make that call, no.

**Ravi Subramaniam:** It would be a short call.

[Laughter]

**Sarah Chen:** Okay. Keiko — security spend was up forty-five percent year-over-year or something like that?

**Keiko Yamamoto:** Quarter over quarter. Forty-five point six percent. From two twenty-four thousand in Q2 to three twenty-seven thousand in Q3.

**Priya Nair:** That's quite a jump.

**Keiko Yamamoto:** It is. Majority of it was the WAF expansion and the GuardDuty log retention increase following the CVE. We've already walked back the GuardDuty retention — we had it at three hundred and sixty-five days and we've dropped it to ninety, which is still compliant with our security policy but saves about nine thousand five hundred per quarter. The WAF cost is stickier because the rules are still active.

**Sarah Chen:** Do we need all of those WAF rules long-term?

**Keiko Yamamoto:** Most of them, yes. The CVE that triggered this is an ongoing threat pattern. We're not dealing with a one-time thing. I'm reviewing the rule set this month and I think I can retire maybe fifteen to twenty percent of the rules we added, which would save around six to eight thousand per quarter.

**Marcus Webb:** Good to know. I'll factor that into the Q4 forecast.

**Keiko Yamamoto:** Also — separate topic but related to security spend — I want to flag that the Elastic Cloud contract is month-to-month. We're paying about seventy-four thousand a quarter on a rolling basis and there's a reserved tier that would save fourteen thousand per quarter. I've been meaning to do this for two quarters. Can we just — do it?

**Marcus Webb:** I support that. What do you need from me?

**Keiko Yamamoto:** Just the vendor approval to commit to an annual contract. It's a simple paperwork change.

**Marcus Webb:** Consider it approved. Get me the contract and I'll countersign.

**Keiko Yamamoto:** Thank you. That's been on my list for months.

**Devon Holt:** Can I just say — I love it when meetings actually unblock things.

**Sarah Chen:** That's the idea, Devon.

**Tomas Riera:** Can we use this format for all our meetings?

**Sarah Chen:** Ha. Alright — latency. Devon, the p99 numbers looked generally good but there were a couple I wanted to flag.

**Devon Holt:** Sure. So overall p99 improved by eighteen milliseconds quarter-over-quarter across the platform. The Commerce checkout endpoint came down from three twelve to two eighty-nine — that was the query plan optimisation the DBA team did in August.

**Ravi Subramaniam:** Yeah, we rewrote the product lookup query that was getting called on every checkout session. It was doing a full scan on a non-indexed column. Embarrassing in hindsight.

**Aisha Okonkwo:** I noticed. Checkout felt snappier. Users noticed too — our checkout completion rate ticked up in September. Correlation, but still.

**Ravi Subramaniam:** I'll take it.

**Devon Holt:** The two regressions I want to flag. ML scoring endpoint — p99 went from two seven one zero to two eight nine zero. That's about six and a half percent worse.

**Tomas Riera:** Yeah, that's the model size. The upgrade we shipped in August added about twelve percent to the model parameter count. We knew p99 would go up, we just — it went up a bit more than we modelled. The streaming work I mentioned should bring that back down significantly in Q4. I'm expecting it to more than recover the regression.

**Devon Holt:** The other one is the events ingest endpoint — p99 went from fifty-eight to sixty-one milliseconds. Small in absolute terms but it's correlated with Kafka partition rebalance events. Jordan, is that something your team is investigating?

**Jordan Blake:** We are, yes. It's a Kafka topic configuration thing — we have too many partitions on the events topic for the current consumer group size and rebalances are more frequent than they should be. The fix is actually reducing partition count which feels counterintuitive but the math works out. We're planning to do that in the November maintenance window.

**Devon Holt:** That's all I had on latency. The auth and analytics endpoints both improved meaningfully so net-net it's a good story.

**Sarah Chen:** The analytics dashboard is still very slow though. P99 at eighteen forty-one milliseconds.

**Jordan Blake:** Yeah. The dashboard queries are genuinely complex — we're aggregating across billions of events. The BigQuery slot expansion actually helped, it came down from two one zero four to eighteen forty-one. But eighteen seconds p99 for a dashboard query is still not great.

**Sarah Chen:** Users are feeling that?

**Jordan Blake:** Internal users mostly. The customer-facing dashboards use pre-aggregated views so they're fast. The internal analyst tool is the slow one and those users are — patient. But we do want to get below ten seconds p99 by end of H1 FY2025.

**Sarah Chen:** Put that in the roadmap.

**Jordan Blake:** It's already in there.

[11:28 AM]

**Priya Nair:** Sarah, I want to make sure we have time for the Q4 budget conversation before we close.

**Sarah Chen:** Yes, good point. Let's do five more minutes on reliability and then we'll do Q4 in the last twenty. Devon, anything else you need to flag?

**Devon Holt:** One thing on DORA — the Internal Admin Portal came in at a Medium classification, change failure rate of twelve point eight percent. That's quite high. It's basically because that service is maintained by a rotating roster of engineers who each have their own approach to deployments and there's no dedicated owner. I'd like to propose assigning a single named owner for Admin Portal going forward.

**Sarah Chen:** Who would that be?

**Devon Holt:** I was thinking someone from the Developer Experience team, since they're the primary users.

**Lily Marchetti:** I can follow up with the Dev Ex lead offline if you want to propose it, Devon?

**Devon Holt:** Yeah, let's do that. Thanks Lily.

**Sarah Chen:** Good. Okay — Tomas, anything else on ML before we move?

**Tomas Riera:** Just one thing. The ML scoring p99 — I want to flag that the streaming implementation we're planning is not just a latency fix. It fundamentally changes how the endpoint works. Instead of waiting for the full inference to complete before returning, we'll stream tokens as they're generated. That has implications for how downstream services consume the response.

**Aisha Okonkwo:** Does that affect the Commerce recommendation widget?

**Tomas Riera:** It would require a small change on your end to consume the streaming response. It's maybe a day of work. But the latency improvement for your users would be significant — instead of waiting two point nine seconds for the first token, they'd start seeing recommendations in under five hundred milliseconds.

**Aisha Okonkwo:** I want that. Let's make sure we coordinate the rollout.

**Tomas Riera:** Absolutely. I'll set up a design review once we have the API spec drafted.

**Sarah Chen:** Okay. Q4 budget. Priya, you want to frame this?

**Priya Nair:** Sure. So the approved Q4 budget is two point seven five million — slightly above Q3 budget given we're heading into a heavier compute period with the holiday season. Based on current run rates and the cost-saving actions we've committed to today and in the report, I'm forecasting Q4 actual spend at somewhere between two point six eight and two point seven three million. Which puts us comfortably within budget assuming the savings actions complete on schedule.

**Marcus Webb:** The biggest variable is the GPU nodes. If Tomas gets the full eight nodes approved today that adds about eighty-two thousand per month which is roughly two hundred and forty thousand for the quarter — but that's already in the Q4 budget.

**Priya Nair:** Correct, it was included in the planning assumption. The four-node option would give us about a hundred and twenty thousand in additional headroom.

**Tomas Riera:** I'll go in hard for the full eight.

**Sarah Chen:** Good. The other variable is whether the NAT gateway optimisation lands. Devon, if that saves twenty-five to thirty-five thousand, that's directly positive to the bottom line.

**Devon Holt:** October fifteenth is our target. I'm confident in that date.

**Sarah Chen:** Okay. Any other Q4 budget risks?

**Marcus Webb:** The Snowflake commitment is sitting dormant — six hundred thousand commitment that we haven't activated yet. The Data team pushed the Redshift migration to Q1 FY2025. We need to make sure we're not incurring penalties.

**Priya Nair:** I've already been in touch with the Snowflake account team. There's a hundred-and-twenty-day activation window from contract signing, which means we need to activate by the first of February at the latest. If Jordan's team commits to starting the migration in January, we're fine. If it slips further, we have a problem.

**Jordan Blake:** January start is in our plan. The Redshift team has some year-end reporting work that runs through December so January second week is the earliest we can realistically kick it off.

**Priya Nair:** That works. I'll confirm that timeline in writing with Snowflake today.

**Sarah Chen:** Good. Okay — let's do a quick round of Q4 commitments, just to make sure everyone leaves with a clear picture of what they own.

**Marcus Webb:** NAT gateway optimisation by October fifteenth. Redshift RI purchase at the CapEx committee October twenty-second. Elastic Cloud contract signed.

**Devon Holt:** Alert audit — thirty-five rules by October thirty-first, remaining six by mid-November. Staging-to-prod parity audit by November fifteenth. Admin Portal owner assignment by next week.

**Tomas Riera:** GPU node request at today's CapEx call. ML scoring streaming implementation — API spec by end of October, implementation targeting end of Q4. Coordination with Aisha on Commerce rollout.

**Jordan Blake:** BigQuery slot purchase processed today. November maintenance window for Kafka partition rebalance fix. Data Ingestion integration test coverage starting November first.

**Keiko Yamamoto:** Elastic Cloud reserved tier switch this month. WAF rule review and cleanup by end of October. GuardDuty retention already done.

**Ravi Subramaniam:** Commerce DB storage auto-scaling policy implemented by October twentieth. DBA review and sign-off for Redshift node spec — already done.

**Aisha Okonkwo:** Coordinate with Tomas on ML streaming rollout plan. Check in with Marcus on Commerce resource tagging in September.

**Priya Nair:** Update Q4 and H1 FY25 budget models this week. Confirm Snowflake activation timeline with account team today.

**Lily Marchetti:** I'll compile all of these into the action tracker and send it out within the hour.

**Sarah Chen:** Perfect. Okay — any other business before we close?

**Devon Holt:** Just one thing. The mid-quarter reliability review — November fifteenth was mentioned in the report. Are we doing that as a full call or just a written update?

**Sarah Chen:** Let's do a shorter call. Thirty minutes. Lily, can you schedule that?

**Lily Marchetti:** Already on it.

**Marcus Webb:** Same group?

**Sarah Chen:** Same group plus I'll add the VP of Product this time because I want her to see the DORA metrics. I think it'll be useful context for roadmap prioritisation.

**Keiko Yamamoto:** Good idea.

**Sarah Chen:** Alright, we're at noon. That was actually really productive. Thank you everyone. Tomas, good luck at the CapEx call.

**Tomas Riera:** Thanks. I'll report back in Slack.

**Aisha Okonkwo:** Good luck!

**Ravi Subramaniam:** Thanks for waiting for me at the start, sorry again about the dentist.

**Sarah Chen:** Ravi, no one cares. Take care of your teeth.

[Laughter]

**Devon Holt:** See everyone.

**Keiko Yamamoto:** Bye all.

[Meeting ends — 12:02 PM EDT]

---

## Action Items Captured (Lily Marchetti)

| # | Action | Owner | Due Date | Notes |
|---|---|---|---|---|
| 1 | Confirm Commerce September resource tagging is correct | Devon Holt | 2024-10-14 | Verify EKS node group tags for Sept deployment |
| 2 | Purchase 500 additional BigQuery annual slots | Jordan Blake | 2024-10-10 | Already submitted per Jordan |
| 3 | Submit GPU node CapEx request (8× A100) | Tomas Riera | 2024-10-10 | Present both 8-node and 4-node+spot scenarios |
| 4 | Alert audit — 35 rules remediated | Devon Holt | 2024-10-31 | Remaining 6 by mid-November |
| 5 | Alert audit — remaining 6 rules | Devon Holt | 2024-11-10 | — |
| 6 | Commerce DB storage auto-scaling policy | Ravi Subramaniam | 2024-10-20 | Marcus verbally approved in meeting |
| 7 | Elastic Cloud reserved tier contract | Keiko Yamamoto | 2024-10-31 | Marcus will countersign |
| 8 | WAF rule review and cleanup | Keiko Yamamoto | 2024-10-31 | Est. save $6–8k/quarter |
| 9 | Staging-to-prod parity audit | Devon Holt | 2024-11-15 | All services, not just incident-affected |
| 10 | NAT gateway egress routing optimisation | Devon Holt | 2024-10-15 | Est. save $25–35k/quarter |
| 11 | Redshift RI node purchase approval | Marcus Webb | 2024-10-22 | CapEx committee |
| 12 | Confirm Snowflake activation timeline | Priya Nair | 2024-10-10 | Deadline 1 Feb 2025 or penalties apply |
| 13 | Update Q4 and H1 FY25 budget models | Priya Nair | 2024-10-14 | Include BigQuery discount correction |
| 14 | ML scoring streaming — API spec | Tomas Riera | 2024-10-31 | Coordinate with Aisha for Commerce widget |
| 15 | Admin Portal — assign named owner | Devon Holt / Lily Marchetti | 2024-10-17 | Propose Dev Ex team lead |
| 16 | Kafka partition rebalance fix | Jordan Blake | 2024-11-15 | November maintenance window |
| 17 | Data Ingestion integration test coverage | Jordan Blake | 2024-11-01 | Start date, 6-week project |
| 18 | Mid-quarter reliability review — schedule 30 min call | Lily Marchetti | 2024-10-11 | 15 November, same group + VP Product |
