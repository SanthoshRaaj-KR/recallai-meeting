# Stress-corpus test guide (write your own transcript)

The default sandbox folder `stress_corpus/docs` holds **100 synthetic company documents** (.md / .txt / .docx / .odt). A change only becomes a proposal when the transcript names a **concrete change to a topic that exists in these docs**. Pick any lines below (or rephrase them) to build your own meeting transcript — every distinct change becomes its own review card.

**Pattern:** `In the <Org> <document>, change the <specific item> from <old> to <new>.` (or `… remove the <section> section.`)

## Edits — number / policy changes (16 available)

- For the Ironbloom information security policy, change the cold-storage archival window from 47 days to 30 days.  _( md · 47 days → 30 days )_
- For the Junivault people operations handbook, change the incident bridge auto-timeout from 53 minutes to 20 minutes.  _( txt · 53 minutes → 20 minutes )_
- For the Kesterline engineering handbook, change the privileged session recording limit from 236 gigabytes to 500 gigabytes.  _( docx · 236 gigabytes → 500 gigabytes )_
- For the Larkspire customer support sop, change the vendor reassessment interval from 19 months to 12 months.  _( odt · 19 months → 12 months )_
- For the Morrowgate product operations guide, change the refund auto-approval ceiling from 418 dollars to 250 dollars.  _( md · 418 dollars → 250 dollars )_
- For the Netherby finance and procurement policy, change the beta cohort size cap from 1,740 participants to 5,000 participants.  _( txt · 1,740 participants → 5,000 participants )_
- For the Oakmeridian it and infrastructure policy, change the learning stipend from 2,360 dollars to 3,000 dollars.  _( docx · 2,360 dollars → 3,000 dollars )_
- For the Pallasvane data governance and privacy, change the purchase order fast-track threshold from 7,950 dollars to 10,000 dollars.  _( odt · 7,950 dollars → 10,000 dollars )_
- For the Cindergate product operations guide, change the warm standby failover budget from 73 seconds to 30 seconds.  _( md · 73 seconds → 30 seconds )_
- For the Drayveil finance and procurement policy, change the data subject request deadline from 27 days to 30 days.  _( txt · 27 days → 30 days )_
- For the Everwynd it and infrastructure policy, change the artifact retention depth from 184 builds to 50 builds.  _( docx · 184 builds → 50 builds )_
- For the Fenmark data governance and privacy, change the on-call acknowledgement window from 11 minutes to 5 minutes.  _( odt · 11 minutes → 5 minutes )_
- For the Oakmeridian information security policy, change the badge re-enrollment grace from 62 hours to 24 hours.  _( md · 62 hours → 24 hours )_
- For the Pallasvane people operations handbook, change the telemetry sampling rate from 3.7 percent to 10 percent.  _( txt · 3.7 percent → 10 percent )_
- For the Quillhaven engineering handbook, change the snapshot replication lag ceiling from 94 seconds to 30 seconds.  _( docx · 94 seconds → 30 seconds )_
- For the Rampart Lyle customer support sop, change the contractor access expiry from 208 days to 90 days.  _( odt · 208 days → 90 days )_

## Removals — delete a whole section (4 available)

- In the Fenmark finance and procurement policy, remove the pager duty carbon-copy rule section.  _( txt )_
- In the Xanthe Reef data governance and privacy, get rid of the on-site smoking shelter policy section.  _( odt )_
- In the Stormfell product operations guide, drop the legacy fax intake procedure section.  _( md )_
- In the Umberlane it and infrastructure policy, delete the floppy disk archival standard section.  _( docx )_

## Rename — propagates across every doc that mentions it (14 docs)

- Company-wide announcement: we are renaming Vantcorex Robotics to Helios Automata. Update the documents to reflect the new name.  _( 'Vantcorex Robotics' → 'Helios Automata', 14 docs )_

## Ready-made multi-change sample (produces 3–4 cards)

```
Quick documentation review from today's meeting — a few concrete updates to lock in.

For the Ironbloom information security policy, change the cold-storage archival window from 47 days to 30 days.

For the Kesterline engineering handbook, change the privileged session recording limit from 236 gigabytes to 500 gigabytes.

For the Larkspire customer support SOP, change the vendor reassessment interval from 19 months to 12 months.

In the Fenmark finance and procurement policy, remove the pager duty carbon-copy rule section.

Everything else stays as-is for now.
```

## Tips

- Be **concrete**: a number, a date, or a named section. Vague intent ('tighten security') → no card.
- You can mix many changes in one transcript; each becomes its own card.
- Want more targets? Open any file in `stress_corpus/docs/` and change any value in any section — the same `In the <doc>, change <item> from X to Y` pattern works for all ~4,800 sections, not just the lines above.
- Renames and section-removals work too (see the lists above).
