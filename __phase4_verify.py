from confluence_logic.agents.fact_extraction_agent import (
    ChangeIntent, ExtractedFacts, _merge_facts, FACT_EXTRACTION_PROMPT
)

# Test 1: verbatim_content field exists
ci = ChangeIntent(subject='test', action='add', verbatim_content='A, B, C')
assert ci.verbatim_content == 'A, B, C', 'verbatim_content field missing'
print('Test 1 PASS: verbatim_content field present')

# Test 2: _merge_facts dedup keeps LAST on same (subject, action)
chunk1 = ExtractedFacts(change_intents=[ChangeIntent(subject='SOC quarter', action='replace', new_value='Q1')])
chunk2 = ExtractedFacts(change_intents=[ChangeIntent(subject='SOC quarter', action='replace', new_value='Q3')])
merged = _merge_facts([chunk1, chunk2])
assert len(merged.change_intents) == 1, f'Expected 1 intent, got {len(merged.change_intents)}'
assert merged.change_intents[0].new_value == 'Q3', 'Last intent should win (Q3 = final state)'
print('Test 2 PASS: dedup keeps LAST occurrence (Q3 wins over Q1)')

# Test 3: FACT_EXTRACTION_PROMPT has the three new rules
assert 'FINAL STATE ONLY' in FACT_EXTRACTION_PROMPT, 'FINAL STATE ONLY rule missing'
assert 'NO DUPLICATES' in FACT_EXTRACTION_PROMPT, 'NO DUPLICATES rule missing'
assert 'verbatim_content for add/create' in FACT_EXTRACTION_PROMPT, 'verbatim_content rule missing'
print('Test 3 PASS: all three prompt rules present')

# Test 4: DrafterAgent changes
from confluence_logic.agents.drafter_agent import (
    _find_relevant_transcript_window, INTENT_DRAFTER_PROMPT
)

transcript = 'x ' * 1500 + 'concerns: A is slow, B crashes, C loses data ' + 'y ' * 1500
window = _find_relevant_transcript_window(transcript, 'concerns')
assert 'A is slow' in window, f'Window should capture concerns discussion, got: {window[:200]}'
assert len(window) <= 1600, f'Window too large: {len(window)}'
print(f'Test 4 PASS: relevant window captures discussion (len={len(window)})')

assert 'RULE 0' in INTENT_DRAFTER_PROMPT, 'RULE 0 missing from drafter prompt'
assert 'verbatim_content' in INTENT_DRAFTER_PROMPT, 'verbatim_content not in drafter prompt'
rule0_pos = INTENT_DRAFTER_PROMPT.index('RULE 0')
rule1_pos = INTENT_DRAFTER_PROMPT.index('RULE 1')
assert rule0_pos < rule1_pos, f'RULE 0 must come before RULE 1 (positions: {rule0_pos} vs {rule1_pos})'
print('Test 5 PASS: RULE 0 present and before RULE 1 in drafter prompt')

print('\nALL PHASE 4 CHECKS PASSED')
