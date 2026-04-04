---
phase: 02-storage-foundation
plan: "01"
subsystem: storage
tags: [pydantic, models, json, aiofiles, metadata-store, tdd]
dependency_graph:
  requires: []
  provides: [MeetingRecord, ActionItem, MetadataStore]
  affects: [02-02-PLAN.md, Phase-3-ingestion, Phase-4-retrieval]
tech_stack:
  added: [storage package]
  patterns: [pydantic-v2-models, aiofiles-async-io, pathlib, tdd-red-green]
key_files:
  created:
    - storage/__init__.py
    - storage/models.py
    - storage/metadata_store.py
    - tests/test_models.py
    - tests/test_metadata_store.py
  modified:
    - .env.example
decisions:
  - "All timestamp fields (start_ts, end_ts, summarized_at) are Python int — no datetime objects in model, enforced by Pydantic type annotation"
  - "MetadataStore raises FileNotFoundError on read of nonexistent path — explicit error, not None sentinel"
  - "status field defaults to 'complete'; 'partial' used for mid-meeting summaries per research pitfall prevention"
metrics:
  duration: "2 minutes"
  completed_date: "2026-04-04"
  tasks_completed: 1
  files_created: 5
  files_modified: 1
---

# Phase 2 Plan 1: MeetingRecord Pydantic Model + MetadataStore Summary

**One-liner:** MeetingRecord Pydantic v2 model (15 fields, Unix epoch timestamps) + async MetadataStore using aiofiles for JSON persistence at meetings/{channel_id}/{meeting_id}.json

## What Was Built

### storage/models.py

Two Pydantic v2 models serving as the single canonical schema for both JSON disk storage and Pinecone vector metadata:

- `ActionItem` — owner (str), task (str), due (Optional[str])
- `MeetingRecord` — 15 fields including all required meeting data; all timestamp fields are `int` (Unix epoch), never `datetime`; `status` defaults to `"complete"` with `"partial"` for mid-meeting summaries

Key design constraints enforced:
- `start_ts: int` — not `datetime` — required for Pinecone `$gte`/`$lte` filter operators
- All fields are flat scalars or lists of scalars — no nested JSON blobs (prevents Pinecone schema mismatch)
- `status` field prevents partial meeting data from being indexed as complete records

### storage/metadata_store.py

`MetadataStore` class with three async methods:

- `write(record)` — serializes to JSON via `model_dump_json(indent=2)`, creates `{base_dir}/{channel_id}/{meeting_id}.json`, returns Path
- `read(channel_id, meeting_id)` — reads and deserializes via `model_validate_json()`, raises `FileNotFoundError` for nonexistent meetings
- `list_by_channel(channel_id)` — returns list of meeting_id strings, empty list if channel dir missing

Uses `aiofiles` for all file I/O (non-blocking), `pathlib.Path` for all path operations.

### Tests

25 tests written and passing:
- `tests/test_models.py` — 14 tests: ActionItem fields, round-trips, MeetingRecord minimal/full validation, timestamp type enforcement, status field behavior
- `tests/test_metadata_store.py` — 11 async tests: write path/file creation, read round-trip, action item deserialization, FileNotFoundError, list_by_channel isolation

## Verification Results

```
python -m pytest tests/test_models.py tests/test_metadata_store.py -v
25 passed in 0.05s

python -c "from storage.models import MeetingRecord, ActionItem; print('OK')" → OK
python -c "from storage.metadata_store import MetadataStore; print('OK')" → OK
grep -c 'start_ts: int' storage/models.py → 1
```

## TDD Commits

| Phase | Commit | Description |
|-------|--------|-------------|
| RED   | 40b2656 | test(02-01): add failing tests for MeetingRecord and MetadataStore |
| GREEN | 458209e | feat(02-01): implement MeetingRecord Pydantic model and MetadataStore |

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None — all fields are wired through model validation and JSON round-trip. No placeholder data.

## Self-Check: PASSED

- storage/__init__.py: FOUND
- storage/models.py: FOUND
- storage/metadata_store.py: FOUND
- tests/test_models.py: FOUND
- tests/test_metadata_store.py: FOUND
- Commit 40b2656: FOUND
- Commit 458209e: FOUND
- All 25 tests: PASS
