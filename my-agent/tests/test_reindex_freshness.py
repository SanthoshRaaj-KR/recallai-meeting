"""Tests for page_needs_reindex — the incremental RAG-sync freshness decision.

Regression guard for the bug where a page whose stored Pinecone version could not
be read (batch fetch failed, or metadata lacked a version) was silently SKIPPED
instead of re-indexed — leaving it stale even though the sync reported success.
Shared by ConfluenceVectorIndex and PineconeHybridIndex.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from review_pipeline.rag import page_needs_reindex


def test_unchanged_page_is_skipped():
    assert page_needs_reindex(5, 5, is_indexed=True) is False


def test_edited_page_is_reindexed():
    # Confluence bumps version.number on every edit.
    assert page_needs_reindex(6, 5, is_indexed=True) is True


def test_new_page_is_reindexed():
    assert page_needs_reindex(1, None, is_indexed=False) is True


def test_unverifiable_stored_version_is_reindexed():
    # THE BUG: page is indexed but we couldn't read its stored version (Pinecone
    # fetch failed for the batch, or metadata had no version). Must re-index —
    # never silently skip a page we can't verify.
    assert page_needs_reindex(5, None, is_indexed=True) is True


def test_missing_live_version_but_known_stored_is_skipped():
    # Listing lacked a version but we have a stored one — don't churn every sync.
    assert page_needs_reindex(None, 5, is_indexed=True) is False


def test_both_versions_unknown_is_reindexed():
    assert page_needs_reindex(None, None, is_indexed=True) is True


def test_lower_live_version_is_skipped():
    # Only a strictly higher live version means an edit; equal/lower is fresh.
    assert page_needs_reindex(4, 5, is_indexed=True) is False
