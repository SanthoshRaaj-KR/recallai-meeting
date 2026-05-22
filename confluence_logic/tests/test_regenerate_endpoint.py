"""Wave 0 RED test scaffold for the Phase 10 Regenerate endpoint (PROP-V2-05).

Phase 10 — auto-propose-pipeline-quality-redesign-v2.

The Phase 10 regenerate endpoint (D-08) re-runs the structure-aware
drafter against the LIVE Confluence page (forced REST fetch, bypassing
RAG cache for the affected page only) and replaces the proposal in
Supabase if the new card passes the full Phase 10 grounding gate. The
endpoint module is the existing ``confluence_logic.review.api``; this
file is RED in Wave 0 because the Phase 10 pipeline (StructureAware
drafter + GroundingGate + dispatcher) has not yet shipped.
"""
import pytest

pytestmark = pytest.mark.asyncio

# Importing the review API module itself succeeds — it already exists.
# The RED gate comes from importing the Phase 10 helpers the
# regenerate endpoint will use; until Wave 1 ships them, the test
# bodies pytest.fail with a Wave 0 marker.
import confluence_logic.review.api as review_api  # noqa: F401, E402
from confluence_logic.agents.grounding_gate import check_grounding  # noqa: F401, E402
from confluence_logic.agents.structure_aware_drafter import (  # noqa: F401, E402
    draft_operation,
)


async def test_regenerate_replaces_proposal_in_place():
    """PROP-V2-05: ``POST /sessions/{sid}/review/regenerate/{pid}`` replaces the existing proposal row in Supabase rather than inserting a new one."""
    pytest.fail("Wave 0 RED — Phase 10 regenerate endpoint pending (Wave 2)")


async def test_regenerate_invokes_grounding_gate():
    """PROP-V2-05: the regenerated card MUST be passed through ``check_grounding`` before persistence; if it fails, the original proposal is preserved and the failure reason surfaces to the UI."""
    pytest.fail("Wave 0 RED — Phase 10 regenerate endpoint pending (Wave 2)")


async def test_regenerate_bypasses_graph_cache_for_affected_page_only():
    """PROP-V2-05: regenerate calls ``connector.fetch_page_html(page_id, force_live=True)`` and triggers ``refresh_page_in_graph`` for that one page only — other pages keep their cached graph entries."""
    pytest.fail("Wave 0 RED — Phase 10 regenerate endpoint pending (Wave 2)")
