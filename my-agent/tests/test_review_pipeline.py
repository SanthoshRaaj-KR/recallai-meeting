import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from review_pipeline.models import ChangeIntent, ExtractedMeeting, PageCandidate
from review_pipeline.pipeline import ProposalPipeline


class FakeConfluenceClient:
    def __init__(self):
        self.updated_html = None
        self.updated_title = None

    async def search_pages(self, query, limit=10):
        if "2nd September" in query or "Q3 meeting plan" in query:
            return [{"page_id": "page-1", "title": "Q3 Meeting Plan"}]
        return []

    async def fetch_page(self, page_id):
        return PageCandidate(
            page_id=page_id,
            title="Q3 Meeting Plan",
            html=(
                "<h2>Timeline</h2>"
                "<p>The Q3 meeting plan is currently scheduled for 2nd September.</p>"
                "<h2>Owners</h2><p>Asha owns the plan.</p>"
            ),
            text=(
                "Timeline\nThe Q3 meeting plan is currently scheduled for 2nd September.\n"
                "Owners\nAsha owns the plan."
            ),
            version=3,
            source="fake",
        )

    async def update_page(self, page_id, html_content, *, title=None, expected_version=None):
        self.updated_html = html_content
        self.updated_title = title
        assert expected_version == 3
        return True

    async def create_page(self, title, html_content, space_key=None):
        return {"id": "new-page"}


def test_pipeline_drafts_exact_date_replacement(monkeypatch):
    pipeline = ProposalPipeline()
    pipeline._client = FakeConfluenceClient()

    async def fake_extract(_transcript, _text, query=None):
        return ExtractedMeeting(
            title="Planning Meeting",
            summary="Q3 plan moved.",
            change_intents=[
                ChangeIntent(
                    instruction="Move Q3 meeting plan from 2nd September to 3rd December",
                    subject="Q3 meeting plan date",
                    target_hint="Q3 meeting plan",
                    old_value="2nd September",
                    new_value="3rd December",
                    action="replace",
                    evidence=["we push Q3 meeting plan to 3rd of December"],
                )
            ],
        )

    monkeypatch.setattr(pipeline, "_extract_meeting", fake_extract)
    monkeypatch.setattr(pipeline, "_generate_page_grounded_candidates", async_no_candidates)
    monkeypatch.setattr(pipeline, "_adversarial_verify", async_passthrough_verify)
    monkeypatch.setattr(pipeline, "_rovo_independent_critic", async_passthrough_critic)
    transcript = [{"participant": "Asha", "text": "We push Q3 meeting plan to 3rd of December."}]

    meeting, proposals = asyncio.run(pipeline.run(session_id="s1", transcript=transcript))

    assert meeting.title == "Planning Meeting"
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal["page_id"] == "page-1"
    assert proposal["change_type"] == "edit"
    assert proposal["edit_mode"] == "replace"
    assert proposal["before_content"] == "2nd September"
    assert proposal["after_content"] == "3rd December"
    assert proposal["confidence"] == "high"
    assert proposal["section_heading"] == "Timeline"


def test_pipeline_returns_empty_without_transcript():
    pipeline = ProposalPipeline()
    meeting, proposals = asyncio.run(pipeline.run(session_id="s1", transcript=[]))

    assert "No transcript" in meeting.summary
    assert proposals == []


async def async_no_candidates(*_args, **_kwargs):
    return []


async def async_passthrough_verify(_meeting, proposals, _text):
    return proposals


async def async_passthrough_critic(_meeting, proposals, _text, query=None):
    return proposals


def test_page_grounded_candidate_can_rescue_missed_extractor(monkeypatch):
    pipeline = ProposalPipeline()
    pipeline._client = FakeConfluenceClient()

    async def missed_extract(_transcript, _text, query=None):
        return ExtractedMeeting(
            title="Planning Meeting",
            summary="Extractor missed the change.",
            change_intents=[],
        )

    async def page_grounded_candidates(_meeting, _text, query=None):
        return [
            ChangeIntent(
                instruction="Move Q3 meeting plan from 2nd September to 3rd December",
                subject="Q3 meeting plan date",
                target_hint="Q3 meeting plan",
                old_value="2nd September",
                new_value="3rd December",
                action="replace",
                evidence=["we push Q3 meeting plan to 3rd of December"],
                source="page_grounded_candidate",
                page_id="page-1",
                page_title="Q3 Meeting Plan",
            )
        ]

    monkeypatch.setattr(pipeline, "_extract_meeting", missed_extract)
    monkeypatch.setattr(pipeline, "_generate_page_grounded_candidates", page_grounded_candidates)
    monkeypatch.setattr(pipeline, "_adversarial_verify", async_passthrough_verify)
    monkeypatch.setattr(pipeline, "_rovo_independent_critic", async_passthrough_critic)
    transcript = [{"participant": "Asha", "text": "We push Q3 meeting plan to 3rd of December."}]

    _meeting, proposals = asyncio.run(pipeline.run(session_id="s1", transcript=transcript))

    assert len(proposals) == 1
    assert proposals[0]["before_content"] == "2nd September"
    assert proposals[0]["after_content"] == "3rd December"


def test_pipeline_records_no_matching_page_diagnostic(monkeypatch):
    pipeline = ProposalPipeline()

    class EmptyConfluenceClient:
        async def search_pages(self, query, limit=10):
            return []

        async def fetch_page(self, page_id):
            raise AssertionError("fetch_page should not be called")

    pipeline._client = EmptyConfluenceClient()

    async def fake_extract(_transcript, _text, query=None):
        return ExtractedMeeting(
            change_intents=[
                ChangeIntent(
                    instruction="Update launch owner from Asha to Ben",
                    subject="launch owner",
                    target_hint="launch plan",
                    old_value="Asha",
                    new_value="Ben",
                    action="replace",
                )
            ]
        )

    monkeypatch.setattr(pipeline, "_extract_meeting", fake_extract)
    monkeypatch.setattr(pipeline, "_generate_page_grounded_candidates", async_no_candidates)
    monkeypatch.setattr(pipeline, "_adversarial_verify", async_passthrough_verify)
    monkeypatch.setattr(pipeline, "_rovo_independent_critic", async_passthrough_critic)

    _meeting, proposals = asyncio.run(
        pipeline.run(session_id="s1", transcript=[{"participant": "Asha", "text": "Owner is now Ben."}])
    )

    assert proposals == []
    assert pipeline.last_diagnostics[0]["reason"] == "no_matching_page_found"


def test_section_level_drafter_refines_additive_proposal(monkeypatch):
    pipeline = ProposalPipeline()
    page = PageCandidate(
        page_id="page-1",
        title="Launch Plan",
        html="<h2>Timeline</h2><p>Launch plan details.</p>",
        text="Timeline\nLaunch plan details.",
    )
    intent = ChangeIntent(
        instruction="Add that launch review is on Friday",
        subject="launch review",
        target_hint="Launch Plan",
        new_value="Launch review is on Friday.",
        action="add",
        evidence=["launch review is on Friday"],
    )

    def fake_section_draft(_intent, _page, _heading, _after, _mode):
        return {
            "edit_mode": "append",
            "section_heading": "Timeline",
            "after_content": "Launch review is on Friday.",
        }

    monkeypatch.setattr(pipeline, "_section_level_draft", fake_section_draft)

    proposal = pipeline._draft_for_page("s1", intent, page, "now", None)

    assert proposal is not None
    assert proposal.section_heading == "Timeline"
    assert proposal.edit_mode == "append"
    assert proposal.after_content == "Launch review is on Friday."


def test_execute_append_inserts_into_selected_section():
    pipeline = ProposalPipeline()
    fake = FakeConfluenceClient()
    pipeline._client = fake
    proposal = {
        "change_type": "edit",
        "page_id": "page-1",
        "page_title": "Q3 Meeting Plan",
        "section_heading": "Timeline",
        "before_content": None,
        "after_content": "Review moved to 3rd December.",
        "edit_mode": "append",
    }

    result = asyncio.run(pipeline.execute(proposal))

    assert result["success"] is True
    assert "Review moved to 3rd December." in fake.updated_html
    assert fake.updated_html.index("Review moved to 3rd December.") < fake.updated_html.index("<h2>Owners</h2>")


def test_execute_replace_handles_text_wrapped_in_block():
    pipeline = ProposalPipeline()
    fake = FakeConfluenceClient()
    pipeline._client = fake
    proposal = {
        "change_type": "edit",
        "page_id": "page-1",
        "page_title": "Q3 Meeting Plan",
        "section_heading": "Timeline",
        "before_content": "The Q3 meeting plan is currently scheduled for 2nd September.",
        "after_content": "The Q3 meeting plan is currently scheduled for 3rd December.",
        "edit_mode": "replace",
    }

    result = asyncio.run(pipeline.execute(proposal))

    assert result["success"] is True
    assert "3rd December" in fake.updated_html
    assert "2nd September" not in fake.updated_html


class TaskConfluenceClient:
    def __init__(self):
        self.updated_html = None
        self.updated_title = None

    async def search_pages(self, query, limit=10):
        if "Q2" in query or "plan" in query.lower():
            return [{"page_id": "page-task", "title": "Quarterly Planning"}]
        return []

    async def fetch_page(self, page_id):
        html = (
            "<h2>Quarterly plans</h2>"
            "<ac:task-list>"
            "<ac:task><ac:task-id>10</ac:task-id>"
            "<ac:task-status>incomplete</ac:task-status>"
            "<ac:task-body>Q2 plan</ac:task-body></ac:task>"
            "<ac:task><ac:task-id>11</ac:task-id>"
            "<ac:task-status>incomplete</ac:task-status>"
            "<ac:task-body>Q3 planning draft</ac:task-body></ac:task>"
            "</ac:task-list>"
        )
        return PageCandidate(
            page_id=page_id,
            title="Quarterly Planning",
            html=html,
            text="Quarterly plans\n[task: incomplete] Q2 plan\n[task: incomplete] Q3 planning draft",
            version=7,
            source="fake",
        )

    async def update_page(self, page_id, html_content, *, title=None, expected_version=None):
        self.updated_html = html_content
        self.updated_title = title
        assert page_id == "page-task"
        assert expected_version == 7
        return True

    async def create_page(self, title, html_content, space_key=None):
        return {"id": "new-page"}


def test_pipeline_drafts_indirect_task_completion(monkeypatch):
    pipeline = ProposalPipeline()
    pipeline._client = TaskConfluenceClient()

    async def fake_extract(_transcript, _text, query=None):
        return ExtractedMeeting(
            title="Planning Meeting",
            summary="Q2 plan completed.",
            change_intents=[
                ChangeIntent(
                    instruction="Mark the Q2 plan as complete",
                    subject="Q2 plan",
                    target_hint="Quarterly plans",
                    new_value="complete",
                    action="complete_task",
                    evidence=["we have completed the Q2 plan completely"],
                )
            ],
        )

    monkeypatch.setattr(pipeline, "_extract_meeting", fake_extract)
    monkeypatch.setattr(pipeline, "_generate_page_grounded_candidates", async_no_candidates)
    monkeypatch.setattr(pipeline, "_adversarial_verify", async_passthrough_verify)
    monkeypatch.setattr(pipeline, "_rovo_independent_critic", async_passthrough_critic)

    _meeting, proposals = asyncio.run(
        pipeline.run(
            session_id="s1",
            transcript=[{"participant": "Asha", "text": "We have completed the Q2 plan completely."}],
        )
    )

    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal["page_id"] == "page-task"
    assert proposal["edit_mode"] == "task_status"
    assert proposal["before_content"] == "[ ] Q2 plan"
    assert proposal["after_content"] == "[x] Q2 plan"


def test_execute_task_status_updates_only_matching_checkbox():
    pipeline = ProposalPipeline()
    fake = TaskConfluenceClient()
    pipeline._client = fake
    proposal = {
        "change_type": "edit",
        "page_id": "page-task",
        "page_title": "Quarterly Planning",
        "section_heading": "Quarterly plans",
        "before_content": "[ ] Q2 plan",
        "after_content": "[x] Q2 plan",
        "edit_mode": "task_status",
    }

    result = asyncio.run(pipeline.execute(proposal))

    assert result["success"] is True
    assert "<ac:task-id>10</ac:task-id><ac:task-status>complete</ac:task-status>" in fake.updated_html
    assert "<ac:task-id>11</ac:task-id><ac:task-status>incomplete</ac:task-status>" in fake.updated_html
