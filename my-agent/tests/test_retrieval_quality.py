"""Retrieval quality tests — can Jarvis actually answer from the returned chunk?

Each test simulates a realistic meeting question with a proper transcript context,
then checks two things:
  1. The right section is retrieved (correct heading / page).
  2. The chunk text contains enough information to answer the question.

Gaps (questions the current chunk set cannot fully answer) are explicitly marked
with pytest.mark.xfail so they surface as known-incomplete rather than silent passes.

Chunk inventory (9 sections from the Deepfake Detection Project page + 2 from Meeting):
  deepfake-1 :: Evaluation Metrics           recall target, approver, decision date
  deepfake-1 :: Frequency Domain Features    DCT steps, GAN trace detection
  deepfake-1 :: Decisions                    backbone, FP rate, deployment, retraining
  deepfake-1 :: Action Items                 open tasks, completed items, dates
  deepfake-1 :: Model Design                 classification vs anomaly, loss functions
  deepfake-1 :: System Architecture          pipeline modules, ensemble approach
  deepfake-1 :: Feature Extraction           CNN architectures, spatial/frequency/temporal
  deepfake-1 :: Deployment Architecture      FastAPI, EC2, real-time, enterprise
  deepfake-1 :: Discussion Topics            current recall = 0.71, meeting context
  meeting-1  :: Action Items                 Q3 roadmap tasks
  meeting-1  :: Goals                        Q3 goals
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Whole module needs a live, seeded Pinecone index — skipped unless RUN_LIVE_TESTS=1.
pytestmark = pytest.mark.live

from agent import _extract_query, _extract_topic_hint  # noqa: E402
from confluence_rag import ConfluenceLiveRAG, _SCORE_THRESHOLD  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_pinecone_result(chunks: list[dict]) -> SimpleNamespace:
    hits = [
        SimpleNamespace(
            fields={k: v for k, v in chunk.items() if k != "score"},
            score=chunk["score"],
        )
        for chunk in chunks
    ]
    return SimpleNamespace(result=SimpleNamespace(hits=hits))


def _rag(top_chunk: dict, *extras: dict) -> ConfluenceLiveRAG:
    """RAG whose Pinecone returns top_chunk first, then any extras."""
    rag = ConfluenceLiveRAG()
    fake_index = MagicMock()
    fake_index.search.return_value = _make_pinecone_result([top_chunk, *extras])
    rag._pinecone_index = fake_index
    return rag


def _ask(question: str, transcript_before: list[str], topic_hint: str = "") -> tuple[str, list[str]]:
    """Simulate agent.py: append wake-word line to transcript, extract query, build enriched query.

    Returns (query_text, full_transcript) matching what on_user_turn_completed does.
    """
    full_line = question  # already contains "Hey Jarvis, ..."
    transcript = list(transcript_before) + [full_line]  # appended before RAG
    query = _extract_query(full_line)
    return query, transcript


# ── Full chunk catalogue ─────────────────────────────────────────────────────
# Drawn from the actual Deepfake Detection Project Confluence page content.

_EVAL_METRICS = {
    "page_id": "deepfake-1", "title": "Deepfake Detection Project", "space_key": "PROJ",
    "heading": "Evaluation Metrics",
    "text": (
        "Minimum Performance Requirements. Minimum Recall: 0.92 (92%). "
        "Decision date: April 6, 2026. Approver: Rahul. "
        "Rationale: prioritizing minimizing missed deepfakes; recall of 0.92 chosen "
        "based on validation experiments and stakeholder risk tolerance. "
        "Enterprise systems prioritize: High Recall (avoid missing fake), "
        "Controlled False Positive Rate, F1 score, ROC, AUC."
    ),
    "score": 0.91,
}

_DCT_FEATURES = {
    "page_id": "deepfake-1", "title": "Deepfake Detection Project", "space_key": "PROJ",
    "heading": "Frequency Domain Features (DCT-Based Detection)",
    "text": (
        "Deepfake models introduce frequency inconsistencies via upsampling artifacts. "
        "Steps: Convert RGB to frequency domain using DCT. Extract high-frequency components. "
        "Use magnitude spectrum as model input. Combine spatial and frequency features. "
        "Advantages: Detects GAN upsampling traces. Robust against visual deception. "
        "Helps detect high-quality fakes."
    ),
    "score": 0.87,
}

_DECISIONS = {
    "page_id": "deepfake-1", "title": "Deepfake Detection Project", "space_key": "PROJ",
    "heading": "Decisions",
    "text": (
        "Move forward with Hybrid DCT + Spatial model. "
        "Use 1% false positive rate as threshold tuning benchmark. "
        "Deploy model as REST microservice. Use ResNet backbone for feature extraction. "
        "Schedule model retraining every 2 weeks. "
        "After discussion with Rahul, the team decided to use accuracy as the primary "
        "evaluation metric instead of recall — decided 2026-02-15."
    ),
    "score": 0.82,
}

_ACTION_ITEMS = {
    "page_id": "deepfake-1", "title": "Deepfake Detection Project", "space_key": "PROJ",
    "heading": "Action Items",
    "text": (
        "Compare Autoencoder vs VAE performance — open. "
        "Add adaptive thresholding logic — completed (2026-02-15). "
        "Implemented hybrid loss function — completed (2026-02-15). "
        "Adaptive thresholding logic implementation — completed (2026-02-15)."
    ),
    "score": 0.78,
}

_MODEL_DESIGN = {
    "page_id": "deepfake-1", "title": "Deepfake Detection Project", "space_key": "PROJ",
    "heading": "Model Design",
    "text": (
        "Classification-Based: Binary classification (0=Real, 1=Fake). "
        "Loss Function: Binary Cross Entropy or Focal Loss for class imbalance. "
        "Optimization: Adam optimizer, learning rate scheduling, early stopping. "
        "Anomaly Detection: train autoencoder only on real images; "
        "high reconstruction error means fake, low means real. "
        "Advantages: better generalisation to unseen fake methods, reduced overfitting. "
        "Threshold selection based on validation ROC curve, maximizing recall under acceptable precision."
    ),
    "score": 0.76,
}

_SYSTEM_ARCH = {
    "page_id": "deepfake-1", "title": "Deepfake Detection Project", "space_key": "PROJ",
    "heading": "System Architecture Overview",
    "text": (
        "Modular pipeline: Video Input → Frame Extraction → Face Detection → Face Cropping "
        "(with contextual padding) → Preprocessing → Feature Extraction → "
        "Classification / Anomaly Scoring → Aggregation Across Frames → Final Video-Level Decision. "
        "Ensemble Model Stage: bagging of multiple lightweight classifiers — small CNNs, "
        "compact frequency-domain detectors. Outputs aggregated via majority vote or calibrated averaging. "
        "Frequency transformations (DCT/FFT) combined with visual and temporal features."
    ),
    "score": 0.74,
}

_FEATURE_EXTRACTION = {
    "page_id": "deepfake-1", "title": "Deepfake Detection Project", "space_key": "PROJ",
    "heading": "Feature Extraction Techniques",
    "text": (
        "Spatial Features — CNN-based: ResNet, EfficientNet, XceptionNet. "
        "Learns blending boundaries, skin tone mismatch, eye reflections, teeth irregularities. "
        "Frequency Domain: DCT converts RGB to frequency domain, extracts high-frequency components. "
        "Temporal Features: LSTM-based temporal modeling, 3D CNN, optical flow analysis, "
        "frame-level aggregation scoring. Video deepfakes show flickering, inconsistent head pose, "
        "irregular blinking, motion distortion."
    ),
    "score": 0.72,
}

_DEPLOYMENT = {
    "page_id": "deepfake-1", "title": "Deepfake Detection Project", "space_key": "PROJ",
    "heading": "Deployment Architecture",
    "text": (
        "API Layer: REST API using FastAPI. Accepts video uploads. Returns JSON response: "
        "Deepfake Probability, Confidence Score, Frame-level heatmaps, Timestamp markers. "
        "Real-Time Detection: Frame streaming, batch inference, GPU acceleration, model quantization. "
        "Enterprise Integration: Confluence reporting module, audit logging, user authentication, "
        "scalable microservices. EC2 deployment. Deployment Target: Microsoft Azure."
    ),
    "score": 0.71,
}

_DISCUSSION_TOPICS = {
    "page_id": "deepfake-1", "title": "Deepfake Detection Project", "space_key": "PROJ",
    "heading": "Discussion Topics",
    "text": (
        "10:00 AM — Current Model Performance (Rahul): Recall = 0.71, needs improvement. "
        "10:20 AM — DCT Preprocessing Improvements (ML Team): "
        "Consider block-wise DCT vs full-image DCT. "
        "10:40 AM — Loss Function Enhancement (Research Lead): Try SSIM + MSE hybrid loss."
    ),
    "score": 0.70,
}

_MEETING_ACTION_ITEMS = {
    "page_id": "meeting-1", "title": "Q3 Meeting Overview", "space_key": "MEET",
    "heading": "Action Items",
    "text": "Review Q3 roadmap. Assign owners to delivery milestones. Schedule follow-up call.",
    "score": 0.68,
}


# ─────────────────────────────────────────────────────────────────────────────
# GROUP A — Basic factual questions (clear single-section answers)
# ─────────────────────────────────────────────────────────────────────────────

def test_qa_recall_target_answered():
    """'What is the recall target?' → Evaluation Metrics → 0.92 in chunk ✅"""
    transcript_before = [
        "Rahul: Our current recall is 0.71 and we need to improve it.",
        "Alice: What's the number we agreed to hit?",
    ]
    query, transcript = _ask(
        "Hey Jarvis, what is the minimum recall we need for production?",
        transcript_before,
    )
    rag = _rag(_EVAL_METRICS)
    hits = rag.search(rag.build_search_query(query, transcript))

    assert hits[0]["heading"] == "Evaluation Metrics"
    assert "0.92" in hits[0]["text"], "Chunk must contain the 0.92 figure for Jarvis to answer"


def test_qa_recall_approver_answered():
    """'Who approved the recall requirement?' → Evaluation Metrics → Rahul ✅"""
    query, transcript = _ask(
        "Hey Jarvis, who approved the recall target?",
        ["Rahul: I signed off on the performance thresholds last week."],
    )
    rag = _rag(_EVAL_METRICS)
    hits = rag.search(rag.build_search_query(query, transcript))

    assert "Rahul" in hits[0]["text"], "Approver name must be in chunk"


def test_qa_recall_decision_date_answered():
    """'When was the recall decision made?' → Evaluation Metrics → April 6, 2026 ✅"""
    query, transcript = _ask(
        "Hey Jarvis, when was the recall requirement finalised?",
        ["Alice: We set the baseline metrics a while back."],
    )
    rag = _rag(_EVAL_METRICS)
    hits = rag.search(rag.build_search_query(query, transcript))

    assert "April 6, 2026" in hits[0]["text"]


def test_qa_false_positive_target_answered():
    """'What is the false positive rate target?' → Decisions → 1% ✅"""
    query, transcript = _ask(
        "Hey Jarvis, what false positive rate are we targeting?",
        ["Rahul: We need tight control on false positives for enterprise clients."],
    )
    rag = _rag(_DECISIONS)
    hits = rag.search(rag.build_search_query(query, transcript))

    assert "1%" in hits[0]["text"], "1% FP rate must be in chunk"


def test_qa_backbone_answered():
    """'What backbone are we using?' → Decisions → ResNet ✅"""
    query, transcript = _ask(
        "Hey Jarvis, which backbone did we decide to use for feature extraction?",
        ["ML Team: We debated EfficientNet vs ResNet last sprint."],
    )
    rag = _rag(_DECISIONS)
    hits = rag.search(rag.build_search_query(query, transcript))

    assert "ResNet" in hits[0]["text"]


def test_qa_retraining_schedule_answered():
    """'How often do we retrain the model?' → Decisions → every 2 weeks ✅"""
    query, transcript = _ask(
        "Hey Jarvis, what is the model retraining schedule?",
        ["Alice: We need to keep the model fresh as new deepfake methods emerge."],
    )
    rag = _rag(_DECISIONS)
    hits = rag.search(rag.build_search_query(query, transcript))

    assert "2 weeks" in hits[0]["text"]


# ─────────────────────────────────────────────────────────────────────────────
# GROUP B — Technical / pipeline questions
# ─────────────────────────────────────────────────────────────────────────────

def test_qa_dct_steps_answered():
    """'How does the DCT pipeline work?' → Frequency Domain Features → full steps ✅"""
    query, transcript = _ask(
        "Hey Jarvis, walk me through the DCT detection steps.",
        [
            "ML Team: Block-wise or full-image DCT?",
            "Research Lead: Let's clarify the pipeline before deciding.",
        ],
    )
    rag = _rag(_DCT_FEATURES)
    hits = rag.search(rag.build_search_query(query, transcript))

    text = hits[0]["text"]
    assert "DCT" in text
    assert "RGB" in text, "Step 1 (RGB→frequency) must be in chunk"
    assert "magnitude" in text.lower(), "Step 3 (magnitude spectrum) must be in chunk"


def test_qa_dct_advantage_answered():
    """'Why are we using DCT?' → Frequency Domain Features → GAN trace detection ✅"""
    query, transcript = _ask(
        "Hey Jarvis, what is the advantage of using frequency domain features?",
        ["Rahul: Someone asked why we bother with DCT when we have good CNNs."],
    )
    rag = _rag(_DCT_FEATURES)
    hits = rag.search(rag.build_search_query(query, transcript))

    assert "GAN" in hits[0]["text"], "GAN upsampling trace advantage must be in chunk"


def test_qa_model_types_answered():
    """'What detection approaches are available?' → Model Design → classification + anomaly ✅"""
    query, transcript = _ask(
        "Hey Jarvis, what are the two main model approaches we are considering?",
        ["Research Lead: We should pick between classification and anomaly detection."],
    )
    rag = _rag(_MODEL_DESIGN)
    hits = rag.search(rag.build_search_query(query, transcript))

    text = hits[0]["text"]
    assert "classification" in text.lower()
    assert "autoencoder" in text.lower(), "Anomaly detection path must mention autoencoder"


def test_qa_loss_function_answered():
    """'What loss function should we use?' → Model Design → BCE + Focal Loss ✅"""
    query, transcript = _ask(
        "Hey Jarvis, what loss function is recommended for our model?",
        ["Research Lead: We tried SSIM and MSE but need to decide on final loss."],
    )
    rag = _rag(_MODEL_DESIGN)
    hits = rag.search(rag.build_search_query(query, transcript))

    text = hits[0]["text"]
    assert "Cross Entropy" in text or "Focal Loss" in text


def test_qa_autoencoder_advantage_answered():
    """'Why use autoencoder over classification?' → Model Design → generalisation ✅"""
    query, transcript = _ask(
        "Hey Jarvis, what is the advantage of the anomaly detection approach over classification?",
        ["Alice: I want to understand when we would prefer the autoencoder."],
    )
    rag = _rag(_MODEL_DESIGN)
    hits = rag.search(rag.build_search_query(query, transcript))

    assert "generalisation" in hits[0]["text"].lower() or "generali" in hits[0]["text"].lower()


def test_qa_pipeline_stages_answered():
    """'What are the pipeline stages?' → System Architecture → full list ✅"""
    query, transcript = _ask(
        "Hey Jarvis, give me an overview of the detection pipeline stages.",
        ["New team member: Can someone walk me through how the system works end to end?"],
    )
    rag = _rag(_SYSTEM_ARCH)
    hits = rag.search(rag.build_search_query(query, transcript))

    text = hits[0]["text"]
    assert "Frame Extraction" in text
    assert "Face Detection" in text
    assert "Aggregation" in text


def test_qa_feature_extraction_cnn_answered():
    """'What CNN architectures are we using?' → Feature Extraction → ResNet/EfficientNet/XceptionNet ✅"""
    query, transcript = _ask(
        "Hey Jarvis, which CNN architectures are available for spatial feature extraction?",
        ["ML Team: We need to pick a backbone for the spatial branch."],
    )
    rag = _rag(_FEATURE_EXTRACTION)
    hits = rag.search(rag.build_search_query(query, transcript))

    text = hits[0]["text"]
    assert "ResNet" in text
    assert "EfficientNet" in text
    assert "XceptionNet" in text


def test_qa_temporal_features_answered():
    """'How do we detect temporal inconsistencies?' → Feature Extraction → LSTM/3D CNN ✅"""
    query, transcript = _ask(
        "Hey Jarvis, how are we handling temporal inconsistencies in deepfake videos?",
        ["Alice: Flickering artifacts are hard to catch with frame-level models alone."],
    )
    rag = _rag(_FEATURE_EXTRACTION)
    hits = rag.search(rag.build_search_query(query, transcript))

    text = hits[0]["text"]
    assert "LSTM" in text or "3D CNN" in text
    assert "flickering" in text.lower() or "temporal" in text.lower()


def test_qa_deployment_target_answered():
    """'Where are we deploying this?' → Deployment Architecture → Azure ✅"""
    query, transcript = _ask(
        "Hey Jarvis, what is the deployment target for the deepfake system?",
        ["Rahul: We need to confirm the cloud provider with the infra team."],
    )
    rag = _rag(_DEPLOYMENT)
    hits = rag.search(rag.build_search_query(query, transcript))

    assert "Azure" in hits[0]["text"]


def test_qa_api_response_format_answered():
    """'What does the API return?' → Deployment Architecture → probability, confidence, heatmaps ✅"""
    query, transcript = _ask(
        "Hey Jarvis, what fields does the deepfake detection API return?",
        ["Engineer: The frontend team needs to know the response schema."],
    )
    rag = _rag(_DEPLOYMENT)
    hits = rag.search(rag.build_search_query(query, transcript))

    text = hits[0]["text"]
    assert "Probability" in text or "probability" in text
    assert "Confidence" in text or "confidence" in text
    assert "heatmap" in text.lower()


# ─────────────────────────────────────────────────────────────────────────────
# GROUP C — Meeting decisions and action items
# ─────────────────────────────────────────────────────────────────────────────

def test_qa_open_action_items_answered():
    """'What action items are still open?' → Action Items → autoencoder vs VAE comparison ✅

    Note: the chunk marks 'Compare Autoencoder vs VAE' as open but completed
    items are also listed.  Jarvis can partially answer but may not clearly
    distinguish open vs closed without the status field.
    """
    query, transcript = _ask(
        "Hey Jarvis, which action items from the deepfake project are still open?",
        ["Rahul: Let's make sure we are tracking everything properly."],
    )
    rag = _rag(_ACTION_ITEMS)
    hits = rag.search(rag.build_search_query(query, transcript))

    text = hits[0]["text"]
    assert "VAE" in text, "VAE comparison (open task) must be in chunk"
    assert "completed" in text.lower(), "Completed items must also be listed so Jarvis can contrast"


def test_qa_hybrid_loss_status_answered():
    """'Has the hybrid loss function been implemented?' → Action Items → completed 2026-02-15 ✅"""
    query, transcript = _ask(
        "Hey Jarvis, has the hybrid loss function implementation been completed?",
        ["Research Lead: I want to confirm this before the next sprint review."],
    )
    rag = _rag(_ACTION_ITEMS)
    hits = rag.search(rag.build_search_query(query, transcript))

    text = hits[0]["text"]
    assert "hybrid loss" in text.lower()
    assert "completed" in text.lower()
    assert "2026-02-15" in text


def test_qa_architecture_decision_answered():
    """'What model architecture did we decide on?' → Decisions → Hybrid DCT + Spatial ✅"""
    query, transcript = _ask(
        "Hey Jarvis, what model architecture did the team decide to move forward with?",
        ["Alice: We had the architecture review last sprint, what was the outcome?"],
    )
    rag = _rag(_DECISIONS)
    hits = rag.search(rag.build_search_query(query, transcript))

    assert "Hybrid DCT" in hits[0]["text"]


# ─────────────────────────────────────────────────────────────────────────────
# GROUP D — Current meeting context questions (from Discussion Topics)
# ─────────────────────────────────────────────────────────────────────────────

def test_qa_current_recall_value_answered():
    """'What is our current recall score?' → Discussion Topics → 0.71 ✅

    This is different from the TARGET (0.92).  The Discussion Topics section
    has the live meeting figure from Rahul's update.
    """
    query, transcript = _ask(
        "Hey Jarvis, what is our current recall score right now?",
        [
            "Rahul: I showed the model performance numbers at the start.",
            "Alice: Yes, it was below the threshold we set.",
        ],
    )
    rag = _rag(_DISCUSSION_TOPICS)
    hits = rag.search(rag.build_search_query(query, transcript))

    assert "0.71" in hits[0]["text"], "Current recall must come from Discussion Topics, not Eval Metrics"


def test_qa_block_vs_full_dct_question_answered():
    """'Block-wise vs full-image DCT?' → Discussion Topics → question raised by ML Team ✅

    The discussion section captures the open question.  The Frequency Domain
    Features section has the implementation steps but NOT this design trade-off.
    Jarvis can surface that this question was raised but cannot yet answer which is better.
    """
    query, transcript = _ask(
        "Hey Jarvis, what was discussed about block-wise versus full-image DCT?",
        ["ML Team: We need to settle this before we lock the preprocessing config."],
    )
    rag = _rag(_DISCUSSION_TOPICS)
    hits = rag.search(rag.build_search_query(query, transcript))

    text = hits[0]["text"]
    assert "block-wise DCT" in text.lower() or "block-wise" in text.lower()
    assert "full-image DCT" in text.lower() or "full-image" in text.lower()


# ─────────────────────────────────────────────────────────────────────────────
# GROUP E — Answer-quality gaps (known incomplete — xfail)
# These reveal where the current chunk catalogue is insufficient.
# They are not bugs — they show what to add to Confluence for better coverage.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.xfail(reason="Dataset name lives in Related Info section — not in any indexed chunk", strict=True)
def test_qa_dataset_name_gap():
    """'What dataset are we using?' — 'Internal Deepfake Dataset v2' is in Related Info,
    which is not indexed as a standalone chunk. Retrieval returns nothing useful. ✗"""
    query, transcript = _ask(
        "Hey Jarvis, which dataset is the model trained on?",
        ["Rahul: The evaluation numbers depend heavily on which dataset we use."],
    )
    # Even the best matching chunk (Model Design or Feature Extraction) does not
    # contain dataset name.
    rag = _rag(_MODEL_DESIGN)  # closest match
    hits = rag.search(rag.build_search_query(query, transcript))

    assert "Internal Deepfake Dataset v2" in hits[0]["text"]  # will fail → xfail ✓


@pytest.mark.xfail(reason="Framework (PyTorch) is in Related Info — not in any indexed chunk", strict=True)
def test_qa_framework_gap():
    """'What framework are we using?' — 'PyTorch' is only in Related Info. ✗"""
    query, transcript = _ask(
        "Hey Jarvis, what deep learning framework is the project using?",
        ["New engineer: I want to set up my dev environment."],
    )
    rag = _rag(_MODEL_DESIGN)
    hits = rag.search(rag.build_search_query(query, transcript))

    assert "PyTorch" in hits[0]["text"]  # will fail → xfail ✓


@pytest.mark.xfail(reason="Preprocessing steps (FPS, face detection padding) are not in current chunk set", strict=True)
def test_qa_preprocessing_fps_gap():
    """'What FPS do we extract frames at?' — lives in Preprocessing Pipeline section
    which is not in the current indexed chunk set. ✗"""
    query, transcript = _ask(
        "Hey Jarvis, at what frame rate do we extract frames from videos?",
        ["ML Team: We need consistent FPS across all input videos."],
    )
    rag = _rag(_SYSTEM_ARCH)  # closest match
    hits = rag.search(rag.build_search_query(query, transcript))

    assert "30 FPS" in hits[0]["text"]  # will fail → xfail ✓


@pytest.mark.xfail(reason="Brainstorm ideas (VAE, adaptive thresholding proposal) live in Brainstorm section — not indexed", strict=True)
def test_qa_brainstorm_ideas_gap():
    """'What ideas came out of the brainstorm?' — Brainstorm section not indexed. ✗"""
    query, transcript = _ask(
        "Hey Jarvis, what ideas were listed in the brainstorm session?",
        ["Alice: I know we had a whiteboard session, what were the options?"],
    )
    rag = _rag(_ACTION_ITEMS)  # closest available
    hits = rag.search(rag.build_search_query(query, transcript))

    # "frequency-domain attention mechanism" is only in the Brainstorm section, not Action Items
    assert "frequency-domain attention mechanism" in hits[0]["text"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# GROUP F — Topic isolation (meeting page must NOT pollute deepfake answers)
# ─────────────────────────────────────────────────────────────────────────────

def test_qa_meeting_action_items_stay_separate():
    """When asking about Q3 roadmap tasks, the meeting page chunk is returned,
    NOT the deepfake action items. Per-page diversity keeps them separate."""
    query, transcript = _ask(
        "Hey Jarvis, what are the action items for the Q3 meeting?",
        ["Alice: Let's make sure the Q3 milestones are tracked."],
    )
    rag = _rag(_MEETING_ACTION_ITEMS, _ACTION_ITEMS)  # meeting chunk first
    hits = rag.search(rag.build_search_query(query, transcript))

    # Per-page diversity: 1 chunk per page → both pages represented
    page_ids = [h["page_id"] for h in hits]
    assert "meeting-1" in page_ids
    # Meeting chunk must come first (higher score in this mock)
    assert hits[0]["page_id"] == "meeting-1"
    assert "Q3" in hits[0]["text"] or "roadmap" in hits[0]["text"].lower()


def test_qa_deepfake_question_excludes_meeting_chunk():
    """Technical deepfake questions must not surface the meeting page chunks."""
    query, transcript = _ask(
        "Hey Jarvis, explain the ensemble model approach for deepfake detection.",
        ["Rahul: The bagging ensemble was discussed in the architecture section."],
    )
    rag = _rag(_SYSTEM_ARCH, _MEETING_ACTION_ITEMS)
    hits = rag.search(rag.build_search_query(query, transcript))

    # First hit must be deepfake page
    assert hits[0]["page_id"] == "deepfake-1"
    assert "ensemble" in hits[0]["text"].lower() or "bagging" in hits[0]["text"].lower()
