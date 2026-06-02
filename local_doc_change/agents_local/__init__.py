"""Agent modules for the local document change pipeline.

Contains four agents that form the intelligence layer between RAG retrieval
and final change proposals:
- IntentExtractionAgent: extracts document-change intents from meeting transcripts
- EvaluationAgent: scores document section relevance for a given intent
- LocalDocEditorAgent: drafts before/after content edits
- VerifierAgent: verifies quality of proposed edits
"""
