"""
Graph RAG module — Neo4j knowledge graph over meeting transcript.

Owns: Neo4j connection lifecycle, real-time entity ingestion from transcript entries,
and context query for general question answering.

Per D-10: Neo4j AuraDB (cloud), configured via NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD env vars.
Per D-11: Node types: Topic, Person, Decision.
Per D-12: Edge types: MENTIONED_BY, RELATED_TO, DECIDED_IN.
Per D-15: This module owns all graph operations.
"""
import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

_driver = None
_openai_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def _get_driver():
    """Lazy singleton Neo4j AsyncDriver. Returns None if NEO4J_URI not set."""
    global _driver
    if _driver is None:
        uri = os.getenv("NEO4J_URI", "")
        if not uri:
            return None
        try:
            from neo4j import AsyncGraphDatabase
            user = os.getenv("NEO4J_USER", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "")
            _driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        except Exception as e:
            logger.warning("Neo4j driver init failed: %s", e)
            return None
    return _driver


# ---------- Entity Extraction ----------

_ENTITY_EXTRACTION_PROMPT = (
    "You are an entity extractor for meeting transcripts.\n"
    "Extract from the spoken text:\n"
    "- topics: list of subjects discussed (e.g., [\"GPT-4\", \"project deadline\"])\n"
    "- people: list of person names mentioned (e.g., [\"Alice\", \"Bob\"])\n"
    "- decisions: list of decisions made, if any (e.g., [\"use React for frontend\"])\n"
    "Return ONLY valid JSON: {\"topics\": [...], \"people\": [...], \"decisions\": [...]}\n"
    "If nothing found in a category, return an empty list.\n"
    "If the text is too short or contains no extractable entities, return "
    "{\"topics\": [], \"people\": [], \"decisions\": []}"
)


async def _extract_entities(participant: str, text: str) -> Dict[str, List[str]]:
    """Extract topics, people, decisions from a transcript entry via LLM."""
    empty: Dict[str, List[str]] = {"topics": [], "people": [], "decisions": []}
    try:
        response = await asyncio.to_thread(
            lambda: _get_client().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": _ENTITY_EXTRACTION_PROMPT},
                    {"role": "user", "content": f"Speaker: {participant}\nText: {text}"},
                ],
                max_tokens=80,
                temperature=0.0,
            )
        )
        raw = (response.choices[0].message.content or "").strip()
        entities = json.loads(raw)
        # Ensure all keys exist
        for key in ("topics", "people", "decisions"):
            if key not in entities or not isinstance(entities[key], list):
                entities[key] = []
        # Add participant as a Person if not already listed
        if participant and participant not in entities["people"]:
            entities["people"].append(participant)
        return entities
    except (json.JSONDecodeError, Exception) as e:
        logger.debug("Entity extraction failed (non-fatal): %s", e)
        # Still record the speaker even on extraction failure
        return {"topics": [], "people": [participant] if participant else [], "decisions": []}


# ---------- Graph Ingestion ----------

async def _upsert_entities(driver, entities: Dict[str, List[str]], speaker: str) -> None:
    """MERGE nodes and edges into Neo4j. Per D-12: MENTIONED_BY, RELATED_TO, DECIDED_IN."""
    import neo4j

    # MERGE Person node for speaker
    if speaker:
        await driver.execute_query(
            "MERGE (p:Person {name: $name}) ON CREATE SET p.first_seen = timestamp() SET p.last_seen = timestamp()",
            {"name": speaker},
            routing_=neo4j.RoutingControl.WRITE,
        )

    # MERGE Topic nodes + MENTIONED_BY edges to speaker
    for topic in entities.get("topics", []):
        if not topic:
            continue
        topic_normalized = topic.strip().title()
        await driver.execute_query(
            "MERGE (t:Topic {name: $name}) ON CREATE SET t.first_seen = timestamp() SET t.last_seen = timestamp()",
            {"name": topic_normalized},
            routing_=neo4j.RoutingControl.WRITE,
        )
        if speaker:
            await driver.execute_query(
                "MATCH (t:Topic {name: $topic_name}) MATCH (p:Person {name: $speaker_name}) MERGE (t)-[:MENTIONED_BY]->(p)",
                {"topic_name": topic_normalized, "speaker_name": speaker},
                routing_=neo4j.RoutingControl.WRITE,
            )

    # MERGE RELATED_TO edges between co-occurring topics
    topics_normalized = [t.strip().title() for t in entities.get("topics", []) if t]
    for i, t1 in enumerate(topics_normalized):
        for t2 in topics_normalized[i + 1:]:
            await driver.execute_query(
                "MATCH (t1:Topic {name: $a}) MATCH (t2:Topic {name: $b}) MERGE (t1)-[:RELATED_TO]-(t2)",
                {"a": t1, "b": t2},
                routing_=neo4j.RoutingControl.WRITE,
            )

    # MERGE Decision nodes + DECIDED_IN edges to first topic (if any)
    for decision in entities.get("decisions", []):
        if not decision:
            continue
        decision_key = decision.strip().lower()[:80]
        await driver.execute_query(
            "MERGE (d:Decision {key: $key}) ON CREATE SET d.text = $text, d.first_seen = timestamp() SET d.last_seen = timestamp()",
            {"key": decision_key, "text": decision},
            routing_=neo4j.RoutingControl.WRITE,
        )
        if speaker:
            await driver.execute_query(
                "MATCH (d:Decision {key: $key}) MATCH (p:Person {name: $speaker}) MERGE (d)-[:MENTIONED_BY]->(p)",
                {"key": decision_key, "speaker": speaker},
                routing_=neo4j.RoutingControl.WRITE,
            )
        if topics_normalized:
            await driver.execute_query(
                "MATCH (d:Decision {key: $key}) MATCH (t:Topic {name: $topic}) MERGE (d)-[:DECIDED_IN]->(t)",
                {"key": decision_key, "topic": topics_normalized[0]},
                routing_=neo4j.RoutingControl.WRITE,
            )

    # MERGE mentioned People nodes (not the speaker — other people referenced in speech)
    for person in entities.get("people", []):
        if not person or person == speaker:
            continue
        await driver.execute_query(
            "MERGE (p:Person {name: $name}) ON CREATE SET p.first_seen = timestamp() SET p.last_seen = timestamp()",
            {"name": person},
            routing_=neo4j.RoutingControl.WRITE,
        )


async def ingest_transcript_entry(entry: Dict[str, Any]) -> None:
    """Fire-and-forget: extract entities from transcript entry and upsert into Neo4j.
    Per D-13: triggered via asyncio.create_task for each transcript entry."""
    driver = _get_driver()
    if driver is None:
        return
    try:
        entities = await _extract_entities(entry.get("participant", ""), entry.get("text", ""))
        await _upsert_entities(driver, entities, entry.get("participant", ""))
    except Exception as e:
        logger.debug("Graph ingest skipped (non-fatal): %s", e)


# ---------- Graph Query ----------

_QUERY_CYPHER = """
MATCH (n)
WHERE toLower(n.name) CONTAINS toLower($keyword)
   OR (n.text IS NOT NULL AND toLower(n.text) CONTAINS toLower($keyword))
WITH n
OPTIONAL MATCH (n)-[r]-(neighbor)
RETURN n, type(r) AS rel_type, neighbor
LIMIT 20
"""


async def _extract_keywords(question: str) -> List[str]:
    """Extract 1-3 entity keywords from the user's question.
    Uses simple word extraction (no LLM call for MVP — fast, zero-latency)."""
    stop_words = {"what", "who", "when", "where", "why", "how", "is", "are", "was", "were",
                  "did", "do", "does", "the", "a", "an", "in", "on", "at", "to", "for",
                  "of", "with", "about", "my", "your", "their", "our", "this", "that",
                  "it", "i", "me", "we", "you", "he", "she", "they", "and", "or", "but",
                  "can", "could", "would", "should", "will", "shall", "has", "have", "had",
                  "been", "be", "not", "no", "from", "by", "just", "also", "think", "suggest",
                  "said", "tell", "told", "ask", "asked", "say"}
    words = question.lower().split()
    keywords = [w.strip("?.,!\"'") for w in words if w.strip("?.,!\"'") not in stop_words and len(w) > 2]
    return keywords[:3]


async def _cypher_query(driver, keywords: List[str]) -> str:
    """Run 1-hop Cypher traversal for each keyword, format results as context string."""
    import neo4j
    results = []
    for keyword in keywords:
        try:
            records, _, _ = await driver.execute_query(
                _QUERY_CYPHER,
                {"keyword": keyword},
                routing_=neo4j.RoutingControl.READ,
            )
            for record in records:
                n = record.data() if hasattr(record, 'data') else {}
                node = n.get("n", {})
                rel = n.get("rel_type", "")
                neighbor = n.get("neighbor", {})
                node_name = node.get("name", "") or node.get("text", "")
                neighbor_name = neighbor.get("name", "") or neighbor.get("text", "") if neighbor else ""
                if node_name and rel and neighbor_name:
                    results.append(f"{node_name} {rel} {neighbor_name}")
                elif node_name:
                    results.append(node_name)
        except Exception as e:
            logger.debug("Cypher query for '%s' failed (non-fatal): %s", keyword, e)

    if not results:
        return ""
    # Deduplicate and format
    unique = list(dict.fromkeys(results))[:10]
    return "In the meeting: " + "; ".join(unique) + "."


async def query_context(question: str) -> str:
    """Return a short context string from the graph relevant to the question.
    Per D-14: extract keywords, run Cypher 1-hop traversal, return formatted context.
    Returns empty string if driver unavailable or query fails."""
    driver = _get_driver()
    if driver is None:
        return ""
    try:
        keywords = await _extract_keywords(question)
        if not keywords:
            return ""
        return await _cypher_query(driver, keywords)
    except Exception as e:
        logger.debug("Graph query skipped (non-fatal): %s", e)
        return ""
