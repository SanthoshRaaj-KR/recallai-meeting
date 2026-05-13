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
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

_driver = None
_openai_client: Optional[OpenAI] = None

# ---------- In-Memory Graph (local, zero-latency) ----------
# Populated on every ingest_transcript_entry call regardless of Neo4j availability.
# query_context reads from here first — eliminates 3 serial AuraDB round-trips per question.
#
# Structure:
#   _local_nodes: {name_lower: {"name": str, "type": "Topic"|"Person"|"Decision", "text": str|None}}
#   _local_edges: list of (from_lower, rel_type, to_lower)
_local_nodes: Dict[str, Dict] = {}
_local_edges: List[tuple] = []


def _local_upsert_node(name: str, node_type: str, text: Optional[str] = None) -> None:
    key = name.lower()
    if key not in _local_nodes:
        _local_nodes[key] = {"name": name, "type": node_type, "text": text}


def _local_upsert_edge(from_name: str, rel: str, to_name: str) -> None:
    edge = (from_name.lower(), rel, to_name.lower())
    if edge not in _local_edges:
        _local_edges.append(edge)


def _local_upsert_entities(entities: Dict[str, List[str]], speaker: str) -> None:
    """Write extracted entities into the in-memory graph (synchronous, zero-latency)."""
    topics = [t.strip().title() for t in entities.get("topics", []) if t]
    decisions = [d for d in entities.get("decisions", []) if d]
    people_others = [p for p in entities.get("people", []) if p and p != speaker]

    if speaker:
        _local_upsert_node(speaker, "Person")

    for topic in topics:
        _local_upsert_node(topic, "Topic")
        if speaker:
            _local_upsert_edge(topic, "MENTIONED_BY", speaker)

    # Co-occurrence edges between topics in the same utterance
    for i in range(len(topics)):
        for j in range(i + 1, len(topics)):
            _local_upsert_edge(topics[i], "RELATED_TO", topics[j])

    for decision in decisions:
        key = decision.strip().lower()[:80]
        _local_upsert_node(key, "Decision", text=decision)
        if speaker:
            _local_upsert_edge(key, "MENTIONED_BY", speaker)
        if topics:
            _local_upsert_edge(key, "DECIDED_IN", topics[0].lower())

    for person in people_others:
        _local_upsert_node(person, "Person")


def _local_graph_query(keywords: List[str]) -> str:
    """1-hop traversal over the in-memory graph for given keywords. O(keywords × edges)."""
    if not _local_nodes:
        return ""
    results = []
    seen_keys = set()
    for kw in keywords:
        kw_lower = kw.lower()
        # Find all nodes whose name contains the keyword
        matching = [
            (key, meta) for key, meta in _local_nodes.items()
            if kw_lower in key or (meta.get("text") and kw_lower in meta["text"].lower())
        ]
        for node_key, node_meta in matching:
            node_display = node_meta.get("text") or node_meta["name"]
            # Emit the node itself
            if node_display not in seen_keys:
                results.append(node_display)
                seen_keys.add(node_display)
            # Follow 1-hop edges
            for (from_k, rel, to_k) in _local_edges:
                if from_k == node_key and to_k in _local_nodes:
                    neighbor_display = _local_nodes[to_k].get("text") or _local_nodes[to_k]["name"]
                    label = f"{node_display} {rel} {neighbor_display}"
                    if label not in seen_keys:
                        results.append(label)
                        seen_keys.add(label)
                elif to_k == node_key and from_k in _local_nodes:
                    neighbor_display = _local_nodes[from_k].get("text") or _local_nodes[from_k]["name"]
                    label = f"{neighbor_display} {rel} {node_display}"
                    if label not in seen_keys:
                        results.append(label)
                        seen_keys.add(label)

    if not results:
        return ""
    return "In the meeting: " + "; ".join(results[:10]) + "."


def reset_local_graph() -> None:
    """Clear the in-memory graph (call at meeting start/reset)."""
    _local_nodes.clear()
    _local_edges.clear()


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
            user = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "")
            # max_connection_lifetime=300 forces connections to refresh every 5 min,
            # preventing stale TCP connections to AuraDB (which closes idle connections).
            _driver = AsyncGraphDatabase.driver(
                uri,
                auth=(user, password),
                max_connection_lifetime=300,
                connection_timeout=30,
                keep_alive=True,
            )
        except Exception as e:
            logger.warning("Neo4j driver init failed: %s", e)
            return None
    return _driver


def reset_driver() -> None:
    """Force-reset the Neo4j driver singleton so the next call recreates it.

    Call this when execute_query raises a connection-level error so the broken
    driver is not reused indefinitely.
    """
    global _driver
    old = _driver
    _driver = None
    if old is not None:
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(old.close())
            else:
                loop.run_until_complete(old.close())
        except Exception:
            pass


# ---------- STT Normalization ----------

_STT_REPLACEMENTS = [
    # Common STT artifacts → canonical forms
    (r"\bcalf\s*ka\b", "Kafka"),
    (r"\braz[eo]r\s*pay\b", "Razorpay"),
    (r"\ba\s*p\s*i\b", "API"),
    (r"\bpipe\s*line\b", "pipeline"),
    (r"\bcat\s*a\s*go(?:ry|ree)\b", "category"),
    (r"\bmy\s*s\s*q\s*l\b", "MySQL"),
    (r"\bpost\s*gres\b", "Postgres"),
    (r"\bkubernete?s\b", "Kubernetes"),
    (r"\bdocker\s*file\b", "Dockerfile"),
    (r"\bgit\s*hub\b", "GitHub"),
    (r"\bdata\s*base\b", "database"),
    (r"\bback\s*end\b", "backend"),
    (r"\bfront\s*end\b", "frontend"),
    (r"\bmicro\s*service\b", "microservice"),
]
_STT_PATTERNS = [(re.compile(pat, re.IGNORECASE), repl) for pat, repl in _STT_REPLACEMENTS]


def _normalize_stt_text(text: str) -> str:
    """Apply lightweight STT artifact normalization before entity extraction."""
    for pattern, replacement in _STT_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


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
    """MERGE nodes and edges into Neo4j using batched UNWIND queries (max 8 round-trips).
    Per D-12: MENTIONED_BY, RELATED_TO, DECIDED_IN."""
    import neo4j

    topics_normalized = [t.strip().title() for t in entities.get("topics", []) if t]
    decisions_raw = [d for d in entities.get("decisions", []) if d]
    people_others = [p for p in entities.get("people", []) if p and p != speaker]

    # 1. MERGE speaker Person node
    if speaker:
        await driver.execute_query(
            "MERGE (p:Person {name: $name}) ON CREATE SET p.first_seen = timestamp() SET p.last_seen = timestamp()",
            {"name": speaker},
            routing_=neo4j.RoutingControl.WRITE,
        )

    # 2. MERGE all Topic nodes in one batched UNWIND call
    if topics_normalized:
        await driver.execute_query(
            "UNWIND $topics AS topic_name "
            "MERGE (t:Topic {name: topic_name}) "
            "ON CREATE SET t.first_seen = timestamp() "
            "SET t.last_seen = timestamp()",
            {"topics": topics_normalized},
            routing_=neo4j.RoutingControl.WRITE,
        )

    # 3. MERGE all MENTIONED_BY edges (topic → speaker) in one batched call
    if topics_normalized and speaker:
        await driver.execute_query(
            "UNWIND $topics AS topic_name "
            "MATCH (t:Topic {name: topic_name}) "
            "MATCH (p:Person {name: $speaker}) "
            "MERGE (t)-[:MENTIONED_BY]->(p)",
            {"topics": topics_normalized, "speaker": speaker},
            routing_=neo4j.RoutingControl.WRITE,
        )

    # 4. MERGE all RELATED_TO edges between co-occurring topics in one batched call
    topic_pairs = [
        [topics_normalized[i], topics_normalized[j]]
        for i in range(len(topics_normalized))
        for j in range(i + 1, len(topics_normalized))
    ]
    if topic_pairs:
        await driver.execute_query(
            "UNWIND $pairs AS pair "
            "MATCH (t1:Topic {name: pair[0]}) "
            "MATCH (t2:Topic {name: pair[1]}) "
            "MERGE (t1)-[:RELATED_TO]-(t2)",
            {"pairs": topic_pairs},
            routing_=neo4j.RoutingControl.WRITE,
        )

    # 5. MERGE all Decision nodes in one batched UNWIND call
    if decisions_raw:
        decision_params = [{"key": d.strip().lower()[:80], "text": d} for d in decisions_raw]
        await driver.execute_query(
            "UNWIND $decisions AS d "
            "MERGE (dec:Decision {key: d.key}) "
            "ON CREATE SET dec.text = d.text, dec.first_seen = timestamp() "
            "SET dec.last_seen = timestamp()",
            {"decisions": decision_params},
            routing_=neo4j.RoutingControl.WRITE,
        )

    # 6. MERGE Decision MENTIONED_BY edges (decision → speaker) in one batched call
    if decisions_raw and speaker:
        decision_keys = [d.strip().lower()[:80] for d in decisions_raw]
        await driver.execute_query(
            "UNWIND $keys AS key "
            "MATCH (dec:Decision {key: key}) "
            "MATCH (p:Person {name: $speaker}) "
            "MERGE (dec)-[:MENTIONED_BY]->(p)",
            {"keys": decision_keys, "speaker": speaker},
            routing_=neo4j.RoutingControl.WRITE,
        )

    # 7. MERGE Decision DECIDED_IN edges (all decisions → first topic) in one batched call
    if decisions_raw and topics_normalized:
        decision_keys = [d.strip().lower()[:80] for d in decisions_raw]
        await driver.execute_query(
            "UNWIND $keys AS key "
            "MATCH (dec:Decision {key: key}) "
            "MATCH (t:Topic {name: $topic}) "
            "MERGE (dec)-[:DECIDED_IN]->(t)",
            {"keys": decision_keys, "topic": topics_normalized[0]},
            routing_=neo4j.RoutingControl.WRITE,
        )

    # 8. MERGE all other mentioned People nodes in one batched call
    if people_others:
        await driver.execute_query(
            "UNWIND $people AS name "
            "MERGE (p:Person {name: name}) "
            "ON CREATE SET p.first_seen = timestamp() "
            "SET p.last_seen = timestamp()",
            {"people": people_others},
            routing_=neo4j.RoutingControl.WRITE,
        )


def _filter_topics(topics: List[str]) -> List[str]:
    """Remove graph-polluting topic fragments from extracted entity lists.

    Filters out:
    - Single words under 4 characters
    - Syllabified fragments (contain isolated single-char tokens like 'a', 'ing', 'ory')
    - Pure number words
    """
    _NUMBER_WORDS = {
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "hundred", "thousand", "million", "billion",
    }
    _SYLLABLE_JUNK = {"a", "e", "i", "o", "u", "ing", "ory", "ery", "ary", "ion", "ed", "er"}
    clean = []
    for topic in topics:
        if not topic:
            continue
        tokens = topic.lower().split()
        # Drop single short words
        if len(tokens) == 1 and len(tokens[0]) < 4:
            continue
        # Drop pure number-word phrases
        if all(t in _NUMBER_WORDS or t.isdigit() for t in tokens):
            continue
        # Drop syllabified fragments: any single-char token or known syllable junk
        if any(t in _SYLLABLE_JUNK or (len(t) == 1 and t.isalpha()) for t in tokens):
            continue
        clean.append(topic)
    return clean


async def ingest_transcript_entry(entry: Dict[str, Any]) -> None:
    """Fire-and-forget: extract entities and upsert into local graph + Neo4j (if available).
    Per D-13: triggered via asyncio.create_task for each transcript entry.

    The local in-memory graph is always populated — Neo4j is optional persistence."""
    try:
        normalized_text = _normalize_stt_text(entry.get("text", ""))
        speaker = entry.get("participant", "")
        entities = await _extract_entities(speaker, normalized_text)
        entities["topics"] = _filter_topics(entities.get("topics", []))

        # Always write to local graph (zero-latency, no network dependency)
        _local_upsert_entities(entities, speaker)

        # Optionally persist to Neo4j in the background
        driver = _get_driver()
        if driver is not None:
            await _upsert_entities(driver, entities, speaker)
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


_ACRONYM_COLLAPSE_PATTERNS = [
    # Spaced-out acronyms: "a p i" → "api", "b m twenty five" → "bm25", etc.
    (re.compile(r'\ba\s+p\s+i\b', re.IGNORECASE), "api"),
    (re.compile(r'\bs\s+q\s+l\b', re.IGNORECASE), "sql"),
    (re.compile(r'\bs\s+s\s+l\b', re.IGNORECASE), "ssl"),
    (re.compile(r'\bs\s+d\s+k\b', re.IGNORECASE), "sdk"),
    (re.compile(r'\bc\s+l\s+i\b', re.IGNORECASE), "cli"),
    (re.compile(r'\bc\s+r\s+u\s+d\b', re.IGNORECASE), "crud"),
    (re.compile(r'\bu\s+r\s+l\b', re.IGNORECASE), "url"),
    (re.compile(r'\bu\s+i\b', re.IGNORECASE), "ui"),
    (re.compile(r'\bb\s+m\s+(?:twenty\s+five|25)\b', re.IGNORECASE), "bm25"),
    (re.compile(r'\br\s+e\s+s\s+t\b', re.IGNORECASE), "rest"),
]


def _collapse_acronyms(text: str) -> str:
    """Collapse STT-expanded acronyms before keyword extraction."""
    for pattern, replacement in _ACRONYM_COLLAPSE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


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
    question = _collapse_acronyms(question)
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
    Per D-14: extract keywords, run 1-hop traversal, return formatted context.

    Uses the local in-memory graph first (zero-latency). Falls back to Neo4j
    only when the local graph is empty (e.g. ingest hasn't run yet)."""
    try:
        keywords = await _extract_keywords(question)
        if not keywords:
            return ""

        # Fast path: local in-memory graph (microseconds)
        if _local_nodes:
            return _local_graph_query(keywords)

        # Slow fallback: Neo4j (only if local graph not yet populated)
        driver = _get_driver()
        if driver is None:
            return ""
        return await _cypher_query(driver, keywords)
    except Exception as e:
        logger.debug("Graph query skipped (non-fatal): %s", e)
        return ""
