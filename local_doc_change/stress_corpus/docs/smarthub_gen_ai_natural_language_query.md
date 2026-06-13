# Gen-AI Natural Language Query — Technical Design

INFER v2.5 ships a natural language query (NLQ) interface that lets operators ask plain-English questions over their device inventory and security posture. This page covers the full RAG pipeline, prompt engineering, and evaluation approach.

## Architecture


    |  Component | Technology | Role |

    |  Query Parser | GPT-4o (OpenAI) | Classify intent + extract structured filters (site, device_class, time range) |

    |  Vector Retrieval | Pinecone (text-embedding-3-large, 3072-dim) | Semantic similarity search over device summaries |

    |  Graph Retrieval | Neo4j (Cypher) | Relationship-aware context (e.g. blast radius, peer devices) |

    |  Structured Query | PostgreSQL (generated Cypher/SQL) | Precise filter queries (exact site, CVE, firmware version) |

    |  Answer Synthesizer | GPT-4o | Fuses retrieval results into a natural language answer with citations |

    |  Semantic Cache | Pinecone (separate namespace) | Cache answers for semantically similar questions (cosine > 0.97) |



## Query Processing Flow

  - **Receive query:** User types "Show me all cameras with firmware older than 6 months in Building 3 with SPS below 50"
  - **Cache check:** Embed query with text-embedding-3-small; check Pinecone semantic cache (threshold: cosine ≥ 0.97)
  - **Intent classification:** GPT-4o classifies as `device_filter_query` and extracts: `device_class=ip_camera, location=Building 3, firmware_age_gt=180d, sps_lt=50`
  - **Dual retrieval:** Pinecone vector search (top-20 device summaries) + PostgreSQL structured query (exact filter)
  - **Graph augmentation:** For each retrieved device, fetch Neo4j context (CVE exposure, peer count, recent alerts)
  - **Reranking:** Cohere Rerank API re-scores the top-20 Pinecone results against the original query
  - **Answer synthesis:** GPT-4o generates answer with inline citations to device IDs and page links
  - **Cache write:** Store query embedding + answer in semantic cache (TTL: 10 minutes)

## System Prompt — Answer Synthesizer

text3 items.
5. End every answer with a recommended action if a security risk is identified.
6. Never reveal tenant data from other tenants.
7. Keep answers under 400 words unless the user explicitly asks for detail.

Context:
{retrieved_chunks}

User question: {user_query}]]>

## Device Summary Embedding Schema

Each device in Pinecone is represented as a text chunk embedded every 15 minutes:

text

## Evaluation Metrics


    |  Metric | Method | Target | Current (v2.5) |

    |  Retrieval Recall@10 | Annotated golden set (500 queries) | >0.90 | 0.93 |

    |  Answer Faithfulness | RAGAS faithfulness score (GPT-4 judge) | >0.92 | 0.94 |

    |  Answer Relevance | RAGAS answer relevance | >0.88 | 0.91 |

    |  Context Precision | RAGAS context precision | >0.85 | 0.87 |

    |  Hallucination rate | Fact-check vs. live DB (random 100 answers/day) | <2% | 1.1% |

    |  Cache hit rate | Production telemetry | >25% | 31% |

    |  End-to-end p95 latency | Production telemetry | <5 s | 3.2 s |
