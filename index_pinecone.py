import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv("confluence_logic/.env")

from confluence_logic.ingestion.doc_pipeline import IngestionPipeline
from confluence_logic.connectors.confluence import ConfluenceConnector

c = ConfluenceConnector()
pages = c.list_pages(limit=500)
print(f"Found {len(pages)} pages — namespace: {os.getenv('PINECONE_NAMESPACE', '(default)')}")

p = IngestionPipeline()
t0 = time.time()
for i, pg in enumerate(pages, 1):
    pid = pg.get("page_id")
    title = pg.get("title", pid)
    print(f"  [{i}/{len(pages)}] {title}")
    p.process_page(pid)

print(f"\nDone in {time.time()-t0:.1f}s")
