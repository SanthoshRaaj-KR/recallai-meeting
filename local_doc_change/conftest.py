"""Root conftest.py: adds local_doc_change project root to sys.path.

Required because pyproject.toml maps '' = 'src' for the installed package,
but the Phase 12 pipeline modules (models/, rag/, agents_local/, pipeline/)
live at the project root level to keep them independent of the LiveKit agent src/.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the local_doc_change/ project root is on sys.path so that
# `from models import ...`, `from rag.chunker import ...`, etc. work
# in all test files without explicit sys.path manipulation.
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
