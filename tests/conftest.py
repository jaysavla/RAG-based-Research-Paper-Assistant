import sys
import os
from unittest.mock import MagicMock

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

# Stub out `store` before any backend module imports it — prevents ML models
# (SentenceTransformer, CrossEncoder, OpenAI) from loading during tests.
mock_store = MagicMock()
sys.modules["store"] = mock_store
