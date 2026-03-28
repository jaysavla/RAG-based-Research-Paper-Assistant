import os

CHUNK_SIZE = 200
CHUNK_OVERLAP = 50

MAX_FILE_MB = 50
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024
MIN_TEXT_CHARS = 200
PDF_MAGIC = b"%PDF"

STORE_DIR = os.path.join(os.path.dirname(__file__), "store")

# BGE models expect this prefix on query strings (not on passages)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
