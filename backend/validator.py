from config import MAX_FILE_BYTES, MAX_FILE_MB, PDF_MAGIC


def validate_file(content: bytes) -> str | None:
    if len(content) == 0:
        return "File is empty (0 bytes)."
    if len(content) > MAX_FILE_BYTES:
        return f"File exceeds {MAX_FILE_MB} MB limit ({len(content) // (1024 * 1024)} MB)."
    if not content.startswith(PDF_MAGIC):
        return "File does not appear to be a valid PDF (missing %PDF header)."
    return None
