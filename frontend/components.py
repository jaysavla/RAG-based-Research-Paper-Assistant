from pathlib import Path


def load_css() -> str:
    css = (Path(__file__).parent / "style.css").read_text()
    return f"<style>{css}</style>"


def source_cards(sources: list) -> str:
    """Render a list of source dicts as styled cards."""
    cards = ""
    for s in sources:
        cards += (
            f'<div class="source-card">'
            f'  <span class="source-label">{s["label"]}</span>'
            f'  <div class="source-meta">📄 {s["filename"]} &nbsp;·&nbsp; pages {s["pages"]}</div>'
            f"</div>"
        )
    return f"<strong>Sources ({len(sources)}):</strong><br><br>{cards}"


def chunk_card(chunk: dict) -> str:
    """Render a single chunk as a styled preview card."""
    text = chunk["text"]
    preview = text[:500] + ("..." if len(text) > 500 else "")
    return (
        f'<div class="chunk-card">'
        f'  <div class="chunk-meta">'
        f"    Chunk {chunk['chunk_id']} &nbsp;·&nbsp; "
        f"    {chunk['word_count']} words &nbsp;·&nbsp; "
        f"    pages {chunk['pages']}"
        f"  </div>"
        f"  {preview}"
        f"</div>"
    )
