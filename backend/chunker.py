import io
import re
from typing import Dict, List

import pdfplumber

from config import CHUNK_OVERLAP, CHUNK_SIZE


def extract_text_by_page(pdf_bytes: io.BytesIO) -> List[Dict]:
    pages = []
    with pdfplumber.open(pdf_bytes) as pdf:
        for i, page in enumerate(pdf.pages):
            # x_tolerance=1 keeps character-level gaps, preserving word spaces
            # that default tolerance=3 can accidentally merge across.
            text = page.extract_text(x_tolerance=1, y_tolerance=3)
            if text and text.strip():
                pages.append({"page": i + 1, "text": text.strip()})
    return pages


def clean_text(text: str) -> str:
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.fullmatch(r"\d+", stripped):   # lone page number
            continue
        # Rejoin hyphenated line-breaks: "sub-" + next word → "sub" + next word
        if cleaned and cleaned[-1].endswith("-"):
            cleaned[-1] = cleaned[-1][:-1] + stripped
        else:
            cleaned.append(stripped)
    text = " ".join(cleaned)
    # Insert a space between a lowercase letter immediately followed by an
    # uppercase letter — catches residual merged words from some PDF encodings.
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return text


def split_into_chunks(pages: List[Dict]) -> List[Dict]:
    sentence_pages = []
    for page_data in pages:
        text = clean_text(page_data["text"])
        for sent in re.split(r"(?<=[.!?])\s+", text):
            sent = sent.strip()
            if sent:
                sentence_pages.append((sent, page_data["page"]))

    if not sentence_pages:
        return []

    chunks, chunk_id, i = [], 0, 0
    while i < len(sentence_pages):
        chunk_sentences, chunk_pages, word_count = [], set(), 0
        j = i
        while j < len(sentence_pages) and word_count < CHUNK_SIZE:
            sent, pg = sentence_pages[j]
            chunk_sentences.append(sent)
            chunk_pages.add(pg)
            word_count += len(sent.split())
            j += 1

        chunk_text = " ".join(chunk_sentences)
        chunks.append({
            "chunk_id":   chunk_id,
            "text":       chunk_text,
            "word_count": word_count,
            "char_count": len(chunk_text),
            "pages":      sorted(chunk_pages),
        })
        chunk_id += 1

        words_to_skip, skipped = max(1, word_count - CHUNK_OVERLAP), 0
        while i < len(sentence_pages) and skipped < words_to_skip:
            skipped += len(sentence_pages[i][0].split())
            i += 1

    return chunks
