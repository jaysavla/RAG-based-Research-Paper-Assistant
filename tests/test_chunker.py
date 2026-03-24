from chunker import clean_text, split_into_chunks


# ── clean_text ─────────────────────────────────────────────────────────────────


def test_clean_text_removes_blank_lines():
    assert clean_text("Hello\n\nWorld") == "Hello World"


def test_clean_text_strips_lone_page_numbers():
    result = clean_text("Introduction\n42\nConclusion")
    assert "42" not in result
    assert "Introduction" in result
    assert "Conclusion" in result


def test_clean_text_keeps_numbers_inside_sentences():
    result = clean_text("There are 42 experiments in total.")
    assert "42" in result


def test_clean_text_strips_leading_trailing_whitespace():
    result = clean_text("  Hello  \n  World  ")
    assert result == "Hello World"


def test_clean_text_empty_string():
    assert clean_text("") == ""


# ── split_into_chunks ──────────────────────────────────────────────────────────


def _pages(text: str, page: int = 1):
    return [{"page": page, "text": text}]


def test_split_produces_at_least_one_chunk():
    chunks = split_into_chunks(_pages("The quick brown fox. " * 20))
    assert len(chunks) >= 1


def test_chunk_has_required_keys():
    chunks = split_into_chunks(_pages("Hello world. This is a test sentence."))
    for chunk in chunks:
        assert "chunk_id" in chunk
        assert "text" in chunk
        assert "word_count" in chunk
        assert "char_count" in chunk
        assert "pages" in chunk


def test_empty_pages_returns_no_chunks():
    assert split_into_chunks([]) == []


def test_chunk_ids_are_sequential():
    chunks = split_into_chunks(_pages(("Word " * 50 + ". ") * 10))
    ids = [c["chunk_id"] for c in chunks]
    assert ids == list(range(len(ids)))


def test_pages_recorded_correctly():
    pages = [{"page": 3, "text": "This is page three. It has some content."}]
    chunks = split_into_chunks(pages)
    assert all(3 in c["pages"] for c in chunks)


def test_word_count_matches_text():
    chunks = split_into_chunks(_pages("One two three four five. Six seven eight."))
    for chunk in chunks:
        assert chunk["word_count"] == len(chunk["text"].split())


def test_char_count_matches_text():
    chunks = split_into_chunks(_pages("Hello world. Foo bar baz."))
    for chunk in chunks:
        assert chunk["char_count"] == len(chunk["text"])
