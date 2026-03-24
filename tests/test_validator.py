from validator import validate_file

PDF_HEADER = b"%PDF-1.4 fake content here"


def test_valid_pdf_returns_none():
    assert validate_file(PDF_HEADER) is None


def test_empty_file_rejected():
    assert validate_file(b"") == "File is empty (0 bytes)."


def test_non_pdf_magic_rejected():
    error = validate_file(b"PK\x03\x04 this is a zip file")
    assert error is not None
    assert "PDF" in error


def test_oversized_file_rejected():
    big = PDF_HEADER + b"x" * (51 * 1024 * 1024)
    error = validate_file(big)
    assert error is not None
    assert "MB" in error


def test_exactly_at_size_limit_accepted():
    # 50 MB exactly should pass
    at_limit = PDF_HEADER + b"x" * (50 * 1024 * 1024 - len(PDF_HEADER))
    assert validate_file(at_limit) is None


def test_one_byte_over_limit_rejected():
    over = PDF_HEADER + b"x" * (50 * 1024 * 1024 - len(PDF_HEADER) + 1)
    assert validate_file(over) is not None


def test_pdf_header_anywhere_but_start_rejected():
    # %PDF not at byte 0 → invalid
    error = validate_file(b"junk" + PDF_HEADER)
    assert error is not None
