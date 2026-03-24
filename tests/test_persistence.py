from persistence import _safe_name


def test_safe_name_allows_alphanumeric_dots_hyphens():
    assert _safe_name("paper-2024.pdf") == "paper-2024.pdf"


def test_safe_name_replaces_spaces():
    result = _safe_name("my paper.pdf")
    assert " " not in result


def test_safe_name_replaces_slashes():
    result = _safe_name("docs/paper.pdf")
    assert "/" not in result


def test_safe_name_replaces_parentheses():
    result = _safe_name("paper (v2).pdf")
    assert "(" not in result
    assert ")" not in result


def test_safe_name_only_safe_chars_remain():
    result = _safe_name("weird!@#$%name.pdf")
    assert all(c.isalnum() or c in "-_." for c in result)
