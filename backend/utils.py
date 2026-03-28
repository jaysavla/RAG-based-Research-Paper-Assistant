import re

_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "is",
    "it",
    "its",
    "was",
    "are",
    "be",
    "as",
    "that",
    "this",
    "which",
    "we",
    "our",
    "their",
    "have",
    "has",
    "been",
    "not",
    "also",
    "can",
    "may",
    "more",
    "such",
    "than",
    "into",
    "these",
    "those",
    "they",
    "were",
    "there",
    "then",
    "when",
    "where",
}


def tokenize(text: str) -> list:
    """Lowercase, keep only real words (≥2 chars), remove stopwords."""
    return [w for w in re.findall(r"\b[a-z]{2,}\b", text.lower()) if w not in _STOPWORDS]


def rrf_merge(list1: list[int], list2: list[int], k: int, rrf_k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion — combine two ranked lists into one.

    Score for each item = sum of 1 / (rrf_k + rank) across all lists.
    Items appearing in both lists are boosted; unique items are included too.
    """
    scores: dict[int, float] = {}
    for rank, idx in enumerate(list1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, idx in enumerate(list2):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)[:k]
