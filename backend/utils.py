from typing import Dict, List


def rrf_merge(list1: List[int], list2: List[int], k: int, rrf_k: int = 60) -> List[int]:
    """Reciprocal Rank Fusion — combine two ranked lists into one.

    Score for each item = sum of 1 / (rrf_k + rank) across all lists.
    Items appearing in both lists are boosted; unique items are included too.
    """
    scores: Dict[int, float] = {}
    for rank, idx in enumerate(list1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, idx in enumerate(list2):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)[:k]
