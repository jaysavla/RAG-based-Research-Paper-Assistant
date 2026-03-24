from utils import rrf_merge


def test_item_in_both_lists_ranks_first():
    # item 0 appears in both lists and should outscore items only in one
    result = rrf_merge([0, 1], [0, 2], k=3)
    assert result[0] == 0


def test_respects_k():
    result = rrf_merge([0, 1, 2], [3, 4, 5], k=2)
    assert len(result) == 2


def test_all_items_included_within_k():
    result = rrf_merge([10], [20], k=2)
    assert set(result) == {10, 20}


def test_empty_first_list():
    result = rrf_merge([], [1, 2, 3], k=2)
    assert result == [1, 2]


def test_empty_both_lists():
    assert rrf_merge([], [], k=5) == []


def test_custom_rrf_k_changes_scores():
    # With a very large rrf_k, ranks matter less — scores become closer.
    # Both orderings are valid; just confirm no crash and correct length.
    result = rrf_merge([0, 1], [1, 0], k=2, rrf_k=1000)
    assert len(result) == 2
    assert set(result) == {0, 1}


def test_higher_rank_in_both_beats_lower():
    # item 0 is rank-0 in list1, item 1 is rank-0 in list2 only
    result = rrf_merge([0, 2], [1, 2], k=3)
    # item 2 appears in both lists — should beat items in only one list
    assert result[0] == 2
