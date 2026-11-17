import pytest

from fastrag.retrieval.profiles import effective_groups, resolve_profile
from fastrag.retrieval.search import rrf
from fastrag.retrieval.signals import PassageEvidence, conflicts, coverage


def test_customer_profile_ceiling_cannot_be_lifted():
    p = resolve_profile("customer", {"profiles": {"customer": {"max_groups": ["public", "staff"], "top_k": 3}}})
    assert p.max_groups == ["public"] and p.top_k == 3
    assert effective_groups(["public", "staff"], ["public", "staff"], p) == ["public"]


def test_effective_groups_is_an_intersection():
    p = resolve_profile("internal_search")
    assert effective_groups(None, ["public", "staff"], p) == ["public", "staff"]
    assert effective_groups(["public", "managers"], ["public", "staff"], p) == ["public"]
    assert effective_groups(["staff"], ["public"], p) == []


def test_unknown_profile():
    with pytest.raises(KeyError):
        resolve_profile("everything")


def test_rrf_prefers_items_in_both_lists():
    scores = rrf([[1, 2, 3], [3, 4, 1]])
    assert max(scores, key=scores.get) in (1, 3)
    assert scores[1] > scores[2] and scores[3] > scores[4]


def test_coverage_and_conflicts():
    a = PassageEvidence(1, "d1", "Phí quản lý căn hộ là 15 triệu mỗi năm", {"phí", "quản_lý", "căn_hộ"}, 0.9, None)
    b = PassageEvidence(2, "d2", "Phí quản lý căn hộ là 18 triệu mỗi năm", {"phí", "quản_lý", "căn_hộ"}, 0.8, None)
    c = PassageEvidence(3, "d1", "Hồ bơi mở cửa lúc 6 giờ", {"hồ_bơi"}, 0.5, None)
    assert coverage(["phí", "quản_lý", "gửi_xe"], [a, c]) == pytest.approx(2 / 3, 0.01)
    found = conflicts([a, b, c])
    assert len(found) == 1 and found[0]["kind"] == "price" and set(found[0]["passages"]) == {1, 2}


def test_select_terms_keeps_rare_terms_within_budget():
    from fastrag.retrieval.search import select_terms

    q = {"căn_hộ": 1, "~can_ho": 1, "masteri": 1, "quận": 1}
    dfs = {"căn_hộ": 40_000, "~can_ho": 41_000, "masteri": 120, "quận": 30_000}
    terms, weights, skipped = select_terms(q, dfs, n=50_000, budget=40_000, accentless_weight=0.5)
    assert terms[0] == "masteri" and "quận" in terms and skipped == 2
    assert weights[0] > weights[1]
    # The rarest term is always used, even over budget.
    terms, _, _ = select_terms({"căn_hộ": 1}, dfs, n=50_000, budget=10, accentless_weight=0.5)
    assert terms == ["căn_hộ"]
