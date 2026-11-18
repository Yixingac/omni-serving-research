import pytest

from omni_research.analysis import intervals as iv


def test_union_does_not_double_count_overlap():
    # Two kernels on different streams overlapping for 2 units: busy time is 8, not 10.
    ks = [(0, 5), (3, 8)]
    assert iv.total(ks) == 8
    assert iv.normalize(ks) == [(0, 8)]


def test_touching_and_nested_and_empty():
    assert iv.normalize([(0, 1), (1, 2), (5, 5), (4, 3), (0.5, 0.7)]) == [(0, 2)]


def test_gaps_and_coverage_in_window():
    busy = [(1, 2), (4, 6), (5, 7)]
    assert iv.gaps(busy, (0, 10)) == [(0, 1), (2, 4), (7, 10)]
    assert iv.coverage(busy, (0, 10)) == pytest.approx(0.4)
    assert iv.gaps([], (0, 3)) == [(0, 3)]
    assert iv.coverage(busy, (3, 3)) == 0.0


def test_clip_intersect_subtract():
    a = [(0, 10)]
    b = [(2, 3), (5, 7)]
    assert iv.intersect(a, b) == [(2, 3), (5, 7)]
    assert iv.subtract(a, b) == [(0, 2), (3, 5), (7, 10)]
    assert iv.clip([(-5, 2), (8, 20)], (0, 10)) == [(0, 2), (8, 10)]


def test_concurrency_profile_half_open():
    prof = iv.concurrency_profile([(0, 2), (1, 3), (2, 4)])
    # At t=2 one interval ends and one starts: the level stays 2, not 3.
    assert prof == [(0, 1), (1, 2), (2, 2), (3, 1), (4, 0)]
    assert iv.time_weighted_mean_level(prof, (0, 4)) == pytest.approx(1.5)


def test_gap_breakdown():
    g = [(0, 0.0005), (1, 1.003), (2, 2.05), (3, 4)]
    out = iv.gap_breakdown(g, [0.001, 0.01, 0.1])
    assert out["<0.001"]["count"] == 1
    assert out["0.001-0.01"]["count"] == 1
    assert out["0.01-0.1"]["count"] == 1
    assert out[">=0.1"]["count"] == 1 and out[">=0.1"]["total"] == pytest.approx(1.0)
