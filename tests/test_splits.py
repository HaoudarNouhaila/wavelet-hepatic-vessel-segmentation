import pytest

from src.data.splits import check_disjoint, load_split, make_split, save_split, select_subset

CASES = [f"hepaticvessel_{i:03d}" for i in range(303)]


def test_split_is_disjoint_and_complete():
    split = make_split(CASES, seed=42)
    check_disjoint(split)
    assert sorted(sum(split.values(), [])) == sorted(CASES)
    assert len(split["train"]) == 212 and len(split["val"]) == 45 and len(split["test"]) == 46


def test_split_is_deterministic():
    assert make_split(CASES, seed=42) == make_split(list(reversed(CASES)), seed=42)
    assert make_split(CASES, seed=42) != make_split(CASES, seed=1)


def test_subset_is_contained_in_full_split():
    split = make_split(CASES, seed=42)
    subset = select_subset(split, {"train": 10, "val": 3, "test": 5})
    for name in subset:
        assert set(subset[name]) <= set(split[name])
    check_disjoint(subset)


def test_leakage_detected():
    with pytest.raises(ValueError):
        check_disjoint({"train": ["a", "b"], "val": ["b"], "test": []})
    with pytest.raises(ValueError):
        make_split(["a", "a", "b"])


def test_save_and_load(tmp_path):
    split = make_split(CASES, seed=3)
    subset = select_subset(split, {"train": 4, "val": 2, "test": 2})
    path = tmp_path / "split.json"
    save_split(path, split, subset, 3, (0.7, 0.15, 0.15))
    assert load_split(path, "subset") == subset
    assert load_split(path, "full") == split
