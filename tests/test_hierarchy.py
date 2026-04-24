"""Tests for HierarchyTree utility."""

import numpy as np
import pandas as pd
import pytest

from hierarchical_forecast.utils.hierarchy import HierarchyTree


SIMPLE_SPEC = {
    "Total": {
        "North": {"StoreA": None, "StoreB": None},
        "South": {"StoreC": None, "StoreD": None},
    }
}


@pytest.fixture
def simple_tree():
    return HierarchyTree(SIMPLE_SPEC)


class TestHierarchyTree:

    def test_n_total(self, simple_tree):
        # Total + North + South + 4 stores = 7
        assert simple_tree.n_total == 7

    def test_n_bottom(self, simple_tree):
        assert simple_tree.n_bottom == 4

    def test_bottom_series(self, simple_tree):
        assert set(simple_tree.bottom_series) == {"StoreA", "StoreB", "StoreC", "StoreD"}

    def test_all_series_contains_all(self, simple_tree):
        all_s = simple_tree.all_series
        assert "Total" in all_s
        assert "North" in all_s
        assert "South" in all_s
        for s in ["StoreA", "StoreB", "StoreC", "StoreD"]:
            assert s in all_s

    def test_summing_matrix_shape(self, simple_tree):
        S, all_series, bottom_series = simple_tree.get_summing_matrix()
        assert S.shape == (7, 4)

    def test_summing_matrix_total_row(self, simple_tree):
        """Total row should be all 1s."""
        S, all_series, _ = simple_tree.get_summing_matrix()
        total_idx = all_series.index("Total")
        assert S[total_idx].sum() == 4  # all 4 bottom series

    def test_summing_matrix_bottom_rows(self, simple_tree):
        """Each bottom-level row should have exactly one 1."""
        S, all_series, bottom_series = simple_tree.get_summing_matrix()
        for b in bottom_series:
            idx = all_series.index(b)
            assert S[idx].sum() == 1.0

    def test_summing_matrix_region_row(self, simple_tree):
        """North should aggregate StoreA + StoreB."""
        S, all_series, bottom_series = simple_tree.get_summing_matrix()
        north_idx = all_series.index("North")
        assert S[north_idx].sum() == 2.0

    def test_levels(self, simple_tree):
        levels = simple_tree.get_levels()
        assert 0 in levels  # Total
        assert 1 in levels  # North, South
        assert 2 in levels  # Stores

    def test_validate_dataframe_passes(self, simple_tree):
        df = pd.DataFrame({
            "unique_id": ["Total", "North", "South", "StoreA", "StoreB", "StoreC", "StoreD"],
            "ds": pd.date_range("2020-01-01", periods=7, freq="M")[:7].repeat(1),
            "y": [1.0] * 7,
        })
        # Expand to have all 7 series
        rows = []
        for sid in ["Total", "North", "South", "StoreA", "StoreB", "StoreC", "StoreD"]:
            rows.append({"unique_id": sid, "ds": pd.Timestamp("2020-01-01"), "y": 1.0})
        df = pd.DataFrame(rows)
        simple_tree.validate_dataframe(df)  # Should not raise

    def test_validate_dataframe_fails_missing(self, simple_tree):
        df = pd.DataFrame({
            "unique_id": ["Total", "North"],
            "ds": [pd.Timestamp("2020-01-01")] * 2,
            "y": [1.0, 1.0],
        })
        with pytest.raises(ValueError, match="missing"):
            simple_tree.validate_dataframe(df)

    def test_from_dataframe(self):
        df = pd.DataFrame({
            "region": ["North", "North", "South"],
            "store": ["A", "B", "C"],
        })
        tree = HierarchyTree.from_dataframe(df, level_cols=["region", "store"])
        assert tree.n_bottom == 3
        assert "Total" in tree.all_series

    def test_repr(self, simple_tree):
        r = repr(simple_tree)
        assert "HierarchyTree" in r
        assert "n_total=7" in r
