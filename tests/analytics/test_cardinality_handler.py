"""
test_cardinality_handler.py

Full test suite for cardinality_handler.py.
Target: 100% line + branch coverage.

Run with:
    pytest test_cardinality_handler.py -v --cov=cardinality_handler --cov-report=term-missing
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

import src.analytics.cardinality_handler as ch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_low_series(n: int = 4) -> pd.Series:
    """2–8 unique values (LOW tier)."""
    return pd.Series(["A", "B", "C", "D"] * 10, name="low_col")


def _make_medium_series() -> pd.Series:
    """9–15 unique values (MEDIUM tier)."""
    values = [f"cat_{i}" for i in range(10)]
    return pd.Series(values * 5, name="med_col")


def _make_high_series() -> pd.Series:
    """16–30 unique values (HIGH tier)."""
    values = [f"item_{i}" for i in range(20)]
    return pd.Series(values * 3, name="high_col")


def _make_extreme_series() -> pd.Series:
    """31+ unique values (EXTREME tier)."""
    values = [f"x_{i}" for i in range(35)]
    return pd.Series(values * 2, name="extreme_col")


def _make_df_with_num() -> pd.DataFrame:
    """DataFrame: one high-cardinality category + one numeric column."""
    cats = [f"prod_{i}" for i in range(20)] * 5
    nums = list(range(100))
    return pd.DataFrame({"category": cats, "revenue": nums})


def _make_df_with_two_cats() -> pd.DataFrame:
    """DataFrame: two high-cardinality categorical columns."""
    cats_a = [f"A{i}" for i in range(12)] * 5
    cats_b = [f"B{i}" for i in range(12)] * 5
    return pd.DataFrame({"col_a": cats_a, "col_b": cats_b})


# ===========================================================================
# get_cardinality_tier
# ===========================================================================

class TestGetCardinalityTier:
    def test_low(self):
        assert ch.get_cardinality_tier(_make_low_series()) == "low"

    def test_medium(self):
        assert ch.get_cardinality_tier(_make_medium_series()) == "medium"

    def test_high(self):
        assert ch.get_cardinality_tier(_make_high_series()) == "high"

    def test_extreme(self):
        assert ch.get_cardinality_tier(_make_extreme_series()) == "extreme"

    def test_boundary_low_exactly_8(self):
        s = pd.Series([str(i) for i in range(8)] * 2)
        assert ch.get_cardinality_tier(s) == "low"

    def test_boundary_medium_exactly_9(self):
        s = pd.Series([str(i) for i in range(9)] * 2)
        assert ch.get_cardinality_tier(s) == "medium"

    def test_boundary_medium_exactly_15(self):
        s = pd.Series([str(i) for i in range(15)] * 2)
        assert ch.get_cardinality_tier(s) == "medium"

    def test_boundary_high_exactly_16(self):
        s = pd.Series([str(i) for i in range(16)] * 2)
        assert ch.get_cardinality_tier(s) == "high"

    def test_boundary_high_exactly_30(self):
        s = pd.Series([str(i) for i in range(30)] * 2)
        assert ch.get_cardinality_tier(s) == "high"

    def test_boundary_extreme_exactly_31(self):
        s = pd.Series([str(i) for i in range(31)] * 2)
        assert ch.get_cardinality_tier(s) == "extreme"


# ===========================================================================
# is_high_cardinality
# ===========================================================================

class TestIsHighCardinality:
    def test_low_is_not_high(self):
        assert ch.is_high_cardinality(_make_low_series()) is False

    def test_exactly_at_threshold_is_not_high(self):
        s = pd.Series([str(i) for i in range(ch.CARDINALITY_LOW)] * 2)
        assert ch.is_high_cardinality(s) is False

    def test_one_above_threshold_is_high(self):
        s = pd.Series([str(i) for i in range(ch.CARDINALITY_LOW + 1)] * 2)
        assert ch.is_high_cardinality(s) is True

    def test_extreme_is_high(self):
        assert ch.is_high_cardinality(_make_extreme_series()) is True


# ===========================================================================
# _compute_top_n_counts
# ===========================================================================

class TestComputeTopNCounts:
    def test_returns_dataframe(self):
        result = ch._compute_top_n_counts(_make_medium_series())
        assert isinstance(result, pd.DataFrame)

    def test_columns_default(self):
        result = ch._compute_top_n_counts(_make_medium_series())
        assert list(result.columns) == ["category", "count"]

    def test_custom_column_names(self):
        result = ch._compute_top_n_counts(
            _make_medium_series(), value_col="label", count_col="value"
        )
        assert list(result.columns) == ["label", "value"]

    def test_top_n_respected(self):
        result = ch._compute_top_n_counts(_make_extreme_series(), top_n=5)
        # at most top_n + 1 (others) rows
        assert len(result) <= 6

    def test_others_row_added_when_significant(self):
        """Others bucket added when remainder covers >= MIN_OTHERS_SHARE."""
        # 30 unique values, top_n=5 -> 25 left, well above 1%
        result = ch._compute_top_n_counts(_make_extreme_series(), top_n=5)
        others_rows = result[result["category"].str.startswith("Others")]
        assert len(others_rows) == 1

    def test_no_others_when_all_fit_in_top_n(self):
        """No others bucket when series has fewer values than top_n."""
        s = pd.Series(["A", "B", "C"] * 10, name="c")
        result = ch._compute_top_n_counts(s, top_n=10)
        others_rows = result[result["category"].str.startswith("Others")]
        assert len(others_rows) == 0

    def test_count_column_is_int(self):
        result = ch._compute_top_n_counts(_make_medium_series())
        assert result["count"].dtype == int or all(
            isinstance(v, int) for v in result["count"]
        )

    def test_category_column_is_str(self):
        result = ch._compute_top_n_counts(_make_medium_series())
        assert all(isinstance(v, str) for v in result["category"])

    def test_others_not_added_when_below_min_share(self):
        """
        If the remainder is tiny (< MIN_OTHERS_SHARE), no Others row.
        Create 11 categories but top_n=10 and the 11th has only 1 occurrence
        out of 10*10+1 = 101 total  ->  ~0.99% which is < 1%.
        """
        base = [f"cat_{i}" for i in range(10)] * 10   # 100 rows, 10 categories
        base.append("rare")                             # 1 row, 11th category
        s = pd.Series(base, name="col")
        result = ch._compute_top_n_counts(s, top_n=10)
        others_rows = result[result["category"].str.startswith("Others")]
        assert len(others_rows) == 0


# ===========================================================================
# _compute_top_n_aggregation
# ===========================================================================

class TestComputeTopNAggregation:
    def setup_method(self):
        self.df = _make_df_with_num()

    def test_returns_dataframe(self):
        result = ch._compute_top_n_aggregation(self.df, "category", "revenue")
        assert isinstance(result, pd.DataFrame)

    def test_agg_col_name(self):
        result = ch._compute_top_n_aggregation(
            self.df, "category", "revenue", agg_func="mean"
        )
        assert "mean_revenue" in result.columns

    def test_top_n_rows(self):
        result = ch._compute_top_n_aggregation(
            self.df, "category", "revenue", top_n=5
        )
        assert len(result) <= 6  # top_n + possible others

    def test_others_appended(self):
        result = ch._compute_top_n_aggregation(
            self.df, "category", "revenue", top_n=3
        )
        others = result[result["category"].str.startswith("Others")]
        assert len(others) == 1

    def test_sum_aggregation(self):
        result = ch._compute_top_n_aggregation(
            self.df, "category", "revenue", agg_func="sum", top_n=3
        )
        assert "sum_revenue" in result.columns

    def test_numeric_col_after_aggregation(self):
        result = ch._compute_top_n_aggregation(self.df, "category", "revenue")
        agg_col = "mean_revenue"
        assert pd.api.types.is_float_dtype(result[agg_col]) or all(
            isinstance(v, (int, float)) for v in result[agg_col]
        )

    def test_no_others_when_all_fit(self):
        """When the data has fewer categories than top_n, no Others row."""
        small_df = pd.DataFrame({
            "cat": ["A", "B", "C"] * 5,
            "val": list(range(15)),
        })
        result = ch._compute_top_n_aggregation(small_df, "cat", "val", top_n=10)
        others = result[result["cat"].str.startswith("Others")]
        assert len(others) == 0


# ===========================================================================
# build_horizontal_bar
# ===========================================================================

class TestBuildHorizontalBar:
    def test_returns_figure(self):
        fig = ch.build_horizontal_bar(_make_medium_series(), "Test HBar")
        assert isinstance(fig, go.Figure)

    def test_title_set(self):
        fig = ch.build_horizontal_bar(_make_medium_series(), "My Title")
        assert fig.layout.title.text == "My Title"

    def test_has_bar_trace(self):
        fig = ch.build_horizontal_bar(_make_medium_series(), "Test")
        assert any(isinstance(t, go.Bar) for t in fig.data)

    def test_orientation_horizontal(self):
        fig = ch.build_horizontal_bar(_make_medium_series(), "Test")
        bar_trace = next(t for t in fig.data if isinstance(t, go.Bar))
        assert bar_trace.orientation == "h"

    def test_custom_top_n(self):
        fig = ch.build_horizontal_bar(_make_high_series(), "Test", top_n=5)
        assert isinstance(fig, go.Figure)


# ===========================================================================
# build_treemap
# ===========================================================================

class TestBuildTreemap:
    def test_returns_figure(self):
        fig = ch.build_treemap(_make_high_series(), "Treemap Title")
        assert isinstance(fig, go.Figure)

    def test_title_set(self):
        fig = ch.build_treemap(_make_high_series(), "TM")
        assert fig.layout.title.text == "TM"

    def test_has_treemap_trace(self):
        fig = ch.build_treemap(_make_high_series(), "TM")
        assert any(isinstance(t, go.Treemap) for t in fig.data)

    def test_custom_top_n(self):
        fig = ch.build_treemap(_make_extreme_series(), "TM", top_n=15)
        assert isinstance(fig, go.Figure)


# ===========================================================================
# build_ranked_table
# ===========================================================================

class TestBuildRankedTable:
    def test_returns_figure(self):
        fig = ch.build_ranked_table(_make_extreme_series(), "Ranked Table")
        assert isinstance(fig, go.Figure)

    def test_title_set(self):
        fig = ch.build_ranked_table(_make_extreme_series(), "RT")
        assert fig.layout.title.text == "RT"

    def test_has_table_trace(self):
        fig = ch.build_ranked_table(_make_extreme_series(), "RT")
        assert any(isinstance(t, go.Table) for t in fig.data)

    def test_custom_top_n(self):
        fig = ch.build_ranked_table(_make_extreme_series(), "RT", top_n=5)
        table = next(t for t in fig.data if isinstance(t, go.Table))
        # rank column values should have at most 5 entries
        assert len(table.cells.values[0]) <= 5


# ===========================================================================
# build_aggregated_horizontal_bar
# ===========================================================================

class TestBuildAggregatedHorizontalBar:
    def setup_method(self):
        self.df = _make_df_with_num()

    def test_returns_figure(self):
        fig = ch.build_aggregated_horizontal_bar(
            self.df, "category", "revenue", "Agg Bar"
        )
        assert isinstance(fig, go.Figure)

    def test_title_set(self):
        fig = ch.build_aggregated_horizontal_bar(
            self.df, "category", "revenue", "My Agg"
        )
        assert fig.layout.title.text == "My Agg"

    def test_horizontal_orientation(self):
        fig = ch.build_aggregated_horizontal_bar(
            self.df, "category", "revenue", "T"
        )
        bar_trace = next(t for t in fig.data if isinstance(t, go.Bar))
        assert bar_trace.orientation == "h"

    def test_sum_agg_func(self):
        fig = ch.build_aggregated_horizontal_bar(
            self.df, "category", "revenue", "T", agg_func="sum"
        )
        assert isinstance(fig, go.Figure)

    def test_custom_top_n(self):
        fig = ch.build_aggregated_horizontal_bar(
            self.df, "category", "revenue", "T", top_n=3
        )
        assert isinstance(fig, go.Figure)


# ===========================================================================
# build_heatmap
# ===========================================================================

class TestBuildHeatmap:
    def setup_method(self):
        self.df = _make_df_with_two_cats()

    def test_returns_figure(self):
        fig = ch.build_heatmap(self.df, "col_a", "col_b", "Heatmap")
        assert isinstance(fig, go.Figure)

    def test_title_set(self):
        fig = ch.build_heatmap(self.df, "col_a", "col_b", "HM Title")
        assert fig.layout.title.text == "HM Title"

    def test_has_heatmap_trace(self):
        fig = ch.build_heatmap(self.df, "col_a", "col_b", "HM")
        assert any(isinstance(t, go.Heatmap) for t in fig.data)

    def test_custom_top_n(self):
        fig = ch.build_heatmap(self.df, "col_a", "col_b", "HM", top_n=5)
        assert isinstance(fig, go.Figure)

    def test_fallback_when_filter_removes_all(self):
        """
        Covers the fallback branch: filtered.empty -> use df_copy.

        Strategy: make top_x and top_y disjoint.
        - col_x='A' appears most (top_x=['A']); its rows all have col_y='X'
        - col_y='Z' appears most (top_y=['Z']); its rows all have col_x='B'/'C'
        - Intersection of (cx=='A') AND (cy=='Z') is empty -> fallback fires.
        """
        cx = ["A"] * 10 + ["B"] * 8 + ["C"] * 7
        cy = ["X"] * 10 + ["Z"] * 8 + ["Z"] * 7
        df = pd.DataFrame({"cx": cx, "cy": cy})
        fig = ch.build_heatmap(df, "cx", "cy", "Fallback HM", top_n=1)
        assert isinstance(fig, go.Figure)


# ===========================================================================
# _dispatch_single_column  (via build_chart_for_cardinality)
# ===========================================================================

class TestDispatchSingleColumn:
    def test_low_produces_pie(self):
        df = pd.DataFrame({"col": _make_low_series()})
        fig = ch.build_chart_for_cardinality(df, "col", "Pie")
        assert any(isinstance(t, go.Pie) for t in fig.data)

    def test_medium_produces_bar(self):
        df = pd.DataFrame({"col": _make_medium_series()})
        fig = ch.build_chart_for_cardinality(df, "col", "HBar")
        assert any(isinstance(t, go.Bar) for t in fig.data)

    def test_high_produces_treemap(self):
        df = pd.DataFrame({"col": _make_high_series()})
        fig = ch.build_chart_for_cardinality(df, "col", "TM")
        assert any(isinstance(t, go.Treemap) for t in fig.data)

    def test_extreme_produces_table(self):
        df = pd.DataFrame({"col": _make_extreme_series()})
        fig = ch.build_chart_for_cardinality(df, "col", "RT")
        assert any(isinstance(t, go.Table) for t in fig.data)


# ===========================================================================
# _dispatch_two_column  (via build_chart_for_cardinality)
# ===========================================================================

class TestDispatchTwoColumn:
    def test_cat_plus_numeric_produces_bar(self):
        df = _make_df_with_num()
        fig = ch.build_chart_for_cardinality(
            df, "category", "Agg", second_col="revenue"
        )
        assert any(isinstance(t, go.Bar) for t in fig.data)

    def test_two_cats_produces_heatmap(self):
        df = _make_df_with_two_cats()
        fig = ch.build_chart_for_cardinality(
            df, "col_a", "Heatmap", second_col="col_b"
        )
        assert any(isinstance(t, go.Heatmap) for t in fig.data)

    def test_second_col_not_in_df_falls_back_to_single(self):
        """second_col provided but absent -> single-column dispatch."""
        df = pd.DataFrame({"col": _make_low_series()})
        fig = ch.build_chart_for_cardinality(
            df, "col", "Pie", second_col="nonexistent"
        )
        assert any(isinstance(t, go.Pie) for t in fig.data)


# ===========================================================================
# build_chart_for_cardinality — guard clauses
# ===========================================================================

class TestBuildChartForCardinalityGuards:
    def test_missing_col_returns_none(self):
        df = pd.DataFrame({"other": [1, 2, 3]})
        result = ch.build_chart_for_cardinality(df, "nonexistent", "Title")
        assert result is None

    def test_all_null_col_returns_none(self):
        df = pd.DataFrame({"col": [None, None, None]})
        result = ch.build_chart_for_cardinality(df, "col", "Title")
        assert result is None

    def test_valid_col_returns_figure(self):
        df = pd.DataFrame({"col": _make_low_series()})
        result = ch.build_chart_for_cardinality(df, "col", "Title")
        assert isinstance(result, go.Figure)

    def test_agg_func_forwarded(self):
        df = _make_df_with_num()
        fig = ch.build_chart_for_cardinality(
            df, "category", "Sum Bar", second_col="revenue", agg_func="sum"
        )
        assert isinstance(fig, go.Figure)

    def test_top_n_forwarded(self):
        df = pd.DataFrame({"col": _make_high_series()})
        fig = ch.build_chart_for_cardinality(df, "col", "TM", top_n=5)
        assert isinstance(fig, go.Figure)


# ===========================================================================
# get_cardinality_prompt_rules
# ===========================================================================

class TestGetCardinalityPromptRules:
    def test_returns_string(self):
        result = ch.get_cardinality_prompt_rules()
        assert isinstance(result, str)

    def test_default_top_n_in_output(self):
        result = ch.get_cardinality_prompt_rules()
        assert str(ch.TOP_N_DEFAULT) in result

    def test_custom_top_n_in_output(self):
        result = ch.get_cardinality_prompt_rules(top_n=7)
        assert "7" in result

    def test_contains_key_terms(self):
        result = ch.get_cardinality_prompt_rules()
        for term in ("LOW", "MEDIUM", "HIGH", "EXTREME", "treemap", "horizontal bar"):
            assert term in result


# ===========================================================================
# Module-level constants sanity checks
# ===========================================================================

class TestModuleConstants:
    def test_cardinality_thresholds_ordered(self):
        assert ch.CARDINALITY_LOW < ch.CARDINALITY_MEDIUM < ch.CARDINALITY_HIGH

    def test_top_n_default_positive(self):
        assert ch.TOP_N_DEFAULT > 0

    def test_min_others_share_between_0_and_1(self):
        assert 0 < ch.MIN_OTHERS_SHARE < 1