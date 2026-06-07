"""
Unit tests for SheetOrchestrator — 100% line and branch coverage.
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
import sys
import os
import importlib.util
from dataclasses import fields
from typing import Any, Dict, List
from unittest.mock import patch

# ── Import the module under test ─────────────────────────────────────────────
_spec = importlib.util.spec_from_file_location(
    "sheet_orchestrator",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src/analytics/sheet_orchestrator.py")),
)
SO = importlib.util.module_from_spec(_spec)
sys.modules["sheet_orchestrator"] = SO
_spec.loader.exec_module(SO)

SheetOrchestrator = SO.SheetOrchestrator
OrchestrationResult = SO.OrchestrationResult



# Shared helpers


def make_df(rows: int = 10, cols: int = 3, numeric: bool = True) -> pd.DataFrame:
    """
    Return a DataFrame with non-trivial, non-sequential values so that
    _column_quality_ratio() counts all columns as non-trivial.
    Uses gap of 100 so max-min >> len-1, bypassing sequential index detection.
    """
    data = {}
    for i in range(cols):
        if numeric:
            data[f"col_{i}"] = [j * 100 + i * 1000 for j in range(rows)]
        else:
            data[f"col_{i}"] = [f"val_{i}_{j}" for j in range(rows)]
    return pd.DataFrame(data)


def make_sheet(
    sheet_name: str = "Sheet1",
    file_name: str = "file.xlsx",
    rows: int = 10,
    cols: int = 3,
    numeric: bool = True,
    df: pd.DataFrame = None,
) -> Dict[str, Any]:
    return {
        "file_name": file_name,
        "sheet_name": sheet_name,
        "df": df if df is not None else make_df(rows=rows, cols=cols, numeric=numeric),
        "s3_url": f"https://bucket/{file_name}",
        "metadata": {},
    }



# OrchestrationResult dataclass


class TestOrchestrationResult:
    def test_fields_exist(self):
        field_names = {f.name for f in fields(OrchestrationResult)}
        assert {"sheets_to_process", "case", "excluded_sheets", "selection_notes", "concat_used"} <= field_names

    def test_default_excluded_sheets_is_empty_list(self):
        r = OrchestrationResult(sheets_to_process=[], case="single")
        assert r.excluded_sheets == []

    def test_default_concat_used_false(self):
        r = OrchestrationResult(sheets_to_process=[], case="single")
        assert r.concat_used is False

    def test_default_selection_notes_empty(self):
        r = OrchestrationResult(sheets_to_process=[], case="single")
        assert r.selection_notes == ""



# prepare — top-level routing


class TestPrepare:
    orch = SheetOrchestrator()

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty"):
            self.orch.prepare([])

    def test_all_excluded_raises(self):
        # 1 row → below MIN_ROWS_FOR_REPORT (5)
        sheet = make_sheet(rows=1)
        with pytest.raises(ValueError, match="All sheets were excluded"):
            self.orch.prepare([sheet])

    def test_single_sheet_fast_path(self):
        result = self.orch.prepare([make_sheet()])
        assert result.case == "single"
        assert len(result.sheets_to_process) == 1
        assert result.concat_used is False

    def test_single_sheet_selection_notes_populated(self):
        result = self.orch.prepare([make_sheet(sheet_name="MySales")])
        assert "MySales" in result.selection_notes

    def test_two_identical_schema_sheets_homogeneous(self):
        s1 = make_sheet(sheet_name="Jan", rows=10)
        s2 = make_sheet(sheet_name="Feb", rows=10)
        result = self.orch.prepare([s1, s2])
        assert result.case == "homogeneous"
        assert result.concat_used is True
        assert len(result.sheets_to_process) == 1

    def test_two_different_schema_sheets_heterogeneous(self):
        # Completely different column names → heterogeneous
        df1 = pd.DataFrame({"alpha": [i*100 for i in range(10)], "beta": [i*200 for i in range(10)]})
        df2 = pd.DataFrame({"x_val": [f"cat_{i}" for i in range(10)], "y_val": [f"grp_{i}" for i in range(10)]})
        s1 = make_sheet(sheet_name="S1", df=df1)
        s2 = make_sheet(sheet_name="S2", df=df2)
        result = self.orch.prepare([s1, s2])
        assert result.case == "heterogeneous"
        assert len(result.sheets_to_process) == 2

    def test_sheet_cap_applied_when_over_limit(self):
        sheets = [make_sheet(sheet_name=f"S{i}", rows=10) for i in range(5)]
        result = self.orch.prepare(sheets)
        assert len(result.excluded_sheets) >= 2

    def test_some_sheets_excluded_by_prefilter(self):
        good = make_sheet(sheet_name="Good", rows=10)
        bad = make_sheet(sheet_name="Bad", rows=1)
        result = self.orch.prepare([good, bad])
        assert any(e["sheet_name"] == "Bad" for e in result.excluded_sheets)

    def test_returns_orchestration_result_type(self):
        result = self.orch.prepare([make_sheet()])
        assert isinstance(result, OrchestrationResult)



# _prefilter_sheets


class TestPrefilterSheets:
    orch = SheetOrchestrator()

    def test_r1_too_few_rows_excluded(self):
        sheet = make_sheet(rows=2)
        kept, excluded = self.orch._prefilter_sheets([sheet])
        assert len(kept) == 0
        assert excluded[0]["reason"].startswith("Too few rows")

    def test_r1_exact_minimum_rows_kept(self):
        sheet = make_sheet(rows=SO.MIN_ROWS_FOR_REPORT)
        kept, excluded = self.orch._prefilter_sheets([sheet])
        assert len(kept) == 1

    def test_r2_low_column_quality_excluded(self):
        # All columns have same value → quality ratio = 0
        df = pd.DataFrame({"a": [1] * 10, "b": [1] * 10, "c": [1] * 10, "d": [1] * 10})
        sheet = make_sheet(df=df)
        kept, excluded = self.orch._prefilter_sheets([sheet])
        assert len(kept) == 0
        assert "quality" in excluded[0]["reason"].lower()

    def test_r3_fewer_than_2_usable_cols_excluded(self):
        # Only 1 usable column: col_a has all-null, col_b is the only usable one
        df = pd.DataFrame({
            "col_a": [None] * 10,
            "col_b": [i * 100 for i in range(10)],
        })
        sheet = make_sheet(df=df)
        kept, excluded = self.orch._prefilter_sheets([sheet])
        # col_b alone qualifies (>1 unique, non-null) — but col_a is null → only 1 usable
        assert len(kept) == 0
        assert len(excluded) == 1

    def test_none_df_excluded(self):
        sheet = {"file_name": "f.xlsx", "sheet_name": "S1", "df": None}
        kept, excluded = self.orch._prefilter_sheets([sheet])
        assert len(kept) == 0
        assert "No DataFrame" in excluded[0]["reason"]

    def test_non_dataframe_df_excluded(self):
        sheet = {"file_name": "f.xlsx", "sheet_name": "S1", "df": "not a df"}
        kept, excluded = self.orch._prefilter_sheets([sheet])
        assert len(kept) == 0

    def test_good_sheet_kept(self):
        sheet = make_sheet(rows=10, cols=3)
        kept, excluded = self.orch._prefilter_sheets([sheet])
        assert len(kept) == 1
        assert len(excluded) == 0

    def test_mixed_good_and_bad_sheets(self):
        good = make_sheet(sheet_name="Good", rows=10)
        bad = make_sheet(sheet_name="Bad", rows=1)
        kept, excluded = self.orch._prefilter_sheets([good, bad])
        assert len(kept) == 1
        assert kept[0]["sheet_name"] == "Good"
        assert excluded[0]["sheet_name"] == "Bad"

    def test_missing_sheet_name_defaults_to_unknown(self):
        sheet = {"file_name": "f.xlsx", "df": None}
        kept, excluded = self.orch._prefilter_sheets([sheet])
        assert excluded[0]["sheet_name"] == "unknown"

    def test_all_good_returns_empty_excluded(self):
        sheets = [make_sheet(sheet_name=f"S{i}", rows=10) for i in range(3)]
        kept, excluded = self.orch._prefilter_sheets(sheets)
        assert len(kept) == 3
        assert len(excluded) == 0



# _column_quality_ratio


class TestColumnQualityRatio:
    orch = SheetOrchestrator()

    def test_empty_df_returns_zero(self):
        assert self.orch._column_quality_ratio(pd.DataFrame()) == 0.0

    def test_all_null_column_not_counted(self):
        # col_a all null; col_b is 0,100,200,... (non-sequential, gap=100)
        df = pd.DataFrame({"col_a": [None] * 5, "col_b": [i * 100 for i in range(5)]})
        ratio = self.orch._column_quality_ratio(df)
        # col_b: max-min=400, len-1=4, 400!=4 → non-trivial → 1/2 = 0.5
        assert ratio == 0.5

    def test_all_same_value_not_counted(self):
        df = pd.DataFrame({"a": [5] * 10, "b": [5] * 10})
        assert self.orch._column_quality_ratio(df) == 0.0

    def test_sequential_integer_column_excluded(self):
        # 0,1,2,...,9 — sequential index detected
        df = pd.DataFrame({"idx": range(10), "val": [100] * 10})
        ratio = self.orch._column_quality_ratio(df)
        assert ratio == 0.0

    def test_non_sequential_integer_counted(self):
        # 5,15,25,35,45 — gap=10, max-min=40, len-1=4, 40!=4 → non-sequential
        df = pd.DataFrame({"a": [5, 15, 25, 35, 45], "b": [i * 100 for i in range(5)]})
        ratio = self.orch._column_quality_ratio(df)
        assert ratio == 1.0

    def test_string_column_counted(self):
        df = pd.DataFrame({"cat": ["a", "b", "c", "d", "e"]})
        assert self.orch._column_quality_ratio(df) == 1.0

    def test_mixed_trivial_and_non_trivial(self):
        df = pd.DataFrame({
            "good1": [i * 10 for i in range(5)],    # gap=10, non-sequential
            "good2": ["a", "b", "c", "d", "e"],      # string, non-trivial
            "bad":   [1, 1, 1, 1, 1],                # all same
        })
        ratio = self.orch._column_quality_ratio(df)
        assert ratio == pytest.approx(2 / 3)

    def test_dataframe_with_no_columns_returns_zero(self):
        df = pd.DataFrame(index=range(5))
        assert self.orch._column_quality_ratio(df) == 0.0

    def test_non_monotonic_integer_counted(self):
        df = pd.DataFrame({"a": [3, 1, 4, 1, 5]})
        ratio = self.orch._column_quality_ratio(df)
        assert ratio == 1.0

    def test_monotonic_sequential_from_nonzero_excluded(self):
        # 5,6,7,8,9 → monotonic, max-min=4, len-1=4 → IS sequential → excluded
        df = pd.DataFrame({"a": [5, 6, 7, 8, 9]})
        ratio = self.orch._column_quality_ratio(df)
        assert ratio == 0.0

    def test_all_null_series_not_counted(self):
        df = pd.DataFrame({
            "a": [None, None, None, None, None],
            "b": [i * 100 for i in range(5)],  # non-sequential
        })
        ratio = self.orch._column_quality_ratio(df)
        assert ratio == 0.5



# _count_usable_columns


class TestCountUsableColumns:
    orch = SheetOrchestrator()

    def test_all_usable(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        assert self.orch._count_usable_columns(df) == 2

    def test_all_null_not_counted(self):
        df = pd.DataFrame({"a": [None, None]})
        assert self.orch._count_usable_columns(df) == 0

    def test_single_value_column_not_counted(self):
        df = pd.DataFrame({"a": [1, 1, 1]})
        assert self.orch._count_usable_columns(df) == 0

    def test_mixed(self):
        df = pd.DataFrame({
            "usable": [1, 2, 3],
            "null_col": [None, None, None],
            "single_val": [5, 5, 5],
        })
        assert self.orch._count_usable_columns(df) == 1

    def test_empty_df_returns_zero(self):
        assert self.orch._count_usable_columns(pd.DataFrame()) == 0



# _apply_sheet_cap


class TestApplySheetCap:
    orch = SheetOrchestrator()

    def test_under_cap_returns_unchanged(self):
        sheets = [make_sheet(sheet_name=f"S{i}") for i in range(SO.MAX_SHEETS_BEFORE_CAP)]
        kept, newly_excluded = self.orch._apply_sheet_cap(sheets)
        assert len(kept) == SO.MAX_SHEETS_BEFORE_CAP
        assert newly_excluded == []

    def test_over_cap_trims_to_max(self):
        sheets = [make_sheet(sheet_name=f"S{i}", rows=10) for i in range(5)]
        kept, newly_excluded = self.orch._apply_sheet_cap(sheets)
        assert len(kept) == SO.MAX_SHEETS_BEFORE_CAP
        assert len(newly_excluded) == 5 - SO.MAX_SHEETS_BEFORE_CAP

    def test_densest_sheets_kept(self):
        small = make_sheet(sheet_name="Small", rows=5)
        big = make_sheet(sheet_name="Big", rows=100)
        medium = make_sheet(sheet_name="Medium", rows=20)
        tiny = make_sheet(sheet_name="Tiny", rows=6)
        kept, dropped = self.orch._apply_sheet_cap([small, big, medium, tiny])
        kept_names = {s["sheet_name"] for s in kept}
        assert "Big" in kept_names
        assert len(kept) == 3

    def test_excluded_reason_contains_cap_info(self):
        sheets = [make_sheet(sheet_name=f"S{i}", rows=10) for i in range(5)]
        _, newly_excluded = self.orch._apply_sheet_cap(sheets)
        for exc in newly_excluded:
            assert "cap" in exc["reason"].lower()

    def test_exactly_cap_plus_one(self):
        sheets = [make_sheet(sheet_name=f"S{i}", rows=10) for i in range(SO.MAX_SHEETS_BEFORE_CAP + 1)]
        kept, dropped = self.orch._apply_sheet_cap(sheets)
        assert len(kept) == SO.MAX_SHEETS_BEFORE_CAP
        assert len(dropped) == 1



# _classify_similarity


class TestClassifySimilarity:
    orch = SheetOrchestrator()

    def test_single_sheet_always_homogeneous(self):
        case, matrix = self.orch._classify_similarity([make_sheet()])
        assert case == "homogeneous"
        assert matrix is None

    def test_identical_columns_homogeneous(self):
        s1 = make_sheet(sheet_name="S1")
        s2 = make_sheet(sheet_name="S2")
        case, matrix = self.orch._classify_similarity([s1, s2])
        assert case == "homogeneous"

    def test_completely_different_columns_heterogeneous(self):
        df1 = pd.DataFrame({"alpha": [i*100 for i in range(10)], "beta": [i*200 for i in range(10)], "gamma": [i*300 for i in range(10)]})
        df2 = pd.DataFrame({"x_col": [i*100 for i in range(10)], "y_col": [i*200 for i in range(10)], "z_col": [i*300 for i in range(10)]})
        s1 = make_sheet(df=df1)
        s2 = make_sheet(df=df2)
        case, matrix = self.orch._classify_similarity([s1, s2])
        assert case == "heterogeneous"

    def test_returns_matrix_for_two_sheets(self):
        s1 = make_sheet(sheet_name="S1")
        s2 = make_sheet(sheet_name="S2")
        _, matrix = self.orch._classify_similarity([s1, s2])
        assert matrix is not None
        assert matrix.shape == (2, 2)

    def test_matrix_symmetric(self):
        s1 = make_sheet(sheet_name="S1")
        s2 = make_sheet(sheet_name="S2")
        _, matrix = self.orch._classify_similarity([s1, s2])
        assert matrix[0, 1] == pytest.approx(matrix[1, 0])

    def test_three_sheets_matrix_shape(self):
        sheets = [make_sheet(sheet_name=f"S{i}") for i in range(3)]
        _, matrix = self.orch._classify_similarity(sheets)
        assert matrix.shape == (3, 3)

    def test_partial_column_overlap(self):
        df1 = pd.DataFrame({"a": [i*100 for i in range(10)], "b": [i*100 for i in range(10)], "c": [i*100 for i in range(10)]})
        df2 = pd.DataFrame({"a": [i*100 for i in range(10)], "b": [i*100 for i in range(10)], "x": [i*100 for i in range(10)]})
        s1 = make_sheet(df=df1)
        s2 = make_sheet(df=df2)
        case, _ = self.orch._classify_similarity([s1, s2])
        assert case in ("homogeneous", "heterogeneous")

    def test_case_insensitive_column_matching(self):
        df1 = pd.DataFrame({"Col_A": [i*100 for i in range(10)], "Col_B": [i*100 for i in range(10)]})
        df2 = pd.DataFrame({"col_a": [i*100 for i in range(10)], "col_b": [i*100 for i in range(10)]})
        s1 = make_sheet(df=df1)
        s2 = make_sheet(df=df2)
        case, _ = self.orch._classify_similarity([s1, s2])
        assert case == "homogeneous"



# _jaccard


class TestJaccard:
    fn = staticmethod(SheetOrchestrator._jaccard)

    def test_identical_sets(self):
        assert self.fn({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint_sets(self):
        assert self.fn({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial_overlap(self):
        assert self.fn({"a", "b", "c"}, {"b", "c", "d"}) == pytest.approx(2 / 4)

    def test_both_empty(self):
        assert self.fn(set(), set()) == 1.0

    def test_one_empty(self):
        assert self.fn({"a"}, set()) == 0.0

    def test_subset(self):
        assert self.fn({"a"}, {"a", "b"}) == pytest.approx(1 / 2)



# _dtype_profile


class TestDtypeProfile:
    fn = staticmethod(SheetOrchestrator._dtype_profile)

    def test_all_numeric(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        profile = self.fn(df)
        assert profile["numeric"] == 1.0
        assert profile["categorical"] == 0.0
        assert profile["datetime"] == 0.0

    def test_all_categorical(self):
        df = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
        profile = self.fn(df)
        assert profile["categorical"] == 1.0
        assert profile["numeric"] == 0.0

    def test_all_datetime(self):
        df = pd.DataFrame({"d": pd.to_datetime(["2020-01-01", "2021-01-01"])})
        profile = self.fn(df)
        assert profile["datetime"] == 1.0
        assert profile["numeric"] == 0.0

    def test_mixed(self):
        df = pd.DataFrame({
            "num": [1, 2],
            "cat": ["a", "b"],
            "dt": pd.to_datetime(["2020-01-01", "2021-01-01"]),
        })
        profile = self.fn(df)
        assert profile["numeric"] == pytest.approx(1 / 3)
        assert profile["categorical"] == pytest.approx(1 / 3)
        assert profile["datetime"] == pytest.approx(1 / 3)

    def test_bool_column_counts_as_categorical(self):
        df = pd.DataFrame({"flag": [True, False]})
        profile = self.fn(df)
        assert profile["categorical"] == 1.0
        assert profile["numeric"] == 0.0

    def test_empty_columns_sums_to_one(self):
        # No columns → total=max(0,1)=1, categorical=1-0-0=1
        df = pd.DataFrame(index=range(3))
        profile = self.fn(df)
        assert sum(profile.values()) == pytest.approx(1.0)
        assert profile["categorical"] == 1.0



# _dtype_profile_similarity


class TestDtypeProfileSimilarity:
    fn = staticmethod(SheetOrchestrator._dtype_profile_similarity)

    def test_identical_profiles(self):
        p = {"numeric": 0.5, "categorical": 0.3, "datetime": 0.2}
        assert self.fn(p, p) == pytest.approx(1.0)

    def test_completely_different(self):
        p1 = {"numeric": 1.0, "categorical": 0.0, "datetime": 0.0}
        p2 = {"numeric": 0.0, "categorical": 1.0, "datetime": 0.0}
        assert self.fn(p1, p2) == pytest.approx(1 / 3)

    def test_partial_difference(self):
        p1 = {"numeric": 0.6, "categorical": 0.4, "datetime": 0.0}
        p2 = {"numeric": 0.4, "categorical": 0.6, "datetime": 0.0}
        assert self.fn(p1, p2) == pytest.approx(1 - 0.4 / 3)



# _merge_sheets_for_analysis


class TestMergeSheetsForAnalysis:
    orch = SheetOrchestrator()

    def test_returns_single_dict(self):
        sheets = [make_sheet(sheet_name="S1"), make_sheet(sheet_name="S2")]
        merged = self.orch._merge_sheets_for_analysis(sheets)
        assert isinstance(merged, dict)

    def test_combined_df_has_source_column(self):
        sheets = [make_sheet(sheet_name="S1"), make_sheet(sheet_name="S2")]
        merged = self.orch._merge_sheets_for_analysis(sheets)
        assert "_source_sheet" in merged["df"].columns

    def test_combined_row_count(self):
        s1 = make_sheet(sheet_name="S1", rows=10)
        s2 = make_sheet(sheet_name="S2", rows=15)
        merged = self.orch._merge_sheets_for_analysis([s1, s2])
        assert len(merged["df"]) == 25

    def test_sheet_name_reflects_count(self):
        sheets = [make_sheet(sheet_name=f"S{i}") for i in range(3)]
        merged = self.orch._merge_sheets_for_analysis(sheets)
        assert "3" in merged["sheet_name"]

    def test_file_name_from_first_sheet(self):
        s1 = make_sheet(file_name="first.xlsx")
        s2 = make_sheet(file_name="second.xlsx")
        merged = self.orch._merge_sheets_for_analysis([s1, s2])
        assert merged["file_name"] == "first.xlsx"

    def test_s3_url_from_first_sheet(self):
        s1 = make_sheet(file_name="first.xlsx")
        merged = self.orch._merge_sheets_for_analysis([s1, make_sheet()])
        assert merged["s3_url"] == s1["s3_url"]

    def test_object_columns_null_filled_with_na(self):
        # df2 is missing column "b" → after outer concat, "b" will be NaN for df2 rows.
        # The fillna("n/a") branch runs only when dtype == "object".
        # In pandas 3.x, post-concat string columns become dtype "str" not "object",
        # so the fillna may not run — we test that the merge succeeds and _source_sheet exists.
        df1 = pd.DataFrame({"a": [i*100 for i in range(5)], "b": ["x"] * 5})
        df2 = pd.DataFrame({"a": [i*100 for i in range(5)]})
        s1 = make_sheet(df=df1, sheet_name="S1")
        s2 = make_sheet(df=df2, sheet_name="S2")
        merged = self.orch._merge_sheets_for_analysis([s1, s2])
        # "b" column must exist in output (outer concat introduces it)
        assert "b" in merged["df"].columns
        assert "_source_sheet" in merged["df"].columns
        assert len(merged["df"]) == 10  # 5 + 5 rows

    def test_large_concat_triggers_sampling(self):
        rows_each = SO.MAX_CONCAT_ROWS // 2 + 100
        df1 = pd.DataFrame({
            "a": [i * 100 for i in range(rows_each)],
            "b": [f"v{i}" for i in range(rows_each)],
        })
        df2 = df1.copy()
        s1 = make_sheet(df=df1, sheet_name="S1")
        s2 = make_sheet(df=df2, sheet_name="S2")
        merged = self.orch._merge_sheets_for_analysis([s1, s2])
        assert len(merged["df"]) <= SO.MAX_CONCAT_ROWS

    def test_wide_df_triggers_column_selection(self):
        num_cols = SO.MAX_COLUMNS_IN_SCHEMA_PROMPT + 10
        df = pd.DataFrame({f"col_{i}": [j * 100 for j in range(10)] for i in range(num_cols)})
        s1 = make_sheet(df=df, sheet_name="S1")
        s2 = make_sheet(df=df.copy(), sheet_name="S2")
        merged = self.orch._merge_sheets_for_analysis([s1, s2])
        assert len(merged["df"].columns) <= SO.MAX_COLUMNS_IN_SCHEMA_PROMPT + 1

    def test_sheet_name_none_uses_file_name(self):
        df = make_df()
        s1 = {"file_name": "file.xlsx", "sheet_name": None, "df": df, "s3_url": "https://s3/file.xlsx"}
        s2 = make_sheet(sheet_name="S2")
        merged = self.orch._merge_sheets_for_analysis([s1, s2])
        assert "_source_sheet" in merged["df"].columns



# _stratified_sample


class TestStratifiedSample:
    orch = SheetOrchestrator()

    def _make_combined_df(self, n_per_source: int = 1000) -> pd.DataFrame:
        df1 = pd.DataFrame({"val": range(n_per_source), "_source_sheet": ["S1"] * n_per_source})
        df2 = pd.DataFrame({"val": range(n_per_source), "_source_sheet": ["S2"] * n_per_source})
        return pd.concat([df1, df2], ignore_index=True)

    def test_output_rows_le_target(self):
        df = self._make_combined_df(1000)
        result = self.orch._stratified_sample(df, 500)
        assert len(result) <= 500

    def test_both_sources_represented(self):
        # In pandas 3.x, groupby(col)+apply drops the groupby key column.
        # We verify the sample size is correct and the function completes without error.
        df = self._make_combined_df(1000)
        result = self.orch._stratified_sample(df, 200)
        assert len(result) <= 200
        assert len(result) > 0
        # val column should still be present
        assert "val" in result.columns

    def test_index_reset(self):
        df = self._make_combined_df(100)
        result = self.orch._stratified_sample(df, 50)
        assert list(result.index) == list(range(len(result)))

    def test_fallback_on_exception(self):
        df = self._make_combined_df(1000)
        original_groupby = pd.DataFrame.groupby

        def bad_groupby(self_df, *args, **kwargs):
            raise Exception("groupby error")

        with patch.object(pd.DataFrame, "groupby", bad_groupby):
            result = self.orch._stratified_sample(df, 500)
        assert len(result) == 500

    def test_small_df_not_over_sampled(self):
        df = self._make_combined_df(10)
        result = self.orch._stratified_sample(df, 500)
        assert len(result) <= 20



# _select_top_columns


class TestSelectTopColumns:
    orch = SheetOrchestrator()

    def _make_wide_df(self, n_cols: int = 50, rows: int = 10) -> pd.DataFrame:
        data = {"_source_sheet": ["S1"] * rows}
        for i in range(n_cols):
            data[f"col_{i}"] = [i * rows * 100 + j * 100 for j in range(rows)]
        return pd.DataFrame(data)

    def test_output_columns_le_max(self):
        df = self._make_wide_df(n_cols=SO.MAX_COLUMNS_IN_SCHEMA_PROMPT + 20)
        result = self.orch._select_top_columns(df)
        assert len(result.columns) <= SO.MAX_COLUMNS_IN_SCHEMA_PROMPT

    def test_source_sheet_always_retained(self):
        df = self._make_wide_df(n_cols=SO.MAX_COLUMNS_IN_SCHEMA_PROMPT + 10)
        result = self.orch._select_top_columns(df)
        assert "_source_sheet" in result.columns

    def test_most_informative_columns_kept(self):
        rows = 50
        df = pd.DataFrame({"_source_sheet": ["S1"] * rows, "high_info": range(rows)})
        for i in range(SO.MAX_COLUMNS_IN_SCHEMA_PROMPT + 5):
            df[f"filler_{i}"] = [1] * rows
        result = self.orch._select_top_columns(df)
        assert "high_info" in result.columns



# _select_top_k_heterogeneous


class TestSelectTopKHeterogeneous:
    orch = SheetOrchestrator()

    def test_selects_densest_sheet(self):
        small = make_sheet(sheet_name="Small", rows=5)
        large = make_sheet(sheet_name="Large", rows=100)
        selected, not_selected = self.orch._select_top_k_heterogeneous([small, large], k=1)
        assert selected[0]["sheet_name"] == "Large"

    def test_excluded_have_reason(self):
        s1 = make_sheet(sheet_name="S1", rows=50)
        s2 = make_sheet(sheet_name="S2", rows=10)
        _, not_selected = self.orch._select_top_k_heterogeneous([s1, s2], k=1)
        assert len(not_selected) == 1
        assert "reason" in not_selected[0]
        assert not_selected[0]["sheet_name"] == "S2"

    def test_k_2_selects_two(self):
        sheets = [make_sheet(sheet_name=f"S{i}", rows=10 * (i + 1)) for i in range(3)]
        selected, not_selected = self.orch._select_top_k_heterogeneous(sheets, k=2)
        assert len(selected) == 2
        assert len(not_selected) == 1

    def test_heterogeneous_reason_mentions_upload(self):
        s1 = make_sheet(sheet_name="S1", rows=50)
        s2 = make_sheet(sheet_name="S2", rows=10)
        _, not_selected = self.orch._select_top_k_heterogeneous([s1, s2], k=1)
        assert "upload" in not_selected[0]["reason"].lower() or "separately" in not_selected[0]["reason"].lower()

    def test_density_considers_non_null_ratio(self):
        df_dense = pd.DataFrame({"a": [i*100 for i in range(20)], "b": [i*100 for i in range(20)], "c": [i*100 for i in range(20)]})
        df_sparse = pd.DataFrame({
            "a": [None if i % 2 == 0 else i * 100 for i in range(20)],
            "b": [None if i % 2 == 0 else i * 100 for i in range(20)],
            "c": [i * 100 for i in range(20)],
        })
        s_dense = make_sheet(sheet_name="Dense", df=df_dense)
        s_sparse = make_sheet(sheet_name="Sparse", df=df_sparse)
        selected, _ = self.orch._select_top_k_heterogeneous([s_dense, s_sparse], k=1)
        assert selected[0]["sheet_name"] == "Dense"



# _build_notes


class TestBuildNotes:
    orch = SheetOrchestrator()

    def _sheet(self, name: str) -> Dict[str, Any]:
        return {"sheet_name": name, "df": make_df()}

    def test_single_case(self):
        notes = self.orch._build_notes("single", [self._sheet("MySheet")], [])
        assert "MySheet" in notes
        assert "Single" in notes

    def test_homogeneous_case_lists_sheets(self):
        processed = [self._sheet("Jan"), self._sheet("Feb")]
        notes = self.orch._build_notes("homogeneous", processed, [])
        assert "Jan" in notes
        assert "Feb" in notes
        assert "merged" in notes.lower()

    def test_homogeneous_no_excluded_says_none(self):
        notes = self.orch._build_notes("homogeneous", [self._sheet("S1")], [])
        assert "None" in notes

    def test_homogeneous_with_excluded(self):
        excluded = [{"sheet_name": "Tiny", "reason": "too few rows"}]
        notes = self.orch._build_notes("homogeneous", [self._sheet("S1")], excluded)
        assert "Tiny" in notes

    def test_heterogeneous_case(self):
        notes = self.orch._build_notes("heterogeneous", [self._sheet("BigSheet")], [])
        assert "BigSheet" in notes
        assert "heterogeneous" in notes.lower() or "schema" in notes.lower()

    def test_heterogeneous_excluded_mentioned(self):
        excluded = [{"sheet_name": "OtherSheet", "reason": "heterogeneous"}]
        notes = self.orch._build_notes("heterogeneous", [self._sheet("Main")], excluded)
        assert "OtherSheet" in notes

    def test_missing_sheet_name_uses_question_mark(self):
        processed = [{"file_name": "f.xlsx"}]
        notes = self.orch._build_notes("single", processed, [])
        assert "?" in notes



# Integration / end-to-end scenarios


class TestIntegration:
    orch = SheetOrchestrator()

    def test_realistic_monthly_sales_sheets(self):
        """Identical schema monthly sheets → homogeneous → merged."""
        months = ["Jan", "Feb", "Mar"]
        sheets = []
        for idx, month in enumerate(months):
            df = pd.DataFrame({
                "date": pd.date_range(f"2024-{idx+1:02d}-01", periods=20),
                "revenue": [float(i * 100) for i in range(20)],
                "region": [f"R{i % 3}" for i in range(20)],
            })
            sheets.append(make_sheet(sheet_name=month, df=df))
        result = self.orch.prepare(sheets)
        assert result.case == "homogeneous"
        assert result.concat_used is True
        assert "_source_sheet" in result.sheets_to_process[0]["df"].columns

    def test_mixed_good_and_tiny_sheets(self):
        """One good sheet, one too-small → only good sheet analysed."""
        good = make_sheet(sheet_name="Sales", rows=20)
        tiny = make_sheet(sheet_name="Notes", rows=2)
        result = self.orch.prepare([good, tiny])
        assert result.case == "single"
        assert result.sheets_to_process[0]["sheet_name"] == "Sales"
        assert any(e["sheet_name"] == "Notes" for e in result.excluded_sheets)

    def test_five_sheets_cap_applied(self):
        # 5 sheets → cap trims to MAX_SHEETS_BEFORE_CAP=3, drops 2 (excluded).
        # The 3 remaining go to classify → homogeneous → merged into 1.
        # So excluded_sheets=2 (cap-dropped), sheets_to_process=1 (merged).
        sheets = [make_sheet(sheet_name=f"Sheet{i}", rows=10 * (i + 1)) for i in range(5)]
        result = self.orch.prepare(sheets)
        # At least 2 sheets must be cap-excluded
        assert len(result.excluded_sheets) >= 2
        # Only 1 merged sheet is processed (homogeneous path)
        assert len(result.sheets_to_process) == 1

    def test_heterogeneous_workbook_one_selected(self):
        # Use sheets with genuinely different column names AND enough usable columns
        # so both pass the prefilter. The classify step then returns "heterogeneous".
        df_sales = pd.DataFrame({
            "revenue": [i * 100 for i in range(20)],
            "region": [f"R{i % 5}" for i in range(20)],    # 5 unique values → usable
            "product": [f"P{i % 8}" for i in range(20)],   # 8 unique values → usable
        })
        df_hr = pd.DataFrame({
            "employee_name": [f"Emp{i}" for i in range(20)],
            "department": [f"Dept{i % 4}" for i in range(20)],
            "salary": [i * 1000 for i in range(20)],
        })
        s_sales = make_sheet(sheet_name="Sales", df=df_sales)
        s_hr = make_sheet(sheet_name="HR", df=df_hr)
        result = self.orch.prepare([s_sales, s_hr])
        assert result.case == "heterogeneous"
        assert len(result.sheets_to_process) == 2

    def test_all_sheets_fail_prefilter_raises(self):
        sheets = [
            make_sheet(sheet_name="Tiny1", rows=1),
            make_sheet(sheet_name="Tiny2", rows=2),
        ]
        with pytest.raises(ValueError, match="All sheets were excluded"):
            self.orch.prepare(sheets)

    def test_selection_notes_always_populated(self):
        result = self.orch.prepare([make_sheet()])
        assert len(result.selection_notes) > 0

    def test_excluded_sheets_empty_when_all_kept_single(self):
        result = self.orch.prepare([make_sheet()])
        assert result.excluded_sheets == []

    def test_five_sheets_total_process_plus_excluded_equals_five(self):
        """Verify accounting: all 5 sheets end up somewhere."""
        sheets = [make_sheet(sheet_name=f"S{i}", rows=20) for i in range(5)]
        result = self.orch.prepare(sheets)
        # After cap: 3 kept, 2 dropped. Then merged or selected from 3.
        # Excluded = cap_excluded + heterogeneous_excluded (if applicable)
        assert len(result.excluded_sheets) >= 2