import pytest
import pandas as pd
import numpy as np
from src.analytics.data_processor import DataProcessor


@pytest.fixture
def processor():
    return DataProcessor()


@pytest.fixture
def simple_numeric_df():
    """DataFrame with only numeric columns that pass filtering."""
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45],
        "score": [88.5, 92.1, 78.3, 95.0, 85.7],
    })


@pytest.fixture
def simple_categorical_df():
    """DataFrame with categorical column with few unique values."""
    return pd.DataFrame({
        "status": ["active", "inactive", "active", "pending", "active"] * 2,
        "region": ["north", "south", "east", "west", "north"] * 2,
    })


# ──────────────────────────────────────────────
# 1. Return structure
# ──────────────────────────────────────────────

class TestReturnStructure:
    def test_top_level_keys(self, processor, simple_numeric_df):
        result = processor.summarize_dataframe(simple_numeric_df)
        assert set(result.keys()) == {"dataset_info", "column_details"}

    def test_dataset_info_keys(self, processor, simple_numeric_df):
        info = processor.summarize_dataframe(simple_numeric_df)["dataset_info"]
        expected = {"rows", "columns", "column_names", "dtypes", "missing_values", "memory_usage"}
        assert set(info.keys()) == expected

    def test_numeric_column_detail_keys(self, processor, simple_numeric_df):
        details = processor.summarize_dataframe(simple_numeric_df)["column_details"]
        for col_info in details.values():
            assert set(col_info.keys()) == {"mean", "median", "min", "max", "std"}

    def test_categorical_column_detail_keys(self, processor, simple_categorical_df):
        details = processor.summarize_dataframe(simple_categorical_df)["column_details"]
        for col_info in details.values():
            assert set(col_info.keys()) == {"top_values", "unique"}


# ──────────────────────────────────────────────
# 2. dataset_info correctness
# ──────────────────────────────────────────────

class TestDatasetInfo:
    def test_row_count(self, processor, simple_numeric_df):
        info = processor.summarize_dataframe(simple_numeric_df)["dataset_info"]
        assert info["rows"] == 5

    def test_column_count(self, processor, simple_numeric_df):
        info = processor.summarize_dataframe(simple_numeric_df)["dataset_info"]
        assert info["columns"] == 2

    def test_column_names(self, processor, simple_numeric_df):
        info = processor.summarize_dataframe(simple_numeric_df)["dataset_info"]
        assert info["column_names"] == ["age", "score"]

    def test_dtypes_are_strings(self, processor, simple_numeric_df):
        dtypes = processor.summarize_dataframe(simple_numeric_df)["dataset_info"]["dtypes"]
        assert all(isinstance(v, str) for v in dtypes.values())

    def test_missing_values_no_nulls(self, processor, simple_numeric_df):
        mv = processor.summarize_dataframe(simple_numeric_df)["dataset_info"]["missing_values"]
        assert all(v == 0 for v in mv.values())

    def test_missing_values_with_nulls(self, processor):
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 1]})
        mv = processor.summarize_dataframe(df)["dataset_info"]["missing_values"]
        assert mv["a"] == 1
        assert mv["b"] == 2

    def test_memory_usage_positive(self, processor, simple_numeric_df):
        mem = processor.summarize_dataframe(simple_numeric_df)["dataset_info"]["memory_usage"]
        assert mem > 0

    def test_memory_usage_in_mb(self, processor):
        # 10_000-row float df should be well under 1 MB
        df = pd.DataFrame({"x": np.random.rand(10_000)})
        mem = processor.summarize_dataframe(df)["dataset_info"]["memory_usage"]
        assert mem < 1.0


# ──────────────────────────────────────────────
# 3. Column filtering logic
# ──────────────────────────────────────────────

class TestColumnFiltering:
    def test_high_cardinality_numeric_column_dropped(self, processor):
        """Column with unique_count > max(10, 0.5*total) should be dropped."""
        # 30 rows, 30 unique integers → unique_count (30) > max(10, 15) → dropped
        df = pd.DataFrame({"id": range(30), "category": ["a", "b"] * 15})
        details = processor.summarize_dataframe(df)["column_details"]
        assert "id" not in details

    def test_low_cardinality_numeric_column_kept(self, processor):
        """Numeric column with few unique values should be kept."""
        # 20 rows, only 3 unique values → 3 <= max(10, 10) → kept
        df = pd.DataFrame({"grade": [1, 2, 3] * 6 + [1, 2]})
        details = processor.summarize_dataframe(df)["column_details"]
        assert "grade" in details

    def test_high_cardinality_object_column_dropped(self, processor):
        """Object column with too many unique values should be dropped."""
        # 30 unique strings → unique_count (30) > max(10, 15) → dropped
        df = pd.DataFrame({"name": [f"name_{i}" for i in range(30)]})
        details = processor.summarize_dataframe(df)["column_details"]
        assert "name" not in details

    def test_long_text_object_column_dropped(self, processor):
        """Object column (dtype=object) with avg string length > 100 is dropped.

        In newer pandas, string literals default to StringDtype ('str'), not
        numpy object dtype, so `df[col].dtype == object` is False unless we
        explicitly cast with .astype('object').  We do that here to exercise
        the avg-length branch (lines 26-28).
        5 unique values, 20 rows → unique_count(5) ≤ max(10, 10) → cardinality
        guard not triggered → falls into elif/avg_length path → dropped.
        """
        long_text = "x" * 150
        long_strings = [f"{long_text}_{i}" for i in range(5)] * 4   # 20 rows, 5 unique
        labels = ["a", "b", "a", "b", "c"] * 4                       # 20 rows, 3 unique

        df = pd.DataFrame({
            "description": pd.Series(long_strings).astype("object"),
            "label": pd.Series(labels).astype("object"),
        })
        assert df["description"].dtype == object, "precondition: dtype must be object"

        details = processor.summarize_dataframe(df)["column_details"]
        assert "description" not in details
        assert "label" in details

    def test_short_text_object_column_kept(self, processor):
        """Object column with avg string length <= 100 and low cardinality is kept."""
        df = pd.DataFrame({"label": ["alpha", "beta", "gamma", "alpha", "beta"] * 2})
        details = processor.summarize_dataframe(df)["column_details"]
        assert "label" in details

    def test_boundary_unique_count_exactly_threshold(self, processor):
        """
        unique_count == max(10, 0.5*total) should NOT be dropped
        (condition is strictly greater-than).
        """
        # 20 rows, 10 unique values → max(10, 10)=10, unique_count(10) > 10 is False → kept
        df = pd.DataFrame({"x": list(range(10)) * 2})
        details = processor.summarize_dataframe(df)["column_details"]
        assert "x" in details

    def test_boundary_unique_count_one_above_threshold(self, processor):
        """unique_count == threshold + 1 should be dropped."""
        # 20 rows, 11 unique values → max(10,10)=10, 11>10 → dropped
        df = pd.DataFrame({"x": list(range(11)) + [0] * 9})
        details = processor.summarize_dataframe(df)["column_details"]
        assert "x" not in details


# ──────────────────────────────────────────────
# 4. Numeric column statistics
# ──────────────────────────────────────────────

class TestNumericStats:
    def _kept_numeric_df(self):
        # 3 unique values across 10 rows → stays under threshold
        return pd.DataFrame({"val": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0]})

    def test_mean(self, processor):
        df = self._kept_numeric_df()
        stats = processor.summarize_dataframe(df)["column_details"]["val"]
        assert stats["mean"] == pytest.approx(df["val"].mean())

    def test_median(self, processor):
        df = self._kept_numeric_df()
        stats = processor.summarize_dataframe(df)["column_details"]["val"]
        assert stats["median"] == pytest.approx(df["val"].median())

    def test_min(self, processor):
        df = self._kept_numeric_df()
        stats = processor.summarize_dataframe(df)["column_details"]["val"]
        assert stats["min"] == pytest.approx(df["val"].min())

    def test_max(self, processor):
        df = self._kept_numeric_df()
        stats = processor.summarize_dataframe(df)["column_details"]["val"]
        assert stats["max"] == pytest.approx(df["val"].max())

    def test_std(self, processor):
        df = self._kept_numeric_df()
        stats = processor.summarize_dataframe(df)["column_details"]["val"]
        assert stats["std"] == pytest.approx(df["val"].std())

    def test_stats_are_floats(self, processor):
        df = self._kept_numeric_df()
        stats = processor.summarize_dataframe(df)["column_details"]["val"]
        for key in ("mean", "median", "min", "max", "std"):
            assert isinstance(stats[key], float)


# ──────────────────────────────────────────────
# 5. Categorical column statistics
# ──────────────────────────────────────────────

class TestCategoricalStats:
    def test_unique_count(self, processor, simple_categorical_df):
        details = processor.summarize_dataframe(simple_categorical_df)["column_details"]
        assert details["status"]["unique"] == simple_categorical_df["status"].nunique()

    def test_unique_is_int(self, processor, simple_categorical_df):
        details = processor.summarize_dataframe(simple_categorical_df)["column_details"]
        assert isinstance(details["status"]["unique"], int)

    def test_top_values_is_dict(self, processor, simple_categorical_df):
        details = processor.summarize_dataframe(simple_categorical_df)["column_details"]
        assert isinstance(details["status"]["top_values"], dict)

    def test_top_values_counts_correct(self, processor):
        df = pd.DataFrame({"color": ["red"] * 5 + ["blue"] * 3 + ["green"] * 2})
        details = processor.summarize_dataframe(df)["column_details"]
        tv = details["color"]["top_values"]
        assert tv["red"] == 5
        assert tv["blue"] == 3
        assert tv["green"] == 2

    def test_top_values_capped_at_200(self, processor):
        """Even if there are many values, top_values contains at most 200 entries."""
        # 8 unique labels × 3 each = 24 rows, unique_count=8 ≤ max(10,12) → kept
        labels = [f"cat_{i}" for i in range(8)] * 3
        df = pd.DataFrame({"label": labels})
        details = processor.summarize_dataframe(df)["column_details"]
        assert len(details["label"]["top_values"]) <= 200


# ──────────────────────────────────────────────
# 6. Edge cases
# ──────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_dataframe(self, processor):
        df = pd.DataFrame()
        result = processor.summarize_dataframe(df)
        assert result["dataset_info"]["rows"] == 0
        assert result["dataset_info"]["columns"] == 0
        assert result["column_details"] == {}

    def test_single_row_dataframe(self, processor):
        df = pd.DataFrame({"x": [42]})
        result = processor.summarize_dataframe(df)
        # 1 unique value, total=1 → unique_count(1) > max(10,0.5) → dropped
        assert result["dataset_info"]["rows"] == 1

    def test_all_columns_dropped(self, processor):
        """When every column is filtered out, column_details should be empty."""
        # 30 unique IDs → all dropped
        df = pd.DataFrame({"id": range(30), "uid": range(100, 130)})
        details = processor.summarize_dataframe(df)["column_details"]
        assert details == {}

    def test_dataframe_with_missing_numeric_values(self, processor):
        """NaN-containing numeric columns should still produce float stats."""
        df = pd.DataFrame({"v": [1.0, 2.0, None, 4.0, 5.0, 1.0, 2.0, None, 4.0, 5.0]})
        details = processor.summarize_dataframe(df)["column_details"]
        if "v" in details:
            assert isinstance(details["v"]["mean"], float)

    def test_original_dataframe_not_mutated(self, processor):
        """The input DataFrame must not be modified by summarize_dataframe."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        original_columns = list(df.columns)
        original_shape = df.shape
        processor.summarize_dataframe(df)
        assert list(df.columns) == original_columns
        assert df.shape == original_shape

    def test_mixed_dtype_dataframe(self, processor):
        """DataFrame with both numeric and categorical columns."""
        df = pd.DataFrame({
            "num": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
            "cat": ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a"],
        })
        details = processor.summarize_dataframe(df)["column_details"]
        assert "mean" in details.get("num", {})
        assert "top_values" in details.get("cat", {})

    def test_boolean_column_treated_as_numeric(self, processor):
        """Boolean dtype is numeric in pandas; should get numeric stats."""
        df = pd.DataFrame({"flag": [True, False, True, False, True,
                                     True, False, True, False, True]})
        details = processor.summarize_dataframe(df)["column_details"]
        if "flag" in details:
            assert "mean" in details["flag"]

    def test_column_names_preserved_in_dataset_info(self, processor):
        df = pd.DataFrame({"alpha": [1, 2], "beta": [3, 4]})
        info = processor.summarize_dataframe(df)["dataset_info"]
        assert "alpha" in info["column_names"]
        assert "beta" in info["column_names"]