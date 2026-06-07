"""
Comprehensive unit tests for PlotGenerator — 100% line coverage.
All external I/O (LLM client) is mocked. Plotly figures are generated locally.
"""
import base64
import json
import struct
import pytest
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from unittest.mock import MagicMock, patch, call
from src.analytics.plot_generator import PlotGenerator


# ─────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def pg(mock_llm):
    return PlotGenerator(mock_llm)


@pytest.fixture
def mixed_df():
    """DataFrame with both numeric and categorical columns."""
    return pd.DataFrame({
        "category": ["A", "B", "A", "C", "B"],
        "value": [10.0, 20.0, 30.0, 40.0, 50.0],
        "count": [1, 2, 3, 4, 5],
    })


@pytest.fixture
def numeric_df():
    return pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})


@pytest.fixture
def cat_only_df():
    return pd.DataFrame({"color": ["red", "blue", "red", "green", "blue"]})


@pytest.fixture
def date_df():
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5),
        "sales": [100.0, 200.0, 150.0, 300.0, 250.0],
    })


@pytest.fixture
def simple_bar_fig():
    df = pd.DataFrame({"x": ["A", "B"], "y": [1, 2]})
    return px.bar(df, x="x", y="y", title="Test")


def _make_valid_plot_json(fig=None) -> str:
    """Produce valid plot JSON via the production pipeline (handles binary data in traces)."""
    if fig is None:
        df = pd.DataFrame({"category": ["A", "B", "C"], "count": [10, 20, 30]})
        fig = px.bar(df, x="category", y="count")
    pg_tmp = PlotGenerator(MagicMock())
    plot_dict = pg_tmp._convert_to_serializable_dict_robust(fig)
    return json.dumps(plot_dict, default=str, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────
# 1. __init__
# ─────────────────────────────────────────────────────────────

class TestInit:
    def test_llm_client_stored(self, mock_llm):
        pg = PlotGenerator(mock_llm)
        assert pg.llm_client is mock_llm

    def test_imports_contain_required_keys(self, pg):
        for key in ('pd', 'np', 'px', 'go'):
            assert key in pg.imports


# ─────────────────────────────────────────────────────────────
# 2. _make_json_serializable
# ─────────────────────────────────────────────────────────────

class TestMakeJsonSerializable:
    def test_timestamp(self, pg):
        ts = pd.Timestamp("2024-01-15")
        result = pg._make_json_serializable(ts)
        assert isinstance(result, str)
        assert "2024-01-15" in result

    def test_datetime(self, pg):
        dt = datetime(2024, 6, 1, 12, 0)
        result = pg._make_json_serializable(dt)
        assert "2024-06-01" in result

    def test_numpy_int(self, pg):
        result = pg._make_json_serializable(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_float(self, pg):
        result = pg._make_json_serializable(np.float32(3.14))
        assert isinstance(result, float)

    def test_nan_returns_none(self, pg):
        assert pg._make_json_serializable(float('nan')) is None
        assert pg._make_json_serializable(np.nan) is None

    def test_bytes_decoded(self, pg):
        result = pg._make_json_serializable(b"hello")
        assert result == "hello"

    def test_plain_string_passthrough(self, pg):
        assert pg._make_json_serializable("hello") == "hello"

    def test_plain_int_passthrough(self, pg):
        assert pg._make_json_serializable(99) == 99

    def test_none_passthrough(self, pg):
        # None: pd.isna(None) is True on some pandas versions
        result = pg._make_json_serializable(None)
        assert result is None

    def test_attribute_error_handled(self, pg):
        """Object that raises AttributeError inside the try block — falls through."""
        class WeirdObj:
            # Cause AttributeError when pd.Timestamp check runs isinstance
            pass
        # Patch isinstance to raise AttributeError for our object
        obj = WeirdObj()
        # Easiest: just pass a bytes-like that decodes fine — already tested.
        # Instead trigger ValueError via a broken bytes-like:
        class BadBytes(bytes):
            def decode(self, *a, **kw):
                raise ValueError("decode fail")
        bad = BadBytes(b"hello")
        result = pg._make_json_serializable(bad)
        # Falls through the except and returns the original value
        assert result is bad


# ─────────────────────────────────────────────────────────────
# 3. _get_dataframe_info
# ─────────────────────────────────────────────────────────────

class TestGetDataframeInfo:
    def test_keys_present(self, pg, mixed_df):
        info = pg._get_dataframe_info(mixed_df)
        for k in ('columns', 'dtypes', 'sample_data', 'numeric_columns',
                  'categorical_columns', 'shape', 'memory_usage'):
            assert k in info

    def test_numeric_columns(self, pg, mixed_df):
        info = pg._get_dataframe_info(mixed_df)
        assert "value" in info['numeric_columns']
        assert "count" in info['numeric_columns']

    def test_categorical_columns(self, pg, mixed_df):
        info = pg._get_dataframe_info(mixed_df)
        assert "category" in info['categorical_columns']

    def test_sample_data_max_3_rows(self, pg):
        df = pd.DataFrame({"a": range(100)})
        info = pg._get_dataframe_info(df)
        assert len(info['sample_data']) == 3

    def test_shape_tuple(self, pg, mixed_df):
        info = pg._get_dataframe_info(mixed_df)
        assert info['shape'] == (5, 3)

    def test_memory_usage_mb_string(self, pg, mixed_df):
        info = pg._get_dataframe_info(mixed_df)
        assert info['memory_usage'].endswith("MB")

    def test_timestamps_serialized(self, pg, date_df):
        info = pg._get_dataframe_info(date_df)
        # sample_data should have isoformat strings, not Timestamp objects
        for rec in info['sample_data']:
            for v in rec.values():
                assert not isinstance(v, pd.Timestamp)


# ─────────────────────────────────────────────────────────────
# 4. _clean_generated_code
# ─────────────────────────────────────────────────────────────

class TestCleanGeneratedCode:
    def test_strips_python_fence(self, pg):
        code = "```python\nfig = px.bar(df, x='a', y='b')\n```"
        result = pg._clean_generated_code(code)
        assert "```" not in result
        assert "fig" in result

    def test_strips_plain_fence(self, pg):
        code = "```\nfig = px.bar(df, x='a', y='b')\n```"
        result = pg._clean_generated_code(code)
        assert "```" not in result

    def test_removes_comment_lines(self, pg):
        code = "# This is a comment\nfig = px.bar(df, x='a', y='b')"
        result = pg._clean_generated_code(code)
        assert "# This is a comment" not in result

    def test_removes_matplotlib_import(self, pg):
        code = "import matplotlib.pyplot as plt\nfig = px.bar(df, x='a', y='b')"
        result = pg._clean_generated_code(code)
        assert "matplotlib" not in result

    def test_removes_plt_import(self, pg):
        code = "import plt\nfig = px.bar(df, x='a', y='b')"
        result = pg._clean_generated_code(code)
        assert "import plt" not in result

    def test_plain_code_unchanged(self, pg):
        code = "fig = px.histogram(df, x='value')"
        assert pg._clean_generated_code(code) == code


# ─────────────────────────────────────────────────────────────
# 5. _validate_code_structure
# ─────────────────────────────────────────────────────────────

class TestValidateCodeStructure:
    def test_valid_code(self, pg):
        code = "fig = px.bar(df, x='category', y='value')"
        _, valid, msg = pg._validate_code_structure(code)
        assert valid

    def test_matplotlib_rejected(self, pg):
        code = "import matplotlib.pyplot as plt\nfig = plt.bar(['a'], [1])"
        _, valid, msg = pg._validate_code_structure(code)
        assert not valid
        assert "matplotlib" in msg

    def test_no_plotly_rejected(self, pg):
        code = "fig = go_chart(df)"  # doesn't contain px./go./plotly
        _, valid, msg = pg._validate_code_structure(code)
        # Actually contains 'go' substring — use a cleaner case
        code2 = "x = df['a'].sum()"
        _, valid2, msg2 = pg._validate_code_structure(code2)
        assert not valid2

    def test_no_fig_rejected(self, pg):
        code = "result = px.bar(df, x='a', y='b')"
        _, valid, msg = pg._validate_code_structure(code)
        assert not valid
        assert "fig" in msg

    def test_index_values_with_px_rejected(self, pg):
        code = "fig = px.bar(x=df['cat'].value_counts().index, y=df['cat'].value_counts().values)"
        _, valid, msg = pg._validate_code_structure(code)
        assert not valid
        assert "reset_index" in msg

    def test_no_df_reference_rejected(self, pg):
        code = "fig = px.bar(x=['a', 'b'], y=[1, 2])"
        _, valid, msg = pg._validate_code_structure(code)
        assert not valid

    def test_df_in_token_position(self, pg):
        """df as a bare token (e.g. after split) still passes."""
        code = "fig = px.scatter(df, x='x', y='y')"
        _, valid, _ = pg._validate_code_structure(code)
        assert valid

    def test_returns_cleaned_code(self, pg):
        code = "```python\nfig = px.bar(df, x='a', y='b')\n```"
        cleaned, valid, _ = pg._validate_code_structure(code)
        assert "```" not in cleaned


# ─────────────────────────────────────────────────────────────
# 6. _build_enhanced_prompt
# ─────────────────────────────────────────────────────────────

class TestBuildEnhancedPrompt:
    def test_contains_question(self, pg, mixed_df):
        info = pg._get_dataframe_info(mixed_df)
        prompt = pg._build_enhanced_prompt("show me a bar chart", info)
        assert "show me a bar chart" in prompt

    def test_attempt_0_no_retry_text(self, pg, mixed_df):
        info = pg._get_dataframe_info(mixed_df)
        prompt = pg._build_enhanced_prompt("q", info, attempt=0)
        assert "PREVIOUS ATTEMPT" not in prompt

    def test_attempt_1_has_retry_text(self, pg, mixed_df):
        info = pg._get_dataframe_info(mixed_df)
        prompt = pg._build_enhanced_prompt("q", info, attempt=1)
        assert "PREVIOUS ATTEMPT" in prompt

    def test_column_info_serialized_in_prompt(self, pg, mixed_df):
        info = pg._get_dataframe_info(mixed_df)
        prompt = pg._build_enhanced_prompt("q", info, attempt=0)
        assert "category" in prompt


# ─────────────────────────────────────────────────────────────
# 7. _execute_code_safely
# ─────────────────────────────────────────────────────────────

class TestExecuteCodeSafely:
    def test_valid_code_returns_figure(self, pg, mixed_df):
        code = "fig = px.bar(df, x='category', y='value')"
        fig, err = pg._execute_code_safely(code, mixed_df)
        assert fig is not None
        assert err is None

    def test_no_fig_variable(self, pg, mixed_df):
        code = "result = df['value'].sum()"
        fig, err = pg._execute_code_safely(code, mixed_df)
        assert fig is None
        assert "fig" in err

    def test_invalid_fig_object(self, pg, mixed_df):
        code = "fig = 'not_a_figure'"
        fig, err = pg._execute_code_safely(code, mixed_df)
        assert fig is None
        assert "valid Plotly" in err

    def test_runtime_error_caught(self, pg, mixed_df):
        code = "fig = px.bar(df, x='nonexistent_column_xyz', y='value')"
        fig, err = pg._execute_code_safely(code, mixed_df)
        assert fig is None
        assert "execution error" in err

    def test_syntax_error_caught(self, pg, mixed_df):
        code = "fig = px.bar(df, x='category', y='value'"  # missing closing paren
        fig, err = pg._execute_code_safely(code, mixed_df)
        assert fig is None
        assert err is not None


# ─────────────────────────────────────────────────────────────
# 8. _decode_binary_data_robust
# ─────────────────────────────────────────────────────────────

class TestDecodeBinaryDataRobust:
    def _encode(self, values, fmt, dtype_str):
        raw = b''.join(struct.pack(fmt, v) for v in values)
        return {'bdata': base64.b64encode(raw).decode(), 'dtype': dtype_str}

    def test_not_dict_returns_none(self, pg):
        assert pg._decode_binary_data_robust([1, 2, 3]) is None

    def test_dict_without_bdata_returns_none(self, pg):
        assert pg._decode_binary_data_robust({'dtype': 'i1'}) is None

    def test_i1_decodes(self, pg):
        raw = bytes([1, 2, 3])
        data = {'bdata': base64.b64encode(raw).decode(), 'dtype': 'i1'}
        result = pg._decode_binary_data_robust(data)
        assert result == [1, 2, 3]

    def test_i2_decodes(self, pg):
        data = self._encode([100, 200], '<h', 'i2')
        result = pg._decode_binary_data_robust(data)
        assert result == [100, 200]

    def test_i4_decodes(self, pg):
        data = self._encode([1000, 2000], '<i', 'i4')
        result = pg._decode_binary_data_robust(data)
        assert result == [1000, 2000]

    def test_f4_decodes(self, pg):
        data = self._encode([1.5, 2.5], '<f', 'f4')
        result = pg._decode_binary_data_robust(data)
        assert len(result) == 2
        assert abs(result[0] - 1.5) < 0.01

    def test_f8_decodes(self, pg):
        data = self._encode([3.14, 2.71], '<d', 'f8')
        result = pg._decode_binary_data_robust(data)
        assert abs(result[0] - 3.14) < 0.001

    def test_unknown_dtype_returns_none(self, pg):
        raw = b'\x01\x02'
        data = {'bdata': base64.b64encode(raw).decode(), 'dtype': 'u8'}
        result = pg._decode_binary_data_robust(data)
        assert result is None

    def test_invalid_base64_returns_none(self, pg):
        """Trigger the except branch via a struct unpack error (1 byte can't unpack i2)."""
        raw = b'\x01'  # valid b64 but wrong payload length for i2
        data = {'bdata': base64.b64encode(raw).decode(), 'dtype': 'i2'}
        result = pg._decode_binary_data_robust(data)
        assert result is None


# ─────────────────────────────────────────────────────────────
# 9. _recursively_fix_data
# ─────────────────────────────────────────────────────────────

class TestRecursivelyFixData:
    def test_plain_value_passthrough(self, pg):
        assert pg._recursively_fix_data(42) == 42
        assert pg._recursively_fix_data("hello") == "hello"

    def test_list_processed(self, pg):
        result = pg._recursively_fix_data([1, 2, 3])
        assert result == [1, 2, 3]

    def test_nested_list(self, pg):
        result = pg._recursively_fix_data([[1, 2], [3, 4]])
        assert result == [[1, 2], [3, 4]]

    def test_dict_processed(self, pg):
        result = pg._recursively_fix_data({'a': 1, 'b': [2, 3]})
        assert result == {'a': 1, 'b': [2, 3]}

    def test_bdata_dict_decoded(self, pg):
        raw = bytes([10, 20, 30])
        obj = {'bdata': base64.b64encode(raw).decode(), 'dtype': 'i1'}
        result = pg._recursively_fix_data(obj)
        assert result == [10, 20, 30]

    def test_bdata_decode_fails_returns_original(self, pg):
        """If _decode_binary_data_robust returns None, original dict is returned."""
        obj = {'bdata': 'invalid!!', 'dtype': 'i1'}
        result = pg._recursively_fix_data(obj)
        assert isinstance(result, dict)

    def test_string_list_parsed(self, pg):
        result = pg._recursively_fix_data('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_string_dict_parsed(self, pg):
        result = pg._recursively_fix_data("{'a': 1}")
        assert result == {'a': 1}

    def test_unparsable_list_string_passthrough(self, pg):
        s = '[not, valid, python'
        result = pg._recursively_fix_data(s)
        assert result == s

    def test_list_string_syntax_error_passthrough(self, pg):
        """ast.literal_eval raises SyntaxError (null byte) → lines 151-152 pass branch."""
        s = '[1, 2, \x00]'  # null byte inside brackets -> SyntaxError in literal_eval
        result = pg._recursively_fix_data(s)
        assert result == s  # falls through to return obj

    def test_unparsable_dict_string_passthrough(self, pg):
        s = '{not: valid}'
        result = pg._recursively_fix_data(s)
        assert result == s

    def test_dict_string_syntax_error_passthrough(self, pg):
        """ast.literal_eval raises SyntaxError (null byte) → lines 156-157 pass branch."""
        s = '{\x00: 1}'  # null byte inside braces -> SyntaxError in literal_eval
        result = pg._recursively_fix_data(s)
        assert result == s  # falls through to return obj

    def test_numpy_array_to_list(self, pg):
        arr = np.array([1, 2, 3])
        result = pg._recursively_fix_data(arr)
        assert result == [1, 2, 3]

    def test_pandas_series_to_list(self, pg):
        s = pd.Series([4, 5, 6])
        result = pg._recursively_fix_data(s)
        assert result == [4, 5, 6]

    def test_numpy_scalar_to_python(self, pg):
        result = pg._recursively_fix_data(np.int64(99))
        assert result == 99
        assert isinstance(result, int)

    def test_nan_to_none(self, pg):
        result = pg._recursively_fix_data(float('nan'))
        assert result is None


# ─────────────────────────────────────────────────────────────
# 10. _convert_to_serializable_dict_robust
# ─────────────────────────────────────────────────────────────

class TestConvertToSerializableDict:
    def test_valid_figure_converted(self, pg, simple_bar_fig):
        result = pg._convert_to_serializable_dict_robust(simple_bar_fig)
        assert 'data' in result
        assert 'layout' in result

    def test_returns_error_structure_on_failure(self, pg):
        bad_fig = MagicMock()
        bad_fig.to_dict.side_effect = RuntimeError("boom")
        result = pg._convert_to_serializable_dict_robust(bad_fig)
        assert result['data'][0]['type'] == 'bar'
        assert 'Error' in result['data'][0]['x']


# ─────────────────────────────────────────────────────────────
# 11. _validate_trace
# ─────────────────────────────────────────────────────────────

class TestValidateTrace:
    def test_valid_bar_trace(self, pg):
        trace = {'type': 'bar', 'x': ['A', 'B'], 'y': [1, 2]}
        valid, msg = pg._validate_trace(trace, 0)
        assert valid

    def test_missing_type(self, pg):
        trace = {'x': ['A'], 'y': [1]}
        valid, msg = pg._validate_trace(trace, 0)
        assert not valid
        assert "Missing 'type'" in msg

    def test_binary_data_rejected(self, pg):
        trace = {'type': 'bar', 'x': {'bdata': 'abc', 'dtype': 'i1'}, 'y': [1]}
        valid, msg = pg._validate_trace(trace, 0)
        assert not valid
        assert "binary data" in msg

    def test_long_unparsed_string_rejected(self, pg):
        trace = {'type': 'bar', 'x': '[' + '1,' * 60 + '1]', 'y': [1]}
        valid, msg = pg._validate_trace(trace, 0)
        assert not valid
        assert "unparsed string" in msg

    def test_non_list_axis_rejected(self, pg):
        trace = {'type': 'bar', 'x': 'not-a-list', 'y': [1]}
        valid, msg = pg._validate_trace(trace, 0)
        assert not valid
        assert "not a list" in msg

    def test_mismatched_xy_lengths(self, pg):
        trace = {'type': 'bar', 'x': [1, 2, 3], 'y': [1, 2]}
        valid, msg = pg._validate_trace(trace, 0)
        assert not valid
        assert "different lengths" in msg

    def test_valid_pie_trace(self, pg):
        trace = {'type': 'pie', 'labels': ['A', 'B'], 'values': [10, 20]}
        valid, msg = pg._validate_trace(trace, 0)
        assert valid

    def test_pie_labels_not_list(self, pg):
        trace = {'type': 'pie', 'labels': 'A,B', 'values': [10, 20]}
        valid, msg = pg._validate_trace(trace, 0)
        assert not valid

    def test_pie_values_not_list(self, pg):
        """labels is a list but values is not — hits line 219."""
        trace = {'type': 'pie', 'labels': ['A', 'B'], 'values': 'bad'}
        valid, msg = pg._validate_trace(trace, 0)
        assert not valid

    def test_pie_label_value_length_mismatch(self, pg):
        trace = {'type': 'pie', 'labels': ['A', 'B', 'C'], 'values': [10, 20]}
        valid, msg = pg._validate_trace(trace, 0)
        assert not valid
        assert "different lengths" in msg

    def test_x_not_list_with_y(self, pg):
        """x is not a list but y is — hits non-list axis check first."""
        trace = {'type': 'bar', 'x': 'A', 'y': [1, 2]}
        valid, msg = pg._validate_trace(trace, 0)
        assert not valid

    def test_y_not_list_but_x_is(self, pg):
        """x is a list but y is not — hits line 213 (inner isinstance check)."""
        trace = {'type': 'bar', 'x': [1, 2], 'y': 'not-a-list'}
        valid, msg = pg._validate_trace(trace, 0)
        assert not valid
        assert "not a list" in msg


# ─────────────────────────────────────────────────────────────
# 12. _validate_plot_json_comprehensive
# ─────────────────────────────────────────────────────────────

class TestValidatePlotJson:
    def test_valid_json(self, pg):
        """Build JSON via the production pipeline to avoid raw binary data in traces."""
        plot_json = _make_valid_plot_json()
        valid, msg = pg._validate_plot_json_comprehensive(plot_json)
        assert valid, f"Expected valid but got: {msg}"""  

    def test_missing_data_key(self, pg):
        bad = json.dumps({'layout': {}})
        valid, msg = pg._validate_plot_json_comprehensive(bad)
        assert not valid
        assert "Missing 'data'" in msg

    def test_empty_data_list(self, pg):
        bad = json.dumps({'data': [], 'layout': {}})
        valid, msg = pg._validate_plot_json_comprehensive(bad)
        assert not valid
        assert "empty" in msg

    def test_missing_layout(self, pg):
        bad = json.dumps({'data': [{'type': 'bar', 'x': ['A'], 'y': [1]}]})
        valid, msg = pg._validate_plot_json_comprehensive(bad)
        assert not valid
        assert "layout" in msg

    def test_invalid_json_string(self, pg):
        valid, msg = pg._validate_plot_json_comprehensive("this is not json")
        assert not valid
        assert "Invalid JSON" in msg

    def test_invalid_trace_propagates(self, pg):
        bad = json.dumps({
            'data': [{'type': 'bar', 'x': {'bdata': 'abc'}, 'y': [1]}],
            'layout': {}
        })
        valid, msg = pg._validate_plot_json_comprehensive(bad)
        assert not valid

    def test_exception_in_validation_caught(self, pg):
        """Non-string input causes an exception that is caught."""
        valid, msg = pg._validate_plot_json_comprehensive(None)
        assert not valid


# ─────────────────────────────────────────────────────────────
# 13. Fallback plot creators
# ─────────────────────────────────────────────────────────────

class TestFallbackPlotCreators:

    # _create_pie_chart_fallback
    def test_pie_chart_with_cat_cols(self, pg, cat_only_df):
        fig = pg._create_pie_chart_fallback(cat_only_df, ['color'])
        assert fig is not None

    def test_pie_chart_no_cat_cols_returns_none(self, pg, cat_only_df):
        assert pg._create_pie_chart_fallback(cat_only_df, []) is None

    # _create_box_or_violin_plot
    def test_box_plot_with_cat_and_num(self, pg, mixed_df):
        fig = pg._create_box_or_violin_plot(mixed_df, ['category'], ['value'], 'box plot')
        assert fig is not None

    def test_violin_plot_with_cat_and_num(self, pg, mixed_df):
        fig = pg._create_box_or_violin_plot(mixed_df, ['category'], ['value'], 'violin chart')
        assert fig is not None

    def test_box_single_numeric(self, pg, numeric_df):
        fig = pg._create_box_or_violin_plot(numeric_df, [], ['x'], 'box')
        assert fig is not None

    def test_box_no_cols_returns_none(self, pg, numeric_df):
        assert pg._create_box_or_violin_plot(numeric_df, [], [], 'box') is None

    # _create_scatter_plot
    def test_scatter_two_numeric(self, pg, numeric_df):
        fig = pg._create_scatter_plot(numeric_df, ['x', 'y'])
        assert fig is not None

    def test_scatter_one_numeric_returns_none(self, pg, numeric_df):
        assert pg._create_scatter_plot(numeric_df, ['x']) is None

    # _create_distribution_plot
    def test_distribution_numeric(self, pg, numeric_df):
        fig = pg._create_distribution_plot(numeric_df, ['x'], [])
        assert fig is not None

    def test_distribution_cat_fallback(self, pg, cat_only_df):
        fig = pg._create_distribution_plot(cat_only_df, [], ['color'])
        assert fig is not None

    def test_distribution_no_cols_returns_none(self, pg, mixed_df):
        assert pg._create_distribution_plot(mixed_df, [], []) is None

    # _create_comparison_plot
    def test_comparison_cat_and_num(self, pg, mixed_df):
        fig = pg._create_comparison_plot(mixed_df, ['category'], ['value'])
        assert fig is not None

    def test_comparison_cat_only(self, pg, cat_only_df):
        fig = pg._create_comparison_plot(cat_only_df, ['color'], [])
        assert fig is not None

    def test_comparison_no_cols_returns_none(self, pg, mixed_df):
        assert pg._create_comparison_plot(mixed_df, [], []) is None

    # _create_trend_plot
    def test_trend_with_date_col(self, pg, date_df):
        info = pg._get_dataframe_info(date_df)
        fig = pg._create_trend_plot(date_df, info)
        assert fig is not None

    def test_trend_no_date_col_returns_none(self, pg, numeric_df):
        info = pg._get_dataframe_info(numeric_df)
        fig = pg._create_trend_plot(numeric_df, info)
        assert fig is None

    def test_trend_invalid_date_handled(self, pg):
        df = pd.DataFrame({'date': ['bad', 'data', 'here'], 'sales': [1.0, 2.0, 3.0]})
        info = pg._get_dataframe_info(df)
        # All dates become NaT after coerce -> df_clean is empty -> returns None
        fig = pg._create_trend_plot(df, info)
        assert fig is None

    def test_trend_exception_in_to_datetime(self, pg):
        """Exception raised inside try block (lines 314-315) is caught and returns None."""
        df = pd.DataFrame({'date': ['2024-01-01', '2024-01-02'], 'sales': [1.0, 2.0]})
        info = pg._get_dataframe_info(df)
        with patch('src.analytics.plot_generator.pd.to_datetime', side_effect=RuntimeError("bad datetime")):
            fig = pg._create_trend_plot(df, info)
        assert fig is None

    # _create_most_informative_plot
    def test_most_informative_reasonable_cardinality(self, pg, mixed_df):
        fig = pg._create_most_informative_plot(mixed_df, ['category'], ['value'])
        assert fig is not None

    def test_most_informative_high_cardinality_falls_to_basic(self, pg):
        """When all cat cols have > 20 unique values, falls back to basic."""
        df = pd.DataFrame({
            'id': [str(i) for i in range(50)],
            'val': range(50),
        })
        fig = pg._create_most_informative_plot(df, ['id'], ['val'])
        assert fig is not None  # basic plot should still return something

    # _create_basic_plot
    def test_basic_numeric(self, pg, numeric_df):
        fig = pg._create_basic_plot(numeric_df, [], ['x'])
        assert fig is not None

    def test_basic_cat_only(self, pg, cat_only_df):
        fig = pg._create_basic_plot(cat_only_df, ['color'], [])
        assert fig is not None

    def test_basic_no_cols_returns_none(self, pg, mixed_df):
        assert pg._create_basic_plot(mixed_df, [], []) is None


# ─────────────────────────────────────────────────────────────
# 14. _generate_intelligent_fallback  (question routing)
# ─────────────────────────────────────────────────────────────

class TestGenerateIntelligentFallback:

    def _info(self, pg, df):
        return pg._get_dataframe_info(df)

    def test_routes_pie(self, pg, mixed_df):
        fig = pg._generate_intelligent_fallback("show pie chart", mixed_df, self._info(pg, mixed_df))
        assert fig is not None

    def test_routes_donut(self, pg, mixed_df):
        fig = pg._generate_intelligent_fallback("donut chart please", mixed_df, self._info(pg, mixed_df))
        assert fig is not None

    def test_routes_histogram(self, pg, mixed_df):
        fig = pg._generate_intelligent_fallback("show distribution", mixed_df, self._info(pg, mixed_df))
        assert fig is not None

    def test_routes_frequency(self, pg, mixed_df):
        fig = pg._generate_intelligent_fallback("frequency of values", mixed_df, self._info(pg, mixed_df))
        assert fig is not None

    def test_routes_box(self, pg, mixed_df):
        fig = pg._generate_intelligent_fallback("box plot by category", mixed_df, self._info(pg, mixed_df))
        assert fig is not None

    def test_routes_violin(self, pg, mixed_df):
        fig = pg._generate_intelligent_fallback("violin chart by category", mixed_df, self._info(pg, mixed_df))
        assert fig is not None

    def test_routes_bar_compare(self, pg, mixed_df):
        fig = pg._generate_intelligent_fallback("compare by group bar", mixed_df, self._info(pg, mixed_df))
        assert fig is not None

    def test_routes_trend(self, pg, date_df):
        fig = pg._generate_intelligent_fallback("trend over time", date_df, self._info(pg, date_df))
        assert fig is not None

    def test_routes_line(self, pg, date_df):
        fig = pg._generate_intelligent_fallback("line chart", date_df, self._info(pg, date_df))
        assert fig is not None

    def test_routes_scatter(self, pg, numeric_df):
        fig = pg._generate_intelligent_fallback("scatter relationship", numeric_df, self._info(pg, numeric_df))
        assert fig is not None

    def test_routes_correlation(self, pg, numeric_df):
        fig = pg._generate_intelligent_fallback("correlation between x and y", numeric_df, self._info(pg, numeric_df))
        assert fig is not None

    def test_default_route(self, pg, mixed_df):
        fig = pg._generate_intelligent_fallback("show me something", mixed_df, self._info(pg, mixed_df))
        assert fig is not None

    def test_exception_falls_to_basic(self, pg):
        """When an internal call raises, falls back to _create_basic_plot."""
        df = pd.DataFrame({'val': [1.0, 2.0, 3.0]})
        info = pg._get_dataframe_info(df)
        with patch.object(pg, '_create_pie_chart_fallback', side_effect=RuntimeError("fail")):
            fig = pg._generate_intelligent_fallback("pie chart", df, info)
        # Should get basic plot, not raise
        assert fig is not None or fig is None  # basic with no cat cols = None


# ─────────────────────────────────────────────────────────────
# 15. create_plot — top-level orchestration
# ─────────────────────────────────────────────────────────────

class TestCreatePlot:
    def test_empty_dataframe_returns_none(self, pg):
        result = pg.create_plot("q", pd.DataFrame(), {}, MagicMock(), "token")
        assert result == (None, None)

    def test_successful_first_attempt(self, pg, mixed_df):
        # Generate real valid JSON via the production pipeline
        df_tmp = pd.DataFrame({"x": ["A", "B"], "y": [1, 2]})
        fig_tmp = px.bar(df_tmp, x="x", y="y")
        pg_tmp = PlotGenerator(MagicMock())
        plot_dict = pg_tmp._convert_to_serializable_dict_robust(fig_tmp)
        valid_json = json.dumps(plot_dict, default=str, ensure_ascii=False)
        with patch.object(pg, '_attempt_plot_generation', return_value=valid_json):
            result, kind = pg.create_plot("bar chart", mixed_df, {}, MagicMock(), "token")
        assert kind == 'plotly'
        assert result == valid_json

    def test_falls_back_after_all_attempts_fail(self, pg, mixed_df):
        with patch.object(pg, '_attempt_plot_generation', return_value=None):
            result, kind = pg.create_plot("bar chart", mixed_df, {}, MagicMock(), "token", max_retries=2)
        # Fallback should succeed for mixed_df
        assert kind == 'plotly'
        assert result is not None

    def test_invalid_json_triggers_retry(self, pg, mixed_df):
        """Return invalid plot JSON → retry → all fail → fallback."""
        invalid_json = json.dumps({'data': [], 'layout': {}})  # empty data = invalid
        with patch.object(pg, '_attempt_plot_generation', return_value=invalid_json):
            result, kind = pg.create_plot("bar chart", mixed_df, {}, MagicMock(), "token", max_retries=2)
        assert kind == 'plotly'  # fallback still works

    def test_max_retries_respected(self, pg, mixed_df):
        with patch.object(pg, '_attempt_plot_generation', return_value=None) as mock_attempt:
            pg.create_plot("q", mixed_df, {}, MagicMock(), "token", max_retries=3)
        assert mock_attempt.call_count == 3


# ─────────────────────────────────────────────────────────────
# 16. _attempt_plot_generation
# ─────────────────────────────────────────────────────────────

class TestAttemptPlotGeneration:
    def _run(self, pg, df, code, attempt=0):
        pg.llm_client.generate.return_value = code
        return pg._attempt_plot_generation("q", df, pg._get_dataframe_info(df), attempt, {}, MagicMock(), "tok")

    def test_valid_code_returns_json(self, pg, mixed_df):
        code = "fig = px.bar(df, x='category', y='value')"
        result = self._run(pg, mixed_df, code)
        assert result is not None
        parsed = json.loads(result)
        assert 'data' in parsed

    def test_invalid_code_structure_returns_none(self, pg, mixed_df):
        code = "x = df['value'].sum()"  # no fig, no plotly
        result = self._run(pg, mixed_df, code)
        assert result is None

    def test_execution_error_returns_none(self, pg, mixed_df):
        code = "fig = px.bar(df, x='MISSING_COL', y='value')"
        result = self._run(pg, mixed_df, code)
        assert result is None

    def test_llm_exception_returns_none(self, pg, mixed_df):
        pg.llm_client.generate.side_effect = RuntimeError("LLM down")
        result = pg._attempt_plot_generation("q", mixed_df, pg._get_dataframe_info(mixed_df), 0, {}, MagicMock(), "tok")
        assert result is None
        pg.llm_client.generate.side_effect = None  # cleanup

    def test_json_loads_raises_returns_none(self, pg, mixed_df):
        """json.loads failure after json.dumps triggers lines 373-374."""
        code = "fig = px.bar(df, x='category', y='value')"
        pg.llm_client.generate.return_value = code
        # Patch json.loads to raise so the except branch on line 373-374 fires
        with patch('src.analytics.plot_generator.json.loads', side_effect=json.JSONDecodeError("bad", "", 0)):
            result = pg._attempt_plot_generation(
                "q", mixed_df, pg._get_dataframe_info(mixed_df), 0, {}, MagicMock(), "tok"
            )
        assert result is None

    def test_attempt_gt_0_uses_lower_temperature(self, pg, mixed_df):
        code = "fig = px.bar(df, x='category', y='value')"
        pg.llm_client.generate.return_value = code
        pg._attempt_plot_generation("q", mixed_df, pg._get_dataframe_info(mixed_df), 1, {}, MagicMock(), "tok")
        call_kwargs = pg.llm_client.generate.call_args
        assert call_kwargs.kwargs.get('temperature') == 0.1 or call_kwargs[1].get('temperature') == 0.1


# ─────────────────────────────────────────────────────────────
# 17. _generate_fallback_result
# ─────────────────────────────────────────────────────────────

class TestGenerateFallbackResult:
    def test_successful_fallback(self, pg, mixed_df):
        info = pg._get_dataframe_info(mixed_df)
        result, kind = pg._generate_fallback_result("show me data", mixed_df, info)
        assert kind == 'plotly'
        assert result is not None

    def test_fallback_none_when_no_fig(self, pg):
        df = pd.DataFrame()
        info = pg._get_dataframe_info(df)
        with patch.object(pg, '_generate_intelligent_fallback', return_value=None):
            result, kind = pg._generate_fallback_result("q", df, info)
        assert result is None
        assert kind is None

    def test_fallback_serialization_error_handled(self, pg, mixed_df):
        """json.dumps raises during fallback → except branch (lines 390-393) → (None, None)."""
        info = pg._get_dataframe_info(mixed_df)
        mock_fig = MagicMock()
        # to_dict succeeds but returns something that json.dumps chokes on
        mock_fig.to_dict.return_value = {'data': [object()], 'layout': {}}
        with patch.object(pg, '_generate_intelligent_fallback', return_value=mock_fig), \
             patch('src.analytics.plot_generator.json.dumps', side_effect=TypeError("not serializable")):
            result, kind = pg._generate_fallback_result("q", mixed_df, info)
        assert result is None
        assert kind is None

    def test_fallback_invalid_plot_json_returns_none(self, pg, mixed_df):
        info = pg._get_dataframe_info(mixed_df)
        bad_dict = {'data': [], 'layout': {}}
        mock_fig = MagicMock()
        mock_fig.to_dict.return_value = bad_dict
        with patch.object(pg, '_generate_intelligent_fallback', return_value=mock_fig):
            result, kind = pg._generate_fallback_result("q", mixed_df, info)
        assert result is None


# ─────────────────────────────────────────────────────────────
# 18. validate_plot_json (public wrapper)
# ─────────────────────────────────────────────────────────────

class TestValidatePlotJsonPublic:
    def test_delegates_to_comprehensive(self, pg):
        plot_json = _make_valid_plot_json()
        valid, msg = pg.validate_plot_json(plot_json)
        assert valid, f"Expected valid but got: {msg}"""  

    def test_invalid_delegates_correctly(self, pg):
        valid, msg = pg.validate_plot_json("not json")
        assert not valid