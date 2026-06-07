import pandas as pd
import numpy as np
import json
import base64
import struct
import ast
import io
import re
import difflib
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, List
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from src.utils.reasoning_extractor import REASONING_SECTION_PROMPT
from src.analytics.cardinality_handler import (
    build_chart_for_cardinality,
    get_cardinality_prompt_rules,
    is_high_cardinality,
    
    TOP_N_DEFAULT,
)
import logging
logger = logging.getLogger(__name__)

ENHANCED_PROMPT_TAIL = """
CHART TYPE MAPPING (default — overridden by cardinality rules below):
- histogram/distribution -> px.histogram(df, x='column')
- pie/donut chart        -> value_counts().reset_index() + px.pie()   (<=8 categories only)
- bar chart              -> value_counts().reset_index() + px.bar()
- box plot               -> px.box(df, x='category', y='value')
- violin plot            -> px.violin(df, x='category', y='value')
- scatter plot           -> px.scatter(df, x='x', y='y')
- line chart             -> px.line(df, x='x', y='y')
- treemap                -> px.treemap(df_agg, path=['col'], values='count')
- heatmap                -> px.imshow(pivot_df, ...)
 
{cardinality_rules}
 
Return ONLY the executable Python code that creates 'fig':"""

class PlotGenerator:
    """
    A robust class to generate Plotly visualizations from natural language questions.
    Handles code generation, validation, execution, and fallback plotting with comprehensive error handling.
    
    ENHANCED VERSION - MORE ROBUST WITH BETTER DATA HANDLING
    """
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.imports = {
            'pd': pd,
            'np': np,
            'px': px,
            'go': go
        }
    
    def _make_json_serializable(self, value: Any) -> Any:
        """Convert pandas/numpy types to JSON-serializable formats."""
        try:
            if isinstance(value, (pd.Timestamp, datetime)):
                return value.isoformat()
            if isinstance(value, (np.generic,)):
                return value.item()
            if pd.isna(value):
                return None
            if isinstance(value, bytes):
                return value.decode('utf-8', errors='ignore')
        except (TypeError, AttributeError, ValueError):
            pass
        return value
    
    def _get_dataframe_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract structured information about the dataframe."""
        sample_df = df.head(3)
        sample_records = sample_df.to_dict(orient='records')
        sanitized_samples = []
        for rec in sample_records:
            sanitized = {k: self._make_json_serializable(v) for k, v in rec.items()}
            sanitized_samples.append(sanitized)
        
        return {
            'columns': list(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'sample_data': sanitized_samples,
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object']).columns),
            'shape': df.shape,
            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB"
        }
    
    def _validate_code_structure(self, code: str) -> Tuple[str, bool, str]:
        code = self._clean_generated_code(code)
        
        if any(term in code.lower() for term in ['matplotlib', 'plt.', 'import plt', 'pyplot']):
            return code, False, "Code contains matplotlib usage which is forbidden"
        
        if not any(term in code for term in ['px.', 'go.', 'plotly']):
            return code, False, "Code does not use Plotly Express or Graph Objects"
        
        if 'fig' not in code:
            return code, False, "Code does not create 'fig' variable"
        
        # Only block .index/.values if it's NOT a heatmap (px.imshow is valid with crosstab result)
        is_heatmap = 'px.imshow' in code or 'imshow' in code
        if not is_heatmap and ('.index' in code or '.values' in code) and 'px.' in code:
            return code, False, "Code uses .index/.values with Plotly Express - use reset_index() instead"
        
        df_patterns = ['df[', 'df.', 'df,', '(df', ' df ', '=df', ',df']
        has_df_reference = any(pattern in code for pattern in df_patterns)
        if not has_df_reference:
            code_tokens = code.replace('(', ' ').replace(')', ' ').replace(',', ' ').split()
            has_df_reference = 'df' in code_tokens
        
        if not has_df_reference:
            return code, False, "Code does not properly reference the dataframe"
        
        return code, True, "Code structure is valid"
    
    def _clean_generated_code(self, code: str) -> str:
        """Extract code from markdown code blocks and remove non-code content."""
        # Remove markdown code blocks
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()
        
        # Remove any remaining markdown or explanations
        lines = []
        for line in code.split('\n'):
            if line.strip().startswith('#') or line.strip() == '':
                continue  # Skip comments and empty lines at start
            if 'import ' in line and ('plt' in line or 'matplotlib' in line):
                continue  # Skip matplotlib imports
            lines.append(line)
        
        return '\n'.join(lines).strip()

    def _fix_datetime_xaxis(self, fig: go.Figure) -> go.Figure:
        for trace in fig.data:
            # trace.x can be a numpy array, so we cannot use 'if not trace.x'
            if not hasattr(trace, 'x') or trace.x is None:
                continue
                
            x_vals = trace.x
            # FIX: Explicit length check
            try:
                if len(x_vals) == 0:
                    continue
            except (TypeError, ValueError):
                continue

            self._process_xaxis_fixing(fig, trace, x_vals)
        return fig

    def _process_xaxis_fixing(self, fig: go.Figure, trace: Any, x_vals: List[Any]) -> None:
        """Categorize and apply fixes to a specific trace's x-axis."""
        import re
        # FIX: Access first element safely (x_vals might be a Series/Array)
        try:
            first_val = x_vals[0] if len(x_vals) > 0 else None
        except (IndexError, KeyError, TypeError):
            return

        if first_val is None:
            return

        is_unix_nano = (
            isinstance(first_val, (int, float, np.integer, np.floating))
            and abs(float(first_val)) > 1e15
        )
        is_datetime_obj = isinstance(first_val, (pd.Timestamp, np.datetime64, datetime))
        is_string_period = (
            isinstance(first_val, str)
            and bool(re.match(r'^\d{4}-\d{2}(-\d{2})?$', first_val))
        )

        if is_unix_nano or is_datetime_obj:
            self._handle_datetime_axis_conversion(fig, trace, x_vals, is_unix_nano)
        elif is_string_period:
            self._handle_string_period_axis(fig, x_vals)

    def _handle_datetime_axis_conversion(self, fig: go.Figure, trace: Any, x_vals: List[Any], is_unix_nano: bool) -> None:
        """Convert datetime objects or Unix timestamps to fixed string periods."""
        try:
            converted = pd.to_datetime(
                x_vals,
                unit='ns' if is_unix_nano else None,
                errors='coerce'
            )
            date_range = (converted.max() - converted.min()).days
            fmt = '%Y-%m' if date_range > 3 else '%Y-%m-%d'
            str_periods = converted.strftime(fmt).tolist()
            trace.x = str_periods
            unique_sorted = sorted(set(str_periods))
            fig.update_layout(xaxis={
                'type': 'category',
                'categoryorder': 'array',
                'categoryarray': unique_sorted
            })
            logger.info(f"Fixed datetime x-axis: converted to '{fmt}' string periods")
        except Exception as e:
            logger.warning(f"_handle_datetime_axis_conversion failed: {e}")

    def _handle_string_period_axis(self, fig: go.Figure, x_vals: List[Any]) -> None:
        """Lock existing string periods to prevent Plotly from interpolating missing months."""
        unique_sorted = sorted(set(x_vals))
        existing = fig.layout.xaxis
        already_locked = (
            hasattr(existing, 'categoryarray') and
            existing.categoryarray is not None and
            len(existing.categoryarray) == len(unique_sorted)
        )
        if not already_locked:
            fig.update_layout(xaxis={
                'type': 'category',
                'categoryorder': 'array',
                'categoryarray': unique_sorted
            })
            logger.info(f"Locked string period x-axis: {len(unique_sorted)} items")

    def _apply_title_case_to_figure(self, fig: go.Figure) -> go.Figure:
        """Post-execution fix: restore Title Case on axis titles and categorical string labels.

        Data values are lowercased during DataFrame cleaning, so this step converts
        them back to Title Case for display without relying on the LLM to do it.
        Date-period strings (e.g. '2024-01', '2024-01-15') are left untouched.
        """
        import re
        _DATE_RE = re.compile(r'^\d{4}(-\d{2}(-\d{2})?)?$')

        def _tc(s: str) -> str:
            return s if _DATE_RE.match(s) else s.title()

        def _tc_seq(vals):
            """Title-case a sequence safely handling numpy/pandas objects."""
            # FIX: Use identity check 'is None' and explicit 'len()' instead of 'if not vals'
            if vals is None:
                return vals
            
            # Check length safely
            try:
                if len(vals) == 0:
                    return vals
            except (TypeError, ValueError):
                return vals

            items = list(vals)
            non_null = [v for v in items if v is not None]
            if not non_null or not all(isinstance(v, str) for v in non_null):
                return vals
            if any(_DATE_RE.match(v) for v in non_null):
                return vals
            return tuple(_tc(v.replace('_', ' ')) if v is not None else v for v in items)

        # Axis titles
        for axis_name in ('xaxis', 'yaxis'):
            ax = getattr(fig.layout, axis_name, None)
            if ax and ax.title and ax.title.text:
                ax.title.text = _tc(ax.title.text.replace('_', ' '))
                
        # Trace-level labels and legend names
        for trace in fig.data:
            if getattr(trace, 'name', None):
                trace.name = _tc(trace.name.replace('_', ' '))
            for attr in ('x', 'y', 'labels'):
                vals = getattr(trace, attr, None)
                if vals is not None:
                    updated = _tc_seq(vals)
                    if updated is not vals:
                        setattr(trace, attr, updated)

        # Pie/donut text labels on slices
        for trace in fig.data:
            if getattr(trace, 'type', None) == 'pie':
                if getattr(trace, 'labels', None) is not None:
                    trace.labels = tuple(
                        _tc(str(v).replace('_', ' ')) if v is not None else v
                        for v in trace.labels
                    )
                if getattr(trace, 'text', None) is not None:
                    trace.text = tuple(
                        _tc(str(v).replace('_', ' ')) if v is not None else v
                        for v in trace.text
                    )

        # Sync categoryarray with the now-title-cased x values
        xaxis = fig.layout.xaxis
        if getattr(xaxis, 'categoryarray', None) is not None:
            arr = list(xaxis.categoryarray)
            non_null = [v for v in arr if v is not None]
            if non_null and all(isinstance(v, str) for v in non_null) and not any(_DATE_RE.match(v) for v in non_null):
                fig.update_layout(xaxis_categoryarray=[_tc(v) for v in arr])

        return fig

    @staticmethod
    def _format_mean_bar_labels(fig: go.Figure) -> go.Figure:
        """Data labels disabled — no-op."""
        return fig

    def _classify_error(self, error_str: str, df_columns: List[str]) -> Tuple[str, Dict[str, Any]]:
        """Classify an execution/validation error into a named category with actionable detail."""
        detail: Dict[str, Any] = {'raw': error_str}

        # NameError: 'X' is not defined — could be a mistyped column or genuinely missing name
        name_match = re.search(r"name '(\w+)' is not defined", error_str)
        if name_match:
            missing = name_match.group(1)
            close = difflib.get_close_matches(missing, df_columns, n=1, cutoff=0.6)
            detail['missing_name'] = missing
            if close:
                detail['suggested_column'] = close[0]
                return 'wrong_column_name', detail
            return 'missing_builtin', detail

        # KeyError: 'column' — column name passed as dict key doesn't exist
        key_match = re.search(r"KeyError: ['\"](.+?)['\"]", error_str)
        if key_match:
            wrong = key_match.group(1)
            close = difflib.get_close_matches(wrong, df_columns, n=1, cutoff=0.5)
            detail['wrong_col'] = wrong
            if close:
                detail['suggested_column'] = close[0]
            return 'wrong_column_name', detail

        error_lower = error_str.lower()

        if 'insufficient for a trend' in error_lower or 'fewer than 2' in error_lower:
            return 'insufficient_data', detail

        if 'matplotlib' in error_lower or 'plt.' in error_lower:
            return 'forbidden_library', detail

        if "'fig' variable" in error_lower or "no 'fig'" in error_lower:
            return 'no_fig', detail

        if 'reset_index' in error_lower or (
            "'series'" in error_lower and 'dataframe' in error_lower
        ):
            return 'wrong_api_usage', detail

        if 'json' in error_lower or 'serializ' in error_lower:
            return 'serialization', detail

        return 'logic_error', detail

    def _auto_fix_code(self, code: str, error_category: str, error_detail: Dict[str, Any]) -> Optional[str]:
        """
        Attempt a deterministic, zero-LLM fix for simple known error patterns.
        Returns the fixed code string if a fix was applied, otherwise None.
        """
        if error_category == 'wrong_column_name':
            wrong = error_detail.get('missing_name') or error_detail.get('wrong_col', '')
            suggested = error_detail.get('suggested_column', '')
            if wrong and suggested and wrong != suggested:
                # Only replace exact quoted references to avoid spurious substitutions
                fixed = code.replace(f"'{wrong}'", f"'{suggested}'")
                fixed = fixed.replace(f'"{wrong}"', f'"{suggested}"')
                if fixed != code:
                    logger.info("Auto-fix: column '%s' -> '%s'", wrong, suggested)
                    return fixed
        return None

    def _build_targeted_retry_prompt(
        self,
        question: str,
        column_info: Dict[str, Any],
        attempt: int,
        error: str,
        error_category: str,
        error_detail: Dict[str, Any],
    ) -> str:
        """
        Concise, error-targeted prompt used on retry attempts (attempt > 0).
        Instead of repeating all 400 lines of rules, it focuses only on what broke.
        """
        _FIX_MAP: Dict[str, str] = {
            'missing_builtin': (
                f"NameError: '{error_detail.get('missing_name')}' is not defined.\n"
                "Use ONLY: px, pd, np, go, df. Do not reference any other names."
            ),
            'wrong_column_name': (
                f"Column name error — "
                f"'{error_detail.get('missing_name') or error_detail.get('wrong_col')}' does not exist.\n"
                + (
                    f"Correct column name: '{error_detail['suggested_column']}'\n"
                    if error_detail.get('suggested_column')
                    else ""
                )
                + f"All valid columns (exact, case-sensitive): {column_info['columns']}"
            ),
            'wrong_api_usage': (
                "Series/index error — you passed a Series where a DataFrame is required.\n"
                "- ALWAYS call .reset_index() after .value_counts() or .groupby()\n"
                "- ALWAYS rename: df.columns = ['col1', 'col2']\n"
                "- NEVER use .index or .values with px functions"
            ),
            'insufficient_data': (
                "Chart has too few data points for a trend.\n"
                "- Switch to DAILY grouping: dt.strftime('%Y-%m-%d') instead of '%Y-%m'\n"
                "- Do NOT pre-filter rows — use color= parameter for category breakdowns"
            ),
            'no_fig': (
                "Code did not create a variable named 'fig'.\n"
                "- The final assignment must be: fig = px.<chart>(df, ...)"
            ),
            'forbidden_library': "Matplotlib is forbidden. Use ONLY px.* or go.* from Plotly.",
            'serialization': f"Serialization error: {error}\nEnsure all data uses native Python types.",
            'logic_error': f"Exact error:\n{error}\nFix this specific issue and do not repeat the same approach.",
        }

        fix_text = _FIX_MAP.get(error_category, _FIX_MAP['logic_error'])

        # Pick only the chart pattern relevant to this query
        q = question.lower()
        if any(t in q for t in ('pie', 'donut')):
            pattern = (
                "counts_df = df['<cat_col>'].value_counts().reset_index()\n"
                "counts_df.columns = ['<cat_col>', 'count']\n"
                "fig = px.pie(counts_df, names='<cat_col>', values='count', title='...')"
            )
        elif any(t in q for t in ('trend', 'over time', 'time series', 'month', 'line')):
            pattern = (
                "df_plot = df.copy()\n"
                "df_plot['<date_col>'] = pd.to_datetime(df_plot['<date_col>'], errors='coerce')\n"
                "df_plot = df_plot.dropna(subset=['<date_col>'])\n"
                "df_plot['period'] = df_plot['<date_col>'].dt.strftime('%Y-%m')\n"
                "monthly = df_plot.groupby('period').size().reset_index(name='count')\n"
                "monthly = monthly.sort_values('period')\n"
                "fig = px.line(monthly, x='period', y='count', title='...')\n"
                "fig.update_layout(xaxis=dict(type='category', categoryorder='array',\n"
                "    categoryarray=monthly['period'].tolist()))"
            )
        elif any(t in q for t in ('heatmap', 'heat map')):
            pattern = (
                "heat_df = pd.crosstab(df['<cat_col_1>'], df['<cat_col_2>'])\n"
                "heat_df.index = heat_df.index.astype(str)\n"
                "heat_df.columns = heat_df.columns.astype(str)\n"
                "fig = px.imshow(heat_df, aspect='auto', color_continuous_scale='Blues', title='...')"
            )
        elif 'scatter' in q:
            pattern = "fig = px.scatter(df, x='<numeric_col_1>', y='<numeric_col_2>', title='...')"
        elif any(t in q for t in ('histogram', 'distribution', 'frequency')):
            pattern = "fig = px.histogram(df, x='<numeric_col>', title='...', nbins=30)"
        else:
            pattern = (
                "counts_df = df['<cat_col>'].value_counts().reset_index()\n"
                "counts_df.columns = ['<cat_col>', 'count']\n"
                "fig = px.bar(counts_df, x='<cat_col>', y='count', title='...')"
            )

        return f"""Generate ONLY Plotly Express code for this visualization request.

DataFrame Info:
{json.dumps(column_info, indent=2)}

User Request: {question}

ATTEMPT {attempt} FAILED — fix the issue below:
{fix_text}

RULES (non-negotiable):
- 'df' is pre-loaded. Available libraries: px, pd, np, go.
- Create a variable named 'fig'. No imports, no markdown, no explanations.
- Always call reset_index() after value_counts() or groupby().
- Always pass DataFrames (not Series) to px functions.

CORRECT PATTERN for this request:
{pattern}

Return ONLY executable Python code:"""

    def _build_enhanced_prompt(self, question: str, column_info: Dict[str, Any], attempt: int = 0, last_error: str = None) -> str:
        """Build the LLM prompt for plot generation with enhanced guidance."""
        base_prompt = f"""Generate ONLY Plotly Express code for this visualization request.

    DataFrame Info:
    {json.dumps(column_info, indent=2)}

    User Request: {question}

    CRITICAL REQUIREMENTS - FAILURE TO FOLLOW WILL CAUSE ERRORS:
    1. **MUST USE Plotly Express (px)** - matplotlib is FORBIDDEN
    2. Dataframe variable is 'df' (already loaded)
    3. Variables already imported: px, pd, np, go
    4. **MUST create variable named 'fig'** with the plot
    5. **NO imports, NO explanations, NO markdown**
    6. **CRITICAL DATA SERIALIZATION RULES:**
    - For value_counts(): ALWAYS use `.reset_index()` and rename columns
    - For groupby(): ALWAYS use `.reset_index()`
    - NEVER use `.index` or `.values` with px functions
    - ALWAYS pass DataFrames, NEVER Series to Plotly Express
    - For pie charts: use reset_index() and pass column names as strings
    - All data passed to Plotly MUST be in DataFrame format with named columns

    DATETIME RULES (CRITICAL - VIOLATIONS CAUSE WRONG AXES):
    - ALWAYS convert date columns: df['<date_col>'] = pd.to_datetime(df['<date_col>'], errors='coerce')
    - NEVER pass raw datetime64/Unix integers to plotly
    - NEVER use `.dt.to_period('M').dt.to_timestamp()` — this causes Unix nanosecond
        scientific notation on the x-axis.
    - NEVER use `.dt.to_period('M').astype(str)` for line chart x-axis — this creates
        a categorical axis that Plotly extends beyond your data range.
    - For monthly aggregation ALWAYS use `.dt.strftime('%Y-%m')` and explicitly set
        xaxis type='category' with categoryarray so Plotly cannot add months not in data:

        df_plot = df.copy()
        df_plot['<date_col>'] = pd.to_datetime(df_plot['<date_col>'], errors='coerce')
        df_plot = df_plot.dropna(subset=['<date_col>'])
        df_plot['period'] = df_plot['<date_col>'].dt.strftime('%Y-%m')
        monthly = df_plot.groupby('period')['<count_col>'].count().reset_index()
        monthly.columns = ['period', 'item_count']
        monthly = monthly.sort_values('period')
        fig = px.line(monthly, x='period', y='item_count', title='...')
        fig.update_layout(xaxis=dict(
            type='category',
            categoryorder='array',
            categoryarray=monthly['period'].tolist()
        ))

    - CRITICAL: NEVER extend the x-axis beyond the last date present in the data.
        Do NOT use pd.date_range(), resample(), or any method that fills in missing
        future periods. Only plot dates/periods that actually exist in the grouped result.
    - For daily date grouping use `.dt.strftime('%Y-%m-%d')` with same categoryarray approach.
    - If date column samples show DD/MM/YYYY format (day > 12 in first position, 
        or format='mixed' warning), use dayfirst=True:
        df['<date_col>'] = pd.to_datetime(df['<date_col>'], errors='coerce', dayfirst=True)
        Wrong parsing turns March 11 into November 3, creating phantom future dates.

    SINGLE-POINT CHART PREVENTION (CRITICAL):
    - After filtering/grouping, if the resulting DataFrame has fewer than 2 rows,
      the time series will render blank. NEVER filter on state/category columns with
      str.contains() — the actual values in the dataset may not match your patterns.
    - For "open state" or any state-based requests: DO NOT filter rows. Instead, plot
      ALL rows and use color='<state_col>' to show breakdown by state. Example:
        monthly = df_plot.groupby(['period', '<state_col>']).size().reset_index(name='count')
        fig = px.line(monthly, x='period', y='count', color='<state_col>', title='...')
    - NEVER use .str.contains() to pre-filter rows before grouping on a state/category column.

    HEATMAP RULES (CRITICAL - VIOLATIONS CAUSE BLANK/INTEGER-AXIS CHARTS):
    - For two categorical columns ALWAYS use pd.crosstab():
        heat_df = pd.crosstab(df['<cat_col_1>'], df['<cat_col_2>'])
        heat_df.index = heat_df.index.astype(str)
        heat_df.columns = heat_df.columns.astype(str)
        fig = px.imshow(heat_df, aspect='auto', color_continuous_scale='Blues', title='...')
    - NEVER use pivot_table() for heatmaps
    - NEVER pass integer-indexed DataFrames to px.imshow
    - MANDATORY: ALWAYS use pd.crosstab() for heatmaps.
    - ALWAYS cast index and columns to str after crosstab.
    - If result DataFrame has integer index AND integer columns, this WILL fail.

    CORRECT PATTERNS (FOLLOW THESE EXACTLY):
    # Histogram - direct column usage
    fig = px.histogram(df, x='<numeric_col>', title='Distribution', nbins=30)

    # Pie chart - MUST use reset_index()
    counts_df = df['<cat_col>'].value_counts().reset_index()
    counts_df.columns = ['<cat_col>', 'count']
    fig = px.pie(counts_df, names='<cat_col>', values='count', title='Distribution of <cat_col>')

    # Bar chart - MUST use reset_index()
    counts_df = df['<cat_col>'].value_counts().reset_index()
    counts_df.columns = ['<cat_col>', 'count']
    fig = px.bar(counts_df, x='<cat_col>', y='count', title='Distribution of <cat_col>')

    # Grouped bar chart - MUST use reset_index()
    grouped = df.groupby('<cat_col>')['<numeric_col>'].mean().reset_index()
    grouped.columns = ['<cat_col>', 'avg_value']
    fig = px.bar(grouped, x='<cat_col>', y='avg_value', title='Average <numeric_col> by <cat_col>')

    # Box plot - direct usage
    fig = px.box(df, x='<cat_col>', y='<numeric_col>', title='Distribution by <cat_col>')

    # Violin plot - direct usage
    fig = px.violin(df, x='<cat_col>', y='<numeric_col>', title='Distribution by <cat_col>')

    # Scatter plot - direct usage
    fig = px.scatter(df, x='<numeric_col_1>', y='<numeric_col_2>', color='<cat_col>', title='Relationship')

    # Line chart (non-time) - MUST prepare data properly
    df_sorted = df.sort_values('<x_col>')
    fig = px.line(df_sorted, x='<x_col>', y='<y_col>', title='Trend')

    # Line chart (time series monthly) - MUST use strftime + categoryarray
    df_plot = df.copy()
    df_plot['<date_col>'] = pd.to_datetime(df_plot['<date_col>'], errors='coerce')
    df_plot = df_plot.dropna(subset=['<date_col>'])
    df_plot['period'] = df_plot['<date_col>'].dt.strftime('%Y-%m')
    monthly = df_plot.groupby('period').size().reset_index(name='count')
    monthly = monthly.sort_values('period')
    fig = px.line(monthly, x='period', y='count', title='Count Over Time')
    fig.update_layout(xaxis=dict(
        type='category',
        categoryorder='array',
        categoryarray=monthly['period'].tolist()
    ))

    # Stacked bar chart
    fig = px.histogram(df, x='<cat_col_1>', color='<cat_col_2>', title='Distribution', barmode='stack')

    # Grouped bar chart with color
    fig = px.bar(df, x='<cat_col_1>', y='<numeric_col>', color='<cat_col_2>', barmode='group', title='Comparison')

    # Heatmap - MUST use crosstab
    heat_df = pd.crosstab(df['<cat_col_1>'], df['<cat_col_2>'])
    heat_df.index = heat_df.index.astype(str)
    heat_df.columns = heat_df.columns.astype(str)
    fig = px.imshow(heat_df, aspect='auto', color_continuous_scale='Blues', title='<cat_col_1> vs <cat_col_2>')

    WRONG PATTERNS (NEVER DO THIS):
    ❌ counts = df['<col>'].value_counts(); fig = px.bar(x=counts.index, y=counts.values)
    ❌ fig = px.pie(values=df['<col>'].value_counts().values, names=df['<col>'].unique())
    ❌ fig = px.box(y=df['<col>'])  # Missing df parameter
    ❌ grouped = df.groupby('<col>')['<val>'].sum(); fig = px.bar(x=grouped.index, y=grouped.values)
    ❌ df_plot['period'] = df_plot['<date_col>'].dt.to_period('M').astype(str)  # extends axis beyond data
    ❌ df_plot['period'] = df_plot['<date_col>'].dt.to_period('M').dt.to_timestamp()  # causes unix timestamps

    CHART TYPE MAPPING (default — overridden by cardinality rules below):
    - histogram/distribution -> px.histogram(df, x='<col>')
    - pie/donut chart        -> value_counts().reset_index() + px.pie()   (<=8 categories only)
    - bar chart              -> value_counts().reset_index() + px.bar()
    - box plot               -> px.box(df, x='<cat_col>', y='<numeric_col>')
    - violin plot            -> px.violin(df, x='<cat_col>', y='<numeric_col>')
    - scatter plot           -> px.scatter(df, x='<x_col>', y='<y_col>')
    - line chart             -> px.line(df, x='<x_col>', y='<y_col>')
    - treemap                -> px.treemap(df_agg, path=['<col>'], values='count')
    - heatmap                -> pd.crosstab() + px.imshow()

    {get_cardinality_prompt_rules()}

    Return ONLY the executable Python code that creates 'fig':"""

        if attempt > 0:
            error_section = f"\n    EXACT ERROR FROM PREVIOUS ATTEMPT: {last_error}" if last_error else ""
            base_prompt += f"""

    PREVIOUS ATTEMPT {attempt} FAILED.{error_section}
    YOU MUST:
    1. Use reset_index() for ALL value_counts() and groupby() operations
    2. Rename columns after reset_index() using .columns = ['name1', 'name2']
    3. Ensure all data is in DataFrame format, NOT Series
    4. For pie charts: create a DataFrame with named columns first
    5. Pass column names (strings) to px functions, not .values or .tolist()
    6. For time series: use dt.strftime('%Y-%m') and set xaxis categoryarray
    7. Double-check that you're following the CORRECT PATTERNS exactly

    {REASONING_SECTION_PROMPT}"""

        return base_prompt
    
    def _execute_code_safely(self, code: str, df: pd.DataFrame) -> Tuple[Optional[go.Figure], Optional[str]]:
        """Execute the generated code safely and return the figure with error message."""
        
        df = df.copy()
        for col in df.select_dtypes(include='category').columns:
            df[col] = df[col].astype(object)
        
        local_vars = {'df': df, **self.imports}
        
        try:
            exec(code, {'__builtins__': __builtins__, **self.imports}, local_vars)
            fig = local_vars.get('fig')

            # Single-point check — only inspect x length, don't evaluate arrays as bools
            if fig is not None and hasattr(fig, 'data') and len(fig.data) > 0:
                for trace in fig.data:
                    x_vals = getattr(trace, 'x', None)
                    if x_vals is None:
                        continue
                    try:
                        x_len = len(x_vals)
                    except TypeError:
                        continue
                    if x_len < 2 and len(fig.data) == 1:
                        return None, (
                            f"Chart has only {x_len} monthly period(s) — insufficient for a trend. "
                            f"Switch to DAILY grouping using dt.strftime('%Y-%m-%d') instead of '%Y-%m'."
                        )

            if fig is None:
                return None, "No 'fig' variable created in code"

            if not hasattr(fig, 'to_dict'):
                return None, "Figure object is not a valid Plotly figure"

            fig = self._fix_datetime_xaxis(fig)
            fig = self._apply_title_case_to_figure(fig)
            fig = self._format_mean_bar_labels(fig)

            # Strip any data labels the generated code may have set
            _label_attrs = ('text', 'texttemplate', 'textposition', 'textfont',
                'insidetextanchor', 'cliponaxis')
            for trace in fig.data:
                for _attr in _label_attrs:
                    try:
                        setattr(trace, _attr, None)
                    except Exception:
                        pass  

            return fig, None
        
        except Exception as e:
            return None, f"Code execution error: {str(e)}"
    
    def _decode_binary_data_robust(self, data: Any) -> Optional[List[Any]]:
        """Robustly decode binary data from Plotly figures."""
        try:
            if not isinstance(data, dict) or 'bdata' not in data:
                return None
            
            bdata = data['bdata']
            dtype = data.get('dtype', 'i1')
            
            # Decode base64
            decoded = base64.b64decode(bdata)
            
            # Handle different data types
            if dtype == 'i1':
                return list(decoded)
            elif dtype == 'i2':
                return [struct.unpack('<h', decoded[i:i+2])[0] for i in range(0, len(decoded), 2)]
            elif dtype == 'i4':
                return [struct.unpack('<i', decoded[i:i+4])[0] for i in range(0, len(decoded), 4)]
            elif dtype == 'f4':
                return [struct.unpack('<f', decoded[i:i+4])[0] for i in range(0, len(decoded), 4)]
            elif dtype == 'f8':
                return [struct.unpack('<d', decoded[i:i+8])[0] for i in range(0, len(decoded), 8)]
            
            return None
        except Exception as e:
            logger.error(f"Binary decode error: {e}")
            return None
    
    def _fix_dict_node(self, obj: dict) -> Any:
        """Handle dict nodes: decode binary data or recurse."""
        if 'bdata' in obj:
            decoded = self._decode_binary_data_robust(obj)
            return decoded if decoded is not None else obj
        return {k: self._recursively_fix_data(v) for k, v in obj.items()}

    def _fix_string_node(self, obj: str) -> Any:
        """Attempt to parse string nodes that look like Python literals."""
        if (obj.startswith('[') and obj.endswith(']')) or \
        (obj.startswith('{') and obj.endswith('}')):
            try:
                return ast.literal_eval(obj)
            except (ValueError, SyntaxError):
                pass
        return obj

    def _fix_scalar_node(self, obj: Any) -> Any:
        """Convert numpy/pandas scalars and NaN values."""
        if isinstance(obj, (np.ndarray, pd.Series)):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        try:
            if pd.isna(obj):
                return None
        except (TypeError, ValueError):
            pass
        return obj

    def _recursively_fix_data(self, obj: Any) -> Any:
        """Recursively fix all data in the plot dictionary."""
        if isinstance(obj, dict):
            return self._fix_dict_node(obj)
        if isinstance(obj, list):
            return [self._recursively_fix_data(item) for item in obj]
        if isinstance(obj, str):
            return self._fix_string_node(obj)
        return self._fix_scalar_node(obj)
    
    def _convert_to_serializable_dict_robust(self, fig: go.Figure) -> Dict[str, Any]:
        """
        Convert Plotly figure to a properly serializable dictionary with comprehensive fixes.
        Preserves categoryarray from xaxis to prevent phantom months on frontend.
        """
        try:
            # Capture categoryarray BEFORE to_dict() drops it
            xaxis = fig.layout.xaxis
            preserved_xaxis = None
            if (xaxis and
                    hasattr(xaxis, 'type') and xaxis.type == 'category' and
                    hasattr(xaxis, 'categoryarray') and xaxis.categoryarray is not None):
                preserved_xaxis = {
                    'type': 'category',
                    'categoryorder': 'array',
                    'categoryarray': list(xaxis.categoryarray)
                }

            plot_dict = fig.to_dict()

            # Recursively fix all data in the plot
            plot_dict = self._recursively_fix_data(plot_dict)

            # Re-inject categoryarray if it was dropped by to_dict()
            if preserved_xaxis:
                if 'layout' not in plot_dict:
                    plot_dict['layout'] = {}
                if 'xaxis' not in plot_dict['layout']:
                    plot_dict['layout']['xaxis'] = {}
                plot_dict['layout']['xaxis'].update(preserved_xaxis)
                logger.info(
                    f"Re-injected categoryarray into serialized dict: "
                    f"{preserved_xaxis['categoryarray']}"
                )

            return plot_dict

        except Exception as e:
            logger.error(f"Figure serialization error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'data': [{
                    'type': 'bar',
                    'x': ['Error'],
                    'y': [1],
                    'name': 'Error in plot generation'
                }],
                'layout': {
                    'title': {'text': 'Plot Generation Error'},
                    'xaxis': {'title': {'text': 'Error'}},
                    'yaxis': {'title': {'text': 'Value'}}
                }
            }
    
    def _validate_plot_json_comprehensive(self, plot_json: str) -> Tuple[bool, str]:
        """Comprehensively validate that the generated plot JSON is properly formatted."""
        try:
            import re as _re
            _sanitized = _re.sub(r'\bNaN\b', 'null', plot_json)
            _sanitized = _re.sub(r'\bInfinity\b', 'null', _sanitized)
            _sanitized = _re.sub(r'\b-Infinity\b', 'null', _sanitized)
            plot_dict = json.loads(_sanitized)
            
            if 'data' not in plot_dict:
                return False, "Missing 'data' in plot JSON"
            
            if not isinstance(plot_dict['data'], list) or len(plot_dict['data']) == 0:
                return False, "Plot data is empty or not a list"
            
            # Validate each trace
            for i, trace in enumerate(plot_dict['data']):
                trace_validation = self._validate_trace(trace, i)
                if not trace_validation[0]:
                    return trace_validation
            
            if 'layout' not in plot_dict:
                return False, "Missing 'layout' in plot JSON"
            
            return True, "Plot JSON is valid and ready for rendering"
        
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON format: {e}"
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def _check_axis_data(self, trace: Dict[str, Any], trace_index: int) -> Tuple[bool, str]:
        """Check each axis field for binary, unparsed string, or non-list data."""
        for axis in ['x', 'y', 'z', 'labels', 'values']:
            if axis not in trace:
                continue
            data = trace[axis]
            if isinstance(data, dict) and 'bdata' in data:
                return False, f"Trace {trace_index}: {axis} contains unconverted binary data"
            if isinstance(data, str) and len(data) > 100 and data.startswith('['):
                return False, f"Trace {trace_index}: {axis} contains unparsed string data"
            if not isinstance(data, list):
                return False, f"Trace {trace_index}: {axis} is not a list (got {type(data).__name__})"
        return True, ""


    def _check_xy_length_consistency(self, trace: Dict[str, Any], trace_index: int) -> Tuple[bool, str]:
        """Check x and y lists exist and have equal lengths."""
        # These trace types use x/y as independent axis labels, not paired data points
        AXIS_LABEL_TRACES = {"heatmap", "contour", "surface", "histogram2d", "histogram2dcontour"}
        
        if trace.get("type") in AXIS_LABEL_TRACES:
            return True, ""
        
        if 'x' not in trace or 'y' not in trace:
            return True, ""
        x, y = trace['x'], trace['y']
        if not isinstance(x, list) or not isinstance(y, list):
            return False, f"Trace {trace_index}: x or y is not a list"
        if len(x) != len(y):
            return False, f"Trace {trace_index}: x and y data have different lengths ({len(x)} vs {len(y)})"
        return True, ""


    def _check_pie_trace(self, trace: Dict[str, Any], trace_index: int) -> Tuple[bool, str]:
        """Validate pie chart labels/values are lists of equal length."""
        if trace.get('type') != 'pie':
            return True, ""
        labels, values = trace.get('labels'), trace.get('values')
        if labels is None or values is None:
            return True, ""
        if not isinstance(labels, list) or not isinstance(values, list):
            return False, f"Trace {trace_index}: pie chart labels/values must be lists"
        if len(labels) != len(values):
            return False, f"Trace {trace_index}: pie chart labels/values have different lengths"
        return True, ""


    def _validate_trace(self, trace: Dict[str, Any], trace_index: int) -> Tuple[bool, str]:
        """Validate an individual trace in the plot data."""
        if 'type' not in trace:
            return False, f"Trace {trace_index}: Missing 'type'"

        for check in (
            self._check_axis_data(trace, trace_index),
            self._check_xy_length_consistency(trace, trace_index),
            self._check_pie_trace(trace, trace_index),
        ):
            if not check[0]:
                return check

        return True, "Trace is valid"
    
    def _generate_intelligent_fallback(
        self,
        question: str,
        df: pd.DataFrame,
        column_info: dict,
    ) -> "Optional[go.Figure]":
        """
        Generate an intelligent fallback plot when LLM code generation fails.
 
        High-cardinality categorical columns are delegated to
        cardinality_handler.build_chart_for_cardinality() which automatically
        selects the best chart type (horizontal bar / treemap / ranked table /
        heatmap) based on the number of unique values.
        """
        cat_cols = column_info.get("categorical_columns", [])
        num_cols = column_info.get("numeric_columns", [])
        question_lower = question.lower()
 
        try:
            df = df.copy()
            for col in df.select_dtypes(include='category').columns:
                df[col] = df[col].astype(object)
            df = df.fillna("N/A")

            # --- numeric / scatter / box / violin / trend remain unchanged ---
            if any(t in question_lower for t in ("distribution", "histogram", "frequency")):
                return self._create_distribution_plot(df, num_cols, cat_cols)
 
            if any(t in question_lower for t in ("box", "violin")):
                return self._create_box_or_violin_plot(df, cat_cols, num_cols, question_lower)
 
            if any(t in question_lower for t in ("scatter", "relationship", "correlation")):
                return self._create_scatter_plot(df, num_cols)
 
            if any(t in question_lower for t in ("trend", "over time", "time", "line")):
                return self._create_trend_plot(df, column_info)
 
            # --- pie / donut / bar / comparison -> delegate to cardinality handler ---
            if any(t in question_lower for t in ("pie", "donut")):
                return self._fallback_categorical(df, cat_cols, num_cols)
 
            if any(t in question_lower for t in ("compare", "versus", "vs", "by", "across", "bar", "heatmap")):
                return self._fallback_two_column(df, cat_cols, num_cols)
 
            # Default: pick best single-column categorical chart
            return self._fallback_categorical(df, cat_cols, num_cols)
 
        except Exception as exc:
            logger.error("Intelligent fallback error: %s", exc)
            return self._create_basic_plot(df, cat_cols, num_cols)
 
    def _fallback_categorical(
        self,
        df: pd.DataFrame,
        cat_cols: list,
        num_cols: list,
    ) -> "Optional[go.Figure]":
        """Fallback for single-column categorical charts using cardinality handler."""
        if not cat_cols:
            return self._create_basic_plot(df, cat_cols, num_cols)
 
        col = cat_cols[0]
        title = f"Distribution of {col}"
        fig = build_chart_for_cardinality(df, col, title)
        return fig if fig is not None else self._create_basic_plot(df, cat_cols, num_cols)
 
    def _fallback_two_column(self, df, cat_cols, num_cols):
        if not cat_cols:
            return self._create_comparison_plot(df, cat_cols, num_cols)

        col = cat_cols[0]

        if num_cols:
            # ADD THIS: skip ID-like columns, prefer meaningful numerics
            meaningful_num = [
                c for c in num_cols
                if not any(skip in c.lower() for skip in ('id', 'key', 'index', 'code', 'number'))
            ]
            second = meaningful_num[0] if meaningful_num else None
            
            if second:
                title = f"{second} by {col}"
                fig = build_chart_for_cardinality(df, col, title, second_col=second)
                return fig if fig is not None else self._create_comparison_plot(df, cat_cols, num_cols)
            else:
                # No meaningful numeric — fall back to count-based chart
                return self._fallback_categorical(df, cat_cols, num_cols)
    
    def _create_pie_chart_fallback(self, df: pd.DataFrame, cat_cols: List[str]) -> Optional[go.Figure]:
        """Create pie chart for categorical data."""
        if cat_cols:
            col = cat_cols[0]
            counts_df = df[col].value_counts().head(10).reset_index()
            counts_df.columns = [col, 'count']
            fig = px.pie(counts_df, names=col, values='count', title=f'Distribution of {col}')
            return fig
        return None
    
    def _create_box_or_violin_plot(self, df: pd.DataFrame, cat_cols: List[str], 
                                   num_cols: List[str], question_lower: str) -> Optional[go.Figure]:
        """Create box or violin plot."""
        if cat_cols and num_cols:
            cat_col = cat_cols[0]
            num_col = num_cols[0]
            
            if 'violin' in question_lower:
                fig = px.violin(df, x=cat_col, y=num_col, title=f'{num_col} by {cat_col}')
            else:
                fig = px.box(df, x=cat_col, y=num_col, title=f'{num_col} by {cat_col}')
            
            fig.update_layout(xaxis_tickangle=-45)
            return fig
        elif num_cols:
            # Single variable box plot
            num_col = num_cols[0]
            fig = px.box(df, y=num_col, title=f'Distribution of {num_col}')
            return fig
        return None
    
    def _create_scatter_plot(self, df: pd.DataFrame, num_cols: List[str]) -> Optional[go.Figure]:
        """Create scatter plot for numeric relationships."""
        if len(num_cols) >= 2:
            fig = px.scatter(df, x=num_cols[0], y=num_cols[1], 
                           title=f'{num_cols[1]} vs {num_cols[0]}')
            return fig
        return None
    
    def _create_distribution_plot(self, df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]) -> Optional[go.Figure]:
        """Create distribution plot based on available data."""
        if num_cols:
            col = num_cols[0]
            fig = px.histogram(df, x=col, title=f'Distribution of {col}', nbins=20)
            return fig
        elif cat_cols:
            col = cat_cols[0]
            value_counts = df[col].value_counts().head(15).reset_index()
            value_counts.columns = [col, 'Count']
            fig = px.bar(value_counts, x=col, y='Count', title=f'Top 15 {col} Distribution')
            fig.update_layout(xaxis_tickangle=-45)
            return fig
        return None
    
    def _create_comparison_plot(self, df: pd.DataFrame, cat_cols: List[str], num_cols: List[str]) -> Optional[go.Figure]:
        """Create comparison plot (bar chart or box plot)."""
        if cat_cols and num_cols:
            cat_col = cat_cols[0]
            
            # Skip ID-like columns, prefer meaningful numerics
            meaningful_num = [
                c for c in num_cols
                if not any(skip in c.lower() for skip in ('id', 'key', 'index', 'code', 'number'))
            ]
            
            if not meaningful_num:
                # No meaningful numeric — fall back to count-based chart
                col = cat_cols[0]
                counts = df[col].value_counts().head(15).reset_index()
                counts.columns = [col, 'count']
                fig = px.bar(counts, x=col, y='count', title=f'Count by {col}')
                fig.update_layout(xaxis_tickangle=-45)
                return fig
            
            num_col = meaningful_num[0]
            grouped = df.groupby(cat_col)[num_col].mean().reset_index()
            grouped.columns = [cat_col, f'avg_{num_col}']
            
            fig = px.bar(grouped, x=cat_col, y=f'avg_{num_col}',
                        title=f'Average {num_col} by {cat_col}')
            fig.update_layout(xaxis_tickangle=-45)
            return fig
        
        elif cat_cols:
            col = cat_cols[0]
            counts = df[col].value_counts().head(15).reset_index()
            counts.columns = [col, 'count']
            fig = px.bar(counts, x=col, y='count', title=f'Count by {col}')
            fig.update_layout(xaxis_tickangle=-45)
            return fig
        
        return None
    
    def _create_trend_plot(self, df: pd.DataFrame, column_info: Dict[str, Any]) -> Optional[go.Figure]:
        """Create trend plot if date columns exist."""
        date_columns = [col for col in column_info['columns']
                    if any(term in col.lower() for term in ['date', 'time', 'year', 'month', 'day'])]
        
        if date_columns and column_info['numeric_columns']:
            date_col = date_columns[0]
            num_col = column_info['numeric_columns'][0]
            
            try:
                df_copy = df.copy()
                df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
                df_clean = df_copy.dropna(subset=[date_col, num_col])
                
                if len(df_clean) > 0:
                    # NEW: aggregate to monthly to avoid noisy per-row line charts
                    df_clean = df_clean.sort_values(date_col)
                    if len(df_clean) > 60:  # aggregate only when data is dense
                        df_clean['_period'] = df_clean[date_col].dt.to_period('M').astype(str)
                        df_agg = df_clean.groupby('_period')[num_col].sum().reset_index()
                        df_agg.columns = ['Period', num_col]
                        fig = px.line(df_agg, x='Period', y=num_col, 
                                    title=f'{num_col} Over Time (Monthly)')
                    else:
                        fig = px.line(df_clean, x=date_col, y=num_col, 
                                    title=f'{num_col} Over Time')
                    return fig
            except Exception as e:
                logger.error(f"Trend plot generation failed: {e}")
        return None
    
    def _create_most_informative_plot(self, df: pd.DataFrame, cat_cols: List[str], num_cols: List[str]) -> Optional[go.Figure]:
        """Create the most informative plot based on data characteristics."""
        # Prefer categorical counts if available
        for col in cat_cols:
            unique_count = df[col].nunique()
            if 2 <= unique_count <= 20:  # Reasonable number of categories
                value_counts = df[col].value_counts().head(15).reset_index()
                value_counts.columns = [col, 'Count']
                fig = px.bar(value_counts, x=col, y='Count', title=f'Distribution of {col}')
                fig.update_layout(xaxis_tickangle=-45)
                return fig
        
        return self._create_basic_plot(df, cat_cols, num_cols)
    
    def _create_basic_plot(self, df: pd.DataFrame, cat_cols: List[str], num_cols: List[str]) -> Optional[go.Figure]:
        """Create a basic plot as last resort."""
        if num_cols:
            col = num_cols[0]
            return px.histogram(df, x=col, title=f'Distribution of {col}', nbins=20)
        elif cat_cols:
            col = cat_cols[0]
            value_counts = df[col].value_counts().head(10).reset_index()
            value_counts.columns = [col, 'Count']
            fig = px.bar(value_counts, x=col, y='Count', title=f'Top 10 {col}')
            fig.update_layout(xaxis_tickangle=-45)
            return fig
        return None
    
    def create_plot(self, question: str, df: pd.DataFrame, llm_params: Dict[str, Any], 
                   token_tracker, auth_token: str, max_retries: int = 3) -> Tuple[Optional[str], Optional[str]]:
        """
        Main method to create a plot from a natural language question with robust error handling.
        """
        if df.empty:
            logger.info("DataFrame is empty")
            return None, None
        
        df = df.dropna(how='all').dropna(axis=1, how='all')
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna('N/A')

        for col in df.select_dtypes(include='category').columns:
            df[col] = df[col].astype(object)

        column_info = self._get_dataframe_info(df)
        logger.info(f"Generating plot for question: '{question}'")
        logger.info(f"DataFrame shape: {df.shape}, Columns: {len(column_info['columns'])}")
        
        # Try LLM-generated plots with retries
        last_error: Optional[str] = None
        last_error_category: Optional[str] = None
        last_error_detail: Dict[str, Any] = {}

        for attempt in range(max_retries):
            logger.info(f"Attempt {attempt + 1}/{max_retries}")
            plot_json, last_error, last_error_category, last_error_detail = (
                self._attempt_plot_generation(
                    question, df, column_info, attempt,
                    llm_params, token_tracker, auth_token,
                    last_error, last_error_category, last_error_detail,
                )
            )

            if plot_json:
                is_valid, validation_msg = self._validate_plot_json_comprehensive(plot_json)
                if is_valid:
                    logger.info("✓ Plot generated and validated successfully")
                    return plot_json, 'plotly'
                else:
                    last_error = validation_msg
                    last_error_category, last_error_detail = self._classify_error(
                        validation_msg, column_info['columns']
                    )
                    logger.info(f"✗ Generated plot failed validation: {validation_msg}")
        
        logger.info("Using intelligent fallback plot generation")
        return self._generate_fallback_result(question, df, column_info)
    
    def _attempt_plot_generation(
        self,
        question: str,
        df: pd.DataFrame,
        column_info: Dict[str, Any],
        attempt: int,
        llm_params: Dict[str, Any],
        token_tracker,
        auth_token: str,
        last_error: str = None,
        last_error_category: str = None,
        last_error_detail: Dict[str, Any] = None,
    ) -> Tuple[Optional[str], Optional[str], str, Dict[str, Any]]:
        """Single attempt at generating plot via LLM.
        Returns (plot_json, error_message, error_category, error_detail)."""

        # On retry, use a short targeted prompt focused on the specific error.
        # On first attempt, use the full enhanced prompt.
        if attempt > 0 and last_error and last_error_category:
            prompt = self._build_targeted_retry_prompt(
                question, column_info, attempt,
                last_error, last_error_category, last_error_detail or {}
            )
        else:
            prompt = self._build_enhanced_prompt(question, column_info, attempt, last_error)

        _fail: Tuple[None, str, str, Dict] = (None, '', 'logic_error', {})

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                llm_params=llm_params,
                token_tracker=token_tracker,
                auth_token=auth_token,
                temperature=0.1 if attempt > 0 else 0.2
            )
            code = response.strip()
            logger.info(f"Generated code length: {len(code)} characters")

            code, is_valid, code_error = self._validate_code_structure(code)
            if not is_valid:
                logger.info(f"✗ Code validation failed: {code_error}")
                err_cat, err_det = self._classify_error(code_error, column_info['columns'])
                return None, code_error, err_cat, err_det

            logger.info("✓ Code structure validated")
            logger.info(f"Generated plotly code:\n{code}")

            fig, execution_error = self._execute_code_safely(code, df)

            # Auto-fix: attempt a deterministic repair before using up a retry
            if not fig and execution_error:
                err_cat, err_det = self._classify_error(execution_error, column_info['columns'])
                fixed_code = self._auto_fix_code(code, err_cat, err_det)
                if fixed_code:
                    logger.info("Applying auto-fix, re-executing...")
                    fig2, exec_error2 = self._execute_code_safely(fixed_code, df)
                    if fig2:
                        logger.info("✓ Auto-fix succeeded")
                        fig = fig2
                        execution_error = None
                    else:
                        logger.info(f"✗ Auto-fix did not resolve: {exec_error2}")
                        err_cat, err_det = self._classify_error(
                            exec_error2 or execution_error, column_info['columns']
                        )
                        execution_error = exec_error2 or execution_error

                if not fig:
                    logger.info(f"✗ Code execution failed: {execution_error}")
                    return None, execution_error, err_cat, err_det

            logger.info("✓ Code executed successfully")

            # Enhanced serialization with recursive fixing
            plot_dict = self._convert_to_serializable_dict_robust(fig)

            try:
                plot_json = json.dumps(plot_dict, default=str, ensure_ascii=False)
                json.loads(plot_json)
                logger.info("✓ Plot data validated and serialized")
                return plot_json, None, '', {}
            except (TypeError, ValueError) as e:
                logger.error(f"✗ Plot serialization failed: {e}")
                err_cat, err_det = self._classify_error(str(e), column_info['columns'])
                return None, str(e), err_cat, err_det

        except Exception as e:
            logger.error(f"✗ Attempt {attempt + 1} error: {e}")
            import traceback
            traceback.print_exc()
            err_cat, err_det = self._classify_error(str(e), column_info['columns'])
            return None, str(e), err_cat, err_det
    
    def _generate_fallback_result(self, question: str, df: pd.DataFrame,
                                  column_info: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        """Generate fallback plot when all attempts fail."""
        fallback_fig = self._generate_intelligent_fallback(question, df, column_info)
        
        if fallback_fig:
            try:
                plot_dict = self._convert_to_serializable_dict_robust(fallback_fig)
                plot_json = json.dumps(plot_dict, default=str, ensure_ascii=False)
                
                is_valid, validation_msg = self._validate_plot_json_comprehensive(plot_json)
                if is_valid:
                    logger.info("✓ Fallback plot generated and validated")
                    return plot_json, 'plotly'
                else:
                    logger.info(f"✗ Fallback plot validation failed: {validation_msg}")
            except Exception as e:
                logger.error(f"Fallback serialization error: {e}")
                import traceback
                traceback.print_exc()
        
        logger.info("✗ All plot generation attempts failed")
        return None, None
    
    def validate_plot_json(self, plot_json: str) -> Tuple[bool, str]:
        """Public method to validate Plotly JSON."""
        return self._validate_plot_json_comprehensive(plot_json)