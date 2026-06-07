import pandas as pd
import numpy as np
import json
import re
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import io
import base64
import plotly.graph_objects as go
import gc
import plotly.io as pio
from pathlib import Path
from src.utils.reasoning_extractor import REASONING_SECTION_PROMPT
import os
import logging
import aiofiles
from src.analytics.cardinality_handler import (
    build_chart_for_cardinality,
    get_cardinality_prompt_rules,
    get_cardinality_tier,
    is_high_cardinality,
    TOP_N_DEFAULT,
)
from src.analytics.sheet_orchestrator import OrchestrationResult

logger = logging.getLogger(__name__)

JSON_STR = '```json'
DIV = '</div>'
# Composed to avoid literal duplication warnings
_RD, _RDS = r'\d+', r'\.\s*'
NUMBERED_LIST_PATTERN = f"^{_RD}{_RDS}"

class ReportGenerator:
    """
    Generates executive-level data insights reports with visualizations and AI analysis.
    Designed for C-suite and senior management presentations.
    Now outputs modern HTML reports converted to PDFs.
    """
    
    def __init__(self, llm_client, plot_generator):
        self.llm_client = llm_client
        self.plot_generator = plot_generator
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom styles for HTML report formatting - optimized for PDF"""
        self.html_styles = """
        <style>
            :root {
                --primary-blue: #0d47a1;
                --secondary-blue: #1976d2;
                --light-blue: #e3f2fd;
                --dark-text: #263238;
                --medium-text: #37474f;
                --light-text: #546e7a;
                --subtle-text: #607d8b;
                --border-color: #bdbdbd;
                --bg-light: #f5f5f5;
            }
            
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            /* A4 page sizing: 794px wide at 96dpi = 210mm */
            @page {
                size: A4;
                margin: 0;
            }

            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                line-height: 1.5;
                color: var(--dark-text);
                background: #e8e8e8;
                padding: 20px 0;
                margin: 0;
            }

            /* A4 page container — every page of content sits inside one of these */
            .a4-page {
                width: 794px;
                min-height: 1123px;
                margin: 0 auto 20px auto;
                background: #ffffff;
                box-shadow: 0 2px 8px rgba(0,0,0,0.15);
                position: relative;
                overflow: hidden;
            }

            @media print {
                body {
                    background: white;
                    padding: 0;
                }
                .a4-page {
                    width: 210mm;
                    min-height: 297mm;
                    margin: 0;
                    box-shadow: none;
                    page-break-after: always;
                }
            }
            
            .cover-page {
                width: 794px;
                height: 1123px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                background: linear-gradient(135deg, var(--primary-blue) 0%, var(--secondary-blue) 100%);
                color: white;
                text-align: center;
                padding: 60px 40px;
                page-break-after: always;
                position: relative;
                margin: 0 auto 20px auto;
                box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            }
            
            .cover-title {
                font-size: 3rem;
                font-weight: 700;
                margin-bottom: 20px;
                letter-spacing: -0.5px;
            }
            
            .cover-subtitle {
                font-size: 1.3rem;
                font-weight: 300;
                margin-bottom: 50px;
                opacity: 0.95;
            }
            
            .cover-divider {
                width: 200px;
                height: 2px;
                background: rgba(255, 255, 255, 0.5);
                margin: 30px auto;
            }
            
            .cover-metadata {
                font-size: 1rem;
                opacity: 0.9;
                margin-top: 30px;
            }
            
            .cover-footer {
                position: absolute;
                bottom: 40px;
                font-size: 0.85rem;
                opacity: 0.7;
                font-style: italic;
            }
            
            .content-wrapper {
                width: 794px;
                margin: 0 auto;
                padding: 30px 40px;
                box-sizing: border-box;
            }
            
            .page-break {
                page-break-after: always;
                margin: 0;
                height: 1px;
                background: transparent;
            }
            
            .section-heading {
                font-size: 1.6rem;
                color: var(--primary-blue);
                font-weight: 700;
                margin: 25px 0 15px 0;
                padding-bottom: 8px;
                border-bottom: 2px solid var(--secondary-blue);
                page-break-after: avoid;
            }
            
            .subsection-heading {
                font-size: 1.2rem;
                color: var(--secondary-blue);
                font-weight: 600;
                margin: 20px 0 12px 0;
                page-break-after: avoid;
            }
            
            .insight-heading {
                font-size: 1rem;
                color: var(--primary-blue);
                font-weight: 600;
                margin: 15px 0 8px 0;
                page-break-after: avoid;
            }
            
            .insight-text {
                font-size: 0.95rem;
                color: var(--medium-text);
                line-height: 1.6;
                margin: 8px 0;
            }
            
            .bullet-point {
                font-size: 0.95rem;
                color: var(--medium-text);
                line-height: 1.6;
                margin: 8px 0 8px 20px;
                padding-left: 10px;
                border-left: 2px solid var(--light-blue);
            }
            
            .metadata-text {
                font-size: 0.9rem;
                color: var(--subtle-text);
                margin: 6px 0;
                font-style: italic;
            }
            
            .section-intro {
                font-size: 0.95rem;
                color: var(--light-text);
                font-style: italic;
                margin: 15px 0;
                line-height: 1.5;
            }
            
            .executive-summary {
                font-size: 0.95rem;
                color: var(--dark-text);
                line-height: 1.6;
                margin: 12px 0;
            }
            
            .summary-table, .sheet-table {
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0 20px 0;
                page-break-inside: avoid;
            }
            
            .summary-table th, .sheet-table th {
                background: var(--primary-blue);
                color: white;
                padding: 10px;
                text-align: left;
                font-weight: 600;
                font-size: 0.95rem;
                border: 1px solid var(--border-color);
            }
            
            .summary-table td, .sheet-table td {
                background: var(--bg-light);
                padding: 8px 10px;
                border: 1px solid var(--border-color);
                font-size: 0.9rem;
            }
            
            .summary-table tr:first-child td {
                background: var(--primary-blue);
                color: white;
                font-weight: 600;
                font-size: 0.95rem;
            }
            
            .summary-table td:first-child {
                font-weight: 600;
                color: var(--primary-blue);
                background: var(--bg-light);
            }
            
            .summary-table tr:first-child td:first-child {
                color: white;
                background: var(--primary-blue);
            }
            
            /* Critical: Prevent plots from breaking across pages */
            .plot-section {
                page-break-inside: avoid;
                margin: 20px 0 30px 0;
                padding: 15px 0;
            }
            
            .plot-container {
                margin: 15px 0 20px 0;
                padding: 10px;
                background: white;
                page-break-inside: avoid;
                overflow: visible !important;
            }
            
            .plot-wrapper {
                width: 100%;
                margin: 0 auto;
                page-break-inside: avoid;
                overflow: visible !important;
                min-height: 500px;
            }
            
            .plot-wrapper > div {
                overflow: visible !important;
            }
            
            .js-plotly-plot {
                overflow: visible !important;
            }
            
            .insights-section {
                margin: 15px 0;
                page-break-inside: avoid;
            }
            
            .insight-card {
                background: white;
                padding: 20px;
                margin: 20px 0;
                border-left: 4px solid var(--secondary-blue);
                page-break-inside: avoid;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            
            .insight-type-badge {
                display: inline-block;
                padding: 5px 12px;
                border-radius: 3px;
                font-size: 0.75rem;
                font-weight: 600;
                margin-bottom: 10px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .badge-correlation {
                background: #e3f2fd;
                color: #1976d2;
            }
            
            .badge-segmentation {
                background: #fff3e0;
                color: #f57c00;
            }
            
            .insight-card h3 {
                color: var(--primary-blue);
                font-size: 1.2rem;
                margin-bottom: 15px;
                font-weight: 700;
            }
            
            .statistical-basis {
                font-size: 0.8rem;
                color: #90a4ae;
                font-style: italic;
                margin: 8px 0 15px 0;
                padding-bottom: 10px;
                border-bottom: 1px solid #e0e0e0;
            }
            
            .metrics-analyzed {
                font-size: 0.85rem;
                color: var(--subtle-text);
                font-style: italic;
                margin-top: 15px;
                padding-top: 12px;
                border-top: 1px solid #e0e0e0;
            }
            
            /* Insight content formatting */
            .insight-content {
                margin: 10px 0;
            }
            
            .insight-subheading {
                color: var(--secondary-blue);
                font-size: 1rem;
                font-weight: 700;
                margin: 15px 0 8px 0;
            }
            
            .insight-content p {
                font-size: 0.95rem;
                color: var(--medium-text);
                line-height: 1.6;
                margin: 8px 0 12px 0;
            }
            
            .insight-list {
                list-style-type: none;
                padding-left: 0;
                margin: 10px 0;
            }
            
            .insight-list li {
                font-size: 0.95rem;
                color: var(--medium-text);
                line-height: 1.6;
                margin: 6px 0;
                padding-left: 20px;
                position: relative;
            }
            
            .insight-list li:before {
                content: "●";
                color: var(--secondary-blue);
                font-weight: bold;
                position: absolute;
                left: 0;
            }
            
            /* Print / save-as-PDF optimizations */
            @media print {
                body {
                    print-color-adjust: exact;
                    -webkit-print-color-adjust: exact;
                    background: white;
                    padding: 0;
                }

                .cover-page {
                    width: 210mm;
                    height: 297mm;
                    margin: 0;
                    box-shadow: none;
                    page-break-after: always;
                }

                .content-wrapper {
                    width: 210mm;
                }
                
                .page-break {
                    page-break-after: always;
                }
                
                .plot-section {
                    page-break-inside: avoid;
                }
                
                .plot-container {
                    page-break-inside: avoid;
                    overflow: visible !important;
                }
                
                .plot-wrapper {
                    overflow: visible !important;
                }
                
                .insight-card {
                    page-break-inside: avoid;
                }
                
                .section-heading,
                .subsection-heading,
                .insight-heading {
                    page-break-after: avoid;
                }
                
                table {
                    page-break-inside: avoid;
                }
            }
        </style>
        """
    
    def _create_cover_page(self) -> str:
        """Create professional HTML cover page"""
        date_str = datetime.now().strftime('%B %d, %Y')
        
        return f"""
        <div class="cover-page">
            <h1 class="cover-title">Data Insights Report</h1>
            <p class="cover-subtitle">Executive Analytics & Strategic Intelligence</p>
            <div class="cover-divider"></div>
            <p class="cover-metadata">Generated: {date_str}</p>
            <p class="cover-footer">Confidential - For Executive Review Only</p>
        </div>
        """
    
    
    def _remove_preambles(self, text: str) -> str:
        """Strip common AI preamble phrases from insight text."""
        preambles = [
            r"here are \d+-\d+ key insights?.*?:",
            r"here are \d+ key insights?.*?:",
            r"here are some key insights?.*?:",
            r"based on the (?:provided )?data.*?:",
            r"(?:based on|from) the visualization.*?:",
            r"the following insights?.*?:",
            r"key insights?:",
            r"insights?:",
        ]
        for pattern in preambles:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        return text.strip()

    def _extract_numbered_insights(self, text: str) -> List[str]:
        matches = re.findall(r'\d+\.\s*\*?\*?(.+)(?=\d+\.\s*\*?\*?|\Z)', text, re.DOTALL)
        if len(matches) < 2:
            return []
        return [
            re.sub(NUMBERED_LIST_PATTERN, '', m.strip().lstrip('*').rstrip('*').strip())  # strip leading "N."
            for m in matches if m.strip()
        ]

    def _extract_bullet_insights(self, text: str) -> List[str]:
        """Extract insights from asterisk bullet format. Returns [] if not found."""
        matches = re.findall(r'\*+\s*(?:\*\*)?(.+)(?=\*+\s*(?:\*\*)?\s*[A-Z]|\Z)', text, re.DOTALL)
        if len(matches) < 2:
            return []
        result = []
        for m in matches:
            cleaned = re.sub(NUMBERED_LIST_PATTERN, '', re.sub(r'\*\*(.*?)\*\*', r'\1', m.strip().lstrip('*').rstrip('*').strip()))
            if len(cleaned) > 20:
                result.append(cleaned)
        return result

    def _extract_sentence_groups(self, text: str) -> List[str]:
        """Group sentences into 2-sentence insight chunks as a last-resort extraction."""
        sentences = re.split(r'\.(?:\s+|$)', text)
        groups, current = [], []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            current.append(sentence)
            if len(current) >= 2:
                groups.append('. '.join(current) + '.')
                current = []
        if current:
            groups.append('. '.join(current) + '.')
        return groups

    def _extract_insights_list(self, text: str) -> List[str]:
        """Try each extraction strategy in order, returning the first that yields results."""
        return (
            self._extract_numbered_insights(text)
            or self._extract_bullet_insights(text)
            or self._extract_sentence_groups(text)
        )

    def _clean_insight_text(self, insight: str) -> str:
        """Remove asterisks, collapse whitespace, and strip a single insight string."""
        insight = re.sub(r'\*+', '', insight)
        insight = re.sub(r'\s+', ' ', insight).strip()
        # Strip trailing stray numbers like "... 2." or "... 2"
        insight = re.sub(r'\s+\d+\.?\s*$', '', insight).strip()
        return insight

    def _clean_and_format_insights(self, insights: str) -> List[str]:
        """
        Clean insights by removing AI preambles and extracting key points.
        Returns a list of up to 4 formatted insight strings.
        """
        cleaned_text = self._remove_preambles(insights)
        insights_list = self._extract_insights_list(cleaned_text)

        cleaned_insights = [
            self._clean_insight_text(insight)
            for insight in insights_list
            if len(self._clean_insight_text(insight)) > 30
        ]

        return (cleaned_insights or [cleaned_text])[:4]
    
    def _get_col_base_info(self, col: pd.Series, col_name: str, total_rows: int) -> Dict[str, Any]:
        """Extract base metadata for a column."""
        return {
            'name': col_name,
            'dtype': str(col.dtype),
            'non_null_count': int(col.count()),
            'null_count': int(col.isnull().sum()),
            'null_percentage': round(col.isnull().sum() / total_rows * 100, 2),
            'unique_count': int(col.nunique())
        }

    def _try_parse_datetime_object_col(self, df: pd.DataFrame, col: str, col_info: Dict[str, Any]) -> bool:
        """
        Try to detect and parse datetime from object columns.
        Returns True if column was identified as datetime, False otherwise.
        """
        try:
            sample_non_null = df[col].dropna().head(5)
            if len(sample_non_null) == 0:
                return False
            pd.to_datetime(sample_non_null, errors='raise', format='mixed')
            try:
                df_temp = pd.to_datetime(df[col], errors='coerce')
                col_info['min_date'] = df_temp.min().strftime('%Y-%m-%d') if not pd.isna(df_temp.min()) else None
                col_info['max_date'] = df_temp.max().strftime('%Y-%m-%d') if not pd.isna(df_temp.max()) else None
                return True
            except Exception:
                return False
        except (ValueError, TypeError):
            return False

    def _get_numeric_stats(self, col: pd.Series) -> Dict[str, Any]:
        """Compute descriptive stats for a numeric column."""
        if col.isnull().all():
            return dict.fromkeys(['mean', 'median', 'std', 'min', 'max', 'q25', 'q75'])
        return {
            'mean':   float(col.mean()),
            'median': float(col.median()),
            'std':    float(col.std()),
            'min':    float(col.min()),
            'max':    float(col.max()),
            'q25':    float(col.quantile(0.25)),
            'q75':    float(col.quantile(0.75))
        }

    def _get_categorical_top_values(self, col: pd.Series) -> Dict[str, int]:
        """Return top-10 value counts for a categorical column as JSON-safe dict."""
        try:
            top_values = col.value_counts().head(10).to_dict()
            result = {}
            for k, v in top_values.items():
                if pd.isna(k):
                    key_str = "null"
                elif isinstance(k, (pd.Timestamp, datetime)):
                    key_str = k.strftime('%Y-%m-%d') if hasattr(k, 'strftime') else str(k)
                else:
                    key_str = str(k)
                result[key_str] = int(v)
            return result
        except Exception:
            return {}

    def _get_datetime_range(self, col: pd.Series) -> Dict[str, Any]:
        """Return min/max date strings for a datetime column."""
        try:
            return {
                'min_date': col.min().strftime('%Y-%m-%d') if not pd.isna(col.min()) else None,
                'max_date': col.max().strftime('%Y-%m-%d') if not pd.isna(col.max()) else None
            }
        except Exception:
            return {
                'min_date': str(col.min()) if not pd.isna(col.min()) else None,
                'max_date': str(col.max()) if not pd.isna(col.max()) else None
            }

    def _classify_and_enrich_col(self, df: pd.DataFrame, col: str,
                                col_info: Dict[str, Any], schema_info: Dict[str, Any]) -> None:
        """
        Classify a column as numeric / categorical / datetime,
        enrich col_info with stats, and update schema_info lists.
        Mutates both col_info and schema_info in place.
        """
        series = df[col]

        # Object column — check for hidden datetime first
        if series.dtype == 'object':
            if self._try_parse_datetime_object_col(df, col, col_info):
                schema_info['datetime_columns'].append(col)
                schema_info['columns'].append(col_info)
                return  # early-exit; no further classification needed

        # Numeric
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            col_info['stats'] = self._get_numeric_stats(series)
            schema_info['numeric_columns'].append(col)

        # Categorical
        elif series.dtype == 'object' or series.nunique() < 20:
            col_info['top_values'] = self._get_categorical_top_values(series)
            schema_info['categorical_columns'].append(col)

        # Datetime (native dtype)
        elif pd.api.types.is_datetime64_any_dtype(series):
            col_info.update(self._get_datetime_range(series))
            schema_info['datetime_columns'].append(col)

        schema_info['columns'].append(col_info)

    def _get_dataframe_schema_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract comprehensive schema information from dataframe."""
        schema_info = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': [],
            'numeric_columns': [],
            'categorical_columns': [],
            'datetime_columns': [],
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 ** 2),
            'high_cardinality_columns': [] 
        }

        for col in df.columns:
            col_info = self._get_col_base_info(df[col], col, schema_info['total_rows'])
            self._classify_and_enrich_col(df, col, col_info, schema_info)
            
            # NEW: tag high-cardinality columns
            if col in schema_info['categorical_columns']:
                tier = get_cardinality_tier(df[col])
                col_info['cardinality_tier'] = tier
                if tier in ('high', 'extreme'):
                    col_info['visualization_note'] = (
                        f'[HIGH CARDINALITY ({col_info["unique_count"]} unique values) '
                        f'— use treemap / horizontal bar / ranked table instead of pie/bar]'
                    )
                    schema_info['high_cardinality_columns'].append(col)

        return schema_info
    
    def _generate_analysis_plan(self, schema_info: Dict[str, Any],
                           llm_params: Dict[str, Any],
                           token_tracker, auth_token: str) -> Dict[str, Any]:
        """Use LLM to generate a comprehensive analysis plan based on data schema"""

        prompt = f"""You are a data insights expert. Based on the dataset schema below, create a comprehensive Exploratory Data Analysis (EDA) insights plan with DIVERSE visualization types.

Dataset Schema:
- Total Rows: {schema_info['total_rows']}
- Total Columns: {schema_info['total_columns']}
- Numeric Columns: {schema_info['numeric_columns']}
- Categorical Columns: {schema_info['categorical_columns']}
- Datetime Columns: {schema_info['datetime_columns']}
- High Cardinality Columns: {schema_info['high_cardinality_columns']}

Column Details (including cardinality tier per categorical column):
{json.dumps(schema_info['columns'], indent=2)}

Create an analysis plan with these sections:
1. **Individual Metrics**: Analyze individual columns (distributions, frequencies)
2. **Relationships**: Analyze relationships between two variables
3. **Advanced Patterns**: Analyze relationships between three variables (if applicable)
4. **Insights**: Deep, non-obvious insights that require cross-analysis and domain expertise

{get_cardinality_prompt_rules()}

CRITICAL VISUALIZATION DIVERSITY RULES:
1. **For Individual Metrics** - Use DIFFERENT chart types based on cardinality tier:
- Numeric columns: Use "histogram", "box plot", or "violin plot"
- Categorical LOW tier (<=8 unique): Use "pie chart" or "donut chart"
- Categorical MEDIUM tier (8-15 unique): Use "horizontal bar chart"
- Categorical HIGH tier (15-30 unique): Use "treemap" or "top-N horizontal bar chart"
- Categorical EXTREME tier (30+ unique): Use "treemap" or "ranked table"
- Time series data: Use "line chart" or "area chart"

2. **For Relationships** - Use DIFFERENT chart types:
- Numeric vs Numeric: Use "scatter plot"
- High-cardinality Categorical vs Numeric: Use "aggregated horizontal bar chart"
- Low-cardinality Categorical vs Numeric: Use "grouped bar chart"
- Two High-cardinality Categoricals: Use "heatmap"
- Low-cardinality Categorical vs Categorical: Use "stacked bar chart" or "grouped bar chart"
- Time vs Numeric: Use "line chart" or "area chart"

3. **For Advanced Patterns** - Use DIFFERENT chart types:
- Faceted/grouped visualizations
- Stacked or grouped bar charts with multiple dimensions
- Heatmaps for two categorical dimensions

HARD BLOCK RULES — these override everything else:
- NEVER suggest a standard pie chart or donut chart for any column with more than 8 unique values.
  Use horizontal bar chart, treemap, or ranked table instead.
- NEVER suggest a standard vertical bar chart for any column with more than 15 unique values.
  Use horizontal bar chart or treemap instead.
- NEVER use columns listed under "High Cardinality Columns" as a color/facet dimension in
  advanced pattern analyses — this causes overcrowded subplots. Use aggregation or pick a
  lower cardinality column instead.
- The columns available for categorical visualizations are ONLY those listed under
  "Categorical Columns" in the schema above. Do not invent or use any other column.
- DO NOT use date/datetime columns directly in visualizations.
- DO NOT use email address columns or columns with values longer than 30 characters as
  x-axis in any chart.
- NEVER emit chart_type "scatter_plot" when both column_x and column_y are 
  categorical (object dtype). For two categorical columns, use "heatmap" instead.
  Scatter plots are ONLY valid when at least one axis is numeric.
- NEVER plot dates beyond the actual max date in the data. When using datetime columns,
  the generated code MUST filter to only dates present in the dataset. Do NOT use
  pd.Timestamp.today() as the ceiling — use df['<date_col>'].max() instead.
  Do NOT use resample() or date_range() which can fabricate missing periods.

CHART TYPE EXAMPLES FOR QUESTIONS:
- Pie chart (LOW cardinality only): "Show a pie chart of the distribution of [column]"
- Horizontal bar: "Create a horizontal bar chart showing the top 10 [column] by count"
- Treemap: "Create a treemap showing the distribution of [column]"
- Ranked table: "Show a ranked table of the top 20 [column] values by frequency"
- Histogram: "Create a histogram showing the distribution of [column]"
- Box plot: "Generate a box plot for [column]"
- Scatter plot: "Create a scatter plot of [y] vs [x]"
- Line chart: "Show a line chart of [y] over [x]"
- Aggregated horizontal bar: "Create a horizontal bar chart showing average [numeric] by top 10 [category]"
- Heatmap: "Create a heatmap using pd.crosstab showing frequency of [category_1] vs [category_2] — use pd.crosstab() with string-cast index/columns"
- Line chart (time series): "Show a line chart of [numeric] aggregated by month over [date_col] — convert date column with pd.to_datetime() and group by month period"
- Grouped bar chart: "Create a grouped bar chart showing [y] by [x] and [color]"
- Stacked bar chart: "Generate a stacked bar chart of [y] by [x]"

SELECTION RULES:
1. Select 3-4 individual analyses with AT LEAST 3 DIFFERENT chart types
2. Select 3-4 relationship analyses with AT LEAST 3 DIFFERENT chart types
3. Select 2-3 advanced pattern analyses with AT LEAST 2 DIFFERENT chart types
4. Prioritize columns with high business value or interesting patterns
5. For each analysis item, include a "cardinality_tier" field using the value from
   the column details above (low / medium / high / extreme). For numeric columns
   set cardinality_tier to "numeric".
6. Avoid redundant analyses
7. Explicitly specify the chart type in the question

For hidden_insights (DO NOT GENERATE - THESE WILL BE COMPUTED FROM ACTUAL DATA):
- Leave this section empty in your response
- Insights will be generated automatically from correlation analysis

Respond ONLY with valid JSON in this exact format:
{{
"individual_analyses": [
    {{
    "column": "column_name",
    "question": "Create a [CHART_TYPE] showing the distribution of column_name",
    "description": "What to analyze and why",
    "chart_type": "histogram|pie_chart|horizontal_bar|treemap|ranked_table|box_plot|etc",
    "cardinality_tier": "low|medium|high|extreme|numeric"
    }}
],
"relationship_analyses": [
    {{
    "column_x": "column_name_1",
    "column_y": "column_name_2",
    "question": "Create a [CHART_TYPE] of column_name_2 vs column_name_1",
    "description": "What relationship to explore",
    "chart_type": "scatter_plot|line_chart|aggregated_horizontal_bar|heatmap|grouped_bar_chart|etc",
    "cardinality_tier": "low|medium|high|extreme|numeric"
    }}
],
"advanced_pattern_analyses": [
    {{
    "column_x": "column_name_1",
    "column_y": "column_name_2",
    "column_z": "column_name_3",
    "question": "Create a [CHART_TYPE] showing column_name_2 vs column_name_1 by column_name_3",
    "description": "What complex relationship to explore",
    "chart_type": "grouped_bar_chart|heatmap|treemap|etc",
    "cardinality_tier": "low|medium|high|extreme|numeric"
    }}
]
}}

IMPORTANT: Ensure you use DIFFERENT chart types across analyses. Do NOT repeat the same chart type more than once in each section.

Your JSON response:

{REASONING_SECTION_PROMPT}"""

        response = self.llm_client.generate(
            prompt=prompt,
            llm_params=llm_params,
            token_tracker=token_tracker,
            auth_token=auth_token,
            temperature=0.3
        )

        try:
            cleaned = response.strip()
            if cleaned.startswith(JSON_STR):
                cleaned = cleaned.split(JSON_STR)[1].split('```')[0].strip()
            elif cleaned.startswith('```'):
                cleaned = cleaned.split('```')[1].split('```')[0].strip()

            analysis_plan = json.loads(cleaned)
            return analysis_plan
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse analysis plan: {e}")
            return self._get_default_analysis_plan(schema_info)


    def _get_default_analysis_plan(self, schema_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a basic default analysis plan if LLM fails"""
        plan = {
            'individual_analyses': [],
            'relationship_analyses': [],
            'advanced_pattern_analyses': []
        }
        
        # Add individual for first 3 numeric columns
        for col in schema_info['numeric_columns'][:3]:
            plan['individual_analyses'].append({
                'column': col,
                'question': f'Show the distribution of {col}',
                'description': f'Distribution of {col}'
            })
        
        # Add individual for first 2 categorical columns
        for col in schema_info['categorical_columns'][:2]:
            plan['individual_analyses'].append({
                'column': col,
                'question': f'Show the frequency distribution of {col}',
                'description': f'Frequency distribution of {col}'
            })
        
        # Add relationship if we have numeric columns
        if len(schema_info['numeric_columns']) >= 2:
            plan['relationship_analyses'].append({
                'column_x': schema_info['numeric_columns'][0],
                'column_y': schema_info['numeric_columns'][1],
                'question': f"Create a scatter plot of {schema_info['numeric_columns'][1]} vs {schema_info['numeric_columns'][0]}",
                'description': f"Relationship between {schema_info['numeric_columns'][0]} and {schema_info['numeric_columns'][1]}"
            })
        
        return plan
    
    
    def _exact_match(self, col_name: str, actual_cols: List[str]) -> Optional[str]:
        """Return col_name if it exactly matches any actual column, else None."""
        return col_name if col_name in actual_cols else None

    def _case_insensitive_match(self, col_name: str, actual_cols: List[str],
                                actual_cols_lower: List[str]) -> Optional[str]:
        """Return matching actual column by case-insensitive comparison, else None."""
        col_lower = str(col_name).lower()
        for actual, actual_lower in zip(actual_cols, actual_cols_lower):
            if col_lower == actual_lower:
                return actual
        return None

    def _partial_match(self, col_name: str, actual_cols: List[str],
                    actual_cols_lower: List[str]) -> Optional[str]:
        """Return matching actual column by partial/contains comparison, else None."""
        col_lower = str(col_name).lower()
        for actual, actual_lower in zip(actual_cols, actual_cols_lower):
            if col_lower in actual_lower or actual_lower in col_lower:
                return actual
        return None

    def _best_column_match(self, col_name: str, actual_cols: List[str],
                            actual_cols_lower: List[str]) -> Optional[str]:
        """Return best matching actual column name using exact → case-insensitive → partial."""
        if col_name is None:
            return None
        return (
            self._exact_match(col_name, actual_cols)
            or self._case_insensitive_match(col_name, actual_cols, actual_cols_lower)
            or self._partial_match(col_name, actual_cols, actual_cols_lower)
        )

    def _sanitize_single_key(self, key: str, sanitized: Dict[str, Any],
                            original: Dict[str, Any], actual_cols: List[str],
                            actual_cols_lower: List[str]) -> bool:
        """
        Validate and correct a single column key in sanitized dict.
        Returns False if column has no match (analysis should be skipped), True otherwise.
        """
        if key not in sanitized or sanitized[key] is None:
            return True

        matched = self._best_column_match(sanitized[key], actual_cols, actual_cols_lower)
        if matched is None:
            logger.info(f"Column '{sanitized[key]}' not found in dataframe. Skipping analysis.")
            return False

        if matched != sanitized[key]:
            logger.info(f"Column name corrected: '{sanitized[key]}' → '{matched}'")
            sanitized[key] = matched

        if 'question' in sanitized:
            sanitized['question'] = (
                sanitized['question']
                .replace(f"'{original[key]}'", f"'{matched}'")
                .replace(original[key], matched)
            )
        return True

    def _sanitize_analysis_columns(self, analysis: Dict[str, Any], df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        actual_cols = list(df.columns)
        actual_cols_lower = [c.lower() for c in actual_cols]
        sanitized = analysis.copy()

        for key in ['column', 'column_x', 'column_y', 'column_z']:
            if key not in sanitized or sanitized[key] is None:
                continue  # skip None/missing — don't return None for optional column_z
            if not self._sanitize_single_key(key, sanitized, analysis, actual_cols, actual_cols_lower):
                if key in ('column_x', 'column_y', 'column'):  # required fields
                    return None
                else:
                    sanitized[key] = None  # optional field (column_z) — null it out safely
        
        return sanitized
    
    
    def _create_visualization_from_plan(
        self,
        df: pd.DataFrame,
        analysis: dict,
        analysis_type: str,
        llm_params: dict,
        token_tracker,
        auth_token: str,
    ) -> "Optional[str]":
        try:
            analysis = self._sanitize_analysis_columns(analysis, df)
            if analysis is None:
                return None

            df = self._clean_df_for_visualization(df)

            # High-cardinality routing
            routing_result = self._handle_routing_options(df, analysis, analysis_type)
            if routing_result:
                return routing_result

            question = self._prepare_visualization_question(analysis, analysis_type)
            logger.info("Generating plot for question: %s", question)

            plot_json, _ = self.plot_generator.create_plot(
                question=question,
                df=df,
                llm_params=llm_params,
                token_tracker=token_tracker,
                auth_token=auth_token,
                max_retries=2,
            )
            return plot_json

        except Exception as exc:
            logger.error("Visualization creation failed: %s", exc)
            return None

    def _clean_df_for_visualization(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(how="all").dropna(axis=1, how="all")
        for col in df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
            if 'created' in col.lower():
                col_max = df[col].max()
                if pd.notna(col_max):
                    # FIX: preserve NaT rows — only drop rows with a valid date that exceeds max
                    df = df[(df[col].isna()) | (df[col] <= col_max)]

        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].fillna("N/A")
        return df

    def _handle_routing_options(self, df, analysis, analysis_type) -> Optional[str]:
        """Check for and apply specialized routing (cardinality handler, cat+cat intercept)."""
        if self._should_use_cardinality_handler(analysis, df):
            logger.info(
                "High-cardinality column detected — using cardinality handler for: %s",
                analysis.get("description", ""),
            )
            result = self._route_high_cardinality_plot(df, analysis, analysis_type)
            if result:
                return result
            logger.info("Cardinality handler returned None — falling back to LLM generation.")

        # Categorical relationship routing
        if analysis_type in ("relationship", "advanced_pattern"):
            chart_type = analysis.get("chart_type", "").lower()
            col_x = analysis.get("column_x") or analysis.get("column")
            col_y = analysis.get("column_y")
            if (col_x and col_y
                    and col_x in df.columns and col_y in df.columns
                    and df[col_x].dtype == "object"
                    and df[col_y].dtype == "object"
                    and chart_type in ("scatter_plot", "scatter plot", "scatter",
                                    "heatmap", "heat_map")):
                logger.info("Intercepting cat+cat %s — rerouting to cardinality handler", chart_type)
                col_display = col_x.replace("_", " ").title()
                second_display = col_y.replace("_", " ").title()
                fig = build_chart_for_cardinality(
                    df=df, col=col_x,
                    title=f"{col_display} vs {second_display}",
                    second_col=col_y
                )
                if fig:
                    plot_dict = self.plot_generator._convert_to_serializable_dict_robust(fig)
                    return json.dumps(plot_dict, default=str, ensure_ascii=False)
                logger.info("Cardinality handler reroute returned None — falling back to LLM.")
        return None

    def _prepare_visualization_question(self, analysis, analysis_type) -> str:
        """Build or retrieve the natural language question for plot generation."""
        question = analysis.get("question", "")
        if not question:
            if analysis_type == "individual":
                question = f"Show the distribution of {analysis['column']}"
            elif analysis_type == "relationship":
                question = f"Create a plot of {analysis['column_y']} vs {analysis['column_x']}"
            elif analysis_type == "advanced_pattern":
                question = (
                    f"Show {analysis['column_y']} vs {analysis['column_x']}"
                    f" colored by {analysis['column_z']}"
                )
        return question


    
    def _route_high_cardinality_plot(self, df, analysis, analysis_type):
        try:
            col = analysis.get("column") or analysis.get("column_x")
            if analysis_type == "advanced_pattern":
                second_col = analysis.get("column_z") or analysis.get("column_y")
            else:
                second_col = analysis.get("column_y")

            col_display = col.replace("_", " ").title()
            second_display = second_col.replace("_", " ").title() if second_col else None
            title = f"{col_display} vs {second_display}" if second_display else f"Distribution of {col_display}"

            if not col or col not in df.columns:
                return None

            fig = build_chart_for_cardinality(
                df=df, col=col, title=title,
                second_col=second_col if second_col and second_col in df.columns else None,
            )
            if fig is None:
                return None

            # FIX: use plot_generator's robust serializer instead of raw pio.to_json
            # This handles numpy/binary types that cause bdata in output
            plot_dict = self.plot_generator._convert_to_serializable_dict_robust(fig)
            plot_json = json.dumps(plot_dict, default=str, ensure_ascii=False)
            return plot_json

        except Exception as exc:
            logger.error("High-cardinality direct plot failed: %s", exc)
            return None
        

    def _should_use_cardinality_handler(self, analysis: dict, df: pd.DataFrame) -> bool:
        """
        Return True when the analysis plan indicates a high/extreme cardinality
        column OR when we detect it ourselves from the actual data.
        """
        plan_tier = analysis.get("cardinality_tier", "").lower()
        if plan_tier in ("high", "extreme"):
            return True
 
        # Double-check against actual data even if LLM missed it
        col = analysis.get("column") or analysis.get("column_x")
        if col and col in df.columns and df[col].dtype == "object":
            tier = get_cardinality_tier(df[col])
            if tier in ("high", "extreme"):
                logger.info(
                    "Column '%s' detected as '%s' cardinality — routing to cardinality handler.",
                    col, tier,
                )
                return True
 
        return False
    
    def _build_individual_chart_summary(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Build data summary for an individual column chart."""
        column = analysis['column']
        if pd.api.types.is_numeric_dtype(df[column]):
            return {
                'column': column,
                'type': 'numeric',
                'stats': {
                    'mean':   float(df[column].mean()),
                    'median': float(df[column].median()),
                    'std':    float(df[column].std()),
                    'min':    float(df[column].min()),
                    'max':    float(df[column].max())
                }
            }
        top_values = df[column].value_counts().head(10).to_dict()
        return {
            'column': column,
            'type': 'categorical',
            'top_values': {str(k): int(v) for k, v in top_values.items()}
        }

    def _build_relationship_chart_summary(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Build data summary for a two-column relationship chart."""
        col_x, col_y = analysis['column_x'], analysis['column_y']
        summary = {
            'column_x': col_x,
            'column_y': col_y,
            'sample_data': self._serialize_sample_records(df, [col_x, col_y], self._safe_serialize)
        }
        if pd.api.types.is_numeric_dtype(df[col_x]) and pd.api.types.is_numeric_dtype(df[col_y]):
            summary['correlation'] = float(df[col_x].corr(df[col_y]))
        return summary

    def _build_advanced_chart_summary(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> Dict[str, Any]:
        cols = [c for c in [analysis.get('column_x'), analysis.get('column_y'), analysis.get('column_z')] 
                if c is not None and c in df.columns]
        return {
            'columns': cols,
            'sample_data': self._serialize_sample_records(df, cols, self._safe_serialize) if cols else []
        }

    def _build_chart_data_summary(self, df: pd.DataFrame, analysis: Dict[str, Any],
                                analysis_type: str) -> Dict[str, Any]:
        """Route to the correct chart summary builder based on analysis_type."""
        if analysis_type == 'individual':
            return self._build_individual_chart_summary(df, analysis)
        if analysis_type == 'relationship':
            return self._build_relationship_chart_summary(df, analysis)
        return self._build_advanced_chart_summary(df, analysis)

    def _build_charts_data(self, df: pd.DataFrame,
                            analyses_with_plots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build the full charts_data list for the batch insights prompt."""
        charts_data = []
        for idx, item in enumerate(analyses_with_plots):
            analysis = item['analysis']
            analysis_type = item['analysis_type']
            charts_data.append({
                'index': idx,
                'analysis_type': analysis_type,
                'description': analysis.get('description', ''),
                'data_summary': self._build_chart_data_summary(df, analysis, analysis_type)
            })
        return charts_data

    def _parse_batch_insights_response(self, response: str) -> Dict[int, List[str]]:
        """Parse LLM batch insights JSON response into {index: insights} dict."""
        cleaned = response.strip()
        if cleaned.startswith(JSON_STR):
            cleaned = cleaned.split(JSON_STR)[1].split('```')[0].strip()
        elif cleaned.startswith('```'):
            cleaned = cleaned.split('```')[1].split('```')[0].strip()
        parsed = json.loads(cleaned)
        return {item['index']: item['insights'] for item in parsed}

    def _generate_insights_batch(self, df: pd.DataFrame,
                                analyses_with_plots: List[Dict[str, Any]],
                                llm_params: Dict[str, Any],
                                token_tracker, auth_token: str) -> Dict[int, List[str]]:
        """
        Generate insights for multiple charts in a single LLM call.
        Returns a dict of {index: [insight1, insight2, ...]}
        """
        charts_data = self._build_charts_data(df, analyses_with_plots)

        prompt = f"""You are a professional data analyst writing insights for an executive report.

    Below are {len(charts_data)} charts with their data summaries. For EACH chart, write 3-4 key insights.

    Charts Data:
    {json.dumps(charts_data, indent=2)}

    CRITICAL INSTRUCTIONS:
    1. Write insights for EVERY chart listed above
    2. DO NOT include preambles like "Here are the insights" or "Based on the data"
    3. Start DIRECTLY with numbered points (1., 2., 3., 4.)
    4. Each insight should be 2-3 sentences
    5. Focus on business implications and specific numbers from the data
    6. Use professional business language

    Return ONLY a JSON array:
    [
    {{
        "index": 0,
        "insights": [
        "1. First insight text here...",
        "2. Second insight text here...",
        "3. Third insight text here..."
        ]
    }},
    ...
    ]

    {REASONING_SECTION_PROMPT}"""

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                llm_params=llm_params,
                token_tracker=token_tracker,
                auth_token=auth_token,
                temperature=0.4
            )
            return self._parse_batch_insights_response(response)
        except Exception as e:
            logger.error(f"Batch insights generation failed: {e}")
            return {}
    
    def _build_individual_data_summary(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Build data summary for an individual column analysis."""
        column = analysis['column']
        if pd.api.types.is_numeric_dtype(df[column]):
            return {
                'column': column,
                'type': 'numeric',
                'stats': {
                    'mean':   float(df[column].mean()),
                    'median': float(df[column].median()),
                    'std':    float(df[column].std()),
                    'min':    float(df[column].min()),
                    'max':    float(df[column].max())
                }
            }
        top_values = df[column].value_counts().head(10).to_dict()
        return {
            'column': column,
            'type': 'categorical',
            'top_values': {str(k): int(v) for k, v in top_values.items()}
        }

    def _serialize_sample_records(self, df: pd.DataFrame, cols: List[str],
                                safe_serialize) -> List[Dict[str, Any]]:
        """Return serialized sample records for the given columns."""
        return [
            {k: safe_serialize(v) for k, v in record.items()}
            for record in df[cols].head(20).to_dict('records')
        ]

    def _build_relationship_data_summary(self, df: pd.DataFrame,
                                        analysis: Dict[str, Any], safe_serialize) -> Dict[str, Any]:
        """Build data summary for a relationship (two-column) analysis."""
        col_x, col_y = analysis['column_x'], analysis['column_y']
        summary = {
            'column_x': col_x,
            'column_y': col_y,
            'sample_data': self._serialize_sample_records(df, [col_x, col_y], safe_serialize)
        }
        if pd.api.types.is_numeric_dtype(df[col_x]) and pd.api.types.is_numeric_dtype(df[col_y]):
            summary['correlation'] = float(df[col_x].corr(df[col_y]))
        return summary

    def _build_advanced_data_summary(self, df: pd.DataFrame,
                                    analysis: Dict[str, Any], safe_serialize) -> Dict[str, Any]:
        """Build data summary for an advanced (three-column) pattern analysis."""
        cols = [analysis['column_x'], analysis['column_y'], analysis['column_z']]
        return {
            'columns': cols,
            'sample_data': self._serialize_sample_records(df, cols, safe_serialize)
        }

    def _build_data_summary(self, df: pd.DataFrame, analysis: Dict[str, Any],
                            analysis_type: str, safe_serialize) -> Dict[str, Any]:
        """Route to the correct data summary builder based on analysis_type."""
        if analysis_type == 'individual':
            return self._build_individual_data_summary(df, analysis)
        if analysis_type == 'relationship':
            return self._build_relationship_data_summary(df, analysis, safe_serialize)
        return self._build_advanced_data_summary(df, analysis, safe_serialize)

    def _safe_serialize(self, obj) -> Any:
        """Recursively serialize a value to a JSON-safe type."""
        if pd.isna(obj):
            return None
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.strftime('%Y-%m-%d %H:%M:%S') if hasattr(obj, 'strftime') else str(obj)
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): self._safe_serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._safe_serialize(item) for item in obj]
        return str(obj)

    def _generate_insights_for_visualization(self, df: pd.DataFrame,
                                            analysis: Dict[str, Any],
                                            analysis_type: str,
                                            llm_params: Dict[str, Any],
                                            token_tracker, auth_token: str) -> str:
        """Use LLM to generate insights from visualization and data."""
        data_summary = self._build_data_summary(df, analysis, analysis_type, self._safe_serialize)

        prompt = f"""You are a professional data analyst writing insights for an executive report.

    Analysis Type: {analysis_type}
    Context: {analysis.get('description', '')}

    Data Summary:
    {json.dumps(data_summary, indent=2)}

    CRITICAL INSTRUCTIONS:
    1. Write 3-4 KEY INSIGHTS in professional business language
    2. DO NOT include any preambles like "Here are the insights" or "Based on the data"
    3. DO NOT use phrases like "Here are 2-4 points" or "The following insights"
    4. Start DIRECTLY with the insights
    5. Format as numbered points (1., 2., 3., 4.)
    6. Each insight should be 2-3 sentences
    7. Focus on business implications and actionable observations
    8. Use clear, concise, professional language

    CORRECT FORMAT EXAMPLE:
    1. The distribution shows a significant concentration in the medium priority category, accounting for 45% of all tasks. This balanced approach suggests effective prioritization strategies are in place.

    2. High-effort tasks represent 60% of the workload, indicating substantial resource requirements. Organizations should consider capacity planning to ensure timely delivery.

    INCORRECT FORMAT (DO NOT DO THIS):
    "Here are 3 key insights based on the data:"
    "Based on the visualization, I can provide the following insights:"

    Write your insights now (numbered format, no preamble):

    {REASONING_SECTION_PROMPT}"""

        try:
            return self.llm_client.generate(
                prompt=prompt,
                llm_params=llm_params,
                token_tracker=token_tracker,
                auth_token=auth_token,
                temperature=0.4
            ).strip()
        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
            return (
                f"Analysis of {analysis.get('description', 'data visualization')} completed. "
                "Please review the visualization for patterns and trends."
            )
    
    
    def _unwrap_plot_json(self, plot_json: str) -> str:
        """Unwrap double-encoded JSON string up to 3 levels deep."""
        stripped = plot_json.strip()
        for _ in range(3):
            if stripped.startswith('"') and stripped.endswith('"'):
                try:
                    stripped = json.loads(stripped)
                except Exception:
                    break
            else:
                break
        if isinstance(stripped, dict):
            return json.dumps(stripped)
        return stripped

    def _parse_plot_json(self, plot_json: str) -> Optional[dict]:
        """Parse and validate plot JSON string. Returns dict or None on failure."""
        logger.debug(f"plot_json preview (first 200 chars): {plot_json[:200]}")
        logger.debug(f"plot_json preview (last 100 chars): {plot_json[-100:]}")
        try:
            return json.loads(plot_json)
        except json.JSONDecodeError as e:
            logger.error(f"plot_json is not valid JSON: {e}")
            logger.error(f"Offending area: '{plot_json[max(0, e.pos-30):e.pos+30]}'")
            return None

    def _is_valid_numeric(self, v) -> bool:
        """Return True if v is a non-None, non-NaN numeric value."""
        if v is None:
            return False
        # Catch string 'nan' that can appear in plotly trace data
        if isinstance(v, str):
            return False
        if isinstance(v, (int, float, np.integer, np.floating)):
            return not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))
        return False

    @staticmethod
    def _format_bar_label(v: float) -> str:
        """
        Format a bar chart label value.
        - Whole numbers (counts): show as integer  e.g. 42 -> '42'
        - Decimal values (means/averages): truncate decimals, show with .0
          e.g. 3.45 -> '3.0', 4.5654 -> '4.0'
        """
        fv = float(v)
        if fv == int(fv):
            return f'{int(fv)}'          # whole number: 42 -> '42'
        return f'{float(int(fv)):.1f}'   # decimal: 3.45 -> '3.0'

    def _apply_bar_labels(self, trace) -> None:
        """Data labels disabled — no-op."""
        pass

    def _apply_pie_labels(self, trace) -> None:
        """Data labels disabled — no-op."""
        pass

    def _apply_histogram_labels(self, trace) -> None:
        """Data labels disabled — no-op."""
        pass

    def _apply_trace_labels(self, fig) -> None:
        """Data labels disabled — no-op."""
        pass
    
    def _compute_chart_dimensions(self, fig) -> Tuple[int, int, int, int]:
        # A4 content width: 794px page - 80px padding = 714px max
        A4_CHART_W = 714

        is_faceted = any(key.startswith('xaxis') and key != 'xaxis' for key in fig.layout)
        n_facets = len([k for k in fig.to_dict().get('layout', {}) if k.startswith('xaxis')])

        all_x_vals = []
        for trace in fig.data:
            if hasattr(trace, 'x') and trace.x is not None:
                all_x_vals.extend(list(trace.x))

        unique_x = len({str(v) for v in all_x_vals if v is not None})
        max_x_label_len = max((len(str(v)) for v in all_x_vals if v is not None), default=0)

        if is_faceted and n_facets > 2:
            # Cap faceted width at A4 content width
            return A4_CHART_W, 600, 90, 8

        # High cardinality or long labels: force 90-degree rotation + more height
        if unique_x > 20 or max_x_label_len > 30:
            return A4_CHART_W, 550, 90, 8
        if unique_x > 10 or max_x_label_len > 20:
            return A4_CHART_W, 480, -45, 9
        return A4_CHART_W, 420, 0, 10

    def _build_xaxis_update(self, fig, tick_angle: int, font_size: int, unique_x: int) -> dict:
        """
        Build xaxis update dict, preserving categoryarray if already set by LLM
        to prevent Plotly from injecting phantom months/categories.
        """
        existing_xaxis = fig.layout.xaxis
        base = {
            'automargin': True,
            'tickangle': tick_angle,
            'tickfont': {
                'size': max(7, font_size - 3) if abs(tick_angle) >= 45 else max(8, font_size - 2)
            },
            'nticks': 20 if unique_x > 20 else 0,
        }

        # Preserve categoryarray/categoryorder if LLM already set them
        if (existing_xaxis and
                hasattr(existing_xaxis, 'type') and
                existing_xaxis.type == 'category' and
                hasattr(existing_xaxis, 'categoryarray') and
                existing_xaxis.categoryarray is not None):
            base['type'] = 'category'
            base['categoryorder'] = 'array'
            base['categoryarray'] = list(existing_xaxis.categoryarray)
            base['nticks'] = 0  # let categoryarray control ticks, not nticks

        return base
    
    def _apply_figure_layout(self, fig, chart_width: int, chart_height: int,
                        tick_angle: int, font_size: int, unique_x: int = 0) -> None:
        """Apply standard layout settings to figure in-place."""
        self._set_axis_titles(fig)
        
        bottom_margin = self._calculate_bottom_margin(tick_angle)

        fig.update_layout(
            height=chart_height,
            width=chart_width,
            autosize=True,
            margin={
                "l": 100,           
                "r": 150,           
                "t": 100,           
                "b": bottom_margin
            },
            font={'size': font_size},
            showlegend=True,
            hovermode='closest',
            legend={
                "orientation": "v",
                "yanchor": "top",
                "y": 1,
                "xanchor": "left",
                "x": 1.05,
                "bgcolor": "rgba(255, 255, 255, 0.8)",
                "bordercolor": "rgba(0, 0, 0, 0.2)",
                "borderwidth": 1
            },
            xaxis=self._build_xaxis_update(fig, tick_angle, font_size, unique_x),
            yaxis={
                'automargin': True,
                'tickfont': {'size': max(8, font_size - 1)},
            },
            uniformtext={'minsize': 7, 'mode': 'hide'}
        )

        self._handle_faceted_layout(fig, chart_height)

    def _set_axis_titles(self, fig) -> None:
        """Normalize axis titles by replacing underscores and title casing."""
        for axis_key in ['xaxis', 'yaxis']:
            ax = fig.layout[axis_key]
            if ax and hasattr(ax, 'title') and ax.title and ax.title.text:
                ax.title.text = ax.title.text.replace('_', ' ').title()

    def _calculate_bottom_margin(self, tick_angle: int) -> int:
        """Calculate bottom margin based on the x-axis tick rotation angle."""
        if abs(tick_angle) == 90:
            return 200
        if tick_angle != 0:
            return 150
        return 80

    def _handle_faceted_layout(self, fig, chart_height: int) -> None:
        """Adjust annotations and layout widths for faceted (multi-subplot) charts."""
        is_faceted = any(key.startswith('xaxis') and key != 'xaxis' for key in fig.layout)
        if not is_faceted:
            return

        self._process_facet_annotations(fig)
        self._process_facet_axes(fig)
        self._update_facet_dimensions(fig, chart_height)

    def _process_facet_annotations(self, fig) -> None:
        """Format facet subplot annotations (titles)."""
        for annotation in fig.layout.annotations:
            if not annotation.text:
                continue
            
            text = annotation.text
            if "=" in text:
                text = text.split("=", 1)[1]
            if len(text) > 15:
                text = text[:12] + "..."
            
            annotation.update({
                "text": text.title(),
                "font": {"size": 8},
                "textangle": -45,
            })

    def _process_facet_axes(self, fig) -> None:
        """Apply density and styling updates to facet axes."""
        for key, axis in fig.layout.to_plotly_json().items():
            if key.startswith('xaxis') and key != 'xaxis':
                axis.update({
                    'tickangle': 90,
                    'tickfont': {'size': 7},
                    'automargin': True,
                    'title': ""
                })
            elif key.startswith('yaxis'):
                axis.update({'automargin': True})

    def _update_facet_dimensions(self, fig, chart_height: int) -> None:
        """Calculate and apply dynamic width/height and legend for faceted charts."""
        # A4 content width cap — faceted charts cannot exceed page width
        A4_CHART_W = 714
        fig.update_layout(
            width=A4_CHART_W,
            height=max(chart_height, 600),
            margin={"t": 120, "b": 160, "l": 60, "r": 60},
            legend={
                "orientation": "h",
                "yanchor": "top",
                "y": -0.25,
                "xanchor": "center",
                "x": 0.5,
                "font": {"size": 8},
                "tracegroupgap": 2,
            },
        )

    
    def _sanitize_json_string(self, json_str: str) -> str:
        """Replace JS-invalid tokens that orjson rejects."""
        import re
        json_str = re.sub(r'\bNaN\b', 'null', json_str)
        json_str = re.sub(r'\bInfinity\b', 'null', json_str)
        json_str = re.sub(r'\b-Infinity\b', 'null', json_str)
        return json_str

    def _parse_plot_json(self, plot_json: str) -> Optional[dict]:
        """Parse and validate plot JSON string. Returns dict or None on failure."""
        try:
            return json.loads(plot_json)
        except json.JSONDecodeError:
            # Try sanitizing NaN/Infinity tokens and retry
            try:
                sanitized = self._sanitize_json_string(plot_json)
                return json.loads(sanitized)
            except json.JSONDecodeError as e:
                logger.error(f"plot_json is not valid JSON even after sanitization: {e}")
                return None

    def _replace_nan_in_trace_list_attr(self, trace, attr: str) -> None:
        """Replace NaN/nan/none strings with 'N/A' in trace list attribute."""
        if not hasattr(trace, attr) or getattr(trace, attr) is None:
            return
        val = getattr(trace, attr)
        if isinstance(val, (list, tuple)):
            setattr(trace, attr, ['N/A' if str(v).lower() in ('nan', 'none', '') else v for v in val])
        elif isinstance(val, str) and val.lower() in ('nan', 'none', ''):
            setattr(trace, attr, 'N/A')

    def _clean_trace_nan_values(self, fig) -> None:
        """Remove NaN/nan/none values from all trace attributes in-place."""
        for trace in fig.data:
            for attr in ['labels', 'names', 'text', 'x', 'y', 'legendgroup', 'name']:
                self._replace_nan_in_trace_list_attr(trace, attr)

    def _clean_axis_tick_text(self, fig) -> None:
        """Remove NaN/nan/none values from axis tick labels in-place."""
        for axis_name in ['xaxis', 'yaxis']:
            ax = getattr(fig.layout, axis_name, None)
            if ax and hasattr(ax, 'ticktext') and ax.ticktext is not None:
                ax.ticktext = ['N/A' if str(v).lower() in ('nan', 'none', '') else v for v in ax.ticktext]

    def _capitalize_trace_labels(self, fig) -> None:
        for trace in fig.data:
            for attr in ['x', 'y', 'labels', 'text', 'name', 'legendgroup']:
                val = getattr(trace, attr, None)
                if val is None:
                    continue
                if isinstance(val, (list, tuple)):
                    setattr(trace, attr, [
                        ' '.join(w.capitalize() for w in v.split()) if isinstance(v, str) else v
                        for v in val
                    ])
                elif isinstance(val, str):
                    setattr(trace, attr, ' '.join(w.capitalize() for w in val.split()))

    def _get_unique_x_count(self, fig) -> int:
        """Count unique x-axis values across all traces."""
        all_x = []
        for trace in fig.data:
            if hasattr(trace, 'x') and trace.x is not None:
                all_x.extend(list(trace.x))
        return len({str(v) for v in all_x if v is not None})

    def _convert_plot_to_html(self, plot_json: str) -> str:
        """Convert plot JSON to an HTML div string for embedding in reports."""
        error_html = '<div class="plot-container"><p>Visualization could not be rendered.</p></div>'
        try:
            plot_json = self._unwrap_plot_json(plot_json)
            plot_json = self._sanitize_json_string(plot_json)
            plot_dict = self._parse_plot_json(plot_json)
            if plot_dict is None:
                return error_html

            fig = pio.from_json(json.dumps(plot_dict))
            
            # Clean all NaN values and format labels
            self._clean_trace_nan_values(fig)
            self._clean_axis_tick_text(fig)
            self._capitalize_trace_labels(fig)
            self._apply_trace_labels(fig)

            # Compute layout dimensions and apply
            chart_width, chart_height, tick_angle, font_size = self._compute_chart_dimensions(fig)
            unique_x = self._get_unique_x_count(fig)
            self._apply_figure_layout(fig, chart_width, chart_height, tick_angle, font_size, unique_x)

            # Strip all data labels that LLM-generated code or cardinality handler may have set.
            # Each attribute is set individually inside try/except because different trace types
            # (Table, Pie, Heatmap, etc.) support different subsets of text properties.
            _label_attrs = ('text', 'texttemplate', 'textposition', 'textfont',
                            'insidetextanchor', 'cliponaxis')
            for trace in fig.data:
                for _attr in _label_attrs:
                    try:
                        setattr(trace, _attr, None)
                    except Exception:
                        pass  # attribute not valid for this trace type — skip silently

            # Generate HTML output — full_html=False emits only the <div>+<script>
            # block, avoiding nested <html><head><body> tags that corrupt the
            # parent document layout and cause Plotly to miscalculate its container.
            html_out = pio.to_html(
                fig,
                full_html=False,
                include_plotlyjs=False,
                config={
                    'responsive': True,
                    'displayModeBar': True,
                    'modeBarButtonsToRemove': ['sendDataToCloud', 'editInChartStudio'],
                    'displaylogo': False,
                    'scrollZoom': True,
                }
            )
            
            del fig
            gc.collect()
            return html_out
        except Exception as e:
            logger.error(f"Plot to HTML conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return error_html
    
    def _parse_correlation_insights_response(self, response: str) -> list:
        """Parse LLM correlation insights JSON, handling control characters."""
        JSON = '```json'
        cleaned = response.strip()
        if cleaned.startswith(JSON):
            cleaned = cleaned.split(JSON)[1].split('```')[0].strip()
        elif cleaned.startswith('```'):
            cleaned = cleaned.split('```')[1].split('```')[0].strip()
        
        # Fix: remove unescaped control characters (newlines inside strings, tabs, etc.)
        import re
        cleaned = re.sub(r'(?<!\\)\\n', '\\\\n', cleaned)  # escape lone \n
        # Remove actual control characters (0x00-0x1f) except valid JSON whitespace
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', cleaned)
        
        return json.loads(cleaned)

    def _generate_correlation_based_insights(self, df: pd.DataFrame,
                                correlation_data: Dict[str, Any],
                                llm_params: Dict[str, Any],
                                token_tracker, auth_token: str) -> List[Dict[str, Any]]:

        all_items = []

        for corr in correlation_data.get('correlations', []):
            all_items.append({
                'type': 'correlation',
                'col1': corr['column_1'],
                'col2': corr['column_2'],
                'correlation_value': corr['correlation'],
                'strength': corr['strength'],
                'direction': corr['direction'],
                'col1_stats': {
                    'mean': float(df[corr['column_1']].mean()),
                    'std': float(df[corr['column_1']].std()),
                    'min': float(df[corr['column_1']].min()),
                    'max': float(df[corr['column_1']].max())
                },
                'col2_stats': {
                    'mean': float(df[corr['column_2']].mean()),
                    'std': float(df[corr['column_2']].std()),
                    'min': float(df[corr['column_2']].min()),
                    'max': float(df[corr['column_2']].max())
                }
            })

        for pattern in correlation_data.get('categorical_patterns', []):
            all_items.append({
                'type': 'segmentation',
                'cat_col': pattern['categorical_column'],
                'num_col': pattern['numeric_column'],
                'top_category': pattern['top_category'],
                'bottom_category': pattern['bottom_category'],
                'categories': pattern['categories']
            })

        if not all_items:
            return []

        prompt = f"""You are a business analyst translating data findings into actionable insights.

    Below are {len(all_items)} findings from the dataset. For EACH finding, produce a business-friendly HTML insight block.

    Findings:
    {json.dumps(all_items, indent=2)}

    For EACH finding, return an insight using this EXACT HTML structure:

    <div class="insight-content">
        <h4 class="insight-subheading">Key Finding</h4>
        <p>[One clear business-language sentence — no statistical jargon]</p>
        
        <h4 class="insight-subheading">Evidence</h4>
        <ul class="insight-list">
            <li>[Specific data point]</li>
            <li>[Impact or magnitude]</li>
            <li>[Real-world implication]</li>
        </ul>
        
        <h4 class="insight-subheading">Business Impact</h4>
        <p>[2-3 sentences on why this matters — revenue, efficiency, costs, customer satisfaction]</p>
        
        <h4 class="insight-subheading">Actionable Recommendations</h4>
        <ul class="insight-list">
            <li>[Action 1]</li>
            <li>[Action 2]</li>
            <li>[Action 3]</li>
        </ul>
    </div>

    CRITICAL RULES:
    - NO statistical terms (correlation, standard deviation, p-value etc.)
    - Use business language only
    - Be specific with numbers
    - Return ONLY a JSON array, nothing else:

    [
    {{
        "index": 0,
        "html": "<div class=\\"insight-content\\">...</div>"
    }},
    ...
    ]

    {REASONING_SECTION_PROMPT}"""

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                llm_params=llm_params,
                token_tracker=token_tracker,
                auth_token=auth_token,
                temperature=0.4
            )

            parsed = self._parse_correlation_insights_response(response)

        except Exception as e:
            logger.error(f"Batched correlation insights failed: {e}")
            return []

        insights = []
        for i, item in enumerate(all_items):
            html_content = next((r['html'] for r in parsed if r['index'] == i), None)
            if not html_content:
                continue

            if item['type'] == 'correlation':
                title = self._format_column_name_for_title(item['col1']) + " and " + self._format_column_name_for_title(item['col2']) + " Relationship"
                statistical_basis = f"{item['strength'].title()} {item['direction']} correlation"
                columns_involved = [item['col1'], item['col2']]
                insight_type = 'correlation'
            else:
                title = self._format_column_name_for_title(item['cat_col']) + " Impact on " + self._format_column_name_for_title(item['num_col'])
                statistical_basis = 'Category performance analysis'
                columns_involved = [item['cat_col'], item['num_col']]
                insight_type = 'segmentation'

            insights.append({
                'title': title,
                'insight_type': insight_type,
                'analysis': html_content.strip(),
                'columns_involved': columns_involved,
                'statistical_basis': statistical_basis
            })

        return insights[:5]

    def _format_column_name_for_title(self, column_name: str) -> str:
        """Convert column names to user-friendly titles"""
        # Replace underscores and hyphens with spaces
        formatted = column_name.replace('_', ' ').replace('-', ' ')
        
        # Capitalize first letter of each word
        formatted = ' '.join(word.capitalize() for word in formatted.split())
        
        return formatted

    def _add_plot_section_to_html(self, df: pd.DataFrame, 
                               analysis: Dict[str, Any], plot_json: str,
                               analysis_type: str, llm_params: Dict[str, Any],
                               token_tracker, auth_token: str) -> str:
        """Generate HTML section for a plot with insights - optimized for PDF"""
        html_parts = []
        
        # Wrap entire section to prevent page breaks
        html_parts.append('<div class="plot-section">')
        
        # Add description
        html_parts.append(f'<h3 class="subsection-heading">{analysis["description"]}</h3>')
        
        # Convert plot to HTML
        plot_html = self._convert_plot_to_html(plot_json)
        html_parts.append(f'<div class="plot-container"><div class="plot-wrapper">{plot_html}</div></div>')
        
        # Generate insights
        insights = self._generate_insights_for_visualization(
            df, analysis, analysis_type, llm_params, token_tracker, auth_token
        )
        
        # Add insights section
        html_parts.append('<div class="insights-section">')
        html_parts.append('<div class="insight-heading">Key Insights:</div>')
        
        cleaned_insights = self._clean_and_format_insights(insights)
        for insight_idx, insight in enumerate(cleaned_insights, 1):
            insight_clean = re.sub(NUMBERED_LIST_PATTERN, '', insight).strip()
            html_parts.append(f'<div class="bullet-point"><strong>{insight_idx}.</strong> {insight_clean}</div>')
        
        html_parts.append(DIV)  # Close insights-section
        html_parts.append(DIV)  # Close plot-section
        
        return '\n'.join(html_parts)
    
    def _add_plot_section_to_html_with_insights(self, analysis: Dict[str, Any],
                                             plot_json: str,
                                             insights_list: List[str]) -> str:
        """
        Same as _add_plot_section_to_html but accepts pre-generated insights
        instead of making a new LLM call. HTML output is identical.
        """
        html_parts = []

        html_parts.append('<div class="plot-section">')
        html_parts.append(f'<h3 class="subsection-heading">{analysis["description"]}</h3>')

        plot_html = self._convert_plot_to_html(plot_json)
        html_parts.append(f'<div class="plot-container"><div class="plot-wrapper">{plot_html}</div></div>')

        html_parts.append('<div class="insights-section">')
        html_parts.append('<div class="insight-heading">Key Insights:</div>')

        # Use pre-generated insights directly — same cleaning logic as before
        cleaned_insights = self._clean_and_format_insights('\n'.join(insights_list))
        for insight_idx, insight in enumerate(cleaned_insights, 1):
            insight_clean = re.sub(NUMBERED_LIST_PATTERN, '', insight).strip()
            html_parts.append(f'<div class="bullet-point"><strong>{insight_idx}.</strong> {insight_clean}</div>')

        html_parts.append(DIV)
        html_parts.append(DIV)

        return '\n'.join(html_parts)

    
    def _is_significant_correlation(self, corr_value: float) -> bool:
        """Return True if correlation value is meaningful (>0.65, not NaN)."""
        return abs(corr_value) > 0.65 and not np.isnan(corr_value)

    def _build_correlation_entry(self, col1: str, col2: str, corr_value: float) -> Dict[str, Any]:
        """Build a single correlation result dict."""
        return {
            'column_1': col1,
            'column_2': col2,
            'correlation': float(corr_value),
            'strength': 'strong' if abs(corr_value) > 0.7 else 'moderate',
            'direction': 'positive' if corr_value > 0 else 'negative'
        }

    def _find_significant_correlations(self, df: pd.DataFrame,
                                        numeric_cols: List[str]) -> List[Dict[str, Any]]:
        """Find and return top-5 significant numeric column correlations."""
        corr_matrix = df[numeric_cols].corr()
        correlations = [
            self._build_correlation_entry(numeric_cols[i], numeric_cols[j],
                                        corr_matrix.loc[numeric_cols[i], numeric_cols[j]])
            for i in range(len(numeric_cols))
            for j in range(i + 1, len(numeric_cols))
            if self._is_significant_correlation(corr_matrix.loc[numeric_cols[i], numeric_cols[j]])
        ]
        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        return correlations[:5]

    def _build_categorical_pattern(self, cat_col: str, num_col: str,
                                    grouped: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Build a categorical pattern entry if variance ratio exceeds threshold.
        Returns None if variation is not significant.
        """
        if len(grouped) < 2:
            return None
        max_mean, min_mean = grouped['mean'].max(), grouped['mean'].min()
        variance_ratio = (max_mean - min_mean) / (grouped['mean'].mean() + 1e-10)
        if variance_ratio <= 0.3:
            return None
        return {
            'categorical_column': cat_col,
            'numeric_column': num_col,
            'categories': grouped.to_dict('index'),
            'variance_ratio': float(variance_ratio),
            'top_category': grouped['mean'].idxmax(),
            'bottom_category': grouped['mean'].idxmin()
        }

    def _find_categorical_patterns(self, df: pd.DataFrame,
                                    numeric_cols: List[str]) -> List[Dict[str, Any]]:
        """Find categorical-numeric patterns with significant variation."""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        patterns = []
        for cat_col in categorical_cols[:3]:
            if df[cat_col].nunique() > 10:
                continue
            for num_col in numeric_cols[:3]:
                grouped = df.groupby(cat_col)[num_col].agg(['mean', 'std', 'count'])
                pattern = self._build_categorical_pattern(cat_col, num_col, grouped)
                if pattern:
                    patterns.append(pattern)
        return patterns[:3]

    def _compute_correlation_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute statistical correlations and identify significant relationships.
        Returns data-driven insights for business interpretation.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return {'correlations': [], 'categorical_patterns': []}
        return {
            'correlations': self._find_significant_correlations(df, numeric_cols),
            'categorical_patterns': self._find_categorical_patterns(df, numeric_cols)
        }
    
    def _build_html_header(self, file_name: str) -> str:
        import os
        import plotly

        plotly_js_path = os.path.join(
            os.path.dirname(plotly.__file__), 'package_data', 'plotly.min.js'
        )

        if os.path.exists(plotly_js_path):
            with open(plotly_js_path, 'r', encoding='utf-8') as js_file:
                plotly_js_content = js_file.read()
            plotly_script = f'<script type="text/javascript">{plotly_js_content}</script>'
        else:
            logger.warning("Bundled plotly.min.js not found — falling back to CDN")
            plotly_script = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'

        return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Data Insights Report - {file_name}</title>
        {plotly_script}
        {self.html_styles}
        <style>
            html, body {{
                height: auto !important;
                min-height: 100% !important;
                overflow-y: auto !important;
                overflow-x: hidden !important;
                scroll-behavior: smooth;
            }}
        </style>
    </head>
    <body>
    <script>
        (function() {{
            function notifyHeight() {{
                const height = document.documentElement.scrollHeight;
                window.parent.postMessage({{ type: 'resize', height: height }}, '*');
            }}

            window.addEventListener('load', notifyHeight);
            window.addEventListener('resize', notifyHeight);

            document.documentElement.style.overflow = 'auto';
            document.documentElement.style.height = 'auto';
            document.body.style.overflow = 'auto';
            document.body.style.height = 'auto';
            document.body.style.minHeight = '100vh';

            const observer = new MutationObserver(notifyHeight);
            observer.observe(document.body, {{
                subtree: true,
                childList: true,
                attributes: true
            }});
        }})();
    </script>
    <div class="a4-page">
    """
    
    def _build_executive_summary_html(
        self,
        sheets_data: List[Dict[str, Any]],
        file_name: str   # NEW — None = legacy path
    ) -> List[str]:
        """Build executive summary. Now includes orchestration context when provided."""

        total_rows = sum(len(s['df']) for s in sheets_data if s.get('df') is not None)
        total_columns = sum(len(s['df'].columns) for s in sheets_data if s.get('df') is not None)

        sheet_rows = []
        for sheet_info in sheets_data:
            sheet_name = sheet_info.get('sheet_name') or 'Main Sheet'
            df = sheet_info.get('df')
            if df is None:
                rows, cols = "—", "—"
            else:
                rows, cols = f"{len(df):,}", str(len(df.columns))
            sheet_rows.append(
                f'<tr><td>{sheet_name}</td><td>{rows}</td><td>{cols}</td></tr>'
            )

        return [
            '<div class="content-wrapper">',
            '<h1 class="section-heading">Executive Summary</h1>',
            f"""
            <table class="summary-table">
                <tr><td>Workbook Overview</td><td></td></tr>
                <tr><td>Total Sheets</td><td>{len(sheets_data)}</td></tr>
                <tr><td>Total Records</td><td>{total_rows:,}</td></tr>
                <tr><td>Total Data Fields</td><td>{total_columns}</td></tr>
            </table>
            """,
            '<h3 class="subsection-heading">Sheet Breakdown:</h3>',
            f"""
            <table class="sheet-table">
                <thead><tr><th>Sheet Name</th><th>Records</th><th>Fields</th></tr></thead>
                <tbody>{''.join(sheet_rows)}</tbody>
            </table>
            """,
            f"""
            <p class="executive-summary">
                This report analyses data from <strong>{file_name}</strong>.
                The report identifies critical patterns, relationships, and strategic opportunities.
            </p>
            """,
        ]

    def _render_individual_analyses(self, df: pd.DataFrame, analysis_plan: Dict[str, Any],
                                    llm_params: Dict[str, Any], token_tracker,
                                    auth_token: str, analysis_complete: str) -> List[str]:
        """Generate and render individual metric plots."""
        import gc
        html_parts = []
        if not analysis_plan.get('individual_analyses'):
            return html_parts

        html_parts.append('<h2 class="subsection-heading">Key Performance Indicators</h2>')
        individual_plots = []
        for analysis in analysis_plan['individual_analyses']:
            logger.info(f"  Creating individual metric plot: {analysis['column']}")
            plot_json = self._create_visualization_from_plan(
                df, analysis, 'individual', llm_params, token_tracker, auth_token
            )
            if plot_json:
                individual_plots.append({'analysis': analysis, 'plot_json': plot_json, 'analysis_type': 'individual'})

        insights_map = self._generate_insights_batch(df, individual_plots, llm_params, token_tracker, auth_token) if individual_plots else {}

        for idx, item in enumerate(individual_plots):
            insights_list = insights_map.get(idx, [analysis_complete])
            html_parts.append(self._add_plot_section_to_html_with_insights(item['analysis'], item['plot_json'], insights_list))
            del item['plot_json']
            gc.collect()

        return html_parts

    def _collect_plots(self, df: pd.DataFrame, analyses: List[Dict[str, Any]],
                   analysis_type: str, log_template: str,
                   llm_params: Dict[str, Any], token_tracker, auth_token: str) -> List[Dict[str, Any]]:
        """Generate plots for a list of analyses and return collected plot items."""
        plots = []
        for analysis in analyses:
            try:
                logger.info(log_template.format(**analysis))
            except KeyError:
                logger.info(log_template.format_map({k: analysis.get(k, 'N/A') for k in ['column_x', 'column_y', 'column_z']}))
            plot_json = self._create_visualization_from_plan(
                df, analysis, analysis_type, llm_params, token_tracker, auth_token
            )
            if plot_json:
                plots.append({'analysis': analysis, 'plot_json': plot_json, 'analysis_type': analysis_type})
        return plots

    def _render_plots(self, plots: List[Dict[str, Any]], insights_map: Dict[int, List[str]],
                    index_offset: int, analysis_complete: str) -> List[str]:
        """Render a list of plots with their insights into HTML parts."""
        import gc
        html_parts = []
        for idx, item in enumerate(plots):
            insights_list = insights_map.get(idx + index_offset, [analysis_complete])
            html_parts.append(self._add_plot_section_to_html_with_insights(
                item['analysis'], item['plot_json'], insights_list
            ))
            del item['plot_json']
            gc.collect()
        return html_parts

    def _render_relationship_and_advanced_analyses(self, df: pd.DataFrame, analysis_plan: Dict[str, Any],
                                                    llm_params: Dict[str, Any], token_tracker,
                                                    auth_token: str, analysis_complete: str) -> List[str]:
        """Generate and render relationship + advanced pattern plots."""
        html_parts = []
        relationship_plots = []

        if analysis_plan.get('relationship_analyses'):
            html_parts.append('<h2 class="subsection-heading">Correlation & Trend Analysis</h2>')
            relationship_plots = self._collect_plots(
                df, analysis_plan['relationship_analyses'], 'relationship',
                "  Creating relationship plot: {column_x} vs {column_y}",
                llm_params, token_tracker, auth_token
            )

        advanced_plots = []
        if analysis_plan.get('advanced_pattern_analyses'):
            advanced_plots = self._collect_plots(
                df, analysis_plan['advanced_pattern_analyses'], 'advanced_pattern',
                "  Creating advanced pattern plot: {column_x}, {column_y}, {column_z}",
                llm_params, token_tracker, auth_token
            )

        combined_plots = relationship_plots + advanced_plots
        insights_map = self._generate_insights_batch(df, combined_plots, llm_params, token_tracker, auth_token) if combined_plots else {}

        html_parts.extend(self._render_plots(relationship_plots, insights_map, 0, analysis_complete))

        if advanced_plots:
            html_parts.append('<h2 class="subsection-heading">Multi-Dimensional Intelligence</h2>')
            html_parts.extend(self._render_plots(advanced_plots, insights_map, len(relationship_plots), analysis_complete))

        return html_parts
    
    def _render_correlation_insights(self, df: pd.DataFrame, llm_params: Dict[str, Any],
                                    token_tracker, auth_token: str) -> List[str]:
        """Compute and render correlation-based insight cards."""
        html_parts = []
        logger.info("  Computing correlation-based insights...")
        correlation_data = self._compute_correlation_insights(df)

        if not (correlation_data['correlations'] or correlation_data['categorical_patterns']):
            return html_parts

        hidden_insights = self._generate_correlation_based_insights(df, correlation_data, llm_params, token_tracker, auth_token)
        if not hidden_insights:
            return html_parts

        html_parts.append('<div class="page-break"></div>')
        html_parts.append('<h1 class="section-heading">Data-Driven Insights & Strategic Opportunities</h1>')

        for _, insight in enumerate(hidden_insights, 1):
            insight_type = insight.get('insight_type', 'general').upper()
            badge_class = 'badge-correlation' if insight_type == 'CORRELATION' else 'badge-segmentation'
            analysis_html = insight['analysis'].strip()
            inner_content = analysis_html if analysis_html.startswith('<div class="insight-content">') \
                else f'<div class="insight-content">{analysis_html}</div>'

            html_parts.append(f"""
            <div class="insight-card">
                <span class="insight-type-badge {badge_class}">{insight_type}</span>
                <h3>{insight["title"]}</h3>
                <div class="statistical-basis">{insight["statistical_basis"]}</div>
                {inner_content}
                <div class="metrics-analyzed">Metrics Analyzed: {', '.join(insight['columns_involved'])}</div>
            </div>
            """)

        return html_parts

    def _process_single_sheet(self, sheet_idx: int, total_sheets: int, sheet_info: Dict[str, Any],
                                llm_params: Dict[str, Any], token_tracker, auth_token: str,
                                analysis_complete: str) -> List[str]:
        """Process one sheet and return its HTML parts."""
        import gc
        df = sheet_info['df']
        sheet_name = sheet_info['sheet_name']

        logger.info(f"\n[Sheet {sheet_idx}/{total_sheets}] Analyzing: {sheet_name or 'Main Sheet'}")
        logger.info(f"  Rows: {len(df)}, Columns: {len(df.columns)}")

        schema_info = self._get_dataframe_schema_info(df)
        logger.info("  Generating analysis plan...")
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"DataFrame columns: {df.columns.tolist()}")
        logger.info(f"DataFrame dtypes:\n{df.dtypes}")
        logger.info(f"DataFrame sample:\n{df.head(3)}")
        analysis_plan = self._generate_analysis_plan(schema_info, llm_params, token_tracker, auth_token)

        html_parts = [
            f'<h1 class="section-heading">Sheet Analysis {sheet_idx}: {sheet_name or "Main Sheet"}</h1>',
            f'<p class="metadata-text">Records: {schema_info["total_rows"]:,} | Fields: {schema_info["total_columns"]}</p>',
        ]

        html_parts.extend(self._render_individual_analyses(df, analysis_plan, llm_params, token_tracker, auth_token, analysis_complete))
        html_parts.extend(self._render_relationship_and_advanced_analyses(df, analysis_plan, llm_params, token_tracker, auth_token, analysis_complete))
        html_parts.extend(self._render_correlation_insights(df, llm_params, token_tracker, auth_token))

        del df
        sheet_info['df'] = None  # release the reference held by sheets_data list
        gc.collect()
        logger.info("  ✓ Sheet analysis complete - memory released")

        return html_parts

    async def generate_multi_sheet_report(
        self,
        sheets_data: List[Dict[str, Any]],
        file_name: str,
        llm_params: Dict[str, Any],
        token_tracker,
        auth_token: str,
    ) -> Tuple[bytes, str]:
        """Generate comprehensive executive-level insights report.

        Now routes through SheetOrchestrator before any LLM call:
        - Pre-filters trivial/tiny sheets
        - Detects homogeneous vs heterogeneous schema
        - Merges sheets (homogeneous) or selects top-1 (heterogeneous)
        """
        from src.analytics.sheet_orchestrator import SheetOrchestrator
        import tempfile, os

        logger.info("Starting multi-sheet report generation for %s", file_name)
        logger.info("Total sheets received: %d", len(sheets_data))

        # ── Orchestration (zero LLM calls) ──────────────────────────────────────
        orchestrator = SheetOrchestrator()
        try:
            orch_result = orchestrator.prepare(sheets_data)
        except ValueError as exc:
            logger.error("[Orchestrator] Preparation failed: %s", exc)
            raise

        logger.info(
            "[Orchestrator] Case: %s | Sheets to process: %d | Excluded: %d",
            orch_result.case,
            len(orch_result.sheets_to_process),
            len(orch_result.excluded_sheets),
        )

        sheets_to_process = orch_result.sheets_to_process
        ANALYSIS_COMPLETE = "Analysis complete. Please review the visualization for patterns."
        PAGE_BREAK_DIV = '<div class="page-break"></div>'

        safe_name = re.sub(r'[^\w\-.]', '_', file_name)
        html_filename = f"Data_Insights_Report_{safe_name}.html"

        _html_fd, tmp_html_path = tempfile.mkstemp(suffix='.html', prefix=f'report_{safe_name}_')
        os.close(_html_fd)

        async with aiofiles.open(tmp_html_path, 'w', encoding='utf-8') as f:

            await f.write(self._build_html_header(file_name))
            await f.write(self._create_cover_page())

            # ── Executive summary now includes orchestration context ─────────────
            summary_parts = self._build_executive_summary_html(
                sheets_data=sheets_data,               # ALL original sheets (for transparency)
                file_name=file_name              # NEW param
            )
            await f.write('\n'.join(summary_parts))
            summary_parts.clear()
            del summary_parts
            gc.collect()

            await f.write(PAGE_BREAK_DIV)

            # ── Process only orchestrator-selected sheets ────────────────────────
            for sheet_idx, sheet_info in enumerate(sheets_to_process, 1):
                sheet_html_parts = self._process_single_sheet(
                    sheet_idx=sheet_idx,
                    total_sheets=len(sheets_to_process),
                    sheet_info=sheet_info,
                    llm_params=llm_params,
                    token_tracker=token_tracker,
                    auth_token=auth_token,
                    analysis_complete=ANALYSIS_COMPLETE,
                )
                await f.write('\n'.join(sheet_html_parts))
                sheet_html_parts.clear()
                del sheet_html_parts
                sheet_info['df'] = None
                gc.collect()

                if sheet_idx < len(sheets_to_process):
                    await f.write(PAGE_BREAK_DIV)

            await f.write('</div></div></body></html>')

        async with aiofiles.open(tmp_html_path, 'r', encoding='utf-8') as f:
            html_bytes = (await f.read()).encode('utf-8')

        if os.path.exists(tmp_html_path):
            os.remove(tmp_html_path)

        logger.info("Multi-sheet HTML report generation complete: %s", html_filename)
        return html_bytes, html_filename

    async def _convert_html_to_pdf(self, html_path: str, output_path: str):
        from playwright.async_api import async_playwright
        import asyncio
        import gc

        async with async_playwright() as p:
            browser = None
            page = None
            context = None
            try:
                browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        '--no-sandbox',
                        '--disable-setuid-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-gpu',
                        '--no-zygote',
                        '--single-process',
                        '--disable-extensions',
                        '--disable-background-networking',
                        '--disable-default-apps',
                        '--disable-sync',
                        '--disable-translate',
                        '--hide-scrollbars',
                        '--metrics-recording-only',
                        '--mute-audio',
                        '--no-first-run',
                        '--safebrowsing-disable-auto-update',
                        '--js-flags=--max-old-space-size=512',
                        '--memory-pressure-off',
                        '--disable-features=TranslateUI',
                        '--blink-settings=imagesEnabled=true',
                        '--allow-file-access-from-files',  # required for file:// JS reference
                        '--disable-web-security',           # required for file:// cross-origin JS
                    ]
                )

                context = await browser.new_context(
                    viewport={"width": 1200, "height": 1600},
                )
                page = await context.new_page()

                file_uri = Path(html_path).resolve().as_uri()

                # domcontentloaded instead of networkidle — 
                # no CDN calls, JS is local, no reason to wait for network
                logger.info("Navigating to HTML file for PDF conversion")

                await page.goto(
                    file_uri,
                    wait_until='domcontentloaded',
                    timeout=60000
                )

                # Wait for plotly to render all staticPlot charts
                await page.wait_for_timeout(5000)

                try:
                    await asyncio.wait_for(
                        page.evaluate("""
                            () => {
                                const plots = document.querySelectorAll('.js-plotly-plot');
                                plots.forEach(plot => {
                                    if (window.Plotly && plot) {
                                        window.Plotly.Plots.resize(plot);
                                    }
                                });
                            }
                        """),
                        timeout=10.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("Plotly resize timed out — continuing")
                except Exception as e:
                    logger.warning("Plotly resize failed: %s — continuing", e)

                await page.wait_for_timeout(1500)

                size = '0.3in'
                try:
                    await asyncio.wait_for(
                        page.pdf(
                            path=output_path,
                            format='Letter',
                            print_background=True,
                            margin={
                                'top': size,
                                'right': size,
                                'bottom': size,
                                'left': size
                            },
                            prefer_css_page_size=False,
                        ),
                        timeout=120.0
                    )
                except asyncio.TimeoutError:
                    logger.error("PDF generation timed out after 120 seconds")
                    raise RuntimeError("PDF generation timed out")

                logger.info("PDF generated successfully: %s", output_path)

            except Exception as e:
                logger.error("PDF conversion failed: %s", e)
                raise

            finally:
                if page:
                    try:
                        await page.close()
                    except Exception:
                        pass
                if context:
                    try:
                        await context.close()
                    except Exception:
                        pass
                if browser:
                    try:
                        await browser.close()
                    except Exception:
                        pass
                gc.collect()