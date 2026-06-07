"""
cardinality_handler.py

Shared module for handling high-cardinality columns in both:
- ReportGenerator (EDA report)
- PlotGenerator (direct plot requests)

Cardinality tiers:
    LOW    : 2-8   unique values  -> pie / donut
    MEDIUM : 8-15  unique values  -> horizontal bar
    HIGH   : 15-30 unique values  -> top-N bar with "Others" + treemap
    EXTREME: 30+   unique values  -> treemap or ranked table
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cardinality thresholds
# ---------------------------------------------------------------------------
CARDINALITY_LOW = 8
CARDINALITY_MEDIUM = 15
CARDINALITY_HIGH = 30
TOP_N_DEFAULT = 10
MIN_OTHERS_SHARE = 0.01     # only create "Others" bucket if it covers >= 1% of data


# ---------------------------------------------------------------------------
# Tier detection
# ---------------------------------------------------------------------------

def get_cardinality_tier(series: pd.Series) -> str:
    """Return cardinality tier string for a categorical series."""
    n = series.nunique()
    if n <= CARDINALITY_LOW:
        return "low"
    if n <= CARDINALITY_MEDIUM:
        return "medium"
    if n <= CARDINALITY_HIGH:
        return "high"
    return "extreme"


def is_high_cardinality(series: pd.Series) -> bool:
    """Return True if the series cardinality is above the LOW threshold."""
    return series.nunique() > CARDINALITY_LOW


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------
def _compute_top_n_counts(
    series: pd.Series,
    top_n: int = TOP_N_DEFAULT,
    value_col: str = "category",
    count_col: str = "count",
) -> pd.DataFrame:
    counts = series.value_counts()
    top = counts.head(top_n)
    rest = counts.iloc[top_n:]

    rows = top.reset_index()
    rows.columns = [value_col, count_col]

    if not rest.empty and rest.sum() / counts.sum() >= MIN_OTHERS_SHARE:
        others_row = pd.DataFrame(
            [[f"Others ({len(rest)} more)", int(rest.sum())]],
            columns=[value_col, count_col],
        )
        rows = pd.concat([rows, others_row], ignore_index=True)

    # Fix: ensure columns are native Python types, not numpy
    rows[value_col] = rows[value_col].astype(str).tolist()
    rows[count_col] = rows[count_col].astype(int).tolist()

    return rows

def _compute_top_n_aggregation(
    df: pd.DataFrame,
    cat_col: str,
    num_col: str,
    agg_func: str = "mean",
    top_n: int = TOP_N_DEFAULT,
) -> pd.DataFrame:
    agg_col = f"{agg_func}_{num_col}"
    grouped = df.groupby(cat_col)[num_col].agg(agg_func).reset_index()
    grouped.columns = [cat_col, agg_col]
    # FIX: ensure numeric dtype
    grouped[agg_col] = pd.to_numeric(grouped[agg_col], errors='coerce')
    grouped = grouped.dropna(subset=[agg_col])
    grouped = grouped.sort_values(agg_col, ascending=False)

    top = grouped.head(top_n)
    rest = grouped.iloc[top_n:]

    if not rest.empty:
        rest_val = rest[agg_col].mean() if agg_func == "mean" else rest[agg_col].sum()
        others_row = pd.DataFrame(
            [[f"Others ({len(rest)} more)", float(rest_val)]],
            columns=[cat_col, agg_col],
        )
        top = pd.concat([top, others_row], ignore_index=True)

    # FIX: cast to native Python types to avoid plotly serialization issues
    top[cat_col] = top[cat_col].astype(str).tolist()
    top[agg_col] = pd.to_numeric(top[agg_col], errors='coerce').tolist()

    return top

# ---------------------------------------------------------------------------
# Individual chart builders
# ---------------------------------------------------------------------------

def build_horizontal_bar(
    series: pd.Series,
    title: str,
    top_n: int = TOP_N_DEFAULT,
) -> go.Figure:
    """Horizontal bar chart — best for medium cardinality (8-15 unique values)."""
    df_plot = _compute_top_n_counts(series, top_n=top_n)
    fig = px.bar(
        df_plot,
        x="count",
        y="category",
        orientation="h",
        title=title,
        text="count",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        yaxis={"categoryorder": "total ascending", "automargin": True},
        xaxis={"automargin": True},
        margin={"l": 200, "r": 80, "t": 60, "b": 60},
    )
    return fig


def build_treemap(
    series: pd.Series,
    title: str,
    top_n: int = TOP_N_DEFAULT * 3,
) -> go.Figure:
    """Treemap — best for high/extreme cardinality (15+ unique values)."""
    df_plot = _compute_top_n_counts(series, top_n=top_n, value_col="label", count_col="value")
    fig = px.treemap(
        df_plot,
        path=["label"],
        values="value",
        title=title,
    )
    fig.update_traces(textinfo="label+value+percent root")
    fig.update_layout(margin={"l": 20, "r": 20, "t": 60, "b": 20})
    return fig


def build_ranked_table(
    series: pd.Series,
    title: str,
    top_n: int = 20,
) -> go.Figure:
    """Plotly table — fallback for extreme cardinality (30+ unique values)."""
    SHARE = "% Share"
    counts = series.value_counts().head(top_n).reset_index()
    counts.columns = ["Category", "Count"]
    total = counts["Count"].sum()
    counts[SHARE] = (counts["Count"] / total * 100).round(1).astype(str) + "%"

    row_colors = [["#f5f5f5", "#ffffff"] * (len(counts) // 2 + 1)]

    fig = go.Figure(
        data=[
            go.Table(
                header={
                    "values": ["#", "Category", "Count", SHARE],
                    "fill_color": "#0d47a1",
                    "font": {'color': 'white', 'size': 12},
                    "align": "left",
                },
                cells={
                    "values": [
                        list(range(1, len(counts) + 1)),
                        counts["Category"].tolist(),
                        counts["Count"].tolist(),
                        counts[SHARE].tolist(),
                    ],
                    "fill_color": row_colors,
                    "align": "left",
                    "font": {'size': 11},
                },
            )
        ]
    )
    fig.update_layout(title=title, margin={"l": 20, "r": 20, "t": 60, "b": 20})
    return fig


def build_aggregated_horizontal_bar(
    df: pd.DataFrame,
    cat_col: str,
    num_col: str,
    title: str,
    agg_func: str = "mean",
    top_n: int = TOP_N_DEFAULT,
) -> go.Figure:
    """
    Horizontal bar showing numeric aggregation over a high-cardinality category.
    E.g. average revenue by product (50 products -> top 10).
    """
    df_plot = _compute_top_n_aggregation(df, cat_col, num_col, agg_func, top_n)
    agg_col = f"{agg_func}_{num_col}"

    fig = px.bar(
        df_plot,
        x=agg_col,
        y=cat_col,
        orientation="h",
        title=title,
        text=agg_col,
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(
        yaxis={"categoryorder": "total ascending", "automargin": True},
        xaxis={"automargin": True},
        margin={"l": 200, "r": 80, "t": 60, "b": 60},
    )
    return fig

def build_heatmap(
    df: pd.DataFrame,
    col_x: str,
    col_y: str,
    title: str,
    top_n: int = TOP_N_DEFAULT,
) -> go.Figure:
    df_copy = df.copy()
    df_copy[col_x] = df_copy[col_x].astype(str)
    df_copy[col_y] = df_copy[col_y].astype(str)

    # FIX: compute top_x/top_y AFTER converting to str, so isin() matches correctly
    top_x = df_copy[col_x].value_counts().head(top_n).index.tolist()
    top_y = df_copy[col_y].value_counts().head(top_n).index.tolist()

    filtered = df_copy[df_copy[col_x].isin(top_x) & df_copy[col_y].isin(top_y)]

    if filtered.empty:
        # FIX: fallback — use all data if filtering removes everything
        filtered = df_copy

    pivot = filtered.groupby([col_y, col_x]).size().unstack(fill_value=0)
    pivot.index = pivot.index.map(str)        
    pivot.columns = pivot.columns.map(str)

    fig = px.imshow(
        pivot,
        title=title,
        color_continuous_scale="Blues",
        aspect="auto",
        text_auto=True,
        zmin=0, 
        zmax=pivot.values.max()
    )
    fig.update_layout(
        xaxis={"tickangle": -45, "automargin": True},
        yaxis={"automargin": True},
        margin={"l": 120, "r": 40, "t": 60, "b": 120},
        coloraxis_showscale=False,
    )
    return fig

def build_top_combinations_table(
    df: pd.DataFrame,
    col_x: str,
    col_y: str,
    title: str,
    top_n: int = 20,
) -> go.Figure:
    combo = (
        df.groupby([col_x, col_y])
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
        .head(top_n)
    )
    SHARE ="% Share"

    combo[SHARE] = (combo["Count"] / combo["Count"].sum() * 100).round(1).astype(str) + "%"

    fig = go.Figure(data=[go.Table(
        header={
            "values": ["#", col_x, col_y, "Count", SHARE],
            "fill_color": "#0d47a1",
            "font": {"color": "white", "size": 12},
            "align": "left",
        },
        cells={
            "values": [
                list(range(1, len(combo) + 1)),
                combo[col_x].tolist(),
                combo[col_y].tolist(),
                combo["Count"].tolist(),
                combo[SHARE].tolist(),
            ],
            "fill_color": [["#f5f5f5", "#ffffff"] * (len(combo) // 2 + 1)],
            "align": "left",
            "font": {"size": 11},
        },
    )])
    fig.update_layout(title=title, margin={"l": 20, "r": 20, "t": 60, "b": 20})
    return fig

# ---------------------------------------------------------------------------
# Single-column dispatcher
# ---------------------------------------------------------------------------

def _dispatch_single_column(series: pd.Series, title: str, top_n: int) -> go.Figure:
    """Choose the best chart type for a single categorical series."""
    tier = get_cardinality_tier(series)
    logger.info("Column '%s' -> cardinality tier: %s", series.name, tier)

    if tier == "low":
        df_plot = _compute_top_n_counts(
            series, top_n=CARDINALITY_LOW, value_col="category", count_col="count"
        )
        return px.pie(df_plot, names="category", values="count", title=title)

    if tier == "medium":
        return build_horizontal_bar(series, title, top_n=top_n)

    if tier == "high":
        return build_treemap(series, title, top_n=top_n * 3)

    # extreme
    return build_ranked_table(series, title)


# ---------------------------------------------------------------------------
# Two-column dispatcher
# ---------------------------------------------------------------------------

def _dispatch_two_column(
    df: pd.DataFrame,
    col: str,
    second_col: str,
    title: str,
    agg_func: str,
    top_n: int,
) -> go.Figure:
    """Choose the best chart type when two columns are involved."""
    if pd.api.types.is_numeric_dtype(df[second_col]):
        logger.info("Two-column chart: aggregated horizontal bar (%s vs %s).", col, second_col)
        return build_aggregated_horizontal_bar(df, col, second_col, title, agg_func, top_n)

    col_tier = get_cardinality_tier(df[col])
    second_tier = get_cardinality_tier(df[second_col])
    
    if col_tier == "extreme" and second_tier == "extreme":
        return build_top_combinations_table(df, col, second_col, title)
    
    # existing heatmap call for non-extreme cases
    return build_heatmap(df, col, second_col, title, top_n=top_n)


# ---------------------------------------------------------------------------
# Main unified dispatcher
# ---------------------------------------------------------------------------

def build_chart_for_cardinality(
    df: pd.DataFrame,
    col: str,
    title: str,
    second_col: Optional[str] = None,
    agg_func: str = "mean",
    top_n: int = TOP_N_DEFAULT,
) -> Optional[go.Figure]:
    """
    Unified dispatcher — picks the best chart based on cardinality.

    Parameters
    ----------
    df         : source DataFrame
    col        : primary (categorical) column
    title      : chart title
    second_col : optional second column
                 - if numeric  -> aggregated horizontal bar
                 - if category -> heatmap
    agg_func   : aggregation function when second_col is numeric
    top_n      : max categories before grouping remainder as 'Others'

    Returns
    -------
    A Plotly Figure, or None if the column cannot be charted.
    """
    if col not in df.columns:
        logger.warning("Column '%s' not found in DataFrame.", col)
        return None

    series = df[col].dropna()
    if series.empty:
        logger.warning("Column '%s' is entirely null.", col)
        return None

    if second_col and second_col in df.columns:
        return _dispatch_two_column(df, col, second_col, title, agg_func, top_n)

    return _dispatch_single_column(series, title, top_n)


# ---------------------------------------------------------------------------
# LLM prompt fragment injected into existing prompts
# ---------------------------------------------------------------------------

_CARDINALITY_RULES_TEMPLATE = """
CARDINALITY-AWARE CHART SELECTION RULES (these override generic chart rules):

Before selecting a chart type for any categorical column, determine its cardinality tier:
  - LOW    (<=8 unique values)  -> pie chart or donut chart
  - MEDIUM (8-15 unique values) -> HORIZONTAL bar chart (not vertical)
  - HIGH   (15-30 unique values) -> treemap OR top-{top_n} horizontal bar
    with remaining values grouped as "Others (N more)"
  - EXTREME (30+ unique values) -> treemap or ranked table ONLY; do NOT use
    pie / vertical bar / donut for these columns

For TWO high-cardinality categorical columns together -> prefer heatmap.
For ONE high-cardinality categorical + ONE numeric column -> prefer aggregated
  horizontal bar showing top-{top_n} categories by the aggregation metric.

NEVER suggest a standard vertical bar chart or pie chart for columns with >15 unique values.
Explicitly mention top_n={top_n} in the chart question so the renderer knows how many
categories to show.
""".strip()


def get_cardinality_prompt_rules(top_n: int = TOP_N_DEFAULT) -> str:
    """Return cardinality-aware prompt rules with the configured top_n value."""
    return _CARDINALITY_RULES_TEMPLATE.format(top_n=top_n)