"""
visualization.py
==================
Chart-building layer. Every function takes a summary table already produced
by ``eda.py`` (never the raw master dataset) and writes exactly one chart to
disk, returning the path it wrote to.

Static charts (matplotlib/seaborn) go to ``outputs/figures/static/`` as PNG,
for README embedding and quick browsing. Interactive charts (Plotly) go to
``outputs/figures/interactive/`` as standalone HTML, for the same data
explored with hover/zoom -- the Streamlit dashboard also reuses the same
chart-building functions directly (see ``dashboard/app.py``).
"""

from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")  # headless backend: must be set before pyplot is imported
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import seaborn as sns  # noqa: E402

from config import FIGURES_DIR, PALETTE, FIGURE_DPI, SNS_STYLE  # noqa: E402
from src import get_logger  # noqa: E402

logger = get_logger(__name__)

STATIC_DIR = FIGURES_DIR / "static"
INTERACTIVE_DIR = FIGURES_DIR / "interactive"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style(SNS_STYLE)
plt.rcParams["figure.dpi"] = FIGURE_DPI


def _save_static(fig: plt.Figure, filename: str) -> str:
    """Save a matplotlib figure to the static output folder and close it.

    Closing the figure after saving prevents matplotlib from accumulating
    open figures in memory across a long batch run.
    """
    path = STATIC_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved static chart: %s", path)
    return str(path)


def _save_interactive(fig: go.Figure, filename: str) -> str:
    """Save a Plotly figure as a standalone HTML file."""
    path = INTERACTIVE_DIR / filename
    fig.write_html(str(path), include_plotlyjs="cdn")
    logger.info("Saved interactive chart: %s", path)
    return str(path)


# ---------------------------------------------------------------------------
# Static (matplotlib / seaborn) charts
# ---------------------------------------------------------------------------

def plot_revenue_by_category(category_df: pd.DataFrame) -> str:
    """Horizontal bar chart of revenue by product category."""
    fig, ax = plt.subplots(figsize=(9, 7))
    data = category_df.sort_values("revenue")
    ax.barh(data["category"], data["revenue"], color=PALETTE["primary"])
    ax.set_xlabel("Revenue (R$)")
    ax.set_title("Top Product Categories by Revenue")
    ax.xaxis.set_major_formatter(lambda x, _: f"R${x/1000:,.0f}k")
    return _save_static(fig, "revenue_by_category.png")


def plot_monthly_revenue_trend(trend_df: pd.DataFrame) -> str:
    """Line chart of monthly revenue over time."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(
        trend_df["purchase_year_month"],
        trend_df["revenue"],
        marker="o",
        color=PALETTE["primary"],
        linewidth=2,
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue (R$)")
    ax.set_title("Monthly Revenue Trend")
    ax.tick_params(axis="x", rotation=90)
    # Thin out x tick labels if there are many months, so they stay readable.
    for i, label in enumerate(ax.get_xticklabels()):
        if i % 2 != 0:
            label.set_visible(False)
    return _save_static(fig, "monthly_revenue_trend.png")


def plot_review_score_distribution(review_df: pd.DataFrame) -> str:
    """Bar chart of the 1-5 review score distribution."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [
        PALETTE["negative"],
        PALETTE["negative"],
        PALETTE["neutral"],
        PALETTE["positive"],
        PALETTE["positive"],
    ]
    ax.bar(review_df["review_score"], review_df["pct"], color=colors[: len(review_df)])
    ax.set_xlabel("Review Score")
    ax.set_ylabel("Share of Orders (%)")
    ax.set_title("Review Score Distribution")
    ax.set_xticks(review_df["review_score"])
    return _save_static(fig, "review_score_distribution.png")


def plot_delivery_time_distribution(df: pd.DataFrame) -> str:
    """Histogram of delivery time in days, capped at the 99th percentile for readability."""
    delivery_days = df["delivery_time_days"].dropna()
    cap = delivery_days.quantile(0.99)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(delivery_days[delivery_days <= cap], bins=40, color=PALETTE["primary"], ax=ax)
    median_days = delivery_days.median()
    ax.axvline(
        median_days,
        color=PALETTE["secondary"],
        linestyle="--",
        label=f"Median: {median_days:.1f} days",
    )
    ax.set_xlabel("Delivery Time (days)")
    ax.set_title("Distribution of Delivery Time (99th percentile capped)")
    ax.legend()
    return _save_static(fig, "delivery_time_distribution.png")


def plot_correlation_heatmap(corr_df: pd.DataFrame) -> str:
    """Heatmap of the numeric-feature correlation matrix."""
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, linewidths=0.5)
    ax.set_title("Correlation Matrix: Key Numeric Features")
    return _save_static(fig, "correlation_heatmap.png")


def plot_rfm_segment_counts(rfm_df: pd.DataFrame) -> str:
    """Bar chart of customer counts per RFM segment."""
    order = ["Champions", "Loyal", "At Risk", "Lost"]
    counts = rfm_df["rfm_segment"].value_counts().reindex(order).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 5))
    segment_colors = [PALETTE["positive"], PALETTE["primary"], PALETTE["secondary"], PALETTE["negative"]]
    ax.bar(counts.index, counts.values, color=segment_colors)
    ax.set_ylabel("Number of Customers")
    ax.set_title("Customer Segments (RFM Analysis)")
    return _save_static(fig, "rfm_segments.png")


# ---------------------------------------------------------------------------
# Interactive (Plotly) charts -- also reused directly by the Streamlit app
# ---------------------------------------------------------------------------

def plotly_revenue_by_category(category_df: pd.DataFrame) -> go.Figure:
    """Interactive horizontal bar chart of revenue by category."""
    data = category_df.sort_values("revenue")
    fig = px.bar(
        data,
        x="revenue",
        y="category",
        orientation="h",
        color_discrete_sequence=[PALETTE["primary"]],
        labels={"revenue": "Revenue (R$)", "category": "Product Category"},
        title="Top Product Categories by Revenue",
        hover_data={"orders": True},
    )
    fig.update_layout(template="plotly_white")
    return fig


def plotly_monthly_revenue_trend(trend_df: pd.DataFrame) -> go.Figure:
    """Interactive line chart of monthly revenue with an order-count overlay."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trend_df["purchase_year_month"],
            y=trend_df["revenue"],
            name="Revenue (R$)",
            mode="lines+markers",
            line=dict(color=PALETTE["primary"], width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=trend_df["purchase_year_month"],
            y=trend_df["orders"],
            name="Orders",
            mode="lines",
            yaxis="y2",
            line=dict(color=PALETTE["secondary"], width=1, dash="dot"),
        )
    )
    fig.update_layout(
        title="Monthly Revenue & Order Volume",
        xaxis_title="Month",
        yaxis=dict(title="Revenue (R$)"),
        yaxis2=dict(title="Orders", overlaying="y", side="right"),
        template="plotly_white",
    )
    return fig


def plotly_customer_seller_map(df: pd.DataFrame, sample_size: int = 3000) -> go.Figure:
    """Interactive scatter-geo map of customer locations, sized by spend.

    Sampling is used because plotting all ~100k points is slow and mostly
    redundant visually; a random sample preserves the spatial pattern.
    """
    geo = df.dropna(subset=["customer_lat", "customer_lng"])
    if len(geo) > sample_size:
        geo = geo.sample(sample_size, random_state=42)

    fig = px.scatter_geo(
        geo,
        lat="customer_lat",
        lon="customer_lng",
        color="customer_state",
        hover_name="customer_city",
        scope="south america",
        title=f"Customer Locations (sample of {len(geo):,})",
    )
    fig.update_layout(template="plotly_white")
    return fig


def plotly_rfm_scatter(rfm_df: pd.DataFrame) -> go.Figure:
    """Interactive scatter of Recency vs Monetary, colored by RFM segment."""
    fig = px.scatter(
        rfm_df,
        x="recency_days",
        y="monetary",
        color="rfm_segment",
        size="frequency",
        opacity=0.6,
        labels={
            "recency_days": "Recency (days since last order)",
            "monetary": "Total Spend (R$)",
        },
        title="Customer RFM Segmentation",
        log_y=True,
    )
    fig.update_layout(template="plotly_white")
    return fig


def generate_all_visualizations(
    eda_results: Dict[str, Any],
    master_df: pd.DataFrame,
    rfm_df: pd.DataFrame,
) -> Dict[str, str]:
    """Generate every static PNG and interactive HTML chart in one call.

    Parameters
    ----------
    eda_results : dict
        Output of ``eda.run_eda``.
    master_df : pd.DataFrame
        Feature-engineered master dataset (needed for the geo map, which
        plots raw rows rather than a pre-aggregated table).
    rfm_df : pd.DataFrame
        Output of ``feature_engineering.compute_customer_rfm``.

    Returns
    -------
    dict[str, str]
        Chart name -> file path, for logging / manifest purposes.
    """
    saved = {}

    saved["revenue_by_category_static"] = plot_revenue_by_category(eda_results["revenue_by_category"])
    saved["monthly_revenue_trend_static"] = plot_monthly_revenue_trend(eda_results["monthly_revenue_trend"])
    saved["review_score_distribution_static"] = plot_review_score_distribution(
        eda_results["review_score_distribution"]
    )
    saved["delivery_time_distribution_static"] = plot_delivery_time_distribution(master_df)
    saved["correlation_heatmap_static"] = plot_correlation_heatmap(eda_results["correlation_matrix"])
    saved["rfm_segments_static"] = plot_rfm_segment_counts(rfm_df)

    saved["revenue_by_category_interactive"] = _save_interactive(
        plotly_revenue_by_category(eda_results["revenue_by_category"]), "revenue_by_category.html"
    )
    saved["monthly_revenue_trend_interactive"] = _save_interactive(
        plotly_monthly_revenue_trend(eda_results["monthly_revenue_trend"]), "monthly_revenue_trend.html"
    )
    saved["customer_map_interactive"] = _save_interactive(
        plotly_customer_seller_map(master_df), "customer_map.html"
    )
    saved["rfm_scatter_interactive"] = _save_interactive(
        plotly_rfm_scatter(rfm_df), "rfm_scatter.html"
    )

    logger.info("Generated %d charts (%d static PNG, %d interactive HTML).", len(saved), 6, 4)
    return saved
