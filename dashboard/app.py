"""
dashboard/app.py
==================
Interactive Streamlit dashboard for the Olist E-Commerce Analytics project.

Run with:
    streamlit run dashboard/app.py

The dashboard is a thin presentation layer: every number and chart it shows
comes from ``src.eda`` and ``src.visualization``, the same functions the
batch pipeline (``main.py``) uses. This means the dashboard and the static
report can never silently disagree with each other, and none of the
analytical logic is duplicated here -- this file only handles filtering,
layout, and calling those functions on the filtered slice.
"""

import sys
from pathlib import Path

# Allow `streamlit run dashboard/app.py` to import the project's top-level
# `config` and `src` packages regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from config import MASTER_DATASET_PATH, PALETTE  # noqa: E402
from src.eda import (  # noqa: E402
    compute_kpis,
    delivery_performance_summary,
    monthly_revenue_trend,
    payment_type_distribution,
    revenue_by_category,
    revenue_by_state,
    review_score_distribution,
    top_sellers,
)
from src.feature_engineering import compute_customer_rfm  # noqa: E402
from src.visualization import (  # noqa: E402
    plotly_customer_seller_map,
    plotly_monthly_revenue_trend,
    plotly_revenue_by_category,
    plotly_rfm_scatter,
)

st.set_page_config(page_title="Olist E-Commerce Analytics", page_icon="📦", layout="wide")


@st.cache_data(show_spinner="Loading master dataset...")
def load_master_data() -> pd.DataFrame:
    """Load the pipeline's parquet output, with a friendly error if missing."""
    if not MASTER_DATASET_PATH.exists():
        st.error(
            f"Master dataset not found at `{MASTER_DATASET_PATH}`.\n\n"
            "Run `python main.py` first to build it."
        )
        st.stop()
    return pd.read_parquet(MASTER_DATASET_PATH)


@st.cache_data(show_spinner="Computing customer segments...")
def load_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RFM once per data load and cache it (it's not filter-dependent)."""
    return compute_customer_rfm(df)


def render_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render sidebar filter widgets and return the filtered DataFrame."""
    st.sidebar.header("Filters")

    min_date = df["order_purchase_timestamp"].min().date()
    max_date = df["order_purchase_timestamp"].max().date()
    date_range = st.sidebar.date_input(
        "Purchase date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    # date_input returns a single date while the user is mid-selection;
    # guard against that before unpacking a two-tuple.
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    states = sorted(df["customer_state"].dropna().unique())
    selected_states = st.sidebar.multiselect("Customer state", states, default=[])

    categories = sorted(df["product_category_name_english"].dropna().unique())
    selected_categories = st.sidebar.multiselect("Product category", categories, default=[])

    mask = df["order_purchase_timestamp"].dt.date.between(start_date, end_date)
    if selected_states:
        mask &= df["customer_state"].isin(selected_states)
    if selected_categories:
        mask &= df["product_category_name_english"].isin(selected_categories)

    filtered = df[mask]
    st.sidebar.caption(f"{filtered['order_id'].nunique():,} orders match the current filters.")
    return filtered


def render_kpi_row(df: pd.DataFrame) -> None:
    """Render the top row of headline KPI metric cards."""
    kpis = compute_kpis(df)
    cols = st.columns(6)
    cols[0].metric("Total Orders", f"{kpis['total_orders']:,}")
    cols[1].metric("Total Revenue", f"R$ {kpis['total_revenue']:,.0f}")
    cols[2].metric("Avg Order Value", f"R$ {kpis['avg_order_value']:,.2f}")
    cols[3].metric("Avg Review Score", f"{kpis['avg_review_score']:.2f} / 5")
    cols[4].metric("Avg Delivery Time", f"{kpis['avg_delivery_time_days']:.1f} days")
    cols[5].metric("Late Delivery Rate", f"{kpis['late_delivery_rate_pct']:.1f}%")


def render_overview_tab(df: pd.DataFrame) -> None:
    """Revenue trend and top-category charts."""
    col1, col2 = st.columns([3, 2])
    with col1:
        st.plotly_chart(plotly_monthly_revenue_trend(monthly_revenue_trend(df)), width='stretch')
    with col2:
        st.plotly_chart(plotly_revenue_by_category(revenue_by_category(df, top_n=10)), width='stretch')

    st.subheader("Revenue by State")
    st.dataframe(revenue_by_state(df).style.format({"revenue": "R$ {:,.2f}"}), width='stretch')


def render_delivery_tab(df: pd.DataFrame) -> None:
    """Delivery-performance metrics and distribution."""
    summary = delivery_performance_summary(df)
    cols = st.columns(4)
    cols[0].metric("Mean Delivery Time", f"{summary['mean_days']:.1f} days")
    cols[1].metric("Median Delivery Time", f"{summary['median_days']:.1f} days")
    cols[2].metric("90th Percentile", f"{summary['p90_days']:.1f} days")
    cols[3].metric("Late Delivery Rate", f"{summary['late_rate_pct']:.1f}%")

    import plotly.express as px

    delivered = df.dropna(subset=["delivery_time_days"])
    cap = delivered["delivery_time_days"].quantile(0.99)
    fig = px.histogram(
        delivered[delivered["delivery_time_days"] <= cap],
        x="delivery_time_days",
        nbins=40,
        color_discrete_sequence=[PALETTE["primary"]],
        labels={"delivery_time_days": "Delivery Time (days)"},
        title="Delivery Time Distribution (99th percentile capped)",
    )
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, width='stretch')

    st.subheader("Top Sellers by Revenue")
    sellers_table = top_sellers(df, top_n=10).style.format(
        {"revenue": "R$ {:,.2f}", "avg_review_score": "{:.2f}"}
    )
    st.dataframe(sellers_table, width='stretch')


def render_customers_tab(df: pd.DataFrame, rfm_full: pd.DataFrame) -> None:
    """Customer segmentation (RFM) view.

    RFM is always computed on the *full* dataset (recency/frequency/monetary
    need a customer's complete order history to be meaningful), then joined
    down to whichever customers remain in the current filtered slice.
    """
    filtered_customer_ids = df["customer_unique_id"].unique()
    rfm_view = rfm_full[rfm_full["customer_unique_id"].isin(filtered_customer_ids)]

    col1, col2 = st.columns([2, 3])
    with col1:
        segment_counts = rfm_view["rfm_segment"].value_counts()
        st.metric("Customers in view", f"{len(rfm_view):,}")
        st.bar_chart(segment_counts)
    with col2:
        st.plotly_chart(plotly_rfm_scatter(rfm_view), width='stretch')

    st.subheader("Payment Method Mix")
    st.dataframe(payment_type_distribution(df), width='stretch')


def render_reviews_tab(df: pd.DataFrame) -> None:
    """Review score distribution and sentiment breakdown."""
    import plotly.express as px

    dist = review_score_distribution(df)
    fig = px.bar(
        dist,
        x="review_score",
        y="pct",
        color="review_score",
        color_continuous_scale=["#C1121F", "#C1121F", "#6C757D", "#3B8768", "#3B8768"],
        labels={"pct": "Share of Orders (%)", "review_score": "Review Score"},
        title="Review Score Distribution",
    )
    fig.update_layout(template="plotly_white", showlegend=False)
    st.plotly_chart(fig, width='stretch')

    if "review_sentiment" in df.columns:
        sentiment_counts = df.drop_duplicates("order_id")["review_sentiment"].value_counts()
        st.bar_chart(sentiment_counts)


def render_geography_tab(df: pd.DataFrame) -> None:
    """Customer location map."""
    st.plotly_chart(plotly_customer_seller_map(df), width='stretch')


def main() -> None:
    """Render the full dashboard."""
    st.title("📦 Olist E-Commerce Analytics Dashboard")
    st.caption("Brazilian E-Commerce Public Dataset by Olist -- 2016-2018")

    master_df = load_master_data()
    rfm_full = load_rfm(master_df)
    filtered_df = render_sidebar_filters(master_df)

    if filtered_df.empty:
        st.warning("No orders match the current filters. Try widening the date range or clearing selections.")
        st.stop()

    render_kpi_row(filtered_df)

    tab_overview, tab_delivery, tab_customers, tab_reviews, tab_geo = st.tabs(
        ["Overview", "Delivery Performance", "Customer Segments", "Reviews", "Geography"]
    )
    with tab_overview:
        render_overview_tab(filtered_df)
    with tab_delivery:
        render_delivery_tab(filtered_df)
    with tab_customers:
        render_customers_tab(filtered_df, rfm_full)
    with tab_reviews:
        render_reviews_tab(filtered_df)
    with tab_geo:
        render_geography_tab(filtered_df)


if __name__ == "__main__":
    main()
