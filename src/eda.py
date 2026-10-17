"""
eda.py
=======
Descriptive analytics on top of the feature-engineered master dataset.

Every function here returns either a small summary ``dict`` or a tidy
``pd.DataFrame`` sized for plotting -- none of them mutate or return the
full master dataset, keeping this module a pure "summarize" layer that the
visualization module consumes.

Revenue and delivery-time statistics are computed on *delivered* orders
only (see ``config.NON_DELIVERED_STATUSES``), since a canceled order has no
real revenue or delivery experience to measure.
"""

import json
from typing import Any, Dict

import numpy as np
import pandas as pd

from config import EDA_SUMMARY_PATH, NON_DELIVERED_STATUSES
from src import get_logger

logger = get_logger(__name__)


def _delivered_only(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows whose order actually reached the customer."""
    return df[~df["order_status"].isin(NON_DELIVERED_STATUSES)]


def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute headline business KPIs for the dashboard's summary cards.

    Returns
    -------
    dict with keys: total_orders, total_revenue, avg_order_value,
    avg_review_score, avg_delivery_time_days, late_delivery_rate_pct,
    total_customers, total_sellers.
    """
    delivered = _delivered_only(df)
    orders_grain = delivered.drop_duplicates(subset="order_id")

    late_rate = delivered["is_late_delivery"].dropna().mean() * 100 if len(delivered) else np.nan

    kpis = {
        "total_orders": int(df["order_id"].nunique()),
        "total_revenue": float(orders_grain["total_payment_value"].sum()),
        "avg_order_value": float(orders_grain["total_payment_value"].mean()),
        "avg_review_score": float(df["review_score"].mean()),
        "avg_delivery_time_days": float(delivered["delivery_time_days"].mean()),
        "late_delivery_rate_pct": float(late_rate),
        "total_customers": int(df["customer_unique_id"].nunique()),
        "total_sellers": int(df["seller_id"].nunique()),
    }
    return kpis


def revenue_by_category(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Total revenue and order count for the top N product categories."""
    delivered = _delivered_only(df)
    summary = (
        delivered.groupby("product_category_name_english")
        .agg(revenue=("item_total", "sum"), orders=("order_id", "nunique"))
        .sort_values("revenue", ascending=False)
        .head(top_n)
        .reset_index()
        .rename(columns={"product_category_name_english": "category"})
    )
    return summary


def revenue_by_state(df: pd.DataFrame) -> pd.DataFrame:
    """Total revenue and customer count per Brazilian state (customer's state)."""
    delivered = _delivered_only(df)
    summary = (
        delivered.groupby("customer_state")
        .agg(revenue=("item_total", "sum"), customers=("customer_unique_id", "nunique"))
        .sort_values("revenue", ascending=False)
        .reset_index()
    )
    return summary


def monthly_revenue_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue and order volume by calendar month, sorted chronologically."""
    delivered = _delivered_only(df)
    summary = (
        delivered.groupby("purchase_year_month")
        .agg(revenue=("item_total", "sum"), orders=("order_id", "nunique"))
        .reset_index()
        .sort_values("purchase_year_month")
    )
    return summary


def review_score_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage share of each 1-5 review score."""
    counts = df["review_score"].value_counts().sort_index()
    summary = counts.rename("count").reset_index().rename(columns={"index": "review_score"})
    summary.columns = ["review_score", "count"]
    summary["pct"] = summary["count"] / summary["count"].sum() * 100
    return summary


def delivery_performance_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Distribution stats for delivery time and lateness, for the EDA report."""
    delivered = _delivered_only(df)
    delivery_days = delivered["delivery_time_days"].dropna()
    return {
        "mean_days": float(delivery_days.mean()),
        "median_days": float(delivery_days.median()),
        "p90_days": float(delivery_days.quantile(0.9)),
        "late_rate_pct": float(delivered["is_late_delivery"].dropna().mean() * 100),
    }


def payment_type_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Share of orders paid with each payment method."""
    orders_grain = df.drop_duplicates(subset="order_id")
    summary = (
        orders_grain["payment_type_primary"]
        .value_counts()
        .rename("count")
        .reset_index()
        .rename(columns={"index": "payment_type"})
    )
    summary.columns = ["payment_type", "count"]
    summary["pct"] = summary["count"] / summary["count"].sum() * 100
    return summary


def top_sellers(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Top N sellers by total revenue, with order count and avg review score."""
    delivered = _delivered_only(df)
    summary = (
        delivered.groupby("seller_id")
        .agg(
            revenue=("item_total", "sum"),
            orders=("order_id", "nunique"),
            avg_review_score=("review_score", "mean"),
        )
        .sort_values("revenue", ascending=False)
        .head(top_n)
        .reset_index()
    )
    return summary


def numeric_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix over the key numeric analytical features."""
    numeric_cols = [
        "price",
        "freight_value",
        "item_total",
        "delivery_time_days",
        "delivery_delta_days",
        "customer_seller_distance_km",
        "product_volume_cm3",
        "review_score",
        "payment_installments_max",
    ]
    available = [c for c in numeric_cols if c in df.columns]
    return df[available].corr(numeric_only=True)


def run_eda(df: pd.DataFrame) -> Dict[str, Any]:
    """Run every EDA function and bundle the results into one dict.

    Also writes the scalar KPIs and delivery summary to
    ``config.EDA_SUMMARY_PATH`` as JSON, so the numbers behind the
    dashboard/README can be inspected or diffed without re-running the
    pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        The feature-engineered master dataset.

    Returns
    -------
    dict[str, Any]
        Keys: "kpis", "delivery_performance", "revenue_by_category",
        "revenue_by_state", "monthly_revenue_trend",
        "review_score_distribution", "payment_type_distribution",
        "top_sellers", "correlation_matrix" -- values are either dicts
        (scalars) or DataFrames (tables), ready for the visualization layer.
    """
    results = {
        "kpis": compute_kpis(df),
        "delivery_performance": delivery_performance_summary(df),
        "revenue_by_category": revenue_by_category(df),
        "revenue_by_state": revenue_by_state(df),
        "monthly_revenue_trend": monthly_revenue_trend(df),
        "review_score_distribution": review_score_distribution(df),
        "payment_type_distribution": payment_type_distribution(df),
        "top_sellers": top_sellers(df),
        "correlation_matrix": numeric_correlation_matrix(df),
    }

    # Persist just the JSON-serializable scalar summaries to disk.
    json_safe = {
        "kpis": results["kpis"],
        "delivery_performance": results["delivery_performance"],
    }
    try:
        with open(EDA_SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(json_safe, f, indent=2)
        logger.info("EDA summary written to %s", EDA_SUMMARY_PATH)
    except OSError as exc:
        logger.warning("Could not write EDA summary to disk: %s", exc)

    return results
