"""
feature_engineering.py
========================
Derives analytical features on top of the merged master dataset:
time-based fields, delivery-performance metrics, financial ratios,
geo-distance between buyer and seller, product-size metrics, review
sentiment buckets, and a customer-level RFM (Recency/Frequency/Monetary)
segmentation table.

Each ``add_*`` function takes the master DataFrame and returns it with new
columns appended, so ``engineer_all_features`` can chain them without
re-implementing any logic.
"""

import numpy as np
import pandas as pd

from src import get_logger

logger = get_logger(__name__)

EARTH_RADIUS_KM = 6371.0


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Break ``order_purchase_timestamp`` into calendar parts for trend EDA.

    Adds: purchase_year, purchase_month, purchase_year_month (e.g. "2018-01"
    as a sortable string), purchase_weekday (Monday=0), purchase_hour.
    """
    df = df.copy()
    ts = df["order_purchase_timestamp"]
    df["purchase_year"] = ts.dt.year
    df["purchase_month"] = ts.dt.month
    df["purchase_year_month"] = ts.dt.to_period("M").astype(str)
    df["purchase_weekday"] = ts.dt.day_name()
    df["purchase_hour"] = ts.dt.hour
    return df


def add_delivery_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute delivery-performance metrics from the order lifecycle dates.

    Adds:
        approval_time_hours          -- purchase -> payment approval
        shipping_time_days           -- approval -> handed to carrier
        delivery_time_days           -- purchase -> delivered to customer
        delivery_delta_days          -- estimated delivery date minus actual
                                         delivery date (positive = early,
                                         negative = late)
        is_late_delivery              -- bool, True if delivered after the
                                         estimated date

    All durations are ``NaN`` for orders that never reached the relevant
    lifecycle stage (e.g. a canceled order has no delivery date), which is
    the correct behavior rather than an error to suppress.
    """
    df = df.copy()
    purchase = df["order_purchase_timestamp"]
    approved = df["order_approved_at"]
    carrier = df["order_delivered_carrier_date"]
    delivered = df["order_delivered_customer_date"]
    estimated = df["order_estimated_delivery_date"]

    df["approval_time_hours"] = (approved - purchase).dt.total_seconds() / 3600
    df["shipping_time_days"] = (carrier - approved).dt.total_seconds() / 86400
    df["delivery_time_days"] = (delivered - purchase).dt.total_seconds() / 86400
    df["delivery_delta_days"] = (estimated - delivered).dt.total_seconds() / 86400
    # Pandas' nullable boolean dtype ("boolean") allows True/False/NA in one
    # column, unlike plain "bool" which cannot hold missing values. Orders
    # without a delivery date were never delivered, so "late" is undefined
    # (NA) for them rather than False.
    df["is_late_delivery"] = (df["delivery_delta_days"] < 0).astype("boolean")
    df.loc[delivered.isna(), "is_late_delivery"] = pd.NA
    return df


def add_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-item financial ratios.

    Adds:
        item_total          -- price + freight_value for this line item
        freight_ratio        -- freight_value / price (cost of shipping as a
                                 fraction of the item's price; higher means
                                 shipping ate more of the sale)
    """
    df = df.copy()
    df["item_total"] = df["price"] + df["freight_value"]
    df["freight_ratio"] = df["freight_value"] / df["price"].replace(0, np.nan)
    return df


def add_geo_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Add great-circle distance (km) between customer and seller.

    Uses the haversine formula on the zip-level lat/lng attached during
    merging. Rows missing either coordinate get ``NaN`` rather than a
    fabricated distance.
    """
    df = df.copy()
    lat1, lng1 = np.radians(df["customer_lat"]), np.radians(df["customer_lng"])
    lat2, lng2 = np.radians(df["seller_lat"]), np.radians(df["seller_lng"])

    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2.0) ** 2
    a = np.clip(a, 0, 1)  # guard against floating-point drift just above 1
    c = 2 * np.arcsin(np.sqrt(a))
    df["customer_seller_distance_km"] = EARTH_RADIUS_KM * c
    return df


def add_product_size_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a single volumetric size feature from the three product dimensions."""
    df = df.copy()
    df["product_volume_cm3"] = (
        df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]
    )
    return df


def add_review_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket the 1-5 review score into a three-level sentiment label."""
    df = df.copy()
    bins = [0, 2, 3, 5]
    labels = ["dissatisfied", "neutral", "satisfied"]
    df["review_sentiment"] = pd.cut(df["review_score"], bins=bins, labels=labels)
    return df


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run every feature-engineering step in sequence on the master dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``data_merger.build_master_dataset``.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with all derived columns appended.
    """
    steps = (
        add_time_features,
        add_delivery_features,
        add_financial_features,
        add_geo_distance,
        add_product_size_features,
        add_review_sentiment,
    )
    for step in steps:
        before_cols = set(df.columns)
        df = step(df)
        new_cols = list(set(df.columns) - before_cols)
        logger.info("%-24s added columns: %s", step.__name__, new_cols)

    logger.info("Feature engineering complete: %d total columns.", df.shape[1])
    return df


def compute_customer_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp = None) -> pd.DataFrame:
    """Build a customer-level RFM (Recency, Frequency, Monetary) table.

    RFM is computed at the ``customer_unique_id`` grain (the stable person
    behind possibly-many ``customer_id`` records) using each customer's
    *distinct orders*, not item rows, so a multi-item order does not inflate
    frequency.

    Parameters
    ----------
    df : pd.DataFrame
        The feature-engineered master dataset (item grain).
    snapshot_date : pd.Timestamp, optional
        The reference "today" for recency. Defaults to one day after the
        latest purchase timestamp in the data, which is the standard
        convention for RFM on a static/historical dataset.

    Returns
    -------
    pd.DataFrame
        One row per ``customer_unique_id`` with columns: recency_days,
        frequency, monetary, and an ``rfm_segment`` label derived from
        tercile scores.
    """
    orders_grain = df.drop_duplicates(subset="order_id")[
        ["customer_unique_id", "order_id", "order_purchase_timestamp", "total_payment_value"]
    ]

    if snapshot_date is None:
        snapshot_date = orders_grain["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

    rfm = (
        orders_grain.groupby("customer_unique_id")
        .agg(
            recency_days=("order_purchase_timestamp", lambda s: (snapshot_date - s.max()).days),
            frequency=("order_id", "nunique"),
            monetary=("total_payment_value", "sum"),
        )
        .reset_index()
    )

    # Score each dimension into terciles (1 = worst, 3 = best), then combine
    # into a readable segment label. Recency is inverted: fewer days = better.
    rfm["r_score"] = pd.qcut(rfm["recency_days"], 3, labels=[3, 2, 1]).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 3, labels=[1, 2, 3]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"], 3, labels=[1, 2, 3]).astype(int)
    rfm["rfm_total"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]

    def _segment(total: int) -> str:
        if total >= 8:
            return "Champions"
        if total >= 6:
            return "Loyal"
        if total >= 4:
            return "At Risk"
        return "Lost"

    rfm["rfm_segment"] = rfm["rfm_total"].apply(_segment)
    logger.info("Computed RFM for %s unique customers.", f"{len(rfm):,}")
    return rfm
