"""
tests/test_pipeline.py
========================
Lightweight unit tests covering the pipeline's pure-function stages.

These use small, hand-built DataFrames rather than the full Olist CSVs, so
the suite runs in well under a second and does not depend on the dataset
being present on disk -- useful for CI, where the ~130MB of raw CSVs are
not checked into the repo.

Run with:
    pytest tests/
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_cleaner import clean_orders, clean_products  # noqa: E402
from src.feature_engineering import (  # noqa: E402
    add_delivery_features,
    add_financial_features,
    add_geo_distance,
)


@pytest.fixture
def sample_orders() -> pd.DataFrame:
    """A tiny orders table with mixed casing and an inconsistent status."""
    return pd.DataFrame(
        {
            "order_id": ["o1", "o2"],
            "customer_id": ["c1", "c2"],
            "order_status": ["Delivered", "CANCELED"],
            "order_purchase_timestamp": pd.to_datetime(["2018-01-01", "2018-01-02"]),
            "order_approved_at": pd.to_datetime(["2018-01-01", "2018-01-02"]),
            "order_delivered_carrier_date": pd.to_datetime(["2018-01-02", pd.NaT]),
            "order_delivered_customer_date": pd.to_datetime(["2018-01-05", pd.NaT]),
            "order_estimated_delivery_date": pd.to_datetime(["2018-01-10", "2018-01-10"]),
        }
    )


@pytest.fixture
def sample_products() -> pd.DataFrame:
    """A products table with a missing category and missing dimensions."""
    return pd.DataFrame(
        {
            "product_id": ["p1", "p2", "p3"],
            "product_category_name": ["toys", "toys", None],
            "product_name_lenght": [40.0, np.nan, 30.0],
            "product_description_lenght": [100.0, 200.0, 150.0],
            "product_photos_qty": [1.0, 2.0, 1.0],
            "product_weight_g": [500.0, np.nan, 700.0],
            "product_length_cm": [20.0, 20.0, 20.0],
            "product_height_cm": [10.0, 10.0, 10.0],
            "product_width_cm": [15.0, 15.0, 15.0],
        }
    )


def test_clean_orders_lowercases_status(sample_orders):
    cleaned = clean_orders(sample_orders)
    assert set(cleaned["order_status"]) == {"delivered", "canceled"}


def test_clean_products_fills_missing_category(sample_products):
    cleaned = clean_products(sample_products)
    assert cleaned["product_category_name"].isnull().sum() == 0
    assert cleaned.loc[cleaned["product_id"] == "p3", "product_category_name"].iat[0] == "unknown"


def test_clean_products_imputes_missing_weight(sample_products):
    cleaned = clean_products(sample_products)
    assert cleaned["product_weight_g"].isnull().sum() == 0
    # p2's missing weight should be filled with the "toys" category median (500).
    assert cleaned.loc[cleaned["product_id"] == "p2", "product_weight_g"].iat[0] == pytest.approx(500.0)


def test_delivery_features_null_for_undelivered_order(sample_orders):
    cleaned = clean_orders(sample_orders)
    featured = add_delivery_features(cleaned)
    o2 = featured[featured["order_id"] == "o2"].iloc[0]
    assert pd.isna(o2["delivery_time_days"])
    assert pd.isna(o2["is_late_delivery"])  # undefined, not False


def test_delivery_features_detects_on_time_order(sample_orders):
    cleaned = clean_orders(sample_orders)
    featured = add_delivery_features(cleaned)
    o1 = featured[featured["order_id"] == "o1"].iloc[0]
    assert o1["delivery_time_days"] == pytest.approx(4.0)
    assert bool(o1["is_late_delivery"]) is False  # delivered before the estimate


def test_financial_features_compute_ratio():
    df = pd.DataFrame({"price": [100.0, 0.0], "freight_value": [10.0, 5.0]})
    result = add_financial_features(df)
    assert result["item_total"].tolist() == [110.0, 5.0]
    assert result["freight_ratio"].iloc[0] == pytest.approx(0.10)
    assert pd.isna(result["freight_ratio"].iloc[1])  # division by zero price -> NaN, not inf


def test_geo_distance_zero_for_same_point():
    df = pd.DataFrame(
        {
            "customer_lat": [-23.5505],
            "customer_lng": [-46.6333],
            "seller_lat": [-23.5505],
            "seller_lng": [-46.6333],
        }
    )
    result = add_geo_distance(df)
    assert result["customer_seller_distance_km"].iloc[0] == pytest.approx(0.0, abs=1e-6)


def test_geo_distance_known_city_pair():
    # Sao Paulo to Rio de Janeiro is roughly 360km as the crow flies.
    df = pd.DataFrame(
        {
            "customer_lat": [-23.5505],
            "customer_lng": [-46.6333],
            "seller_lat": [-22.9068],
            "seller_lng": [-43.1729],
        }
    )
    result = add_geo_distance(df)
    assert 340 < result["customer_seller_distance_km"].iloc[0] < 380
