"""
data_cleaner.py
================
Table-by-table cleaning rules for the raw Olist datasets.

Each ``clean_<table>`` function takes exactly one raw DataFrame and returns
a cleaned copy. Keeping the functions single-table means they can be unit
tested and reasoned about independently, and ``clean_all_datasets`` simply
wires them together in the right order.
"""

from typing import Dict

import pandas as pd

from src import get_logger

logger = get_logger(__name__)


def _strip_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from every object (string) column.

    Defensive cleaning: the raw Kaggle export is largely well-formed, but
    stray whitespace in ID columns silently breaks merges, so we normalize
    it everywhere rather than assuming it is absent.
    """
    df = df.copy()
    object_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        df[col] = df[col].str.strip()
    return df


def clean_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the customers table.

    - Strips whitespace from text fields.
    - Title-cases city names for consistent display (raw data is lowercase).
    - Upper-cases state codes.
    """
    df = _strip_string_columns(df)
    df["customer_city"] = df["customer_city"].str.title()
    df["customer_state"] = df["customer_state"].str.upper()
    before = len(df)
    df = df.drop_duplicates(subset="customer_id")
    if len(df) != before:
        logger.warning("customers: dropped %d duplicate customer_id rows.", before - len(df))
    return df.reset_index(drop=True)


def clean_geolocation(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the geolocation table to one row per zip-code prefix.

    The raw table has ~1M rows but only ~19k unique zip prefixes -- most zip
    codes have dozens of near-duplicate lat/lng pings. For a lookup table we
    only need one representative coordinate per zip, so we aggregate:
    mean latitude/longitude, and the most frequently occurring city/state
    (mode) to guard against occasional mis-tagged entries.
    """
    df = _strip_string_columns(df)
    df["geolocation_city"] = df["geolocation_city"].str.title()
    df["geolocation_state"] = df["geolocation_state"].str.upper()

    # Clip physically-impossible coordinates (Brazil's bounding box) before
    # averaging, so a handful of bad GPS pings don't skew the mean.
    brazil_lat_range, brazil_lng_range = (-34, 6), (-74, -32)
    mask = df["geolocation_lat"].between(*brazil_lat_range) & df["geolocation_lng"].between(*brazil_lng_range)
    dropped = (~mask).sum()
    if dropped:
        logger.warning("geolocation: dropping %d rows with out-of-range coordinates.", dropped)
    df = df[mask]

    agg = (
        df.groupby("geolocation_zip_code_prefix")
        .agg(
            geolocation_lat=("geolocation_lat", "mean"),
            geolocation_lng=("geolocation_lng", "mean"),
            geolocation_city=("geolocation_city", lambda s: s.mode().iat[0]),
            geolocation_state=("geolocation_state", lambda s: s.mode().iat[0]),
        )
        .reset_index()
    )
    logger.info("geolocation: reduced %d raw rows to %d unique zip prefixes.", len(df), len(agg))
    return agg


def clean_order_items(df: pd.DataFrame) -> pd.DataFrame:
    """Clean order items: strip strings, guard against non-positive prices."""
    df = _strip_string_columns(df)
    invalid = df["price"] <= 0
    if invalid.any():
        logger.warning("order_items: removing %d rows with non-positive price.", invalid.sum())
        df = df[~invalid]
    df["freight_value"] = df["freight_value"].clip(lower=0)
    return df.reset_index(drop=True)


def clean_payments(df: pd.DataFrame) -> pd.DataFrame:
    """Clean payments: strip strings, flag non-positive payment values.

    Non-positive payments are rare (a handful of rows) and are kept rather
    than dropped, since an order's *other* payment installments may still be
    valid -- but they are logged so the anomaly is visible.
    """
    df = _strip_string_columns(df)
    non_positive = (df["payment_value"] <= 0).sum()
    if non_positive:
        logger.warning("payments: %d rows have a non-positive payment_value (kept, flagged).", non_positive)
    return df.reset_index(drop=True)


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Clean reviews: fill missing free-text fields, clip score to [1, 5]."""
    df = _strip_string_columns(df)
    df["review_comment_title"] = df["review_comment_title"].fillna("")
    df["review_comment_message"] = df["review_comment_message"].fillna("")
    df["has_comment"] = df["review_comment_message"].str.len() > 0
    df["review_score"] = df["review_score"].clip(lower=1, upper=5)
    return df.reset_index(drop=True)


def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Clean orders: strip strings, normalize status casing.

    Missing ``order_delivered_customer_date`` / ``order_approved_at`` values
    are left as ``NaT`` intentionally -- they are informative (the order
    never reached that stage) rather than a data-quality defect to impute.
    """
    df = _strip_string_columns(df)
    df["order_status"] = df["order_status"].str.lower()
    return df.reset_index(drop=True)


def clean_products(df: pd.DataFrame) -> pd.DataFrame:
    """Clean products: fill missing category, impute missing dimensions.

    - Missing ``product_category_name`` (610 rows) becomes ``"unknown"``
      rather than being dropped, since the row still carries valid price /
      order information used elsewhere.
    - Missing physical dimensions (2 rows) are imputed with the median for
      the product's own category, falling back to the global median if the
      category itself has no other rows with that dimension.
    """
    df = _strip_string_columns(df)
    df["product_category_name"] = df["product_category_name"].fillna("unknown")

    dimension_cols = [
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
    ]
    for col in dimension_cols:
        if df[col].isnull().any():
            category_median = df.groupby("product_category_name")[col].transform("median")
            df[col] = df[col].fillna(category_median)
            df[col] = df[col].fillna(df[col].median())  # global fallback
    return df.reset_index(drop=True)


def clean_sellers(df: pd.DataFrame) -> pd.DataFrame:
    """Clean sellers: strip strings, standardize city/state casing."""
    df = _strip_string_columns(df)
    df["seller_city"] = df["seller_city"].str.title()
    df["seller_state"] = df["seller_state"].str.upper()
    return df.drop_duplicates(subset="seller_id").reset_index(drop=True)


def clean_category_translation(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the category-name translation lookup table."""
    df = _strip_string_columns(df)
    return df.drop_duplicates(subset="product_category_name").reset_index(drop=True)


# Maps dataset key -> cleaning function, so `clean_all_datasets` stays a
# simple loop instead of a long if/elif chain.
_CLEANERS = {
    "customers": clean_customers,
    "geolocation": clean_geolocation,
    "order_items": clean_order_items,
    "payments": clean_payments,
    "reviews": clean_reviews,
    "orders": clean_orders,
    "products": clean_products,
    "sellers": clean_sellers,
    "category_translation": clean_category_translation,
}


def clean_all_datasets(datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Apply the appropriate cleaning function to every raw dataset.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Output of ``data_loader.load_all_datasets``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Same keys, cleaned values.
    """
    cleaned = {}
    for name, df in datasets.items():
        cleaner = _CLEANERS.get(name)
        if cleaner is None:
            logger.warning("No cleaning rule registered for '%s'; passing through unchanged.", name)
            cleaned[name] = df
            continue
        cleaned[name] = cleaner(df)
        logger.info("Cleaned '%s': %s rows remain.", name, f"{len(cleaned[name]):,}")
    return cleaned
