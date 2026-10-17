"""
data_merger.py
===============
Joins the cleaned per-table DataFrames into one order-item-grained master
dataset. That grain (one row per item within an order) is the natural unit
for this data: it is the finest level any table offers, and every coarser
question ("revenue per order", "orders per customer") can be produced from
it with a ``groupby``, whereas the reverse is not true.

Join order matters here -- we start from order_items (the fact table) and
progressively attach dimension tables, so that a many-to-one relationship
is preserved at every step and row counts stay predictable.
"""

from typing import Dict

import pandas as pd

from src import get_logger

logger = get_logger(__name__)


def _log_merge(step: str, before: int, after: int) -> None:
    """Emit a consistent log line after each merge step."""
    delta = after - before
    logger.info("%-28s | %8s -> %8s rows (%+d)", step, f"{before:,}", f"{after:,}", delta)


def build_master_dataset(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge all cleaned tables into a single order-item-level DataFrame.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Output of ``data_cleaner.clean_all_datasets``.

    Returns
    -------
    pd.DataFrame
        One row per (order_id, order_item_id), enriched with order status
        and dates, customer and seller identity/location, product
        attributes (with English category names), geolocation coordinates,
        and payment totals. Review score is attached at the order level
        (Olist reviews are per-order, not per-item).

    Notes
    -----
    Aggregate payment totals are merged rather than exploded: a single
    order can have multiple payment rows (e.g. split across a voucher and
    a credit card), so we sum ``payment_value`` per order *before* joining
    to avoid inflating row counts or double-counting revenue.
    """
    order_items = datasets["order_items"]
    orders = datasets["orders"]
    customers = datasets["customers"]
    products = datasets["products"]
    sellers = datasets["sellers"]
    reviews = datasets["reviews"]
    payments = datasets["payments"]
    geolocation = datasets["geolocation"]
    category_translation = datasets["category_translation"]

    df = order_items.copy()
    logger.info("Starting merge from order_items base: %s rows.", f"{len(df):,}")

    # --- orders (adds status + all lifecycle timestamps) ------------------
    before = len(df)
    df = df.merge(orders, on="order_id", how="left", validate="many_to_one")
    _log_merge("+ orders", before, len(df))

    # --- customers (adds customer identity + home location) ---------------
    before = len(df)
    df = df.merge(customers, on="customer_id", how="left", validate="many_to_one")
    _log_merge("+ customers", before, len(df))

    # --- products, with English category names attached first -------------
    products_translated = products.merge(category_translation, on="product_category_name", how="left")
    original_category = products_translated["product_category_name"]
    products_translated["product_category_name_english"] = (
        products_translated["product_category_name_english"].fillna(original_category)
    )
    before = len(df)
    df = df.merge(products_translated, on="product_id", how="left", validate="many_to_one")
    _log_merge("+ products", before, len(df))

    # --- sellers (adds seller identity + location) --------------------------
    before = len(df)
    df = df.merge(sellers, on="seller_id", how="left", validate="many_to_one")
    _log_merge("+ sellers", before, len(df))

    # --- reviews (one row per order -> attach at order grain) ---------------
    # Some orders have more than one review over time; keep the most recent.
    reviews_latest = (
        reviews.sort_values("review_answer_timestamp")
        .drop_duplicates(subset="order_id", keep="last")[
            ["order_id", "review_score", "has_comment"]
        ]
    )
    before = len(df)
    df = df.merge(reviews_latest, on="order_id", how="left", validate="many_to_one")
    _log_merge("+ reviews", before, len(df))

    # --- payments, pre-aggregated to one row per order ----------------------
    # Numeric aggregates (sum, max) use vectorized groupby ops. The "primary"
    # payment type is taken from the single largest payment_value row per
    # order rather than a statistical mode -- both are defensible, but
    # sorting once and taking the top row is far faster than an apply-based
    # mode() over ~99k groups, with no meaningful difference in outcome.
    payments_numeric = payments.groupby("order_id").agg(
        total_payment_value=("payment_value", "sum"),
        payment_installments_max=("payment_installments", "max"),
    )
    payments_primary_type = (
        payments.sort_values("payment_value", ascending=False)
        .drop_duplicates(subset="order_id", keep="first")
        .set_index("order_id")["payment_type"]
        .rename("payment_type_primary")
    )
    payments_agg = payments_numeric.join(payments_primary_type).reset_index()
    before = len(df)
    df = df.merge(payments_agg, on="order_id", how="left", validate="many_to_one")
    _log_merge("+ payments (aggregated)", before, len(df))

    # --- geolocation, joined twice: once for customer zip, once for seller --
    geo_customer = geolocation.rename(
        columns={
            "geolocation_zip_code_prefix": "customer_zip_code_prefix",
            "geolocation_lat": "customer_lat",
            "geolocation_lng": "customer_lng",
        }
    )[["customer_zip_code_prefix", "customer_lat", "customer_lng"]]
    before = len(df)
    df = df.merge(geo_customer, on="customer_zip_code_prefix", how="left", validate="many_to_one")
    _log_merge("+ geolocation (customer)", before, len(df))

    geo_seller = geolocation.rename(
        columns={
            "geolocation_zip_code_prefix": "seller_zip_code_prefix",
            "geolocation_lat": "seller_lat",
            "geolocation_lng": "seller_lng",
        }
    )[["seller_zip_code_prefix", "seller_lat", "seller_lng"]]
    before = len(df)
    df = df.merge(geo_seller, on="seller_zip_code_prefix", how="left", validate="many_to_one")
    _log_merge("+ geolocation (seller)", before, len(df))

    logger.info("Master dataset assembled: %s rows, %d columns.", f"{len(df):,}", df.shape[1])
    return df
