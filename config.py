"""
config.py
=========
Central configuration for the Olist E-Commerce Analytics project.

Every module imports its paths and constants from here so that there is a
single source of truth for file locations. Keeping this separate from the
logic makes the project portable: moving the project to a new machine only
requires updating the paths in this file.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
PROCESSED_DIR = OUTPUT_DIR / "processed"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Make sure the output directories always exist, even on a fresh clone.
for directory in (FIGURES_DIR, PROCESSED_DIR, REPORTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Raw source files (Kaggle "Brazilian E-Commerce Public Dataset by Olist")
# ---------------------------------------------------------------------------
RAW_FILES = {
    "customers": DATA_RAW_DIR / "olist_customers_dataset.csv",
    "geolocation": DATA_RAW_DIR / "olist_geolocation_dataset.csv",
    "order_items": DATA_RAW_DIR / "olist_order_items_dataset.csv",
    "payments": DATA_RAW_DIR / "olist_order_payments_dataset.csv",
    "reviews": DATA_RAW_DIR / "olist_order_reviews_dataset.csv",
    "orders": DATA_RAW_DIR / "olist_orders_dataset.csv",
    "products": DATA_RAW_DIR / "olist_products_dataset.csv",
    "sellers": DATA_RAW_DIR / "olist_sellers_dataset.csv",
    "category_translation": DATA_RAW_DIR / "product_category_name_translation.csv",
}

# ---------------------------------------------------------------------------
# Processed artifacts produced by the pipeline
# ---------------------------------------------------------------------------
MASTER_DATASET_PATH = PROCESSED_DIR / "olist_master_dataset.parquet"
MASTER_DATASET_CSV_PATH = PROCESSED_DIR / "olist_master_dataset.csv"
EDA_SUMMARY_PATH = REPORTS_DIR / "eda_summary.json"

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------
# Orders in these statuses never reached a customer and are excluded from
# revenue / delivery-time analysis (kept only for funnel/status reporting).
NON_DELIVERED_STATUSES = {"canceled", "unavailable", "created", "invoiced"}

# Consistent color palette so every chart in the project looks related.
PALETTE = {
    "primary": "#2E86AB",
    "secondary": "#F26419",
    "accent": "#A23B72",
    "positive": "#3B8768",
    "negative": "#C1121F",
    "neutral": "#6C757D",
}

RANDOM_SEED = 42

# Plot style used by matplotlib/seaborn across the project.
SNS_STYLE = "whitegrid"
FIGURE_DPI = 150
