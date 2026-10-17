"""
data_loader.py
===============
Responsible for one thing: getting the nine raw Olist CSV files off disk and
into pandas DataFrames safely.

No cleaning, merging, or feature engineering happens here on purpose -- that
separation of concerns is what lets the later stages be tested and re-run
independently of how the data was loaded.
"""

from pathlib import Path
from typing import Dict

import pandas as pd

from config import RAW_FILES
from src import get_logger

logger = get_logger(__name__)


class DataLoadError(Exception):
    """Raised when a required raw file is missing or unreadable."""


# Columns that should be parsed as datetimes for each raw table. Declaring
# this once avoids scattering ``pd.to_datetime`` calls across the pipeline.
DATE_COLUMNS = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "order_items": ["shipping_limit_date"],
    "reviews": ["review_creation_date", "review_answer_timestamp"],
}


def _read_csv(name: str, path: Path) -> pd.DataFrame:
    """Read a single CSV file, applying date parsing where relevant.

    Parameters
    ----------
    name : str
        Logical dataset name (key from ``config.RAW_FILES``), used to look
        up which columns need datetime parsing and for error messages.
    path : Path
        Location of the CSV on disk.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    DataLoadError
        If the file does not exist or pandas fails to parse it.
    """
    if not path.exists():
        raise DataLoadError(f"Raw file for '{name}' not found at: {path}")

    try:
        df = pd.read_csv(
            path,
            parse_dates=DATE_COLUMNS.get(name),
            low_memory=False,
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DataLoadError(f"Failed to parse '{name}' at {path}: {exc}") from exc

    logger.info("Loaded '%s': %s rows, %s columns", name, f"{len(df):,}", df.shape[1])
    return df


def load_all_datasets() -> Dict[str, pd.DataFrame]:
    """Load every raw Olist table declared in ``config.RAW_FILES``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys match ``config.RAW_FILES`` keys, e.g. ``"orders"``,
        ``"customers"``, ``"products"``, etc.

    Raises
    ------
    DataLoadError
        If any individual file fails to load. The exception message
        identifies which dataset caused the failure.
    """
    datasets: Dict[str, pd.DataFrame] = {}
    for name, path in RAW_FILES.items():
        datasets[name] = _read_csv(name, path)

    logger.info("Successfully loaded %d raw datasets.", len(datasets))
    return datasets


def validate_schema(datasets: Dict[str, pd.DataFrame]) -> None:
    """Sanity-check that key ID columns exist before merging is attempted.

    This catches a corrupted or mismatched CSV early, with a clear error
    message, instead of letting a cryptic ``KeyError`` surface deep inside
    the merge stage later on.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Output of :func:`load_all_datasets`.

    Raises
    ------
    DataLoadError
        If a required column is missing from a table.
    """
    required_columns = {
        "customers": {"customer_id", "customer_unique_id"},
        "orders": {"order_id", "customer_id", "order_status"},
        "order_items": {"order_id", "product_id", "seller_id", "price"},
        "payments": {"order_id", "payment_value"},
        "reviews": {"order_id", "review_score"},
        "products": {"product_id"},
        "sellers": {"seller_id"},
        "geolocation": {"geolocation_zip_code_prefix"},
        "category_translation": {"product_category_name"},
    }

    for name, expected_cols in required_columns.items():
        actual_cols = set(datasets[name].columns)
        missing = expected_cols - actual_cols
        if missing:
            raise DataLoadError(
                f"Dataset '{name}' is missing expected column(s): {missing}"
            )

    logger.info("Schema validation passed for all %d datasets.", len(datasets))
