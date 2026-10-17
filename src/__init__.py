"""
src package
===========
Modular pipeline for the Olist E-Commerce Analytics project.

Modules
-------
data_loader          -- read raw CSVs into DataFrames with validation
data_cleaner         -- type coercion, deduplication, missing-value handling
data_merger          -- join the cleaned tables into one master dataset
feature_engineering  -- derive analytical features (delivery time, RFM, etc.)
eda                  -- descriptive statistics and summary tables
visualization        -- matplotlib / seaborn / plotly chart builders
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a configured module-level logger.

    Using one logger factory keeps log formatting consistent across every
    module instead of each file configuring ``logging`` independently.

    Parameters
    ----------
    name : str
        Usually ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:  # avoid duplicate handlers on re-import
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
