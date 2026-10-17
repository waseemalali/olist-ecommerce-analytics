#!/usr/bin/env python3
"""
main.py
========
Entry point for the Olist E-Commerce Analytics pipeline.

Runs every stage in order -- load, clean, merge, engineer features, run EDA,
generate charts -- and persists the artifacts each stage produces. Each
stage is wrapped so a failure is reported with the stage name and the
pipeline exits with a non-zero status instead of a raw traceback, which
matters for running this unattended (e.g. in CI or a cron job).

Usage
-----
    python main.py                  # run the full pipeline
    python main.py --skip-charts    # skip chart generation (faster re-runs)
"""

import argparse
import sys
import time

import pandas as pd

from config import MASTER_DATASET_CSV_PATH, MASTER_DATASET_PATH
from src import get_logger
from src.data_cleaner import clean_all_datasets
from src.data_loader import DataLoadError, load_all_datasets, validate_schema
from src.data_merger import build_master_dataset
from src.eda import run_eda
from src.feature_engineering import compute_customer_rfm, engineer_all_features
from src.visualization import generate_all_visualizations

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the pipeline run."""
    parser = argparse.ArgumentParser(description="Olist E-Commerce Analytics pipeline")
    parser.add_argument(
        "--skip-charts",
        action="store_true",
        help="Skip static/interactive chart generation (useful for quick data-only re-runs).",
    )
    return parser.parse_args()


def run_pipeline(skip_charts: bool = False) -> int:
    """Execute the full pipeline end to end.

    Parameters
    ----------
    skip_charts : bool
        If True, the visualization stage is skipped.

    Returns
    -------
    int
        Process exit code: 0 on success, 1 on failure.
    """
    start_time = time.time()

    try:
        logger.info("STAGE 1/6 -- Loading raw datasets")
        raw_datasets = load_all_datasets()
        validate_schema(raw_datasets)
    except DataLoadError as exc:
        logger.error("Pipeline aborted during loading: %s", exc)
        return 1

    try:
        logger.info("STAGE 2/6 -- Cleaning datasets")
        cleaned_datasets = clean_all_datasets(raw_datasets)
    except (KeyError, ValueError) as exc:
        logger.error("Pipeline aborted during cleaning: %s", exc)
        return 1

    try:
        logger.info("STAGE 3/6 -- Merging into master dataset")
        master_df = build_master_dataset(cleaned_datasets)
    except (KeyError, pd.errors.MergeError) as exc:
        logger.error("Pipeline aborted during merging: %s", exc)
        return 1

    try:
        logger.info("STAGE 4/6 -- Engineering features")
        master_df = engineer_all_features(master_df)
        rfm_df = compute_customer_rfm(master_df)
    except (KeyError, ValueError) as exc:
        logger.error("Pipeline aborted during feature engineering: %s", exc)
        return 1

    # Persist the master dataset now, before EDA/visualization, so the
    # expensive load-clean-merge-engineer work is never lost even if a
    # later, cheaper stage fails.
    try:
        master_df.to_parquet(MASTER_DATASET_PATH, index=False)
        master_df.to_csv(MASTER_DATASET_CSV_PATH, index=False)
        logger.info("Master dataset saved to %s and %s", MASTER_DATASET_PATH, MASTER_DATASET_CSV_PATH)
    except OSError as exc:
        logger.error("Failed to save master dataset: %s", exc)
        return 1

    try:
        logger.info("STAGE 5/6 -- Running EDA")
        eda_results = run_eda(master_df)
        logger.info("KPIs: %s", eda_results["kpis"])
    except (KeyError, ValueError) as exc:
        logger.error("Pipeline aborted during EDA: %s", exc)
        return 1

    if not skip_charts:
        try:
            logger.info("STAGE 6/6 -- Generating charts")
            generate_all_visualizations(eda_results, master_df, rfm_df)
        except (KeyError, ValueError, OSError) as exc:
            logger.error("Pipeline aborted during chart generation: %s", exc)
            return 1
    else:
        logger.info("STAGE 6/6 -- Skipped (--skip-charts)")

    elapsed = time.time() - start_time
    logger.info("Pipeline completed successfully in %.1f seconds.", elapsed)
    return 0


if __name__ == "__main__":
    args = parse_args()
    sys.exit(run_pipeline(skip_charts=args.skip_charts))
