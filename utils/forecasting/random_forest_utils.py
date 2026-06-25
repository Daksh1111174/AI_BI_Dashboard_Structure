"""
===========================================================
Random Forest Utilities
Part 1 - Data Preparation

AI Business Intelligence Dashboard

Author : Daksh Shah
===========================================================
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from sklearn.model_selection import train_test_split

from utils.forecasting.feature_engineering import (
    build_ml_dataset
)


# ==========================================================
# Validate Dataset
# ==========================================================

def validate_dataset(
    df: pd.DataFrame,
    date_column: str = "Order Date",
    target_column: str = "Sales"
) -> bool:
    """
    Validate required columns.
    """

    required = [
        date_column,
        target_column
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    return True


# ==========================================================
# Prepare ML Dataset
# ==========================================================

def prepare_dataset(
    df: pd.DataFrame,
    date_column="Order Date",
    target_column="Sales",
    frequency="D"
) -> Dict:
    """
    Complete feature engineering pipeline.
    """

    validate_dataset(
        df,
        date_column,
        target_column
    )

    dataset = build_ml_dataset(

        df=df,

        date_column=date_column,

        target_column=target_column,

        frequency=frequency,

        use_lags=True,

        use_rolling=True,

        use_advanced=True,

        scaling=False

    )

    return dataset


# ==========================================================
# Feature Columns
# ==========================================================

def get_feature_columns(
    X: pd.DataFrame
) -> List[str]:
    """
    Return feature names.
    """

    return list(X.columns)


# ==========================================================
# Target Column
# ==========================================================

def get_target_name(
    y: pd.Series
):

    return y.name


# ==========================================================
# Train Test Split
# ==========================================================

def create_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size=0.2
):
    """
    Time-series train test split.
    """

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=test_size,

        shuffle=False

    )

    return {

        "X_train": X_train,

        "X_test": X_test,

        "y_train": y_train,

        "y_test": y_test

    }


# ==========================================================
# Dataset Summary
# ==========================================================

def dataset_summary(
    dataset: Dict
):

    X = dataset["X"]

    y = dataset["y"]

    summary = {

        "Rows":

            len(X),

        "Features":

            len(X.columns),

        "Target":

            y.name,

        "Missing":

            int(X.isna().sum().sum()),

        "Start":

            dataset["dataset"]["ds"].min(),

        "End":

            dataset["dataset"]["ds"].max()

    }

    return summary


# ==========================================================
# Feature Statistics
# ==========================================================

def feature_statistics(
    X: pd.DataFrame
):

    stats = pd.DataFrame({

        "Mean":

            X.mean(),

        "Median":

            X.median(),

        "Std":

            X.std(),

        "Min":

            X.min(),

        "Max":

            X.max()

    })

    return stats


# ==========================================================
# Correlation Matrix
# ==========================================================

def correlation_matrix(
    X: pd.DataFrame
):

    return X.corr()


# ==========================================================
# Remove Constant Columns
# ==========================================================

def remove_constant_columns(
    X: pd.DataFrame
):

    keep = [

        col

        for col in X.columns

        if X[col].nunique() > 1

    ]

    return X[keep]


# ==========================================================
# Align Train/Test Columns
# ==========================================================

def align_columns(
    X_train,
    X_test
):
    """
    Ensure both datasets
    have identical columns.
    """

    train_cols = X_train.columns

    X_test = X_test.reindex(

        columns=train_cols,

        fill_value=0

    )

    return X_train, X_test


# ==========================================================
# Data Information
# ==========================================================

def data_information(
    dataset
):

    info = {

        "Dataset Shape":

            dataset["dataset"].shape,

        "Feature Shape":

            dataset["X"].shape,

        "Target Shape":

            dataset["y"].shape

    }

    return info


# ==========================================================
# Build Random Forest Dataset
# ==========================================================

def build_random_forest_dataset(
    df,
    date_column="Order Date",
    target_column="Sales"
):
    """
    One-click dataset builder.
    """

    dataset = prepare_dataset(

        df,

        date_column,

        target_column

    )

    split = create_train_test_split(

        dataset["X"],

        dataset["y"]

    )

    dataset.update(split)

    return dataset
