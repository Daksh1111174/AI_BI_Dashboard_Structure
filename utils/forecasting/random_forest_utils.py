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
# ==========================================================
# Future Date Generator
# ==========================================================

def create_future_dates(
    history: pd.DataFrame,
    periods: int = 30,
    frequency: str = "D"
):
    """
    Generate future dates after the last date
    in the historical dataset.
    """

    last_date = history["ds"].max()

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=periods,
        freq=frequency
    )

    future = pd.DataFrame({
        "ds": future_dates
    })

    return future


# ==========================================================
# Calendar Features
# ==========================================================

def add_future_calendar_features(
    future: pd.DataFrame
):
    """
    Generate calendar features for
    future timestamps.
    """

    df = future.copy()

    df["year"] = df["ds"].dt.year

    df["quarter"] = df["ds"].dt.quarter

    df["month"] = df["ds"].dt.month

    df["week"] = (
        df["ds"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    df["day"] = df["ds"].dt.day

    df["dayofweek"] = df["ds"].dt.dayofweek

    df["dayofyear"] = df["ds"].dt.dayofyear

    df["is_weekend"] = (
        df["dayofweek"]
        .isin([5, 6])
        .astype(int)
    )

    df["is_month_start"] = (
        df["ds"]
        .dt.is_month_start
        .astype(int)
    )

    df["is_month_end"] = (
        df["ds"]
        .dt.is_month_end
        .astype(int)
    )

    return df


# ==========================================================
# Initialize Lag Features
# ==========================================================

def initialize_lags(
    history: pd.DataFrame,
    future: pd.DataFrame,
    target="y"
):
    """
    Use latest historical values as
    initial lag values.
    """

    df = future.copy()

    history = history.copy()

    lags = [
        1,
        2,
        3,
        5,
        7,
        14,
        21,
        30,
        60,
        90
    ]

    for lag in lags:

        if len(history) >= lag:

            value = history[target].iloc[-lag]

        else:

            value = history[target].iloc[0]

        df[f"lag_{lag}"] = value

    return df


# ==========================================================
# Rolling Statistics
# ==========================================================

def initialize_rolling_features(
    history,
    future,
    target="y"
):

    df = future.copy()

    series = history[target]

    windows = [3, 7, 14, 30]

    for window in windows:

        values = series.tail(window)

        df[f"rolling_mean_{window}"] = values.mean()

        df[f"rolling_std_{window}"] = values.std()

        df[f"rolling_min_{window}"] = values.min()

        df[f"rolling_max_{window}"] = values.max()

        df[f"rolling_sum_{window}"] = values.sum()

    return df


# ==========================================================
# EMA Features
# ==========================================================

def initialize_ema(
    history,
    future,
    target="y"
):

    df = future.copy()

    spans = [7, 14, 30]

    for span in spans:

        ema = (
            history[target]
            .ewm(span=span)
            .mean()
            .iloc[-1]
        )

        df[f"ema_{span}"] = ema

    return df


# ==========================================================
# Trend Feature
# ==========================================================

def initialize_trend(
    history,
    future
):

    df = future.copy()

    start = len(history)

    df["trend"] = np.arange(

        start,

        start + len(df)

    )

    return df


# ==========================================================
# Cyclical Features
# ==========================================================

def add_future_cyclical_features(
    future
):

    df = future.copy()

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    df["dow_sin"] = np.sin(
        2 * np.pi * df["dayofweek"] / 7
    )

    df["dow_cos"] = np.cos(
        2 * np.pi * df["dayofweek"] / 7
    )

    df["doy_sin"] = np.sin(
        2 * np.pi * df["dayofyear"] / 365
    )

    df["doy_cos"] = np.cos(
        2 * np.pi * df["dayofyear"] / 365
    )

    return df


# ==========================================================
# Complete Future Dataset
# ==========================================================

def build_future_dataset(
    history,
    periods=30
):

    future = create_future_dates(

        history,

        periods

    )

    future = add_future_calendar_features(

        future

    )

    future = initialize_lags(

        history,

        future

    )

    future = initialize_rolling_features(

        history,

        future

    )

    future = initialize_ema(

        history,

        future

    )

    future = initialize_trend(

        history,

        future

    )

    future = add_future_cyclical_features(

        future

    )

    return future
    # ==========================================================
# Recursive Forecast Utilities
# ==========================================================

from typing import List


# ==========================================================
# Append Prediction
# ==========================================================

def append_prediction(
    history: pd.DataFrame,
    prediction: float,
    next_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Append a predicted value to history.
    """

    new_row = pd.DataFrame({
        "ds": [next_date],
        "y": [prediction]
    })

    history = pd.concat(
        [history, new_row],
        ignore_index=True
    )

    return history


# ==========================================================
# Update Lag Features
# ==========================================================

def update_lag_features(
    history: pd.DataFrame,
    lag_list: List[int] = None
):
    """
    Recalculate lag features.
    """

    if lag_list is None:

        lag_list = [
            1,
            2,
            3,
            5,
            7,
            14,
            21,
            30,
            60,
            90
        ]

    df = history.copy()

    for lag in lag_list:

        df[f"lag_{lag}"] = (
            df["y"]
            .shift(lag)
        )

    return df


# ==========================================================
# Update Rolling Features
# ==========================================================

def update_rolling_features(
    history: pd.DataFrame
):

    df = history.copy()

    windows = [3, 7, 14, 30]

    for window in windows:

        df[f"rolling_mean_{window}"] = (
            df["y"]
            .rolling(window)
            .mean()
        )

        df[f"rolling_std_{window}"] = (
            df["y"]
            .rolling(window)
            .std()
        )

        df[f"rolling_min_{window}"] = (
            df["y"]
            .rolling(window)
            .min()
        )

        df[f"rolling_max_{window}"] = (
            df["y"]
            .rolling(window)
            .max()
        )

        df[f"rolling_sum_{window}"] = (
            df["y"]
            .rolling(window)
            .sum()
        )

    return df


# ==========================================================
# Update EMA
# ==========================================================

def update_ema_features(
    history: pd.DataFrame
):

    df = history.copy()

    spans = [7, 14, 30]

    for span in spans:

        df[f"ema_{span}"] = (
            df["y"]
            .ewm(span=span)
            .mean()
        )

    return df


# ==========================================================
# Update Trend
# ==========================================================

def update_trend(
    history: pd.DataFrame
):

    df = history.copy()

    df["trend"] = np.arange(len(df))

    return df


# ==========================================================
# Update Calendar Features
# ==========================================================

def update_calendar_features(
    history: pd.DataFrame
):

    df = history.copy()

    df["year"] = df["ds"].dt.year

    df["quarter"] = df["ds"].dt.quarter

    df["month"] = df["ds"].dt.month

    df["week"] = (
        df["ds"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    df["day"] = df["ds"].dt.day

    df["dayofweek"] = (
        df["ds"]
        .dt.dayofweek
    )

    df["dayofyear"] = (
        df["ds"]
        .dt.dayofyear
    )

    df["is_weekend"] = (
        df["dayofweek"]
        .isin([5, 6])
        .astype(int)
    )

    return df


# ==========================================================
# Update Cyclical Features
# ==========================================================

def update_cyclical_features(
    history: pd.DataFrame
):

    df = history.copy()

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    df["dow_sin"] = np.sin(
        2 * np.pi * df["dayofweek"] / 7
    )

    df["dow_cos"] = np.cos(
        2 * np.pi * df["dayofweek"] / 7
    )

    df["doy_sin"] = np.sin(
        2 * np.pi * df["dayofyear"] / 365
    )

    df["doy_cos"] = np.cos(
        2 * np.pi * df["dayofyear"] / 365
    )

    return df


# ==========================================================
# Rebuild Feature Matrix
# ==========================================================

def rebuild_features(
    history: pd.DataFrame
):
    """
    Recalculate all engineered features
    after adding a prediction.
    """

    df = history.copy()

    df = update_calendar_features(df)

    df = update_lag_features(df)

    df = update_rolling_features(df)

    df = update_ema_features(df)

    df = update_trend(df)

    df = update_cyclical_features(df)

    return df


# ==========================================================
# Get Latest Feature Vector
# ==========================================================

def latest_feature_vector(
    history: pd.DataFrame,
    feature_columns: List[str]
):
    """
    Return the latest row ready for prediction.
    """

    df = rebuild_features(history)

    latest = df.iloc[-1:]

    return latest[feature_columns]
    # ==========================================================
# Helper Utilities
# ==========================================================

from pathlib import Path
from datetime import datetime
import json
import joblib


# ==========================================================
# Save Dataset
# ==========================================================

def save_prepared_dataset(
    dataset: pd.DataFrame,
    path: str
):
    """
    Save engineered dataset as CSV.
    """

    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    dataset.to_csv(
        path,
        index=False
    )


# ==========================================================
# Load Dataset
# ==========================================================

def load_prepared_dataset(
    path: str
):

    df = pd.read_csv(path)

    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"])

    return df


# ==========================================================
# Save Feature Metadata
# ==========================================================

def save_feature_metadata(
    feature_columns,
    target_name,
    path="models/feature_metadata.json"
):

    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata = {

        "version": "1.0",

        "created":

            datetime.now().isoformat(),

        "target":

            target_name,

        "features":

            list(feature_columns)

    }

    with open(path, "w") as file:

        json.dump(

            metadata,

            file,

            indent=4

        )


# ==========================================================
# Load Metadata
# ==========================================================

def load_feature_metadata(
    path="models/feature_metadata.json"
):

    with open(path) as file:

        return json.load(file)


# ==========================================================
# Save Scaler
# ==========================================================

def save_scaler(
    scaler,
    path="models/scaler.pkl"
):

    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        scaler,
        path
    )


# ==========================================================
# Load Scaler
# ==========================================================

def load_scaler(
    path="models/scaler.pkl"
):

    return joblib.load(path)


# ==========================================================
# Validate Features
# ==========================================================

def validate_feature_columns(
    dataframe,
    expected_features
):
    """
    Ensure required columns exist.
    """

    missing = [

        col

        for col in expected_features

        if col not in dataframe.columns

    ]

    if missing:

        raise ValueError(

            f"Missing features: {missing}"

        )

    return True


# ==========================================================
# Align Feature Order
# ==========================================================

def align_feature_order(
    dataframe,
    expected_features
):
    """
    Reorder columns to match training.
    """

    return dataframe.reindex(

        columns=expected_features,

        fill_value=0

    )


# ==========================================================
# Missing Feature Report
# ==========================================================

def missing_feature_report(
    dataframe,
    expected_features
):

    missing = [

        feature

        for feature in expected_features

        if feature not in dataframe.columns

    ]

    extra = [

        feature

        for feature in dataframe.columns

        if feature not in expected_features

    ]

    return {

        "missing":

            missing,

        "extra":

            extra

    }


# ==========================================================
# Dataset Diagnostics
# ==========================================================

def dataset_diagnostics(
    dataframe
):

    return {

        "Rows":

            len(dataframe),

        "Columns":

            len(dataframe.columns),

        "Missing":

            int(

                dataframe.isna()

                .sum()

                .sum()

            ),

        "Duplicates":

            int(

                dataframe.duplicated()

                .sum()

            ),

        "Memory (MB)":

            round(

                dataframe.memory_usage(

                    deep=True

                ).sum()

                / 1024**2,

                2

            )

    }


# ==========================================================
# Model Version
# ==========================================================

def model_information():

    return {

        "Model":

            "Random Forest",

        "Version":

            "1.0",

        "Author":

            "Daksh Shah",

        "Framework":

            "scikit-learn"

    }


# ==========================================================
# Export Summary
# ==========================================================

def export_summary(
    dataset
):

    summary = {

        "Created":

            datetime.now()

            .strftime("%Y-%m-%d %H:%M"),

        "Rows":

            len(dataset),

        "Columns":

            len(dataset.columns)

    }

    return summary
    
