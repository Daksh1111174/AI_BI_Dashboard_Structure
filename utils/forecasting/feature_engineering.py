"""
===========================================================
Feature Engineering
Part 1 - Date & Calendar Features

AI Business Intelligence Dashboard

Author : Daksh Shah
===========================================================
"""

from __future__ import annotations

import pandas as pd
import numpy as np


# ==========================================================
# Validation
# ==========================================================

def validate_timeseries(
    df: pd.DataFrame,
    date_column: str,
    target_column: str
):
    """
    Validate required columns.
    """

    if date_column not in df.columns:
        raise ValueError(
            f"{date_column} column not found."
        )

    if target_column not in df.columns:
        raise ValueError(
            f"{target_column} column not found."
        )

    return True


# ==========================================================
# Base Time Series Preparation
# ==========================================================

def prepare_timeseries(
    df: pd.DataFrame,
    date_column: str = "Order Date",
    target_column: str = "Sales",
    frequency: str = "D"
):
    """
    Convert dataset into ML-ready time series.

    Returns:
        DataFrame
    """

    validate_timeseries(
        df,
        date_column,
        target_column
    )

    data = df.copy()

    data[date_column] = pd.to_datetime(
        data[date_column],
        errors="coerce"
    )

    data = data.dropna(
        subset=[date_column]
    )

    data = (
        data
        .groupby(
            pd.Grouper(
                key=date_column,
                freq=frequency
            )
        )[target_column]
        .sum()
        .reset_index()
    )

    data = data.sort_values(
        date_column
    )

    data = data.rename(
        columns={
            date_column: "ds",
            target_column: "y"
        }
    )

    return data


# ==========================================================
# Missing Dates
# ==========================================================

def fill_missing_dates(
    data: pd.DataFrame
):
    """
    Ensure continuous dates.
    """

    all_dates = pd.date_range(
        start=data["ds"].min(),
        end=data["ds"].max(),
        freq="D"
    )

    data = (
        data
        .set_index("ds")
        .reindex(all_dates)
        .fillna(0)
        .rename_axis("ds")
        .reset_index()
    )

    return data


# ==========================================================
# Calendar Features
# ==========================================================

def add_calendar_features(
    data: pd.DataFrame
):
    """
    Add basic calendar features.
    """

    df = data.copy()

    # Year

    df["year"] = df["ds"].dt.year

    # Quarter

    df["quarter"] = df["ds"].dt.quarter

    # Month

    df["month"] = df["ds"].dt.month

    # Week Number

    df["week"] = (
        df["ds"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    # Day

    df["day"] = df["ds"].dt.day

    # Day of Week

    df["dayofweek"] = (
        df["ds"]
        .dt.dayofweek
    )

    # Day of Year

    df["dayofyear"] = (
        df["ds"]
        .dt.dayofyear
    )

    return df


# ==========================================================
# Weekend Features
# ==========================================================

def add_weekend_features(
    data: pd.DataFrame
):
    """
    Weekend indicators.
    """

    df = data.copy()

    df["is_weekend"] = (
        df["dayofweek"]
        .isin([5, 6])
        .astype(int)
    )

    return df


# ==========================================================
# Month Features
# ==========================================================

def add_month_features(
    data: pd.DataFrame
):
    """
    Month start/end indicators.
    """

    df = data.copy()

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
# Quarter Features
# ==========================================================

def add_quarter_features(
    data: pd.DataFrame
):
    """
    Quarter start/end indicators.
    """

    df = data.copy()

    df["is_quarter_start"] = (
        df["ds"]
        .dt.is_quarter_start
        .astype(int)
    )

    df["is_quarter_end"] = (
        df["ds"]
        .dt.is_quarter_end
        .astype(int)
    )

    return df


# ==========================================================
# Year Features
# ==========================================================

def add_year_features(
    data: pd.DataFrame
):
    """
    Year start/end indicators.
    """

    df = data.copy()

    df["is_year_start"] = (
        df["ds"]
        .dt.is_year_start
        .astype(int)
    )

    df["is_year_end"] = (
        df["ds"]
        .dt.is_year_end
        .astype(int)
    )

    return df


# ==========================================================
# Complete Date Pipeline
# ==========================================================

def build_date_features(
    df: pd.DataFrame,
    date_column="Order Date",
    target_column="Sales",
    frequency="D"
):
    """
    Full calendar feature pipeline.
    """

    data = prepare_timeseries(
        df,
        date_column,
        target_column,
        frequency
    )

    data = fill_missing_dates(data)

    data = add_calendar_features(data)

    data = add_weekend_features(data)

    data = add_month_features(data)

    data = add_quarter_features(data)

    data = add_year_features(data)

    return data


# ==========================================================
# Feature Summary
# ==========================================================

def feature_summary(
    data: pd.DataFrame
):
    """
    Return basic dataset summary.
    """

    return {

        "Rows": len(data),

        "Columns": len(data.columns),

        "Start Date":
            data["ds"].min(),

        "End Date":
            data["ds"].max(),

        "Missing Values":
            data.isna().sum().sum()

    }


# ==========================================================
# Available Features
# ==========================================================

def feature_list(
    data: pd.DataFrame
):
    """
    Return feature names.
    """

    return list(data.columns)
