"""
===========================================================
Prophet Utilities
AI Business Intelligence Dashboard

Author : Daksh Shah
===========================================================
"""

from __future__ import annotations

import pandas as pd
from prophet.make_holidays import make_holidays_df
from sklearn.model_selection import train_test_split


# ==========================================================
# Validation
# ==========================================================

def validate_dataset(
    df: pd.DataFrame,
    date_column: str = "Order Date",
    target_column: str = "Sales"
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
# Prepare Time Series
# ==========================================================

def prepare_prophet_dataframe(
    df: pd.DataFrame,
    date_column="Order Date",
    target_column="Sales",
    frequency="D"
):
    """
    Convert dataframe into Prophet format.

    Returns:
        ds
        y
    """

    validate_dataset(
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

    data.columns = ["ds", "y"]

    data = data.sort_values("ds")

    data = data.fillna(0)

    return data


# ==========================================================
# Aggregation
# ==========================================================

def aggregate_data(
    prophet_df,
    level="Monthly"
):
    """
    Aggregate Prophet dataframe.

    Levels

    Daily
    Weekly
    Monthly
    Quarterly
    Yearly
    """

    freq = {

        "Daily": "D",

        "Weekly": "W",

        "Monthly": "M",

        "Quarterly": "Q",

        "Yearly": "Y"

    }

    if level not in freq:

        raise ValueError(
            "Invalid aggregation level."
        )

    data = (
        prophet_df
        .groupby(
            pd.Grouper(
                key="ds",
                freq=freq[level]
            )
        )["y"]
        .sum()
        .reset_index()
    )

    return data


# ==========================================================
# Missing Values
# ==========================================================

def fill_missing_dates(df):
    """
    Fill missing dates.

    Prophet performs better with
    continuous time series.
    """

    data = df.copy()

    date_range = pd.date_range(

        start=data["ds"].min(),

        end=data["ds"].max(),

        freq="D"

    )

    data = (

        data

        .set_index("ds")

        .reindex(date_range)

        .fillna(0)

        .rename_axis("ds")

        .reset_index()

    )

    return data


# ==========================================================
# Train Test Split
# ==========================================================

def split_data(
    prophet_df,
    test_size=0.2,
    shuffle=False
):
    """
    Split train test.
    """

    train, test = train_test_split(

        prophet_df,

        test_size=test_size,

        shuffle=shuffle

    )

    train = train.sort_values("ds")

    test = test.sort_values("ds")

    return train, test


# ==========================================================
# Future Dataframe
# ==========================================================

def create_future_dataframe(
    model,
    periods=30,
    frequency="D"
):
    """
    Create future dataframe.
    """

    future = model.make_future_dataframe(

        periods=periods,

        freq=frequency

    )

    return future


# ==========================================================
# Holidays
# ==========================================================

def load_holidays(
    country="IN",
    years=None
):
    """
    Create holiday dataframe.
    """

    if years is None:

        years = [

            2022,

            2023,

            2024,

            2025,

            2026,

            2027

        ]

    holidays = make_holidays_df(

        year_list=years,

        country=country

    )

    return holidays


# ==========================================================
# Outlier Removal
# ==========================================================

def remove_outliers(
    prophet_df,
    threshold=3
):
    """
    Remove Z-score outliers.
    """

    data = prophet_df.copy()

    mean = data["y"].mean()

    std = data["y"].std()

    z = (

        (data["y"] - mean)

        / std

    )

    data = data[
        abs(z) < threshold
    ]

    return data


# ==========================================================
# Scaling (Optional)
# ==========================================================

def normalize_target(prophet_df):
    """
    Normalize target values.
    """

    data = prophet_df.copy()

    minimum = data["y"].min()

    maximum = data["y"].max()

    data["y"] = (

        data["y"] - minimum

    ) / (

        maximum - minimum

    )

    return data


# ==========================================================
# Restore Scaling
# ==========================================================

def inverse_normalize(
    values,
    minimum,
    maximum
):

    return (

        values *

        (maximum - minimum)

    ) + minimum


# ==========================================================
# Dataset Information
# ==========================================================

def dataset_summary(prophet_df):

    return {

        "Records": len(prophet_df),

        "Start": prophet_df["ds"].min(),

        "End": prophet_df["ds"].max(),

        "Minimum": prophet_df["y"].min(),

        "Maximum": prophet_df["y"].max(),

        "Mean": prophet_df["y"].mean(),

        "Median": prophet_df["y"].median(),

        "Std": prophet_df["y"].std()

    }


# ==========================================================
# Save Prepared Data
# ==========================================================

def save_prepared_data(
    prophet_df,
    path
):

    prophet_df.to_csv(
        path,
        index=False
    )


# ==========================================================
# Load Prepared Data
# ==========================================================

def load_prepared_data(
    path
):

    data = pd.read_csv(path)

    data["ds"] = pd.to_datetime(
        data["ds"]
    )

    return data
