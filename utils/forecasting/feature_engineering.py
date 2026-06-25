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
    # ==========================================================
# Lag Features
# ==========================================================

def add_lag_features(
    data: pd.DataFrame,
    target_column: str = "y",
    lags: list = None
):
    """
    Create lag features.

    Default:
        1
        2
        3
        7
        14
        30
    """

    if lags is None:

        lags = [
            1,
            2,
            3,
            7,
            14,
            30
        ]

    df = data.copy()

    for lag in lags:

        df[f"lag_{lag}"] = (
            df[target_column]
            .shift(lag)
        )

    return df


# ==========================================================
# Multiple Lag Groups
# ==========================================================

def add_extended_lags(
    data: pd.DataFrame
):
    """
    Add extended lag features.
    """

    df = data.copy()

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

    return add_lag_features(
        df,
        lags=lag_list
    )


# ==========================================================
# Difference Features
# ==========================================================

def add_difference_features(
    data: pd.DataFrame,
    target_column="y"
):
    """
    First and second order differences.
    """

    df = data.copy()

    df["diff_1"] = (
        df[target_column]
        .diff(1)
    )

    df["diff_7"] = (
        df[target_column]
        .diff(7)
    )

    df["diff_30"] = (
        df[target_column]
        .diff(30)
    )

    return df


# ==========================================================
# Percentage Change
# ==========================================================

def add_percentage_change(
    data: pd.DataFrame,
    target_column="y"
):
    """
    Percentage growth.
    """

    df = data.copy()

    df["pct_change_1"] = (

        df[target_column]

        .pct_change(1)

    )

    df["pct_change_7"] = (

        df[target_column]

        .pct_change(7)

    )

    df["pct_change_30"] = (

        df[target_column]

        .pct_change(30)

    )

    return df


# ==========================================================
# Previous Week Statistics
# ==========================================================

def add_previous_period_statistics(
    data: pd.DataFrame,
    target_column="y"
):
    """
    Previous period summary.
    """

    df = data.copy()

    df["previous_week"] = (

        df[target_column]

        .shift(7)

    )

    df["previous_month"] = (

        df[target_column]

        .shift(30)

    )

    df["previous_quarter"] = (

        df[target_column]

        .shift(90)

    )

    return df


# ==========================================================
# Lead Features (Optional)
# ==========================================================

def add_lead_features(
    data: pd.DataFrame,
    target_column="y"
):
    """
    Future target values.

    Useful for supervised learning
    experiments only.

    Do NOT use in production
    forecasting.
    """

    df = data.copy()

    df["lead_1"] = (

        df[target_column]

        .shift(-1)

    )

    df["lead_7"] = (

        df[target_column]

        .shift(-7)

    )

    return df


# ==========================================================
# Remove Lag Missing Values
# ==========================================================

def remove_lag_nan(
    data: pd.DataFrame
):
    """
    Remove rows introduced by lagging.
    """

    return data.dropna().reset_index(
        drop=True
    )


# ==========================================================
# Lag Feature Pipeline
# ==========================================================

def build_lag_features(
    data: pd.DataFrame
):
    """
    Complete lag feature pipeline.
    """

    df = data.copy()

    df = add_extended_lags(df)

    df = add_difference_features(df)

    df = add_percentage_change(df)

    df = add_previous_period_statistics(df)

    df = remove_lag_nan(df)

    return df
    # ==========================================================
# Rolling Mean
# ==========================================================

def add_rolling_mean(
    data: pd.DataFrame,
    target_column="y",
    windows=None
):
    """
    Add rolling mean features.
    """

    if windows is None:
        windows = [3, 7, 14, 30]

    df = data.copy()

    for window in windows:
        df[f"rolling_mean_{window}"] = (
            df[target_column]
            .rolling(window=window)
            .mean()
        )

    return df


# ==========================================================
# Rolling Median
# ==========================================================

def add_rolling_median(
    data: pd.DataFrame,
    target_column="y",
    windows=None
):

    if windows is None:
        windows = [3, 7, 14, 30]

    df = data.copy()

    for window in windows:
        df[f"rolling_median_{window}"] = (
            df[target_column]
            .rolling(window)
            .median()
        )

    return df


# ==========================================================
# Rolling Standard Deviation
# ==========================================================

def add_rolling_std(
    data: pd.DataFrame,
    target_column="y",
    windows=None
):

    if windows is None:
        windows = [3, 7, 14, 30]

    df = data.copy()

    for window in windows:
        df[f"rolling_std_{window}"] = (
            df[target_column]
            .rolling(window)
            .std()
        )

    return df


# ==========================================================
# Rolling Min
# ==========================================================

def add_rolling_min(
    data: pd.DataFrame,
    target_column="y",
    windows=None
):

    if windows is None:
        windows = [7, 14, 30]

    df = data.copy()

    for window in windows:
        df[f"rolling_min_{window}"] = (
            df[target_column]
            .rolling(window)
            .min()
        )

    return df


# ==========================================================
# Rolling Max
# ==========================================================

def add_rolling_max(
    data: pd.DataFrame,
    target_column="y",
    windows=None
):

    if windows is None:
        windows = [7, 14, 30]

    df = data.copy()

    for window in windows:
        df[f"rolling_max_{window}"] = (
            df[target_column]
            .rolling(window)
            .max()
        )

    return df


# ==========================================================
# Rolling Sum
# ==========================================================

def add_rolling_sum(
    data: pd.DataFrame,
    target_column="y",
    windows=None
):

    if windows is None:
        windows = [7, 14, 30]

    df = data.copy()

    for window in windows:
        df[f"rolling_sum_{window}"] = (
            df[target_column]
            .rolling(window)
            .sum()
        )

    return df


# ==========================================================
# Exponential Moving Average
# ==========================================================

def add_ema(
    data: pd.DataFrame,
    target_column="y",
    spans=None
):

    if spans is None:
        spans = [7, 14, 30]

    df = data.copy()

    for span in spans:
        df[f"ema_{span}"] = (
            df[target_column]
            .ewm(span=span)
            .mean()
        )

    return df


# ==========================================================
# Expanding Statistics
# ==========================================================

def add_expanding_features(
    data: pd.DataFrame,
    target_column="y"
):

    df = data.copy()

    df["expanding_mean"] = (
        df[target_column]
        .expanding()
        .mean()
    )

    df["expanding_std"] = (
        df[target_column]
        .expanding()
        .std()
    )

    df["expanding_min"] = (
        df[target_column]
        .expanding()
        .min()
    )

    df["expanding_max"] = (
        df[target_column]
        .expanding()
        .max()
    )

    return df


# ==========================================================
# Volatility
# ==========================================================

def add_volatility(
    data: pd.DataFrame,
    target_column="y",
    window=14
):

    df = data.copy()

    df["volatility"] = (
        df[target_column]
        .rolling(window)
        .std()
    )

    return df


# ==========================================================
# Z-Score
# ==========================================================

def add_zscore(
    data: pd.DataFrame,
    target_column="y",
    window=30
):

    df = data.copy()

    rolling_mean = (
        df[target_column]
        .rolling(window)
        .mean()
    )

    rolling_std = (
        df[target_column]
        .rolling(window)
        .std()
    )

    df["zscore"] = (
        df[target_column] - rolling_mean
    ) / rolling_std

    return df


# ==========================================================
# Trend Slope
# ==========================================================

def add_trend(
    data: pd.DataFrame,
    target_column="y"
):

    df = data.copy()

    df["trend"] = np.arange(len(df))

    return df


# ==========================================================
# Rolling Feature Pipeline
# ==========================================================

def build_rolling_features(
    data: pd.DataFrame
):

    df = data.copy()

    df = add_rolling_mean(df)

    df = add_rolling_median(df)

    df = add_rolling_std(df)

    df = add_rolling_min(df)

    df = add_rolling_max(df)

    df = add_rolling_sum(df)

    df = add_ema(df)

    df = add_expanding_features(df)

    df = add_volatility(df)

    df = add_zscore(df)

    df = add_trend(df)

    return df
    # ==========================================================
# Advanced Feature Engineering
# ==========================================================

from prophet.make_holidays import make_holidays_df
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# ==========================================================
# Cyclical Features
# ==========================================================

def add_cyclical_features(data: pd.DataFrame):

    """
    Convert cyclical calendar values into
    sin/cos representation.
    """

    df = data.copy()

    # Month

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    # Day of Week

    df["dow_sin"] = np.sin(
        2 * np.pi * df["dayofweek"] / 7
    )

    df["dow_cos"] = np.cos(
        2 * np.pi * df["dayofweek"] / 7
    )

    # Day of Year

    df["doy_sin"] = np.sin(
        2 * np.pi * df["dayofyear"] / 365
    )

    df["doy_cos"] = np.cos(
        2 * np.pi * df["dayofyear"] / 365
    )

    return df


# ==========================================================
# Business Day Feature
# ==========================================================

def add_business_day_feature(data):

    df = data.copy()

    df["is_business_day"] = (
        ~df["is_weekend"].astype(bool)
    ).astype(int)

    return df


# ==========================================================
# Holiday Features
# ==========================================================

def add_holiday_feature(
    data,
    country="IN"
):

    df = data.copy()

    years = sorted(
        df["year"].unique().tolist()
    )

    holidays = make_holidays_df(
        year_list=years,
        country=country
    )

    holiday_dates = pd.to_datetime(
        holidays["ds"]
    )

    df["is_holiday"] = (
        df["ds"]
        .isin(holiday_dates)
        .astype(int)
    )

    return df


# ==========================================================
# Quarter End Feature
# ==========================================================

def add_financial_features(data):

    df = data.copy()

    df["financial_quarter"] = (
        ((df["month"] - 1) // 3) + 1
    )

    df["half_year"] = np.where(
        df["month"] <= 6,
        1,
        2
    )

    return df


# ==========================================================
# Interaction Features
# ==========================================================

def add_interaction_features(data):

    df = data.copy()

    df["month_week"] = (
        df["month"] *
        df["week"]
    )

    df["quarter_month"] = (
        df["quarter"] *
        df["month"]
    )

    df["year_month"] = (
        df["year"] * 100 +
        df["month"]
    )

    return df


# ==========================================================
# Scaling
# ==========================================================

def scale_features(
    data,
    method="standard"
):

    df = data.copy()

    numeric = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if "y" in numeric:
        numeric.remove("y")

    if method == "standard":

        scaler = StandardScaler()

    else:

        scaler = MinMaxScaler()

    df[numeric] = scaler.fit_transform(
        df[numeric]
    )

    return df, scaler


# ==========================================================
# Target Scaling
# ==========================================================

def scale_target(data):

    df = data.copy()

    scaler = StandardScaler()

    df["y"] = scaler.fit_transform(
        df[["y"]]
    )

    return df, scaler


# ==========================================================
# Feature Selection
# ==========================================================

def remove_constant_features(data):

    df = data.copy()

    constant = []

    for col in df.columns:

        if df[col].nunique() <= 1:

            constant.append(col)

    return df.drop(
        columns=constant
    )


# ==========================================================
# Remove Highly Correlated
# ==========================================================

def remove_correlated_features(
    data,
    threshold=0.95
):

    df = data.copy()

    corr = (
        df.select_dtypes(
            include=np.number
        )
        .corr()
        .abs()
    )

    upper = corr.where(
        np.triu(
            np.ones(corr.shape),
            k=1
        ).astype(bool)
    )

    drop = [

        column

        for column in upper.columns

        if any(
            upper[column] > threshold
        )

    ]

    return df.drop(
        columns=drop
    )


# ==========================================================
# Final Advanced Pipeline
# ==========================================================

def build_advanced_features(
    data,
    country="IN"
):

    df = data.copy()

    df = add_cyclical_features(df)

    df = add_business_day_feature(df)

    df = add_holiday_feature(
        df,
        country
    )

    df = add_financial_features(df)

    df = add_interaction_features(df)

    df = remove_constant_features(df)

    return df
    """
===========================================================
Part 5
Master Dataset Builder

AI Business Intelligence Dashboard
===========================================================
"""

from sklearn.model_selection import train_test_split


# ==========================================================
# Remove Missing
# ==========================================================

def clean_dataset(data):

    df = data.copy()

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna()

    df = df.reset_index(drop=True)

    return df


# ==========================================================
# Feature Target Split
# ==========================================================

def split_features_target(
    data,
    target="y"
):

    df = data.copy()

    X = df.drop(
        columns=[
            "ds",
            target
        ],
        errors="ignore"
    )

    y = df[target]

    return X, y


# ==========================================================
# Train Test Split
# ==========================================================

def split_train_test(
    X,
    y,
    test_size=0.2,
    shuffle=False
):

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=test_size,

        shuffle=shuffle

    )

    return (

        X_train,

        X_test,

        y_train,

        y_test

    )


# ==========================================================
# Future Dataset
# ==========================================================

def build_future_dataframe(
    history,
    periods=30
):

    last_date = history["ds"].max()

    future = pd.DataFrame(

        {

            "ds": pd.date_range(

                last_date +

                pd.Timedelta(days=1),

                periods=periods,

                freq="D"

            )

        }

    )

    return future


# ==========================================================
# Dataset Information
# ==========================================================

def dataset_information(
    X,
    y
):

    return {

        "Samples":

            len(X),

        "Features":

            len(X.columns),

        "Feature Names":

            list(X.columns),

        "Target":

            y.name

    }


# ==========================================================
# Feature Importance Template
# ==========================================================

def empty_importance(
    X
):

    return pd.DataFrame(

        {

            "Feature":

                X.columns,

            "Importance":

                np.zeros(

                    len(X.columns)

                )

        }

    )


# ==========================================================
# Complete ML Dataset Builder
# ==========================================================

def build_ml_dataset(

    df,

    date_column="Order Date",

    target_column="Sales",

    frequency="D",

    use_lags=True,

    use_rolling=True,

    use_advanced=True,

    scaling=False,

    scaling_method="standard"

):

    # ----------------------------
    # Date Features
    # ----------------------------

    data = build_date_features(

        df,

        date_column,

        target_column,

        frequency

    )

    # ----------------------------
    # Lag
    # ----------------------------

    if use_lags:

        data = build_lag_features(

            data

        )

    # ----------------------------
    # Rolling
    # ----------------------------

    if use_rolling:

        data = build_rolling_features(

            data

        )

    # ----------------------------
    # Advanced
    # ----------------------------

    if use_advanced:

        data = build_advanced_features(

            data

        )

    # ----------------------------
    # Clean
    # ----------------------------

    data = clean_dataset(

        data

    )

    # ----------------------------
    # Scaling
    # ----------------------------

    scaler = None

    if scaling:

        data, scaler = scale_features(

            data,

            scaling_method

        )

    # ----------------------------
    # X Y
    # ----------------------------

    X, y = split_features_target(

        data

    )

    return {

        "dataset":

            data,

        "X":

            X,

        "y":

            y,

        "scaler":

            scaler,

        "summary":

            dataset_information(

                X,

                y

            )

    }


# ==========================================================
# Save Engineered Dataset
# ==========================================================

def save_dataset(

    data,

    path

):

    data.to_csv(

        path,

        index=False

    )


# ==========================================================
# Load Dataset
# ==========================================================

def load_dataset(

    path

):

    data = pd.read_csv(path)

    data["ds"] = pd.to_datetime(

        data["ds"]

    )

    return data
    
