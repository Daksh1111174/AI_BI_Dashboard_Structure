"""
===========================================================
Forecasting Base Utilities
===========================================================
"""

import pandas as pd


def prepare_time_series(
    df,
    date_column="Order Date",
    target_column="Sales"
):
    """
    Prepare dataframe for forecasting models.
    Returns a dataframe with columns:
        ds -> date
        y  -> target
    """

    if date_column not in df.columns:
        raise ValueError(f"{date_column} not found")

    if target_column not in df.columns:
        raise ValueError(f"{target_column} not found")

    data = df.copy()

    data[date_column] = pd.to_datetime(
        data[date_column],
        errors="coerce"
    )

    data = data.dropna(subset=[date_column])

    data = (
        data
        .groupby(date_column)[target_column]
        .sum()
        .reset_index()
    )

    data.columns = ["ds", "y"]

    return data.sort_values("ds")
