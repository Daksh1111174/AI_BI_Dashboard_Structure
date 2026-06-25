"""
===========================================================
Linear Regression Forecast
===========================================================
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

from .base import prepare_time_series


def train_linear_regression(
    df,
    target="Sales",
    periods=30
):

    ts = prepare_time_series(
        df,
        target_column=target
    )

    ts["t"] = np.arange(len(ts))

    X = ts[["t"]]
    y = ts["y"]

    model = LinearRegression()

    model.fit(X, y)

    future = np.arange(
        len(ts),
        len(ts) + periods
    ).reshape(-1, 1)

    forecast = model.predict(future)

    future_dates = pd.date_range(
        ts["ds"].max(),
        periods=periods + 1,
        freq="D"
    )[1:]

    result = pd.DataFrame({
        "Date": future_dates,
        "Forecast": forecast
    })

    return model, result
