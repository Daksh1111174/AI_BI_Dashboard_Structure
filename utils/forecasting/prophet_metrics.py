"""
===========================================================
Prophet Metrics
AI Business Intelligence Dashboard

Author : Daksh Shah
===========================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

try:
    from prophet.diagnostics import (
        cross_validation,
        performance_metrics,
    )
    PROPHET_DIAGNOSTICS = True
except ImportError:
    PROPHET_DIAGNOSTICS = False


# ==========================================================
# Basic Metrics
# ==========================================================

def mae(y_true, y_pred):
    """Mean Absolute Error"""
    return mean_absolute_error(y_true, y_pred)


def mse(y_true, y_pred):
    """Mean Squared Error"""
    return mean_squared_error(y_true, y_pred)


def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def r2(y_true, y_pred):
    """R² Score"""
    return r2_score(y_true, y_pred)


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = y_true != 0

    if mask.sum() == 0:
        return np.nan

    return np.mean(
        np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    ) * 100


def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denominator = (
        np.abs(y_true) +
        np.abs(y_pred)
    )

    mask = denominator != 0

    if mask.sum() == 0:
        return np.nan

    return (
        np.mean(
            (
                2 * np.abs(y_pred[mask] - y_true[mask])
            ) / denominator[mask]
        ) * 100
    )


# ==========================================================
# Complete Evaluation
# ==========================================================

def evaluate_forecast(actual_df, forecast_df):
    """
    Compare actual values with Prophet forecast.

    Parameters
    ----------
    actual_df : DataFrame
        Must contain columns:
            ds
            y

    forecast_df : DataFrame
        Prophet prediction output
    """

    merged = pd.merge(
        actual_df,
        forecast_df[
            [
                "ds",
                "yhat"
            ]
        ],
        on="ds",
        how="inner"
    )

    y_true = merged["y"]
    y_pred = merged["yhat"]

    metrics = {

        "MAE":
            round(mae(y_true, y_pred), 3),

        "MSE":
            round(mse(y_true, y_pred), 3),

        "RMSE":
            round(rmse(y_true, y_pred), 3),

        "MAPE":
            round(mape(y_true, y_pred), 3),

        "SMAPE":
            round(smape(y_true, y_pred), 3),

        "R2":
            round(r2(y_true, y_pred), 3),

    }

    return metrics


# ==========================================================
# Prophet Cross Validation
# ==========================================================

def prophet_cross_validation(
    model,
    initial="365 days",
    period="90 days",
    horizon="90 days"
):
    """
    Prophet rolling cross-validation.
    """

    if not PROPHET_DIAGNOSTICS:
        raise ImportError(
            "Prophet diagnostics are not installed."
        )

    cv = cross_validation(
        model=model,
        initial=initial,
        period=period,
        horizon=horizon
    )

    return cv


# ==========================================================
# Prophet Performance Metrics
# ==========================================================

def prophet_performance(cv_results):
    """
    Calculate Prophet diagnostic metrics.
    """

    if not PROPHET_DIAGNOSTICS:
        raise ImportError(
            "Prophet diagnostics unavailable."
        )

    perf = performance_metrics(cv_results)

    return perf


# ==========================================================
# Metrics Summary
# ==========================================================

def metrics_summary(metrics_dict):
    """
    Convert metric dictionary to DataFrame.
    """

    return pd.DataFrame(
        {

            "Metric":
                metrics_dict.keys(),

            "Value":
                metrics_dict.values()

        }

    )


# ==========================================================
# Best Model Logic
# ==========================================================

def better_model(metric_a, metric_b, metric_name):
    """
    Compare two metric values.

    Lower is better:
        MAE
        RMSE
        MSE
        MAPE
        SMAPE

    Higher is better:
        R2
    """

    lower = [
        "MAE",
        "RMSE",
        "MSE",
        "MAPE",
        "SMAPE"
    ]

    if metric_name in lower:
        return metric_a < metric_b

    return metric_a > metric_b


# ==========================================================
# Forecast Quality
# ==========================================================

def quality_rating(r2_score_value):
    """
    Convert R² into a simple quality label.
    """

    if r2_score_value >= 0.95:
        return "★★★★★ Excellent"

    elif r2_score_value >= 0.90:
        return "★★★★☆ Very Good"

    elif r2_score_value >= 0.80:
        return "★★★☆☆ Good"

    elif r2_score_value >= 0.60:
        return "★★☆☆☆ Fair"

    return "★☆☆☆☆ Poor"
