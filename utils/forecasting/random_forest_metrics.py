"""
===========================================================
Random Forest Metrics
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
    mean_absolute_percentage_error,
    explained_variance_score,
    median_absolute_error,
    max_error
)


# ==========================================================
# Random Forest Metrics
# ==========================================================

class RandomForestMetrics:
    """
    Evaluation metrics for Random Forest Regression.
    """

    def __init__(self):

        self.results = {}

    # ======================================================
    # MAE
    # ======================================================

    @staticmethod
    def mae(y_true, y_pred):

        return mean_absolute_error(
            y_true,
            y_pred
        )

    # ======================================================
    # MSE
    # ======================================================

    @staticmethod
    def mse(y_true, y_pred):

        return mean_squared_error(
            y_true,
            y_pred
        )

    # ======================================================
    # RMSE
    # ======================================================

    @staticmethod
    def rmse(y_true, y_pred):

        return np.sqrt(

            mean_squared_error(

                y_true,

                y_pred

            )

        )

    # ======================================================
    # R2
    # ======================================================

    @staticmethod
    def r2(y_true, y_pred):

        return r2_score(

            y_true,

            y_pred

        )

    # ======================================================
    # MAPE
    # ======================================================

    @staticmethod
    def mape(
        y_true,
        y_pred
    ):

        return (

            mean_absolute_percentage_error(

                y_true,

                y_pred

            ) * 100

        )

    # ======================================================
    # SMAPE
    # ======================================================

    @staticmethod
    def smape(
        y_true,
        y_pred
    ):

        y_true = np.array(y_true)

        y_pred = np.array(y_pred)

        denominator = (

            np.abs(y_true)

            +

            np.abs(y_pred)

        )

        mask = denominator != 0

        if mask.sum() == 0:

            return np.nan

        value = np.mean(

            (

                2 *

                np.abs(

                    y_true[mask]

                    -

                    y_pred[mask]

                )

            )

            /

            denominator[mask]

        )

        return value * 100

    # ======================================================
    # Explained Variance
    # ======================================================

    @staticmethod
    def explained_variance(
        y_true,
        y_pred
    ):

        return explained_variance_score(

            y_true,

            y_pred

        )

    # ======================================================
    # Median Absolute Error
    # ======================================================

    @staticmethod
    def median_error(
        y_true,
        y_pred
    ):

        return median_absolute_error(

            y_true,

            y_pred

        )

    # ======================================================
    # Maximum Error
    # ======================================================

    @staticmethod
    def maximum_error(
        y_true,
        y_pred
    ):

        return max_error(

            y_true,

            y_pred

        )

    # ======================================================
    # Mean Error
    # ======================================================

    @staticmethod
    def mean_error(
        y_true,
        y_pred
    ):

        residual = (

            np.array(y_true)

            -

            np.array(y_pred)

        )

        return residual.mean()

    # ======================================================
    # Residuals
    # ======================================================

    @staticmethod
    def residuals(
        y_true,
        y_pred
    ):

        residual = (

            np.array(y_true)

            -

            np.array(y_pred)

        )

        return pd.DataFrame(

            {

                "Actual":

                    y_true,

                "Prediction":

                    y_pred,

                "Residual":

                    residual

            }

        )

    # ======================================================
    # Evaluate
    # ======================================================

    def evaluate(
        self,
        y_true,
        y_pred
    ):

        metrics = {

            "MAE":

                self.mae(

                    y_true,

                    y_pred

                ),

            "MSE":

                self.mse(

                    y_true,

                    y_pred

                ),

            "RMSE":

                self.rmse(

                    y_true,

                    y_pred

                ),

            "R2":

                self.r2(

                    y_true,

                    y_pred

                ),

            "MAPE":

                self.mape(

                    y_true,

                    y_pred

                ),

            "SMAPE":

                self.smape(

                    y_true,

                    y_pred

                ),

            "Explained Variance":

                self.explained_variance(

                    y_true,

                    y_pred

                ),

            "Median Error":

                self.median_error(

                    y_true,

                    y_pred

                ),

            "Maximum Error":

                self.maximum_error(

                    y_true,

                    y_pred

                ),

            "Mean Error":

                self.mean_error(

                    y_true,

                    y_pred

                )

        }

        self.results = metrics

        return metrics

    # ======================================================
    # DataFrame
    # ======================================================

    def dataframe(self):

        return pd.DataFrame(

            {

                "Metric":

                    self.results.keys(),

                "Value":

                    self.results.values()

            }

        )
        # ==========================================================
# Additional Imports
# ==========================================================

from sklearn.model_selection import (
    TimeSeriesSplit,
    cross_val_score
)

import json
from pathlib import Path


class RandomForestMetrics(RandomForestMetrics):

    # ======================================================
    # Time Series Cross Validation
    # ======================================================

    @staticmethod
    def cross_validation(
        model,
        X,
        y,
        splits=5,
        scoring="neg_root_mean_squared_error"
    ):

        cv = TimeSeriesSplit(
            n_splits=splits
        )

        scores = cross_val_score(

            estimator=model,

            X=X,

            y=y,

            cv=cv,

            scoring=scoring,

            n_jobs=-1

        )

        return np.abs(scores)

    # ======================================================
    # Cross Validation Summary
    # ======================================================

    @staticmethod
    def cross_validation_summary(scores):

        return {

            "Mean":
                scores.mean(),

            "Std":
                scores.std(),

            "Minimum":
                scores.min(),

            "Maximum":
                scores.max()

        }

    # ======================================================
    # Error Distribution
    # ======================================================

    @staticmethod
    def error_distribution(
        y_true,
        y_pred
    ):

        error = np.array(y_true) - np.array(y_pred)

        return pd.DataFrame({

            "Residual":
                error

        })

    # ======================================================
    # Residual Statistics
    # ======================================================

    @staticmethod
    def residual_statistics(
        y_true,
        y_pred
    ):

        residual = np.array(y_true) - np.array(y_pred)

        return {

            "Mean":
                residual.mean(),

            "Median":
                np.median(residual),

            "Std":
                residual.std(),

            "Minimum":
                residual.min(),

            "Maximum":
                residual.max()

        }

    # ======================================================
    # Performance Rating
    # ======================================================

    @staticmethod
    def performance_rating(r2):

        if r2 >= 0.95:
            return "★★★★★ Excellent"

        elif r2 >= 0.90:
            return "★★★★☆ Very Good"

        elif r2 >= 0.80:
            return "★★★☆☆ Good"

        elif r2 >= 0.60:
            return "★★☆☆☆ Fair"

        return "★☆☆☆☆ Poor"

    # ======================================================
    # Model Comparison Dictionary
    # ======================================================

    @staticmethod
    def comparison(
        model_name,
        metrics
    ):

        return {

            "Model":
                model_name,

            "MAE":
                metrics["MAE"],

            "RMSE":
                metrics["RMSE"],

            "MAPE":
                metrics["MAPE"],

            "R2":
                metrics["R2"]

        }

    # ======================================================
    # Export CSV
    # ======================================================

    def export_csv(
        self,
        path="reports/random_forest_metrics.csv"
    ):

        Path(path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.dataframe().to_csv(
            path,
            index=False
        )

        return path

    # ======================================================
    # Export JSON
    # ======================================================

    def export_json(
        self,
        path="reports/random_forest_metrics.json"
    ):

        Path(path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(path, "w") as file:

            json.dump(

                self.results,

                file,

                indent=4,

                default=float

            )

        return path

    # ======================================================
    # Markdown Report
    # ======================================================

    def markdown_report(self):

        report = "# Random Forest Performance\n\n"

        for key, value in self.results.items():

            report += f"- **{key}** : {value:.4f}\n"

        return report

    # ======================================================
    # Complete Summary
    # ======================================================

    def summary(
        self,
        y_true,
        y_pred
    ):

        metrics = self.evaluate(
            y_true,
            y_pred
        )

        summary = {

            "Metrics":
                metrics,

            "Rating":
                self.performance_rating(
                    metrics["R2"]
                ),

            "Residual Statistics":

                self.residual_statistics(

                    y_true,

                    y_pred

                )

        }

        return summary

    # ======================================================
    # Reset
    # ======================================================

    def reset(self):

        self.results = {}

    # ======================================================
    # String Representation
    # ======================================================

    def __str__(self):

        if not self.results:

            return "RandomForestMetrics(Not Evaluated)"

        return self.dataframe().to_string(
            index=False
        )
        
