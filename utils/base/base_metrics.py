"""
===========================================================
AI Business Intelligence
Base Metrics

Author : Daksh Shah
===========================================================
"""

from __future__ import annotations

from abc import ABC
from typing import Dict, Any

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
    median_absolute_error,
    max_error,
    mean_absolute_percentage_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==========================================================
# Base Metrics
# ==========================================================

class BaseMetrics(ABC):
    """
    Base evaluation metrics for regression and classification.
    """

    def __init__(self):

        self.results: Dict[str, Any] = {}
      # ======================================================
# Additional Imports
# ======================================================

from sklearn.model_selection import (
    KFold,
    TimeSeriesSplit,
    cross_val_score
)

from scipy.stats import skew, kurtosis


    # ======================================================
    # K-Fold Cross Validation
    # ======================================================

    def kfold_cv(
        self,
        model,
        X,
        y,
        folds=5,
        scoring="neg_root_mean_squared_error"
    ):

        cv = KFold(

            n_splits=folds,

            shuffle=True,

            random_state=42

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
    # Time Series Cross Validation
    # ======================================================

    def timeseries_cv(
        self,
        model,
        X,
        y,
        folds=5,
        scoring="neg_root_mean_squared_error"
    ):

        cv = TimeSeriesSplit(

            n_splits=folds

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
    def cv_summary(
        scores
    ):

        scores = np.asarray(scores)

        return {

            "Mean":

                scores.mean(),

            "Median":

                np.median(scores),

            "Minimum":

                scores.min(),

            "Maximum":

                scores.max(),

            "Standard Deviation":

                scores.std()

        }

    # ======================================================
    # Error Distribution
    # ======================================================

    @staticmethod
    def error_distribution(
        y_true,
        y_pred
    ):

        residual = (

            np.asarray(y_true)

            -

            np.asarray(y_pred)

        )

        return pd.DataFrame(

            {

                "Residual":

                    residual

            }

        )

    # ======================================================
    # Residual Statistics
    # ======================================================

    @staticmethod
    def residual_statistics(
        y_true,
        y_pred
    ):

        residual = (

            np.asarray(y_true)

            -

            np.asarray(y_pred)

        )

        return {

            "Mean":

                residual.mean(),

            "Median":

                np.median(

                    residual

                ),

            "Standard Deviation":

                residual.std(),

            "Variance":

                residual.var(),

            "Minimum":

                residual.min(),

            "Maximum":

                residual.max(),

            "Skewness":

                skew(

                    residual

                ),

            "Kurtosis":

                kurtosis(

                    residual

                )

        }

    # ======================================================
    # Performance Rating
    # ======================================================

    @staticmethod
    def performance_rating(
        r2_score_value
    ):

        if r2_score_value >= 0.95:

            return "★★★★★ Excellent"

        if r2_score_value >= 0.90:

            return "★★★★☆ Very Good"

        if r2_score_value >= 0.80:

            return "★★★☆☆ Good"

        if r2_score_value >= 0.70:

            return "★★☆☆☆ Fair"

        return "★☆☆☆☆ Poor"

    # ======================================================
    # Statistical Summary
    # ======================================================

    @staticmethod
    def statistical_summary(
        values
    ):

        values = np.asarray(values)

        return {

            "Count":

                len(values),

            "Mean":

                values.mean(),

            "Median":

                np.median(values),

            "Std":

                values.std(),

            "Minimum":

                values.min(),

            "Maximum":

                values.max()

        }

    # ======================================================
    # Comparison Record
    # ======================================================

    @staticmethod
    def comparison_record(
        model_name,
        metrics
    ):

        return {

            "Model":

                model_name,

            "MAE":

                metrics.get("MAE"),

            "RMSE":

                metrics.get("RMSE"),

            "MAPE":

                metrics.get("MAPE"),

            "R2":

                metrics.get("R2")

        }

    # ======================================================
    # Leaderboard DataFrame
    # ======================================================

    @staticmethod
    def leaderboard(
        records
    ):

        leaderboard = pd.DataFrame(

            records

        )

        if "RMSE" in leaderboard.columns:

            leaderboard = leaderboard.sort_values(

                "RMSE",

                ascending=True

            )

        leaderboard = leaderboard.reset_index(

            drop=True

        )

        leaderboard.index += 1

        leaderboard.index.name = "Rank"

        return leaderboard

    # ======================================================
    # Metric Exists
    # ======================================================

    def has_metric(
        self,
        name
    ):

        return name in self.results
      # ======================================================
# Additional Imports
# ======================================================

import json
from pathlib import Path
from datetime import datetime


    # ======================================================
    # Export CSV
    # ======================================================

    def export_csv(
        self,
        path="reports/metrics.csv"
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
        path="reports/metrics.json"
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

        report = "# Model Evaluation Report\n\n"

        report += (
            f"Generated: "
            f"{datetime.now():%Y-%m-%d %H:%M:%S}\n\n"
        )

        for metric, value in self.results.items():

            if isinstance(value, float):

                report += (
                    f"- **{metric}** : "
                    f"{value:.4f}\n"
                )

            else:

                report += (
                    f"- **{metric}** : "
                    f"{value}\n"
                )

        return report

    # ======================================================
    # HTML Report
    # ======================================================

    def html_report(self):

        html = """
        <html>
        <head>
            <title>Model Metrics</title>
        </head>
        <body>
        <h2>Evaluation Metrics</h2>
        <table border="1" cellpadding="5">
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
        """

        for metric, value in self.results.items():

            if isinstance(value, float):
                value = round(value, 4)

            html += f"""
            <tr>
                <td>{metric}</td>
                <td>{value}</td>
            </tr>
            """

        html += """
        </table>
        </body>
        </html>
        """

        return html

    # ======================================================
    # Save HTML Report
    # ======================================================

    def export_html(
        self,
        path="reports/metrics.html"
    ):

        Path(path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(path, "w", encoding="utf-8") as file:

            file.write(
                self.html_report()
            )

        return path

    # ======================================================
    # Complete Summary
    # ======================================================

    def summary(self):

        return {

            "metrics": self.results,

            "generated":

                datetime.now().isoformat(),

            "count":

                len(self.results)

        }

    # ======================================================
    # Reset
    # ======================================================

    def reset(self):

        self.results = {}

    # ======================================================
    # Available Metrics
    # ======================================================

    @property
    def metric_names(self):

        return list(

            self.results.keys()

        )

    # ======================================================
    # Number of Metrics
    # ======================================================

    @property
    def metric_count(self):

        return len(

            self.results

        )

    # ======================================================
    # Dictionary
    # ======================================================

    def to_dict(self):

        return dict(

            self.results

        )

    # ======================================================
    # DataFrame Alias
    # ======================================================

    def to_dataframe(self):

        return self.dataframe()

    # ======================================================
    # String Representation
    # ======================================================

    def __str__(self):

        if not self.results:

            return "BaseMetrics(Not Evaluated)"

        return self.dataframe().to_string(
            index=False
        )

    # ======================================================
    # Representation
    # ======================================================

    def __repr__(self):

        return (

            f"BaseMetrics("

            f"metrics={len(self.results)})"

        )

    # ======================================================
    # Regression Metrics
    # ======================================================

    @staticmethod
    def mae(y_true, y_pred):

        return float(
            mean_absolute_error(
                y_true,
                y_pred
            )
        )

    @staticmethod
    def mse(y_true, y_pred):

        return float(
            mean_squared_error(
                y_true,
                y_pred
            )
        )

    @staticmethod
    def rmse(y_true, y_pred):

        return float(

            np.sqrt(

                mean_squared_error(
                    y_true,
                    y_pred
                )

            )

        )

    @staticmethod
    def r2(y_true, y_pred):

        return float(

            r2_score(
                y_true,
                y_pred
            )

        )

    @staticmethod
    def explained_variance(
        y_true,
        y_pred
    ):

        return float(

            explained_variance_score(
                y_true,
                y_pred
            )

        )

    @staticmethod
    def median_error(
        y_true,
        y_pred
    ):

        return float(

            median_absolute_error(
                y_true,
                y_pred
            )

        )

    @staticmethod
    def max_error(
        y_true,
        y_pred
    ):

        return float(

            max_error(
                y_true,
                y_pred
            )

        )

    @staticmethod
    def mape(
        y_true,
        y_pred
    ):

        return float(

            mean_absolute_percentage_error(
                y_true,
                y_pred
            ) * 100

        )

    @staticmethod
    def smape(
        y_true,
        y_pred
    ):

        y_true = np.asarray(y_true)

        y_pred = np.asarray(y_pred)

        denominator = np.abs(y_true) + np.abs(y_pred)

        mask = denominator != 0

        if mask.sum() == 0:

            return np.nan

        return float(

            np.mean(

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

            ) * 100

        )

    # ======================================================
    # Classification Metrics
    # ======================================================

    @staticmethod
    def accuracy(
        y_true,
        y_pred
    ):

        return float(

            accuracy_score(
                y_true,
                y_pred
            )

        )

    @staticmethod
    def precision(
        y_true,
        y_pred,
        average="weighted"
    ):

        return float(

            precision_score(

                y_true,

                y_pred,

                average=average,

                zero_division=0

            )

        )

    @staticmethod
    def recall(
        y_true,
        y_pred,
        average="weighted"
    ):

        return float(

            recall_score(

                y_true,

                y_pred,

                average=average,

                zero_division=0

            )

        )

    @staticmethod
    def f1(
        y_true,
        y_pred,
        average="weighted"
    ):

        return float(

            f1_score(

                y_true,

                y_pred,

                average=average,

                zero_division=0

            )

        )

    @staticmethod
    def confusion(
        y_true,
        y_pred
    ):

        return confusion_matrix(
            y_true,
            y_pred
        )

    # ======================================================
    # Residuals
    # ======================================================

    @staticmethod
    def residuals(
        y_true,
        y_pred
    ):

        residual = (

            np.asarray(y_true)

            -

            np.asarray(y_pred)

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
    # Regression Evaluation
    # ======================================================

    def regression_metrics(
        self,
        y_true,
        y_pred
    ):

        self.results = {

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

                self.max_error(

                    y_true,

                    y_pred

                )

        }

        return self.results

    # ======================================================
    # Classification Evaluation
    # ======================================================

    def classification_metrics(
        self,
        y_true,
        y_pred
    ):

        self.results = {

            "Accuracy":

                self.accuracy(

                    y_true,

                    y_pred

                ),

            "Precision":

                self.precision(

                    y_true,

                    y_pred

                ),

            "Recall":

                self.recall(

                    y_true,

                    y_pred

                ),

            "F1":

                self.f1(

                    y_true,

                    y_pred

                )

        }

        return self.results

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
      
