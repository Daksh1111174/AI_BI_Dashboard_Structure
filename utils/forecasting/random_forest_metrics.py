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
