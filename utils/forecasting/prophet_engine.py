"""
===========================================================
Prophet Forecast Engine
AI Business Intelligence Dashboard

Author : Daksh Shah
===========================================================
"""

from pathlib import Path

from utils.forecasting.prophet_model import ProphetForecaster
from utils.forecasting.prophet_metrics import (
    evaluate_forecast,
    metrics_summary,
    quality_rating
)

from utils.forecasting.prophet_visualization import (
    forecast_chart,
    actual_vs_forecast,
    trend_chart,
    weekly_chart,
    monthly_chart,
    yearly_chart,
    quarterly_chart,
    monthly_forecast,
    yearly_forecast,
    forecast_summary
)


class ProphetEngine:

    def __init__(self):

        self.forecaster = ProphetForecaster()

    # ====================================================
    # Train
    # ====================================================

    def train(

        self,

        df,

        target="Sales",

        date_column="Order Date",

        periods=30,

        frequency="D"

    ):

        train_df, forecast = self.forecaster.train_and_forecast(

            df=df,

            target=target,

            date_column=date_column,

            periods=periods,

            frequency=frequency

        )

        metrics = evaluate_forecast(

            train_df,

            forecast

        )

        return {

            "train": train_df,

            "forecast": forecast,

            "metrics": metrics,

            "rating": quality_rating(

                metrics["R2"]

            )

        }

    # ====================================================
    # Charts
    # ====================================================

    def charts(

        self,

        train_df,

        forecast

    ):

        return {

            "forecast":

                forecast_chart(forecast),

            "actual":

                actual_vs_forecast(

                    train_df,

                    forecast

                ),

            "trend":

                trend_chart(forecast),

            "weekly":

                weekly_chart(forecast),

            "monthly":

                monthly_chart(forecast),

            "yearly":

                yearly_chart(forecast),

            "quarterly":

                quarterly_chart(forecast),

            "monthly_summary":

                monthly_forecast(forecast),

            "yearly_summary":

                yearly_forecast(forecast)

        }

    # ====================================================
    # Complete Pipeline
    # ====================================================

    def run(

        self,

        df,

        target="Sales",

        date_column="Order Date",

        periods=30,

        frequency="D"

    ):

        results = self.train(

            df,

            target,

            date_column,

            periods,

            frequency

        )

        charts = self.charts(

            results["train"],

            results["forecast"]

        )

        return {

            "forecast":

                results["forecast"],

            "train":

                results["train"],

            "metrics":

                metrics_summary(

                    results["metrics"]

                ),

            "rating":

                results["rating"],

            "summary":

                forecast_summary(

                    results["forecast"]

                ),

            "figures":

                charts

        }

    # ====================================================
    # Save
    # ====================================================

    def save(

        self,

        path="models/prophet.pkl"

    ):

        Path(path).parent.mkdir(

            exist_ok=True,

            parents=True

        )

        self.forecaster.save(path)

    # ====================================================
    # Load
    # ====================================================

    def load(

        self,

        path="models/prophet.pkl"

    ):

        self.forecaster.load(path)
