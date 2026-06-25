"""
===========================================================
Random Forest Forecast Engine
Part 1

AI Business Intelligence Dashboard

Author : Daksh Shah
===========================================================
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Dict

from utils.forecasting.random_forest_utils import (
    build_random_forest_dataset
)

from utils.forecasting.random_forest_model import (
    RandomForestForecaster
)

from utils.forecasting.random_forest_metrics import (
    RandomForestMetrics
)

from utils.forecasting.random_forest_visualization import (
    RandomForestVisualization
)


# ==========================================================
# Random Forest Engine
# ==========================================================

class RandomForestEngine:

    def __init__(self):

        self.model = RandomForestForecaster()

        self.metrics = RandomForestMetrics()

        self.visualization = RandomForestVisualization()

        self.dataset = None

        self.results = {}

    # ======================================================
    # Build Dataset
    # ======================================================

    def prepare_dataset(

        self,

        dataframe,

        date_column="Order Date",

        target_column="Sales"

    ):

        self.dataset = build_random_forest_dataset(

            dataframe,

            date_column,

            target_column

        )

        return self.dataset

    # ======================================================
    # Train
    # ======================================================

    def train(self):

        self.model.fit(

            self.dataset["X_train"],

            self.dataset["y_train"]

        )

        return self.model

    # ======================================================
    # Predict Test
    # ======================================================

    def predict_test(self):

        prediction = self.model.predict(

            self.dataset["X_test"]

        )

        return prediction

    # ======================================================
    # Evaluate
    # ======================================================

    def evaluate(self):

        prediction = self.predict_test()

        return self.metrics.evaluate(

            self.dataset["y_test"],

            prediction

        )

    # ======================================================
    # Forecast
    # ======================================================

    def forecast(

        self,

        periods=30

    ):

        history = self.dataset["dataset"][

            ["ds", "y"]

        ].copy()

        forecast = self.model.forecast(

            history,

            periods,

            self.model.feature_names

        )

        return forecast

    # ======================================================
    # Importance
    # ======================================================

    def importance(self):

        return self.model.feature_importance()

    # ======================================================
    # Dashboard
    # ======================================================

    def dashboard(

        self,

        forecast,

        metrics

    ):

        dashboard = self.visualization.executive_dashboard(

            self.dataset["dataset"],

            forecast,

            metrics

        )

        return dashboard

    # ======================================================
    # Summary
    # ======================================================

    def summary(self):

        return {

            "Training":

                self.model.training_info(),

            "Metrics":

                self.results["metrics"]

        }
          # ======================================================
    # Run Complete Pipeline
    # ======================================================

    def run(
        self,
        dataframe,
        date_column="Order Date",
        target_column="Sales",
        periods=30
    ):
        """
        Execute the complete Random Forest pipeline.
        """

        # Prepare data
        self.prepare_dataset(
            dataframe,
            date_column,
            target_column
        )

        # Train model
        self.train()

        # Test predictions
        predictions = self.predict_test()

        # Metrics
        metrics = self.metrics.evaluate(
            self.dataset["y_test"],
            predictions
        )

        # Forecast
        forecast = self.forecast(
            periods=periods
        )

        # Feature Importance
        importance = self.importance()

        # Charts
        charts = self.create_charts(
            predictions,
            forecast,
            importance
        )

        self.results = {

            "model": self.model,

            "metrics": metrics,

            "forecast": forecast,

            "predictions": predictions,

            "importance": importance,

            "charts": charts,

            "training":

                self.model.training_info()

        }

        return self.results

    # ======================================================
    # Create Charts
    # ======================================================

    def create_charts(
        self,
        predictions,
        forecast,
        importance
    ):

        y_test = self.dataset["y_test"]

        history = self.dataset["dataset"]

        charts = {

            "actual_prediction":

                self.visualization.actual_vs_prediction(

                    y_test,

                    predictions

                ),

            "scatter":

                self.visualization.prediction_scatter(

                    y_test,

                    predictions

                ),

            "residual":

                self.visualization.residual_plot(

                    y_test,

                    predictions

                ),

            "importance":

                self.visualization.feature_importance(

                    importance

                ),

            "forecast":

                self.visualization.forecast_plot(

                    history,

                    forecast

                ),

            "kpi":

                self.visualization.kpi_dashboard(

                    self.results.get("metrics", {})

                )

        }

        return charts

    # ======================================================
    # Save Model
    # ======================================================

    def save_model(
        self,
        path="models/random_forest.pkl"
    ):

        return self.model.save(path)

    # ======================================================
    # Load Model
    # ======================================================

    def load_model(
        self,
        path="models/random_forest.pkl"
    ):

        self.model.load(path)

        return self.model

    # ======================================================
    # Export Forecast
    # ======================================================

    def export_forecast(
        self,
        path="reports/forecast.csv"
    ):

        Path(path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.results["forecast"].to_csv(
            path,
            index=False
        )

        return path

    # ======================================================
    # Export Metrics
    # ======================================================

    def export_metrics(
        self,
        path="reports/metrics.csv"
    ):

        self.metrics.export_csv(path)

        return path

    # ======================================================
    # Export Charts
    # ======================================================

    def export_charts(
        self,
        folder="reports/charts"
    ):

        return self.visualization.save_dashboard(

            self.results["charts"],

            folder

        )

    # ======================================================
    # Export Everything
    # ======================================================

    def export_all(
        self,
        folder="reports"
    ):

        Path(folder).mkdir(
            parents=True,
            exist_ok=True
        )

        outputs = {

            "forecast":

                self.export_forecast(

                    f"{folder}/forecast.csv"

                ),

            "metrics":

                self.export_metrics(

                    f"{folder}/metrics.csv"

                ),

            "charts":

                self.export_charts(

                    f"{folder}/charts"

                )

        }

        return outputs

    # ======================================================
    # Batch Forecast
    # ======================================================

    def batch_forecast(
        self,
        datasets,
        date_column="Order Date",
        target_column="Sales",
        periods=30
    ):

        results = {}

        for name, df in datasets.items():

            results[name] = self.run(

                dataframe=df,

                date_column=date_column,

                target_column=target_column,

                periods=periods

            )

        return results

    # ======================================================
    # Streamlit Helper
    # ======================================================

    @staticmethod
    def display_dashboard(
        st,
        charts
    ):

        for title, fig in charts.items():

            st.subheader(
                title.replace("_", " ").title()
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # ======================================================
    # Reset Engine
    # ======================================================

    def reset(self):

        self.dataset = None

        self.results = {}

        self.model.reset()

        self.metrics.reset()

    # ======================================================
    # Engine Information
    # ======================================================

    def info(self):

        return {

            "Engine":

                "Random Forest Forecast Engine",

            "Version":

                "1.0",

            "Model":

                "RandomForestRegressor",

            "Status":

                "Ready"

        }
      
