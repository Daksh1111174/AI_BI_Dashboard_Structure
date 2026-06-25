"""
===========================================================
AI Business Intelligence

Base Forecaster

Author : Daksh Shah
===========================================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from utils.base.base_model import BaseModel
from utils.base.base_metrics import BaseMetrics
from utils.base.base_visualization import BaseVisualization


# ==========================================================
# Base Forecaster
# ==========================================================

class BaseForecaster(BaseModel, ABC):
    """
    Base class for every forecasting model.

    Inherited by:

        RandomForestForecaster

        XGBoostForecaster

        ProphetForecaster

        LinearRegressionForecaster
    """

    def __init__(
        self,
        model_name: str,
        metrics: Optional[BaseMetrics] = None,
        visualization: Optional[BaseVisualization] = None
    ):

        super().__init__(model_name)

        self.metrics = metrics or BaseMetrics()

        self.visualization = (

            visualization

            or

            BaseVisualization()

        )

        self.X_train = None

        self.X_test = None

        self.y_train = None

        self.y_test = None

        self.predictions = None

        self.forecast_result = None
          # ======================================================
    # Forecast
    # ======================================================

    def forecast(
        self,
        X_future
    ):
        """
        Forecast future observations.

        Parameters
        ----------
        X_future : pandas.DataFrame

        Returns
        -------
        numpy.ndarray
        """

        if not self.is_trained:
            raise RuntimeError(
                "Model has not been trained."
            )

        self.forecast_result = self.model.predict(
            X_future
        )

        return self.forecast_result

    # ======================================================
    # Recursive Forecast
    # ======================================================

    def recursive_forecast(
        self,
        initial_features,
        periods,
        update_function
    ):
        """
        Generic recursive forecasting.

        update_function receives:

            current_features,
            prediction

        and returns

            updated_features
        """

        if not self.is_trained:

            raise RuntimeError(
                "Model not trained."
            )

        current = initial_features.copy()

        forecasts = []

        for _ in range(periods):

            prediction = self.model.predict(current)[0]

            forecasts.append(prediction)

            current = update_function(
                current,
                prediction
            )

        self.forecast_result = np.array(
            forecasts
        )

        return self.forecast_result

    # ======================================================
    # Forecast DataFrame
    # ======================================================

    def forecast_dataframe(
        self,
        dates=None,
        column_name="Forecast"
    ):

        if self.forecast_result is None:

            raise RuntimeError(
                "Run forecast() first."
            )

        df = pd.DataFrame(
            {
                column_name:
                    self.forecast_result
            }
        )

        if dates is not None:

            df.insert(
                0,
                "Date",
                dates
            )

        return df

    # ======================================================
    # Forecast Summary
    # ======================================================

    def forecast_summary(self):

        if self.forecast_result is None:

            return {}

        values = np.asarray(
            self.forecast_result
        )

        return {

            "Count":
                len(values),

            "Minimum":
                float(values.min()),

            "Maximum":
                float(values.max()),

            "Average":
                float(values.mean()),

            "Standard Deviation":
                float(values.std())

        }

    # ======================================================
    # Forecast Confidence Interval
    # ======================================================

    def confidence_interval(
        self,
        confidence=0.95
    ):
        """
        Approximate confidence interval
        based on forecast distribution.

        Override in subclasses if the
        model provides native intervals.
        """

        if self.forecast_result is None:

            raise RuntimeError(
                "Forecast unavailable."
            )

        values = np.asarray(
            self.forecast_result
        )

        mean = values.mean()

        std = values.std()

        z = 1.96 if confidence >= 0.95 else 1.64

        return {

            "Lower":
                mean - z * std,

            "Upper":
                mean + z * std

        }

    # ======================================================
    # Forecast History
    # ======================================================

    def forecast_history(self):

        return {

            "Training Rows":
                len(self.X_train)
                if self.X_train is not None
                else 0,

            "Testing Rows":
                len(self.X_test)
                if self.X_test is not None
                else 0,

            "Forecast Length":
                len(self.forecast_result)
                if self.forecast_result is not None
                else 0

        }

    # ======================================================
    # Forecast Ready
    # ======================================================

    @property
    def forecast_ready(self):

        return self.forecast_result is not None
          # ======================================================
    # Results Dictionary
    # ======================================================

    def results(self):

        return {

            "model": self.model_name,

            "trained": self.is_trained,

            "metrics": self.metrics.to_dict(),

            "training": self.training_summary,

            "forecast_summary": self.forecast_summary(),

            "forecast_ready": self.forecast_ready

        }

    # ======================================================
    # Evaluation DataFrame
    # ======================================================

    def evaluation_dataframe(self):

        return self.metrics.to_dataframe()

    # ======================================================
    # Training Data
    # ======================================================

    def training_data(self):

        return {

            "X_train": self.X_train,

            "X_test": self.X_test,

            "y_train": self.y_train,

            "y_test": self.y_test

        }

    # ======================================================
    # Generate Evaluation Charts
    # ======================================================

    def evaluation_charts(self):

        if self.predictions is None:

            raise RuntimeError(
                "Run predict() first."
            )

        return {

            "actual_prediction":

                self.visualization.line_chart(

                    self.prediction_dataframe(),

                    x=self.prediction_dataframe().index,

                    y="Actual",

                    title="Actual Values"

                ),

            "prediction":

                self.visualization.line_chart(

                    self.prediction_dataframe(),

                    x=self.prediction_dataframe().index,

                    y="Prediction",

                    title="Predicted Values"

                ),

            "scatter":

                self.visualization.scatter_chart(

                    self.prediction_dataframe(),

                    x="Actual",

                    y="Prediction",

                    title="Actual vs Prediction"

                )

        }

    # ======================================================
    # Export Metrics
    # ======================================================

    def export_metrics(
        self,
        path="reports/metrics.csv"
    ):

        return self.metrics.export_csv(path)

    # ======================================================
    # Export Forecast
    # ======================================================

    def export_forecast(
        self,
        path="reports/forecast.csv"
    ):

        df = self.forecast_dataframe()

        df.to_csv(

            path,

            index=False

        )

        return path

    # ======================================================
    # Ready Status
    # ======================================================

    @property
    def ready(self):

        return (

            self.is_trained

            and

            self.predictions is not None

        )

    # ======================================================
    # String Representation
    # ======================================================

    def __repr__(self):

        return (

            f"{self.model_name}("

            f"trained={self.is_trained}, "

            f"forecast_ready={self.forecast_ready})"

        )
          # ======================================================
    # Pipeline Hooks
    # ======================================================

    def before_fit(self):
        """
        Hook executed before training.
        Override in subclasses if needed.
        """
        pass

    def after_fit(self):
        """
        Hook executed after training.
        """
        pass

    def before_predict(self):
        """
        Hook executed before prediction.
        """
        pass

    def after_predict(self):
        """
        Hook executed after prediction.
        """
        pass

    def before_forecast(self):
        """
        Hook executed before forecasting.
        """
        pass

    def after_forecast(self):
        """
        Hook executed after forecasting.
        """
        pass

    # ======================================================
    # Complete Training Pipeline
    # ======================================================

    def train_pipeline(
        self,
        X,
        y,
        test_size=0.2,
        random_state=42
    ):
        """
        Complete training workflow.
        """

        self.prepare_dataset(
            X=X,
            y=y,
            test_size=test_size,
            random_state=random_state
        )

        self.before_fit()

        self.fit()

        self.after_fit()

        return self

    # ======================================================
    # Prediction Pipeline
    # ======================================================

    def prediction_pipeline(
        self
    ):
        """
        Complete prediction workflow.
        """

        self.before_predict()

        predictions = self.predict()

        metrics = self.evaluate()

        self.after_predict()

        return {

            "predictions": predictions,

            "metrics": metrics

        }

    # ======================================================
    # Forecast Pipeline
    # ======================================================

    def forecast_pipeline(
        self,
        X_future
    ):
        """
        Complete forecasting workflow.
        """

        self.before_forecast()

        forecast = self.forecast(
            X_future
        )

        summary = self.forecast_summary()

        self.after_forecast()

        return {

            "forecast": forecast,

            "summary": summary

        }

    # ======================================================
    # Complete Pipeline
    # ======================================================

    def run(
        self,
        X,
        y,
        X_future=None,
        test_size=0.2,
        random_state=42
    ):
        """
        Execute the full forecasting pipeline.
        """

        self.train_pipeline(
            X=X,
            y=y,
            test_size=test_size,
            random_state=random_state
        )

        prediction_results = self.prediction_pipeline()

        results = {

            "model": self.model_name,

            "training": self.training_summary,

            "predictions": prediction_results["predictions"],

            "metrics": prediction_results["metrics"],

            "evaluation": self.evaluation_dataframe()

        }

        if X_future is not None:

            results["forecast"] = self.forecast_pipeline(
                X_future
            )

        return results

    # ======================================================
    # Can Forecast
    # ======================================================

    @property
    def can_forecast(self):

        return self.is_trained

    # ======================================================
    # Can Evaluate
    # ======================================================

    @property
    def can_evaluate(self):

        return (

            self.is_trained

            and

            self.X_test is not None

        )

    # ======================================================
    # Clear Results
    # ======================================================

    def clear_results(self):

        self.predictions = None

        self.forecast_result = None

    # ======================================================
    # Pipeline Status
    # ======================================================

    def pipeline_status(self):

        return {

            "trained": self.is_trained,

            "predictions": self.predictions is not None,

            "forecast": self.forecast_result is not None,

            "evaluation":

                self.metrics.metric_count

        }

    # ======================================================
    # Health Check
    # ======================================================

    def health(self):

        return {

            **self.health_check(),

            **self.pipeline_status()

        }
      

    # ======================================================
    # Abstract Methods
    # ======================================================

    @abstractmethod
    def build_model(self):
        """
        Create forecasting model.
        """
        pass

    # ======================================================
    # Prepare Dataset
    # ======================================================

    def prepare_dataset(
        self,
        X,
        y,
        test_size=0.2,
        random_state=42
    ):

        self.validate_dataset(
            X,
            y
        )

        self.set_features(

            X.columns

        )

        self.set_target(

            y.name

        )

        (

            self.X_train,

            self.X_test,

            self.y_train,

            self.y_test

        ) = train_test_split(

            X,

            y,

            test_size=test_size,

            random_state=random_state,

            shuffle=False

        )

        return self

    # ======================================================
    # Fit
    # ======================================================

    def fit(
        self,
        X=None,
        y=None
    ):

        if X is None:

            X = self.X_train

            y = self.y_train

        self.validate_dataset(
            X,
            y
        )

        if self.model is None:

            self.build_model()

        self.model.fit(
            X,
            y
        )

        self.is_trained = True

        self.training_summary = {

            "Rows":

                len(X),

            "Columns":

                X.shape[1],

            "Target":

                y.name

        }

        return self

    # ======================================================
    # Predict
    # ======================================================

    def predict(
        self,
        X=None
    ):

        if not self.is_trained:

            raise RuntimeError(

                "Model has not been trained."

            )

        if X is None:

            X = self.X_test

        self.predictions = self.model.predict(

            X

        )

        return self.predictions

    # ======================================================
    # Evaluate
    # ======================================================

    def evaluate(self):

        prediction = self.predict()

        return self.metrics.regression_metrics(

            self.y_test,

            prediction

        )

    # ======================================================
    # Residuals
    # ======================================================

    def residuals(self):

        return self.metrics.residuals(

            self.y_test,

            self.predictions

        )

    # ======================================================
    # Prediction DataFrame
    # ======================================================

    def prediction_dataframe(self):

        return pd.DataFrame(

            {

                "Actual":

                    self.y_test.values,

                "Prediction":

                    self.predictions

            }

        )

    # ======================================================
    # Score
    # ======================================================

    def score(self):

        return self.model.score(

            self.X_test,

            self.y_test

        )
      
