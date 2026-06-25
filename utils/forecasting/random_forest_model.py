"""
===========================================================
Random Forest Forecasting Model
Part 1 - Model Configuration & Initialization

AI Business Intelligence Dashboard

Author : Daksh Shah
===========================================================
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor


# ==========================================================
# Configuration
# ==========================================================

@dataclass
class RandomForestConfig:
    """
    Configuration for Random Forest model.
    """

    n_estimators: int = 300

    max_depth: Optional[int] = 15

    min_samples_split: int = 2

    min_samples_leaf: int = 1

    max_features: str = "sqrt"

    bootstrap: bool = True

    random_state: int = 42

    n_jobs: int = -1


# ==========================================================
# Random Forest Forecaster
# ==========================================================

class RandomForestForecaster:

    def __init__(
        self,
        config: RandomForestConfig = RandomForestConfig()
    ):

        self.config = config

        self.model: Optional[
            RandomForestRegressor
        ] = None

        self.feature_names = []

        self.target_name = None

        self.training_rows = 0

        self.training_columns = 0

        self.is_trained = False

        self.training_summary = {}

    # ======================================================
    # Build Model
    # ======================================================

    def build(self):

        """
        Create RandomForestRegressor.
        """

        self.model = RandomForestRegressor(

            n_estimators=self.config.n_estimators,

            max_depth=self.config.max_depth,

            min_samples_split=self.config.min_samples_split,

            min_samples_leaf=self.config.min_samples_leaf,

            max_features=self.config.max_features,

            bootstrap=self.config.bootstrap,

            random_state=self.config.random_state,

            n_jobs=self.config.n_jobs

        )

        return self.model

    # ======================================================
    # Get Config
    # ======================================================

    def get_config(self):

        return asdict(self.config)

    # ======================================================
    # Update Config
    # ======================================================

    def update_config(
        self,
        **kwargs
    ):

        """
        Update model configuration.
        """

        for key, value in kwargs.items():

            if hasattr(self.config, key):

                setattr(
                    self.config,
                    key,
                    value
                )

        self.model = None

    # ======================================================
    # Model Information
    # ======================================================

    def model_info(self):

        return {

            "Model":

                "Random Forest",

            "Framework":

                "scikit-learn",

            "Version":

                "1.0",

            "Configuration":

                self.get_config(),

            "Trained":

                self.is_trained

        }

    # ======================================================
    # Validate Dataset
    # ======================================================

    def validate_data(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ):

        if X.empty:

            raise ValueError(
                "Feature matrix is empty."
            )

        if y is not None:

            if len(X) != len(y):

                raise ValueError(

                    "X and y lengths do not match."

                )

        return True

    # ======================================================
    # Reset Model
    # ======================================================

    def reset(self):

        """
        Reset model state.
        """

        self.model = None

        self.feature_names = []

        self.target_name = None

        self.training_rows = 0

        self.training_columns = 0

        self.training_summary = {}

        self.is_trained = False

    # ======================================================
    # Check Model
    # ======================================================

    def is_ready(self):

        """
        Check if model exists.
        """

        return self.model is not None

    # ======================================================
    # Summary
    # ======================================================

    def summary(self):

        return {

            "Rows":

                self.training_rows,

            "Columns":

                self.training_columns,

            "Features":

                len(self.feature_names),

            "Target":

                self.target_name,

            "Model Ready":

                self.is_ready(),

            "Model Trained":

                self.is_trained

        }

    # ======================================================
    # Save Configuration
    # ======================================================

    def save_config(
        self,
        path="models/random_forest_config.pkl"
    ):

        Path(path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(
            self.config,
            path
        )

    # ======================================================
    # Load Configuration
    # ======================================================

    def load_config(
        self,
        path="models/random_forest_config.pkl"
    ):

        self.config = joblib.load(path)

        return self.config
      # ==========================================================
# Training
# ==========================================================

import time
import numpy as np


class RandomForestForecaster(RandomForestForecaster):

    # ======================================================
    # Train Model
    # ======================================================

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ):
        """
        Train Random Forest model.
        """

        self.validate_data(X, y)

        if self.model is None:
            self.build()

        start_time = time.perf_counter()

        self.model.fit(X, y)

        end_time = time.perf_counter()

        self.feature_names = list(X.columns)

        self.target_name = y.name

        self.training_rows = len(X)

        self.training_columns = X.shape[1]

        self.training_summary = {

            "Rows": len(X),

            "Columns": X.shape[1],

            "Training Time (sec)": round(
                end_time - start_time,
                3
            ),

            "Target": y.name

        }

        # OOB Score (if enabled)

        if hasattr(self.model, "oob_score_"):

            try:

                self.training_summary[
                    "OOB Score"
                ] = round(
                    self.model.oob_score_,
                    4
                )

            except Exception:
                pass

        self.is_trained = True

        return self

    # ======================================================
    # Training Summary
    # ======================================================

    def training_info(self):

        return self.training_summary

    # ======================================================
    # Feature Count
    # ======================================================

    def feature_count(self):

        return len(self.feature_names)

    # ======================================================
    # Target Name
    # ======================================================

    def target(self):

        return self.target_name

    # ======================================================
    # Predict Training Data
    # ======================================================

    def predict_train(
        self,
        X: pd.DataFrame
    ):

        if not self.is_trained:

            raise RuntimeError(
                "Model has not been trained."
            )

        return self.model.predict(X)

    # ======================================================
    # Residuals
    # ======================================================

    def residuals(
        self,
        X,
        y
    ):

        pred = self.predict_train(X)

        residual = y - pred

        return pd.DataFrame({

            "Actual": y,

            "Prediction": pred,

            "Residual": residual

        })

    # ======================================================
    # Training Accuracy
    # ======================================================

    def training_score(
        self,
        X,
        y
    ):

        return self.model.score(
            X,
            y
        )

    # ======================================================
    # Basic Statistics
    # ======================================================

    def prediction_statistics(
        self,
        predictions
    ):

        predictions = np.asarray(predictions)

        return {

            "Minimum":

                predictions.min(),

            "Maximum":

                predictions.max(),

            "Average":

                predictions.mean(),

            "Median":

                np.median(predictions),

            "Std":

                predictions.std()

        }

    # ======================================================
    # Print Summary
    # ======================================================

    def print_summary(self):

        print("=" * 60)

        print("Random Forest Training Summary")

        print("=" * 60)

        for key, value in self.training_summary.items():

            print(f"{key:25}: {value}")

        print("=" * 60)
      # ==========================================================
# Prediction
# ==========================================================

from utils.forecasting.random_forest_utils import (
    append_prediction,
    rebuild_features,
    latest_feature_vector,
)


class RandomForestForecaster(RandomForestForecaster):

    # ======================================================
    # Predict
    # ======================================================

    def predict(
        self,
        X: pd.DataFrame
    ):
        """
        Predict on an existing feature matrix.
        """

        if not self.is_trained:
            raise RuntimeError(
                "Train the model before prediction."
            )

        return self.model.predict(X)

    # ======================================================
    # Predict DataFrame
    # ======================================================

    def predict_dataframe(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Return predictions as a DataFrame.
        """

        predictions = self.predict(X)

        return pd.DataFrame(
            {
                "Prediction": predictions
            }
        )

    # ======================================================
    # Recursive Forecast
    # ======================================================

    def recursive_forecast(
        self,
        history: pd.DataFrame,
        periods: int,
        feature_columns
    ):
        """
        Multi-step recursive forecasting.

        Parameters
        ----------
        history : DataFrame
            Must contain:
                ds
                y

        periods : int

        feature_columns : list
        """

        if not self.is_trained:

            raise RuntimeError(
                "Model has not been trained."
            )

        history = history.copy()

        forecasts = []

        for _ in range(periods):

            X = latest_feature_vector(
                history,
                feature_columns
            )

            prediction = float(
                self.model.predict(X)[0]
            )

            next_date = (
                history["ds"].max()
                + pd.Timedelta(days=1)
            )

            forecasts.append(
                {
                    "Date": next_date,
                    "Forecast": prediction
                }
            )

            history = append_prediction(
                history,
                prediction,
                next_date
            )

            history = rebuild_features(
                history
            )

        return pd.DataFrame(
            forecasts
        )

    # ======================================================
    # Batch Forecast
    # ======================================================

    def forecast(
        self,
        history,
        periods,
        feature_columns
    ):

        return self.recursive_forecast(
            history,
            periods,
            feature_columns
        )

    # ======================================================
    # Predict Single Sample
    # ======================================================

    def predict_single(
        self,
        row: pd.DataFrame
    ) -> float:

        prediction = self.predict(row)

        return float(prediction[0])

    # ======================================================
    # Prediction Interval (Approximation)
    # ======================================================

    def prediction_interval(
        self,
        X: pd.DataFrame,
        confidence: float = 0.95
    ):
        """
        Approximate prediction interval using
        individual tree predictions.
        """

        if not self.is_trained:
            raise RuntimeError(
                "Model has not been trained."
            )

        tree_predictions = np.array(
            [
                tree.predict(X)
                for tree in self.model.estimators_
            ]
        )

        lower = np.percentile(
            tree_predictions,
            (1 - confidence) / 2 * 100,
            axis=0
        )

        upper = np.percentile(
            tree_predictions,
            (1 + confidence) / 2 * 100,
            axis=0
        )

        mean = tree_predictions.mean(axis=0)

        return pd.DataFrame(
            {
                "Prediction": mean,
                "Lower": lower,
                "Upper": upper
            }
        )

    # ======================================================
    # Predict Last Observation
    # ======================================================

    def predict_latest(
        self,
        X: pd.DataFrame
    ):

        latest = X.tail(1)

        return self.predict_single(
            latest
        )

    # ======================================================
    # Forecast Summary
    # ======================================================

    def forecast_summary(
        self,
        forecast: pd.DataFrame
    ):

        return {

            "Periods":

                len(forecast),

            "Minimum":

                forecast["Forecast"].min(),

            "Maximum":

                forecast["Forecast"].max(),

            "Average":

                forecast["Forecast"].mean(),

            "Median":

                forecast["Forecast"].median()

        }
      # ==========================================================
# Feature Importance
# ==========================================================

from sklearn.inspection import permutation_importance


class RandomForestForecaster(RandomForestForecaster):

    # ======================================================
    # Feature Importance
    # ======================================================

    def feature_importance(
        self,
        normalize=True
    ):
        """
        Return feature importance.
        """

        if not self.is_trained:

            raise RuntimeError(
                "Model has not been trained."
            )

        importance = self.model.feature_importances_

        df = pd.DataFrame({

            "Feature":

                self.feature_names,

            "Importance":

                importance

        })

        df = df.sort_values(

            "Importance",

            ascending=False

        ).reset_index(drop=True)

        if normalize:

            df["Importance (%)"] = (

                df["Importance"]

                / df["Importance"].sum()

            ) * 100

        df["Rank"] = np.arange(

            1,

            len(df) + 1

        )

        return df

    # ======================================================
    # Top Features
    # ======================================================

    def top_features(
        self,
        n=20
    ):

        return self.feature_importance().head(n)

    # ======================================================
    # Bottom Features
    # ======================================================

    def bottom_features(
        self,
        n=20
    ):

        return self.feature_importance().tail(n)

    # ======================================================
    # Cumulative Importance
    # ======================================================

    def cumulative_importance(
        self
    ):

        df = self.feature_importance()

        df["Cumulative"] = (

            df["Importance (%)"]

            .cumsum()

        )

        return df

    # ======================================================
    # Important Feature List
    # ======================================================

    def important_features(
        self,
        threshold=1.0
    ):
        """
        Return features above
        threshold percentage.
        """

        df = self.feature_importance()

        return df[

            df["Importance (%)"]

            >= threshold

        ]

    # ======================================================
    # Least Important Features
    # ======================================================

    def removable_features(
        self,
        threshold=0.10
    ):

        df = self.feature_importance()

        return df[

            df["Importance (%)"]

            < threshold

        ]

    # ======================================================
    # Permutation Importance
    # ======================================================

    def permutation_importance(
        self,
        X,
        y,
        repeats=10
    ):

        result = permutation_importance(

            self.model,

            X,

            y,

            n_repeats=repeats,

            random_state=42,

            n_jobs=-1

        )

        df = pd.DataFrame({

            "Feature":

                X.columns,

            "Importance":

                result.importances_mean,

            "Std":

                result.importances_std

        })

        return df.sort_values(

            "Importance",

            ascending=False

        )

    # ======================================================
    # Importance Dictionary
    # ======================================================

    def importance_dict(
        self
    ):

        df = self.feature_importance()

        return dict(

            zip(

                df["Feature"],

                df["Importance (%)"]

            )

        )

    # ======================================================
    # Feature Summary
    # ======================================================

    def importance_summary(
        self
    ):

        df = self.feature_importance()

        return {

            "Total Features":

                len(df),

            "Most Important":

                df.iloc[0]["Feature"],

            "Highest Importance":

                df.iloc[0]["Importance (%)"],

            "Average Importance":

                df["Importance (%)"].mean()

        }

    # ======================================================
    # Export Importance
    # ======================================================

    def export_importance(
        self,
        path="reports/feature_importance.csv"
    ):

        Path(path).parent.mkdir(

            parents=True,

            exist_ok=True

        )

        self.feature_importance().to_csv(

            path,

            index=False

        )

        return path
      # ==========================================================
# Save / Load
# ==========================================================

import platform
import sklearn
from datetime import datetime


class RandomForestForecaster(RandomForestForecaster):

    # ======================================================
    # Create Model Package
    # ======================================================

    def model_package(self):
        """
        Create a complete model package.
        """

        if not self.is_trained:
            raise RuntimeError(
                "Model has not been trained."
            )

        package = {

            "model":
                self.model,

            "config":
                self.get_config(),

            "feature_names":
                self.feature_names,

            "target":
                self.target_name,

            "training_summary":
                self.training_summary,

            "metadata": {

                "created":
                    datetime.now().isoformat(),

                "model":
                    "Random Forest",

                "version":
                    "1.0",

                "framework":
                    "scikit-learn",

                "sklearn_version":
                    sklearn.__version__,

                "python":
                    platform.python_version()

            }

        }

        return package

    # ======================================================
    # Save Model
    # ======================================================

    def save(
        self,
        path="models/random_forest.pkl"
    ):
        """
        Save complete model package.
        """

        if not self.is_trained:

            raise RuntimeError(
                "Train model before saving."
            )

        Path(path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        package = self.model_package()

        joblib.dump(
            package,
            path
        )

        return path

    # ======================================================
    # Load Model
    # ======================================================

    def load(
        self,
        path="models/random_forest.pkl"
    ):
        """
        Load complete model package.
        """

        package = joblib.load(path)

        self.model = package["model"]

        config = package["config"]

        self.config = RandomForestConfig(
            **config
        )

        self.feature_names = package[
            "feature_names"
        ]

        self.target_name = package[
            "target"
        ]

        self.training_summary = package[
            "training_summary"
        ]

        self.is_trained = True

        return self

    # ======================================================
    # Metadata
    # ======================================================

    def metadata(self):

        package = self.model_package()

        return package["metadata"]

    # ======================================================
    # Export Config
    # ======================================================

    def export_config(self):

        return pd.DataFrame({

            "Parameter":

                self.get_config().keys(),

            "Value":

                self.get_config().values()

        })

    # ======================================================
    # Feature Names
    # ======================================================

    def features(self):

        return self.feature_names

    # ======================================================
    # Training Summary DataFrame
    # ======================================================

    def summary_dataframe(self):

        return pd.DataFrame({

            "Metric":

                self.training_summary.keys(),

            "Value":

                self.training_summary.values()

        })

    # ======================================================
    # Check Feature Compatibility
    # ======================================================

    def validate_features(
        self,
        X
    ):
        """
        Ensure inference data matches
        training features.
        """

        missing = [

            col

            for col in self.feature_names

            if col not in X.columns

        ]

        if missing:

            raise ValueError(

                f"Missing features: {missing}"

            )

        return True

    # ======================================================
    # Align Feature Order
    # ======================================================

    def align_features(
        self,
        X
    ):
        """
        Align columns to training order.
        """

        self.validate_features(X)

        return X[
            self.feature_names
        ]

    # ======================================================
    # Predict Safe
    # ======================================================

    def predict_safe(
        self,
        X
    ):
        """
        Prediction with automatic
        feature alignment.
        """

        X = self.align_features(X)

        return self.predict(X)
          # ======================================================
    # Evaluate
    # ======================================================

    def evaluate(
        self,
        X_test,
        y_test
    ):
        """
        Evaluate model performance.
        """

        from sklearn.metrics import (
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )

        predictions = self.predict(X_test)

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        mse = mean_squared_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(mse)

        r2 = r2_score(
            y_test,
            predictions
        )

        return {

            "MAE": mae,

            "MSE": mse,

            "RMSE": rmse,

            "R2": r2

        }

    # ======================================================
    # Run Complete Pipeline
    # ======================================================

    def run(
        self,
        dataset,
        periods=30
    ):
        """
        Complete Random Forest workflow.
        """

        X_train = dataset["X_train"]
        X_test = dataset["X_test"]

        y_train = dataset["y_train"]
        y_test = dataset["y_test"]

        history = dataset["dataset"][
            ["ds", "y"]
        ].copy()

        # Train

        self.fit(
            X_train,
            y_train
        )

        # Metrics

        metrics = self.evaluate(
            X_test,
            y_test
        )

        # Test Predictions

        predictions = self.predict(
            X_test
        )

        prediction_df = pd.DataFrame({

            "Actual": y_test.values,

            "Prediction": predictions

        })

        # Future Forecast

        future = self.forecast(

            history,

            periods,

            self.feature_names

        )

        # Feature Importance

        importance = self.feature_importance()

        return {

            "model": self,

            "metrics": metrics,

            "training": self.training_summary,

            "predictions": prediction_df,

            "forecast": future,

            "importance": importance,

            "summary": self.summary()

        }

    # ======================================================
    # Quick Forecast
    # ======================================================

    @classmethod
    def quick_forecast(
        cls,
        dataset,
        periods=30
    ):
        """
        Train and forecast with one call.
        """

        model = cls()

        return model.run(
            dataset,
            periods
        )
      
