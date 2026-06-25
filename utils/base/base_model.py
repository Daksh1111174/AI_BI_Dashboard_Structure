"""
===========================================================
AI Business Intelligence

Base Model

Author : Daksh Shah
===========================================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

import joblib
import pandas as pd
import platform
import sklearn


# ============================================================
# Base Model
# ============================================================

class BaseModel(ABC):
    """
    Abstract base class for all ML models.

    Every forecasting model should inherit this class.

    Example
    -------
    RandomForestModel(BaseModel)

    XGBoostModel(BaseModel)

    ProphetModel(BaseModel)

    LinearRegressionModel(BaseModel)
    """

    VERSION = "1.0.0"

    # --------------------------------------------------------

    def __init__(
        self,
        model_name: str
    ):

        self.model_name = model_name

        self.model = None

        self.feature_names = []

        self.target_name = None

        self.training_summary = {}

        self.metadata = {}

        self.is_trained = False

        self.created_at = datetime.now()
          # ========================================================
    # Configuration
    # ========================================================

    def set_config(
        self,
        config: Dict[str, Any]
    ) -> None:
        """
        Store model configuration.
        """

        self.config = config

    def get_config(self) -> Dict[str, Any]:
        """
        Return model configuration.
        """

        return getattr(self, "config", {})

    # ========================================================
    # Export Metadata
    # ========================================================

    def export_metadata(
        self
    ) -> pd.DataFrame:
        """
        Export metadata as DataFrame.
        """

        metadata = self.build_metadata()

        return pd.DataFrame(
            {
                "Property": metadata.keys(),
                "Value": metadata.values()
            }
        )

    # ========================================================
    # Export Training Summary
    # ========================================================

    def export_training_summary(
        self
    ) -> pd.DataFrame:
        """
        Export training summary.
        """

        return pd.DataFrame(
            {
                "Metric": self.training_summary.keys(),
                "Value": self.training_summary.values()
            }
        )

    # ========================================================
    # Health Check
    # ========================================================

    def health_check(self) -> Dict[str, Any]:
        """
        Verify model readiness.
        """

        return {

            "Model Exists":
                self.model is not None,

            "Model Trained":
                self.is_trained,

            "Feature Count":
                len(self.feature_names),

            "Target":
                self.target_name,

            "Metadata":
                bool(self.metadata)

        }

    # ========================================================
    # Clone Model
    # ========================================================

    def clone(self):
        """
        Deep copy model object.
        """

        import copy

        return copy.deepcopy(self)

    # ========================================================
    # Compare Metadata
    # ========================================================

    def compare(
        self,
        other_model
    ) -> Dict[str, Any]:
        """
        Compare two models.
        """

        return {

            "Model A":
                self.model_name,

            "Model B":
                other_model.model_name,

            "Same Version":
                self.VERSION == other_model.VERSION,

            "Same Features":
                self.feature_names == other_model.feature_names,

            "Same Target":
                self.target_name == other_model.target_name

        }

    # ========================================================
    # String Representation
    # ========================================================

    def __repr__(self):

        return (

            f"{self.model_name}"

            f"(trained={self.is_trained}, "

            f"features={len(self.feature_names)})"

        )

    # ========================================================
    # Status
    # ========================================================

    @property
    def status(self):

        return "Trained" if self.is_trained else "Not Trained"

    # ========================================================
    # Ready
    # ========================================================

    @property
    def ready(self):

        return self.model is not None

    # ========================================================
    # Number of Features
    # ========================================================

    @property
    def n_features(self):

        return len(self.feature_names)

    # ========================================================
    # Number of Samples
    # ========================================================

    @property
    def n_samples(self):

        return self.training_summary.get(
            "Rows",
            0
        )

    # ========================================================
    # Serialize
    # ========================================================

    def to_dict(self):

        return {

            "model":
                self.model_name,

            "trained":
                self.is_trained,

            "features":
                self.feature_names,

            "target":
                self.target_name,

            "metadata":
                self.build_metadata(),

            "training":
                self.training_summary

        }

    # ========================================================
    # Deserialize Metadata
    # ========================================================

    def from_dict(
        self,
        data: Dict[str, Any]
    ):

        self.feature_names = data.get(
            "features",
            []
        )

        self.target_name = data.get(
            "target"
        )

        self.training_summary = data.get(
            "training",
            {}
        )

        self.metadata = data.get(
            "metadata",
            {}
        )

        self.is_trained = data.get(
            "trained",
            False
        )

        return self

    # ========================================================
    # Equality
    # ========================================================

    def __eq__(
        self,
        other
    ):

        if not isinstance(
            other,
            BaseModel
        ):

            return False

        return (

            self.model_name == other.model_name

            and

            self.feature_names == other.feature_names

            and

            self.target_name == other.target_name

        )

    # ========================================================
    # Abstract Methods
    # ========================================================

    @abstractmethod
    def build_model(self):
        """
        Build ML model.
        """
        pass

    @abstractmethod
    def fit(
        self,
        X,
        y
    ):
        """
        Train model.
        """
        pass

    @abstractmethod
    def predict(
        self,
        X
    ):
        """
        Predict values.
        """
        pass

    # ========================================================
    # Metadata
    # ========================================================

    def build_metadata(self):

        self.metadata = {

            "model":

                self.model_name,

            "version":

                self.VERSION,

            "created":

                self.created_at.isoformat(),

            "python":

                platform.python_version(),

            "sklearn":

                sklearn.__version__,

            "trained":

                self.is_trained

        }

        return self.metadata

    # ========================================================
    # Feature Information
    # ========================================================

    def set_features(
        self,
        columns
    ):

        self.feature_names = list(columns)

    def get_features(self):

        return self.feature_names

    # ========================================================
    # Target
    # ========================================================

    def set_target(
        self,
        target
    ):

        self.target_name = target

    def get_target(self):

        return self.target_name

    # ========================================================
    # Model Information
    # ========================================================

    def info(self):

        return {

            "Model":

                self.model_name,

            "Version":

                self.VERSION,

            "Trained":

                self.is_trained,

            "Features":

                len(self.feature_names),

            "Target":

                self.target_name

        }

    # ========================================================
    # Validate Dataset
    # ========================================================

    def validate_dataset(
        self,
        X,
        y=None
    ):

        if X is None:

            raise ValueError(

                "Feature matrix is None."

            )

        if len(X) == 0:

            raise ValueError(

                "Dataset is empty."

            )

        if y is not None:

            if len(X) != len(y):

                raise ValueError(

                    "X and y length mismatch."

                )

        return True

    # ========================================================
    # Save
    # ========================================================

    def save(
        self,
        path
    ):

        if not self.is_trained:

            raise RuntimeError(

                "Train model before saving."

            )

        Path(path).parent.mkdir(

            parents=True,

            exist_ok=True

        )

        package = {

            "model":

                self.model,

            "metadata":

                self.build_metadata(),

            "features":

                self.feature_names,

            "target":

                self.target_name,

            "training":

                self.training_summary

        }

        joblib.dump(

            package,

            path

        )

        return path

    # ========================================================
    # Load
    # ========================================================

    def load(
        self,
        path
    ):

        package = joblib.load(path)

        self.model = package["model"]

        self.metadata = package["metadata"]

        self.feature_names = package["features"]

        self.target_name = package["target"]

        self.training_summary = package["training"]

        self.is_trained = True

        return self

    # ========================================================
    # Reset
    # ========================================================

    def reset(self):

        self.model = None

        self.feature_names = []

        self.target_name = None

        self.training_summary = {}

        self.metadata = {}

        self.is_trained = False

    # ========================================================
    # Training Summary
    # ========================================================

    def summary(self):

        return self.training_summary
