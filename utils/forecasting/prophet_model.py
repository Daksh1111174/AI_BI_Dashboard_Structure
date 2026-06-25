"""
===========================================================
Prophet Forecasting Model
AI Business Intelligence Dashboard

Author : Daksh Shah
===========================================================
"""

from pathlib import Path

import joblib
import pandas as pd
from prophet import Prophet

from utils.forecasting.prophet_utils import (
    prepare_prophet_dataframe,
    create_future_dataframe,
    load_holidays
)


class ProphetForecaster:

    def __init__(
        self,
        growth="linear",
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        holidays_country="IN"
    ):

        self.model = None

        self.holidays_country = holidays_country

        self.config = {

            "growth": growth,

            "yearly_seasonality": yearly_seasonality,

            "weekly_seasonality": weekly_seasonality,

            "daily_seasonality": daily_seasonality,

            "seasonality_mode": seasonality_mode,

            "changepoint_prior_scale": changepoint_prior_scale,

            "seasonality_prior_scale": seasonality_prior_scale

        }

    # =======================================================
    # Build Model
    # =======================================================

    def build(self):

        holidays = load_holidays(
            country=self.holidays_country
        )

        self.model = Prophet(

            holidays=holidays,

            growth=self.config["growth"],

            yearly_seasonality=self.config[
                "yearly_seasonality"
            ],

            weekly_seasonality=self.config[
                "weekly_seasonality"
            ],

            daily_seasonality=self.config[
                "daily_seasonality"
            ],

            seasonality_mode=self.config[
                "seasonality_mode"
            ],

            changepoint_prior_scale=self.config[
                "changepoint_prior_scale"
            ],

            seasonality_prior_scale=self.config[
                "seasonality_prior_scale"
            ]

        )

        # Monthly Seasonality
        self.model.add_seasonality(

            name="monthly",

            period=30.5,

            fourier_order=5

        )

        # Quarterly Seasonality
        self.model.add_seasonality(

            name="quarterly",

            period=91.25,

            fourier_order=7

        )

        return self.model

    # =======================================================
    # Train
    # =======================================================

    def fit(

        self,

        df,

        date_column="Order Date",

        target_column="Sales",

        frequency="D"

    ):

        prophet_df = prepare_prophet_dataframe(

            df,

            date_column,

            target_column,

            frequency

        )

        if self.model is None:

            self.build()

        self.model.fit(prophet_df)

        return prophet_df

    # =======================================================
    # Forecast
    # =======================================================

    def forecast(

        self,

        periods=30,

        frequency="D"

    ):

        future = create_future_dataframe(

            self.model,

            periods,

            frequency

        )

        forecast = self.model.predict(future)

        return forecast

    # =======================================================
    # Predict Existing Data
    # =======================================================

    def predict(

        self,

        dataframe

    ):

        return self.model.predict(dataframe)

    # =======================================================
    # Components
    # =======================================================

    def components(

        self,

        forecast

    ):

        cols = [

            "trend",

            "yearly",

            "weekly",

            "monthly",

            "quarterly"

        ]

        available = [

            c for c in cols

            if c in forecast.columns

        ]

        return forecast[available]

    # =======================================================
    # Confidence Interval
    # =======================================================

    def confidence_interval(

        self,

        forecast

    ):

        return forecast[

            [

                "ds",

                "yhat",

                "yhat_lower",

                "yhat_upper"

            ]

        ]

    # =======================================================
    # Trend
    # =======================================================

    def trend(

        self,

        forecast

    ):

        return forecast[

            [

                "ds",

                "trend"

            ]

        ]

    # =======================================================
    # Save Model
    # =======================================================

    def save(

        self,

        path="models/prophet.pkl"

    ):

        Path(path).parent.mkdir(

            parents=True,

            exist_ok=True

        )

        joblib.dump(

            self.model,

            path

        )

    # =======================================================
    # Load Model
    # =======================================================

    def load(

        self,

        path="models/prophet.pkl"

    ):

        self.model = joblib.load(path)

        return self.model

    # =======================================================
    # Model Info
    # =======================================================

    def info(self):

        return {

            "Model": "Facebook Prophet",

            "Growth": self.config["growth"],

            "Seasonality": self.config[
                "seasonality_mode"
            ],

            "Country": self.holidays_country

        }

    # =======================================================
    # Full Pipeline
    # =======================================================

    def train_and_forecast(

        self,

        df,

        target="Sales",

        date_column="Order Date",

        periods=30,

        frequency="D"

    ):

        train = self.fit(

            df,

            date_column,

            target,

            frequency

        )

        forecast = self.forecast(

            periods,

            frequency

        )

        return train, forecast
