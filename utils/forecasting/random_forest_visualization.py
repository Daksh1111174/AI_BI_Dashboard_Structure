"""
===========================================================
Random Forest Visualization
Part 1

AI Business Intelligence Dashboard

Author : Daksh Shah
===========================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots


class RandomForestVisualization:
    """
    Visualization utilities for
    Random Forest Forecasting.
    """

    def __init__(self):

        self.template = "plotly_white"

    # =====================================================
    # Actual vs Prediction
    # =====================================================

    def actual_vs_prediction(
        self,
        y_true,
        y_pred,
        title="Actual vs Prediction"
    ):

        fig = go.Figure()

        fig.add_trace(

            go.Scatter(

                y=y_true,

                mode="lines",

                name="Actual"

            )

        )

        fig.add_trace(

            go.Scatter(

                y=y_pred,

                mode="lines",

                name="Prediction"

            )

        )

        fig.update_layout(

            title=title,

            template=self.template,

            xaxis_title="Observation",

            yaxis_title="Value",

            hovermode="x unified"

        )

        return fig

    # =====================================================
    # Scatter Plot
    # =====================================================

    def prediction_scatter(
        self,
        y_true,
        y_pred
    ):

        fig = px.scatter(

            x=y_true,

            y=y_pred,

            labels={

                "x": "Actual",

                "y": "Prediction"

            },

            title="Prediction Scatter"

        )

        fig.add_shape(

            type="line",

            x0=min(y_true),

            y0=min(y_true),

            x1=max(y_true),

            y1=max(y_true)

        )

        fig.update_layout(

            template=self.template

        )

        return fig

    # =====================================================
    # Residual Plot
    # =====================================================

    def residual_plot(
        self,
        y_true,
        y_pred
    ):

        residual = np.array(

            y_true

        ) - np.array(

            y_pred

        )

        fig = px.scatter(

            x=y_pred,

            y=residual,

            labels={

                "x": "Prediction",

                "y": "Residual"

            },

            title="Residual Plot"

        )

        fig.add_hline(

            y=0,

            line_dash="dash"

        )

        fig.update_layout(

            template=self.template

        )

        return fig

    # =====================================================
    # Error Distribution
    # =====================================================

    def error_distribution(
        self,
        y_true,
        y_pred
    ):

        residual = np.array(

            y_true

        ) - np.array(

            y_pred

        )

        fig = px.histogram(

            residual,

            nbins=40,

            title="Residual Distribution"

        )

        fig.update_layout(

            template=self.template,

            xaxis_title="Residual",

            yaxis_title="Frequency"

        )

        return fig

    # =====================================================
    # Time Series Comparison
    # =====================================================

    def time_series(
        self,
        dates,
        actual,
        predicted
    ):

        fig = go.Figure()

        fig.add_trace(

            go.Scatter(

                x=dates,

                y=actual,

                name="Actual",

                mode="lines"

            )

        )

        fig.add_trace(

            go.Scatter(

                x=dates,

                y=predicted,

                name="Prediction",

                mode="lines"

            )

        )

        fig.update_layout(

            template=self.template,

            title="Time Series Comparison",

            hovermode="x unified"

        )

        return fig

    # =====================================================
    # Prediction Error
    # =====================================================

    def prediction_error(
        self,
        y_true,
        y_pred
    ):

        error = np.array(

            y_true

        ) - np.array(

            y_pred

        )

        fig = go.Figure()

        fig.add_trace(

            go.Bar(

                y=error,

                name="Error"

            )

        )

        fig.update_layout(

            template=self.template,

            title="Prediction Error"

        )

        return fig

    # =====================================================
    # Residual Box Plot
    # =====================================================

    def residual_boxplot(
        self,
        y_true,
        y_pred
    ):

        residual = np.array(

            y_true

        ) - np.array(

            y_pred

        )

        fig = px.box(

            residual,

            title="Residual Box Plot"

        )

        fig.update_layout(

            template=self.template

        )

        return fig

    # =====================================================
    # Combined Evaluation Dashboard
    # =====================================================

    def evaluation_dashboard(
        self,
        y_true,
        y_pred
    ):

        fig = make_subplots(

            rows=2,

            cols=2,

            subplot_titles=(

                "Actual vs Prediction",

                "Residual Plot",

                "Prediction Scatter",

                "Residual Distribution"

            )

        )

        fig.add_trace(

            go.Scatter(

                y=y_true,

                mode="lines",

                name="Actual"

            ),

            row=1,

            col=1

        )

        fig.add_trace(

            go.Scatter(

                y=y_pred,

                mode="lines",

                name="Prediction"

            ),

            row=1,

            col=1

        )

        residual = np.array(

            y_true

        ) - np.array(

            y_pred

        )

        fig.add_trace(

            go.Scatter(

                x=y_pred,

                y=residual,

                mode="markers",

                name="Residual"

            ),

            row=1,

            col=2

        )

        fig.add_trace(

            go.Scatter(

                x=y_true,

                y=y_pred,

                mode="markers",

                name="Scatter"

            ),

            row=2,

            col=1

        )

        fig.add_trace(

            go.Histogram(

                x=residual,

                name="Distribution"

            ),

            row=2,

            col=2

        )

        fig.update_layout(

            height=800,

            template=self.template,

            title="Random Forest Evaluation Dashboard"

        )

        return fig
          # =====================================================
    # Feature Importance
    # =====================================================

    def feature_importance(
        self,
        importance_df,
        top_n=20,
        title="Feature Importance"
    ):

        df = (
            importance_df
            .head(top_n)
            .sort_values("Importance")
        )

        fig = px.bar(

            df,

            x="Importance",

            y="Feature",

            orientation="h",

            title=title,

            text="Importance"

        )

        fig.update_layout(

            template=self.template,

            yaxis_title="Feature",

            xaxis_title="Importance"

        )

        return fig

    # =====================================================
    # Feature Importance Percentage
    # =====================================================

    def feature_importance_percentage(
        self,
        importance_df,
        top_n=15
    ):

        df = importance_df.head(top_n)

        fig = px.bar(

            df,

            x="Feature",

            y="Importance (%)",

            text="Importance (%)",

            title="Feature Importance (%)"

        )

        fig.update_layout(

            template=self.template,

            xaxis_tickangle=-45

        )

        return fig

    # =====================================================
    # Top Features
    # =====================================================

    def top_features(
        self,
        importance_df,
        n=10
    ):

        return self.feature_importance(

            importance_df,

            top_n=n,

            title=f"Top {n} Features"

        )

    # =====================================================
    # Bottom Features
    # =====================================================

    def bottom_features(
        self,
        importance_df,
        n=10
    ):

        df = (

            importance_df

            .tail(n)

            .sort_values("Importance")

        )

        fig = px.bar(

            df,

            x="Importance",

            y="Feature",

            orientation="h",

            title=f"Bottom {n} Features"

        )

        fig.update_layout(

            template=self.template

        )

        return fig

    # =====================================================
    # Cumulative Importance
    # =====================================================

    def cumulative_importance(
        self,
        importance_df
    ):

        df = importance_df.copy()

        if "Importance (%)" not in df.columns:

            total = df["Importance"].sum()

            df["Importance (%)"] = (

                df["Importance"] / total

            ) * 100

        df["Cumulative"] = (

            df["Importance (%)"]

            .cumsum()

        )

        fig = go.Figure()

        fig.add_trace(

            go.Scatter(

                x=df["Feature"],

                y=df["Cumulative"],

                mode="lines+markers",

                name="Cumulative"

            )

        )

        fig.update_layout(

            title="Cumulative Feature Importance",

            template=self.template,

            xaxis_tickangle=-45,

            yaxis_title="Cumulative %"

        )

        return fig

    # =====================================================
    # Correlation Heatmap
    # =====================================================

    def correlation_heatmap(
        self,
        dataframe
    ):

        corr = dataframe.corr(numeric_only=True)

        fig = px.imshow(

            corr,

            aspect="auto",

            title="Correlation Heatmap",

            text_auto=".2f",

            color_continuous_scale="RdBu_r"

        )

        fig.update_layout(

            template=self.template

        )

        return fig

    # =====================================================
    # Target Correlation
    # =====================================================

    def target_correlation(
        self,
        dataframe,
        target="y",
        top_n=20
    ):

        corr = (

            dataframe

            .corr(numeric_only=True)[target]

            .drop(target)

            .abs()

            .sort_values(

                ascending=False

            )

            .head(top_n)

        )

        fig = px.bar(

            x=corr.index,

            y=corr.values,

            title=f"Top {top_n} Features Correlated with {target}"

        )

        fig.update_layout(

            template=self.template,

            xaxis_tickangle=-45,

            yaxis_title="Correlation"

        )

        return fig

    # =====================================================
    # Correlation Matrix Table
    # =====================================================

    def correlation_table(
        self,
        dataframe
    ):

        return dataframe.corr(

            numeric_only=True

        )

    # =====================================================
    # Pairwise Scatter Matrix
    # =====================================================

    def scatter_matrix(
        self,
        dataframe,
        columns=None
    ):

        if columns is None:

            columns = dataframe.select_dtypes(

                include="number"

            ).columns[:6]

        fig = px.scatter_matrix(

            dataframe,

            dimensions=columns,

            title="Scatter Matrix"

        )

        fig.update_layout(

            template=self.template

        )

        return fig

    # =====================================================
    # Feature Distribution
    # =====================================================

    def feature_distribution(
        self,
        dataframe,
        feature
    ):

        fig = px.histogram(

            dataframe,

            x=feature,

            nbins=40,

            title=f"{feature} Distribution"

        )

        fig.update_layout(

            template=self.template

        )

        return fig

    # =====================================================
    # Multiple Feature Distribution
    # =====================================================

    def multiple_distributions(
        self,
        dataframe,
        features
    ):

        figures = {}

        for feature in features:

            figures[feature] = self.feature_distribution(

                dataframe,

                feature

            )

        return figures

    # =====================================================
    # Missing Values Heatmap
    # =====================================================

    def missing_values_heatmap(
        self,
        dataframe
    ):

        missing = dataframe.isna().astype(int)

        fig = px.imshow(

            missing,

            aspect="auto",

            title="Missing Values Heatmap",

            color_continuous_scale="Blues"

        )

        fig.update_layout(

            template=self.template

        )

        return fig
          # =====================================================
    # Feature Importance
    # =====================================================

    def feature_importance(
        self,
        importance_df,
        top_n=20,
        title="Feature Importance"
    ):

        df = (
            importance_df
            .head(top_n)
            .sort_values("Importance")
        )

        fig = px.bar(

            df,

            x="Importance",

            y="Feature",

            orientation="h",

            title=title,

            text="Importance"

        )

        fig.update_layout(

            template=self.template,

            yaxis_title="Feature",

            xaxis_title="Importance"

        )

        return fig

    # =====================================================
    # Feature Importance Percentage
    # =====================================================

    def feature_importance_percentage(
        self,
        importance_df,
        top_n=15
    ):

        df = importance_df.head(top_n)

        fig = px.bar(

            df,

            x="Feature",

            y="Importance (%)",

            text="Importance (%)",

            title="Feature Importance (%)"

        )

        fig.update_layout(

            template=self.template,

            xaxis_tickangle=-45

        )

        return fig

    # =====================================================
    # Top Features
    # =====================================================

    def top_features(
        self,
        importance_df,
        n=10
    ):

        return self.feature_importance(

            importance_df,

            top_n=n,

            title=f"Top {n} Features"

        )

    # =====================================================
    # Bottom Features
    # =====================================================

    def bottom_features(
        self,
        importance_df,
        n=10
    ):

        df = (

            importance_df

            .tail(n)

            .sort_values("Importance")

        )

        fig = px.bar(

            df,

            x="Importance",

            y="Feature",

            orientation="h",

            title=f"Bottom {n} Features"

        )

        fig.update_layout(

            template=self.template

        )

        return fig

    # =====================================================
    # Cumulative Importance
    # =====================================================

    def cumulative_importance(
        self,
        importance_df
    ):

        df = importance_df.copy()

        if "Importance (%)" not in df.columns:

            total = df["Importance"].sum()

            df["Importance (%)"] = (

                df["Importance"] / total

            ) * 100

        df["Cumulative"] = (

            df["Importance (%)"]

            .cumsum()

        )

        fig = go.Figure()

        fig.add_trace(

            go.Scatter(

                x=df["Feature"],

                y=df["Cumulative"],

                mode="lines+markers",

                name="Cumulative"

            )

        )

        fig.update_layout(

            title="Cumulative Feature Importance",

            template=self.template,

            xaxis_tickangle=-45,

            yaxis_title="Cumulative %"

        )

        return fig

    # =====================================================
    # Correlation Heatmap
    # =====================================================

    def correlation_heatmap(
        self,
        dataframe
    ):

        corr = dataframe.corr(numeric_only=True)

        fig = px.imshow(

            corr,

            aspect="auto",

            title="Correlation Heatmap",

            text_auto=".2f",

            color_continuous_scale="RdBu_r"

        )

        fig.update_layout(

            template=self.template

        )

        return fig

    # =====================================================
    # Target Correlation
    # =====================================================

    def target_correlation(
        self,
        dataframe,
        target="y",
        top_n=20
    ):

        corr = (

            dataframe

            .corr(numeric_only=True)[target]

            .drop(target)

            .abs()

            .sort_values(

                ascending=False

            )

            .head(top_n)

        )

        fig = px.bar(

            x=corr.index,

            y=corr.values,

            title=f"Top {top_n} Features Correlated with {target}"

        )

        fig.update_layout(

            template=self.template,

            xaxis_tickangle=-45,

            yaxis_title="Correlation"

        )

        return fig

    # =====================================================
    # Correlation Matrix Table
    # =====================================================

    def correlation_table(
        self,
        dataframe
    ):

        return dataframe.corr(

            numeric_only=True

        )

    # =====================================================
    # Pairwise Scatter Matrix
    # =====================================================

    def scatter_matrix(
        self,
        dataframe,
        columns=None
    ):

        if columns is None:

            columns = dataframe.select_dtypes(

                include="number"

            ).columns[:6]

        fig = px.scatter_matrix(

            dataframe,

            dimensions=columns,

            title="Scatter Matrix"

        )

        fig.update_layout(

            template=self.template

        )

        return fig

    # =====================================================
    # Feature Distribution
    # =====================================================

    def feature_distribution(
        self,
        dataframe,
        feature
    ):

        fig = px.histogram(

            dataframe,

            x=feature,

            nbins=40,

            title=f"{feature} Distribution"

        )

        fig.update_layout(

            template=self.template

        )

        return fig

    # =====================================================
    # Multiple Feature Distribution
    # =====================================================

    def multiple_distributions(
        self,
        dataframe,
        features
    ):

        figures = {}

        for feature in features:

            figures[feature] = self.feature_distribution(

                dataframe,

                feature

            )

        return figures

    # =====================================================
    # Missing Values Heatmap
    # =====================================================

    def missing_values_heatmap(
        self,
        dataframe
    ):

        missing = dataframe.isna().astype(int)

        fig = px.imshow(

            missing,

            aspect="auto",

            title="Missing Values Heatmap",

            color_continuous_scale="Blues"

        )

        fig.update_layout(

            template=self.template

        )

        return fig
      
