"""
===========================================================
Prophet Visualization
AI Business Intelligence Dashboard

Author : Daksh Shah
===========================================================
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ==========================================================
# Forecast Chart
# ==========================================================

def forecast_chart(forecast):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat"],
            mode="lines",
            name="Forecast"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_lower"],
            mode="lines",
            fill="tonexty",
            line=dict(width=0),
            name="Confidence Interval"
        )
    )

    fig.update_layout(
        title="Sales Forecast",
        xaxis_title="Date",
        yaxis_title="Forecast",
        hovermode="x unified",
        height=550
    )

    return fig


# ==========================================================
# Actual vs Forecast
# ==========================================================

def actual_vs_forecast(train_df, forecast):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=train_df["ds"],
            y=train_df["y"],
            mode="lines",
            name="Actual"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat"],
            mode="lines",
            name="Forecast"
        )
    )

    fig.update_layout(
        title="Actual vs Forecast",
        hovermode="x unified",
        height=550
    )

    return fig


# ==========================================================
# Trend
# ==========================================================

def trend_chart(forecast):

    fig = px.line(
        forecast,
        x="ds",
        y="trend",
        title="Trend"
    )

    fig.update_layout(height=500)

    return fig


# ==========================================================
# Weekly Seasonality
# ==========================================================

def weekly_chart(forecast):

    if "weekly" not in forecast.columns:
        return None

    fig = px.line(
        forecast,
        x="ds",
        y="weekly",
        title="Weekly Seasonality"
    )

    return fig


# ==========================================================
# Monthly Seasonality
# ==========================================================

def monthly_chart(forecast):

    if "monthly" not in forecast.columns:
        return None

    fig = px.line(
        forecast,
        x="ds",
        y="monthly",
        title="Monthly Seasonality"
    )

    return fig


# ==========================================================
# Yearly Seasonality
# ==========================================================

def yearly_chart(forecast):

    if "yearly" not in forecast.columns:
        return None

    fig = px.line(
        forecast,
        x="ds",
        y="yearly",
        title="Yearly Seasonality"
    )

    return fig


# ==========================================================
# Quarterly Seasonality
# ==========================================================

def quarterly_chart(forecast):

    if "quarterly" not in forecast.columns:
        return None

    fig = px.line(
        forecast,
        x="ds",
        y="quarterly",
        title="Quarterly Seasonality"
    )

    return fig


# ==========================================================
# Monthly Forecast Summary
# ==========================================================

def monthly_forecast(forecast):

    data = forecast.copy()

    data["Month"] = data["ds"].dt.to_period("M").astype(str)

    monthly = (
        data
        .groupby("Month")["yhat"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        monthly,
        x="Month",
        y="yhat",
        text_auto=".2s",
        title="Monthly Forecast"
    )

    return fig


# ==========================================================
# Yearly Forecast Summary
# ==========================================================

def yearly_forecast(forecast):

    data = forecast.copy()

    data["Year"] = data["ds"].dt.year

    yearly = (
        data
        .groupby("Year")["yhat"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        yearly,
        x="Year",
        y="yhat",
        text_auto=".2s",
        title="Yearly Forecast"
    )

    return fig


# ==========================================================
# Forecast Table
# ==========================================================

def forecast_table(forecast):

    return forecast[
        [
            "ds",
            "yhat",
            "yhat_lower",
            "yhat_upper"
        ]
    ].copy()


# ==========================================================
# Forecast Statistics
# ==========================================================

def forecast_summary(forecast):

    return {

        "Forecast Start":
            forecast["ds"].min(),

        "Forecast End":
            forecast["ds"].max(),

        "Minimum":
            forecast["yhat"].min(),

        "Maximum":
            forecast["yhat"].max(),

        "Average":
            forecast["yhat"].mean(),

        "Median":
            forecast["yhat"].median(),

        "Std":
            forecast["yhat"].std()

    }


# ==========================================================
# Export Forecast
# ==========================================================

def export_forecast(
    forecast,
    path
):

    forecast.to_csv(
        path,
        index=False
    )
