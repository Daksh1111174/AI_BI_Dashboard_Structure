"""
===========================================================
Revenue Charts
AI Business Intelligence Dashboard
===========================================================
"""

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


def revenue_dashboard(df):
    """
    Display revenue-related analytics.
    """

    if "Order Date" not in df.columns:
        st.warning("Order Date column not found.")
        return

    data = df.copy()

    # ----------------------------
    # Date Conversion
    # ----------------------------

    data["Order Date"] = pd.to_datetime(
        data["Order Date"],
        errors="coerce"
    )

    data = data.dropna(subset=["Order Date"])

    # ===================================================
    # Monthly Revenue
    # ===================================================

    monthly = (
        data
        .groupby(pd.Grouper(
            key="Order Date",
            freq="M"
        ))
        .agg(
            Revenue=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Orders=("Order ID", "count")
        )
        .reset_index()
    )

    st.subheader("📈 Monthly Revenue Trend")

    fig = px.line(
        monthly,
        x="Order Date",
        y="Revenue",
        markers=True,
        title="Monthly Revenue"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ===================================================
    # Revenue vs Profit
    # ===================================================

    st.subheader("💰 Revenue vs Profit")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=monthly["Order Date"],
            y=monthly["Revenue"],
            mode="lines+markers",
            name="Revenue"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=monthly["Order Date"],
            y=monthly["Profit"],
            mode="lines+markers",
            name="Profit"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ===================================================
    # Orders Trend
    # ===================================================

    st.subheader("📦 Monthly Orders")

    fig = px.bar(
        monthly,
        x="Order Date",
        y="Orders",
        color="Orders",
        text_auto=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ===================================================
    # Quarterly Revenue
    # ===================================================

    quarterly = (
        data
        .groupby(pd.Grouper(
            key="Order Date",
            freq="Q"
        ))["Sales"]
        .sum()
        .reset_index()
    )

    st.subheader("📅 Quarterly Revenue")

    fig = px.bar(
        quarterly,
        x="Order Date",
        y="Sales",
        color="Sales",
        text_auto=".2s"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ===================================================
    # Yearly Revenue
    # ===================================================

    yearly = (
        data
        .groupby(pd.Grouper(
            key="Order Date",
            freq="Y"
        ))["Sales"]
        .sum()
        .reset_index()
    )

    st.subheader("📆 Yearly Revenue")

    fig = px.line(
        yearly,
        x="Order Date",
        y="Sales",
        markers=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ===================================================
    # Cumulative Revenue
    # ===================================================

    monthly["Cumulative Revenue"] = (
        monthly["Revenue"].cumsum()
    )

    st.subheader("📈 Cumulative Revenue")

    fig = px.area(
        monthly,
        x="Order Date",
        y="Cumulative Revenue"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ===================================================
    # Moving Average
    # ===================================================

    monthly["Moving Average"] = (
        monthly["Revenue"]
        .rolling(3)
        .mean()
    )

    st.subheader("📉 Revenue Moving Average")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=monthly["Order Date"],
            y=monthly["Revenue"],
            name="Revenue"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=monthly["Order Date"],
            y=monthly["Moving Average"],
            name="3-Month Average"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ===================================================
    # Revenue Distribution
    # ===================================================

    st.subheader("📊 Revenue Distribution")

    fig = px.histogram(
        data,
        x="Sales",
        nbins=40
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ===================================================
    # Box Plot
    # ===================================================

    st.subheader("📦 Revenue Box Plot")

    fig = px.box(
        data,
        y="Sales"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ===================================================
    # Monthly Summary Table
    # ===================================================

    st.subheader("📋 Monthly Summary")

    st.dataframe(
        monthly,
        use_container_width=True
    )
