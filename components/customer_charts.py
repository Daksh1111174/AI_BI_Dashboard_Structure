"""
===========================================================
Customer Analytics
AI Business Intelligence Dashboard
===========================================================
"""

import streamlit as st
import pandas as pd
import plotly.express as px


def customer_dashboard(df):

    required = [
        "Customer ID",
        "Customer Name",
        "Sales",
        "Profit",
        "Order ID"
    ]

    for col in required:
        if col not in df.columns:
            st.warning(f"{col} column not found.")
            return

    data = df.copy()

    st.header("👥 Customer Analytics")

    # ==================================================
    # Customer KPIs
    # ==================================================

    total_customers = data["Customer ID"].nunique()

    total_orders = data["Order ID"].nunique()

    avg_sales = (
        data.groupby("Customer ID")["Sales"]
        .sum()
        .mean()
    )

    avg_profit = (
        data.groupby("Customer ID")["Profit"]
        .sum()
        .mean()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Customers",
        total_customers
    )

    c2.metric(
        "Orders",
        total_orders
    )

    c3.metric(
        "Avg Customer Sales",
        f"₹ {avg_sales:,.2f}"
    )

    c4.metric(
        "Avg Customer Profit",
        f"₹ {avg_profit:,.2f}"
    )

    st.divider()

    # ==================================================
    # Top Customers
    # ==================================================

    st.subheader("🏆 Top Customers")

    top_customers = (
        data.groupby("Customer Name")
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum")
        )
        .reset_index()
        .sort_values("Sales", ascending=False)
        .head(10)
    )

    fig = px.bar(
        top_customers,
        x="Sales",
        y="Customer Name",
        orientation="h",
        color="Sales",
        text_auto=".2s"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ==================================================
    # Customer Profitability
    # ==================================================

    st.subheader("💰 Customer Profitability")

    fig = px.scatter(
        top_customers,
        x="Sales",
        y="Profit",
        size="Sales",
        hover_name="Customer Name"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ==================================================
    # Segment Distribution
    # ==================================================

    if "Segment" in data.columns:

        st.subheader("👥 Customer Segments")

        segment = (
            data.groupby("Segment")
            .agg(
                Sales=("Sales", "sum")
            )
            .reset_index()
        )

        fig = px.pie(
            segment,
            names="Segment",
            values="Sales",
            hole=.45
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ==================================================
    # Region Distribution
    # ==================================================

    if "Region" in data.columns:

        st.subheader("🌍 Customers by Region")

        region = (
            data.groupby("Region")
            .agg(
                Customers=("Customer ID", "nunique")
            )
            .reset_index()
        )

        fig = px.bar(
            region,
            x="Region",
            y="Customers",
            color="Customers"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ==================================================
    # Customer Lifetime Value
    # ==================================================

    st.subheader("💎 Customer Lifetime Value")

    clv = (
        data.groupby("Customer Name")
        .agg(
            Revenue=("Sales", "sum"),
            Orders=("Order ID", "count"),
            Profit=("Profit", "sum")
        )
        .reset_index()
    )

    fig = px.scatter(
        clv,
        x="Orders",
        y="Revenue",
        size="Profit",
        hover_name="Customer Name",
        color="Profit"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ==================================================
    # New vs Returning Customers
    # ==================================================

    st.subheader("🔄 New vs Returning")

    visits = (
        data.groupby("Customer ID")
        .size()
        .reset_index(name="Visits")
    )

    visits["Type"] = visits["Visits"].apply(
        lambda x: "Returning"
        if x > 1
        else "New"
    )

    chart = (
        visits.groupby("Type")
        .size()
        .reset_index(name="Customers")
    )

    fig = px.pie(
        chart,
        names="Type",
        values="Customers"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ==================================================
    # Top Customers Table
    # ==================================================

    st.subheader("📋 Customer Summary")

    st.dataframe(
        top_customers,
        use_container_width=True
    )
