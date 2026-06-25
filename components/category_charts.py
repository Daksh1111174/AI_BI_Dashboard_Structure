"""
===========================================================
Category Charts
AI Business Intelligence Dashboard
===========================================================
"""

import streamlit as st
import pandas as pd
import plotly.express as px


def category_dashboard(df):
    """
    Display category and sub-category analytics.
    """

    required = {"Category", "Sub-Category", "Sales", "Profit"}

    if not required.issubset(df.columns):
        st.warning(
            "Required columns are missing for category analysis."
        )
        return

    data = df.copy()

    # =====================================================
    # Sales by Category
    # =====================================================

    st.subheader("📦 Sales by Category")

    category_sales = (
        data.groupby("Category")
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Orders=("Order ID", "count")
        )
        .reset_index()
        .sort_values("Sales", ascending=False)
    )

    fig = px.bar(
        category_sales,
        x="Category",
        y="Sales",
        color="Category",
        text_auto=".2s",
        title="Category Sales"
    )

    st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # Profit by Category
    # =====================================================

    st.subheader("💰 Profit by Category")

    fig = px.bar(
        category_sales,
        x="Category",
        y="Profit",
        color="Profit",
        text_auto=".2s",
        title="Category Profit"
    )

    st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # Sales Share (Donut)
    # =====================================================

    st.subheader("🥧 Sales Share")

    fig = px.pie(
        category_sales,
        names="Category",
        values="Sales",
        hole=0.45
    )

    st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # Treemap
    # =====================================================

    st.subheader("🌳 Category Treemap")

    fig = px.treemap(
        data,
        path=["Category", "Sub-Category"],
        values="Sales",
        color="Profit",
        hover_data=["Sales", "Profit"]
    )

    st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # Sales by Sub-Category
    # =====================================================

    st.subheader("📋 Top Sub-Categories")

    sub_sales = (
        data.groupby("Sub-Category")
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum")
        )
        .reset_index()
        .sort_values("Sales", ascending=False)
    )

    fig = px.bar(
        sub_sales.head(15),
        x="Sales",
        y="Sub-Category",
        orientation="h",
        color="Sales",
        text_auto=".2s"
    )

    st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # Bottom Sub-Categories
    # =====================================================

    st.subheader("⚠️ Lowest Selling Sub-Categories")

    fig = px.bar(
        sub_sales.tail(10),
        x="Sales",
        y="Sub-Category",
        orientation="h",
        color="Sales"
    )

    st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # Profit Margin by Category
    # =====================================================

    st.subheader("📈 Profit Margin")

    category_sales["Margin"] = (
        category_sales["Profit"]
        / category_sales["Sales"]
    ) * 100

    fig = px.bar(
        category_sales,
        x="Category",
        y="Margin",
        color="Margin",
        text_auto=".1f"
    )

    st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # Discount Analysis
    # =====================================================

    if "Discount" in data.columns:

        st.subheader("🏷 Average Discount")

        discount = (
            data.groupby("Category")
            .agg(
                Discount=("Discount", "mean")
            )
            .reset_index()
        )

        fig = px.bar(
            discount,
            x="Category",
            y="Discount",
            color="Discount",
            text_auto=".2f"
        )

        st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # Summary Table
    # =====================================================

    st.subheader("📊 Category Summary")

    st.dataframe(
        category_sales,
        use_container_width=True
    )
