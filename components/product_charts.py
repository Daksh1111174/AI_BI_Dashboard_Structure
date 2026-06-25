"""
===========================================================
Product Analytics
AI Business Intelligence Dashboard

Author : Daksh Shah
===========================================================
"""

import streamlit as st
import pandas as pd
import plotly.express as px


def product_dashboard(df):
    """
    Product Analytics Dashboard
    """

    required = [
        "Product Name",
        "Category",
        "Sub-Category",
        "Sales",
        "Profit",
        "Quantity"
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        st.warning(f"Missing columns: {', '.join(missing)}")
        return

    data = df.copy()

    st.header("📦 Product Analytics")

    # ====================================================
    # Product KPIs
    # ====================================================

    total_products = data["Product Name"].nunique()

    total_categories = data["Category"].nunique()

    total_subcategories = data["Sub-Category"].nunique()

    total_quantity = data["Quantity"].sum()

    avg_sales = (
        data.groupby("Product Name")["Sales"]
        .sum()
        .mean()
    )

    avg_profit = (
        data.groupby("Product Name")["Profit"]
        .sum()
        .mean()
    )

    c1, c2, c3 = st.columns(3)

    c4, c5, c6 = st.columns(3)

    c1.metric(
        "Products",
        f"{total_products:,}"
    )

    c2.metric(
        "Categories",
        total_categories
    )

    c3.metric(
        "Sub Categories",
        total_subcategories
    )

    c4.metric(
        "Quantity Sold",
        f"{total_quantity:,}"
    )

    c5.metric(
        "Avg Product Sales",
        f"₹ {avg_sales:,.2f}"
    )

    c6.metric(
        "Avg Product Profit",
        f"₹ {avg_profit:,.2f}"
    )

    st.divider()

    # ====================================================
    # Top Selling Products
    # ====================================================

    st.subheader("🏆 Top Selling Products")

    top_products = (
        data
        .groupby("Product Name")
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Quantity=("Quantity", "sum")
        )
        .reset_index()
        .sort_values("Sales", ascending=False)
        .head(10)
    )

    fig = px.bar(
        top_products,
        x="Sales",
        y="Product Name",
        orientation="h",
        color="Sales",
        text_auto=".2s",
        title="Top 10 Products"
    )

    fig.update_layout(
        height=500,
        yaxis=dict(categoryorder="total ascending")
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ====================================================
    # Bottom Selling Products
    # ====================================================

    st.subheader("📉 Bottom Selling Products")

    bottom_products = (
        data
        .groupby("Product Name")
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Quantity=("Quantity", "sum")
        )
        .reset_index()
        .sort_values("Sales")
        .head(10)
    )

    fig = px.bar(
        bottom_products,
        x="Sales",
        y="Product Name",
        orientation="h",
        color="Sales",
        text_auto=".2s",
        title="Bottom 10 Products"
    )

    fig.update_layout(
        height=500,
        yaxis=dict(categoryorder="total ascending")
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ====================================================
    # Product Profitability
    # ====================================================

    st.subheader("💰 Product Profitability")

    profitability = (
        data
        .groupby("Product Name")
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Quantity=("Quantity", "sum")
        )
        .reset_index()
    )

    fig = px.scatter(
        profitability,
        x="Sales",
        y="Profit",
        size="Quantity",
        color="Profit",
        hover_name="Product Name",
        title="Sales vs Profit"
    )

    fig.update_layout(height=600)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ====================================================
    # Category Treemap
    # ====================================================

    st.subheader("🌳 Product Treemap")

    fig = px.treemap(
        data,
        path=[
            "Category",
            "Sub-Category",
            "Product Name"
        ],
        values="Sales",
        color="Profit",
        color_continuous_scale="RdYlGn",
        hover_data=[
            "Sales",
            "Profit",
            "Quantity"
        ]
    )

    fig.update_layout(height=700)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ====================================================
    # Product Summary
    # ====================================================

    st.subheader("📋 Product Summary")

    st.dataframe(
        top_products,
        use_container_width=True
    )
      # ====================================================
    # ABC Analysis
    # ====================================================

    st.subheader("📦 ABC Product Analysis")

    abc = (
        data
        .groupby("Product Name")
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Quantity=("Quantity", "sum")
        )
        .reset_index()
        .sort_values("Sales", ascending=False)
    )

    abc["CumSales"] = abc["Sales"].cumsum()

    abc["CumPercent"] = (
        abc["CumSales"] /
        abc["Sales"].sum()
    ) * 100

    def classify(value):

        if value <= 80:
            return "A"

        elif value <= 95:
            return "B"

        return "C"

    abc["Class"] = abc["CumPercent"].apply(classify)

    fig = px.pie(
        abc,
        names="Class",
        values="Sales",
        hole=0.5,
        title="ABC Classification"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
      # ====================================================
    # Product Demand
    # ====================================================

    if "Order Date" in data.columns:

        st.subheader("📈 Product Demand Trend")

        data["Order Date"] = pd.to_datetime(
            data["Order Date"],
            errors="coerce"
        )

        demand = (
            data
            .groupby(
                pd.Grouper(
                    key="Order Date",
                    freq="M"
                )
            )["Quantity"]
            .sum()
            .reset_index()
        )

        fig = px.line(
            demand,
            x="Order Date",
            y="Quantity",
            markers=True,
            title="Monthly Product Demand"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
          # ====================================================
    # Discount Analysis
    # ====================================================

    if "Discount" in data.columns:

        st.subheader("🏷 Discount Impact")

        discount = (
            data
            .groupby("Category")
            .agg(
                Discount=("Discount", "mean"),
                Sales=("Sales", "sum")
            )
            .reset_index()
        )

        fig = px.scatter(
            discount,
            x="Discount",
            y="Sales",
            size="Sales",
            color="Category",
            hover_name="Category"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
          # ====================================================
    # Sales Distribution
    # ====================================================

    st.subheader("📊 Sales Distribution")

    fig = px.histogram(
        data,
        x="Sales",
        nbins=40,
        marginal="box"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
      # ====================================================
    # Category Summary
    # ====================================================

    st.subheader("📋 Product Summary")

    summary = (
        data
        .groupby(
            [
                "Category",
                "Sub-Category"
            ]
        )
        .agg(
            Products=("Product Name", "count"),
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Quantity=("Quantity", "sum")
        )
        .reset_index()
    )

    st.dataframe(
        summary,
        use_container_width=True
    )
      # ====================================================
    # Inventory
    # ====================================================

    if "Stock" in data.columns:

        st.subheader("📦 Inventory Status")

        fig = px.bar(
            data,
            x="Product Name",
            y="Stock",
            color="Stock"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
          # ====================================================
    # Download Report
    # ====================================================

    csv = summary.to_csv(index=False)

    st.download_button(
        "📥 Download Product Report",
        csv,
        file_name="product_report.csv",
        mime="text/csv"
    )
  
