"""
===========================================================
KPI Cards Component
AI Business Intelligence Dashboard
===========================================================
"""

import streamlit as st
import pandas as pd


# =========================================================
# Helper Functions
# =========================================================

def calculate_growth(df):
    """
    Calculate month-over-month revenue growth.
    """

    if "Order Date" not in df.columns:
        return 0

    data = df.copy()

    data["Order Date"] = pd.to_datetime(
        data["Order Date"],
        errors="coerce"
    )

    monthly = (
        data
        .groupby(pd.Grouper(key="Order Date", freq="M"))
        ["Sales"]
        .sum()
    )

    if len(monthly) < 2:
        return 0

    previous = monthly.iloc[-2]
    current = monthly.iloc[-1]

    if previous == 0:
        return 0

    growth = ((current - previous) / previous) * 100

    return round(growth, 2)


# =========================================================
# Main KPI Function
# =========================================================

def show_kpis(df):

    revenue = (
        df["Sales"].sum()
        if "Sales" in df.columns
        else 0
    )

    profit = (
        df["Profit"].sum()
        if "Profit" in df.columns
        else 0
    )

    orders = (
        df["Order ID"].nunique()
        if "Order ID" in df.columns
        else 0
    )

    customers = (
        df["Customer ID"].nunique()
        if "Customer ID" in df.columns
        else 0
    )

    quantity = (
        df["Quantity"].sum()
        if "Quantity" in df.columns
        else 0
    )

    discount = (
        df["Discount"].mean()
        if "Discount" in df.columns
        else 0
    )

    shipping = (
        df["Shipping Cost"].sum()
        if "Shipping Cost" in df.columns
        else 0
    )

    # ----------------------------------------

    aov = revenue / orders if orders else 0

    margin = (
        (profit / revenue) * 100
        if revenue
        else 0
    )

    basket = (
        quantity / orders
        if orders
        else 0
    )

    growth = calculate_growth(df)

    # =====================================================
    # KPI Cards
    # =====================================================

    st.subheader("📊 Executive KPIs")

    row1 = st.columns(5)

    row1[0].metric(
        "Revenue",
        f"₹ {revenue:,.0f}",
        f"{growth:.2f}%"
    )

    row1[1].metric(
        "Profit",
        f"₹ {profit:,.0f}"
    )

    row1[2].metric(
        "Orders",
        f"{orders:,}"
    )

    row1[3].metric(
        "Customers",
        f"{customers:,}"
    )

    row1[4].metric(
        "Quantity",
        f"{quantity:,}"
    )

    # ----------------------------------------

    row2 = st.columns(5)

    row2[0].metric(
        "Average Order Value",
        f"₹ {aov:,.2f}"
    )

    row2[1].metric(
        "Profit Margin",
        f"{margin:.2f}%"
    )

    row2[2].metric(
        "Average Discount",
        f"{discount:.2f}%"
    )

    row2[3].metric(
        "Shipping Cost",
        f"₹ {shipping:,.0f}"
    )

    row2[4].metric(
        "Basket Size",
        f"{basket:.2f}"
    )

    st.divider()

    # =====================================================
    # Quick Business Health
    # =====================================================

    st.subheader("🏥 Business Health")

    c1, c2, c3 = st.columns(3)

    # Revenue

    if revenue > 0:
        c1.success("✅ Revenue generated")
    else:
        c1.error("❌ No revenue")

    # Profit

    if margin >= 20:
        c2.success("Excellent profit margin")

    elif margin >= 10:
        c2.warning("Healthy profit margin")

    else:
        c2.error("Low profit margin")

    # Growth

    if growth > 10:
        c3.success("Strong business growth")

    elif growth >= 0:
        c3.info("Stable growth")

    else:
        c3.error("Negative growth")
