"""
===========================================================
Dashboard
AI Business Intelligence Dashboard
===========================================================
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------
# Page Config
# -----------------------------------------------------

st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Executive Dashboard")

st.markdown(
"""
Business overview with KPIs and interactive filters.
"""
)

# -----------------------------------------------------
# Check Dataset
# -----------------------------------------------------

if "dataset" not in st.session_state:

    st.warning("Please upload a dataset from the Home page.")

    st.stop()

df = st.session_state.dataset.copy()

# -----------------------------------------------------
# Date Conversion
# -----------------------------------------------------

if "Order Date" in df.columns:

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        errors="coerce"
    )

# -----------------------------------------------------
# Sidebar Filters
# -----------------------------------------------------

st.sidebar.header("Dashboard Filters")

filtered_df = df.copy()

# -----------------------------------------------------
# Region Filter
# -----------------------------------------------------

if "Region" in df.columns:

    regions = sorted(df["Region"].dropna().unique())

    selected_regions = st.sidebar.multiselect(
        "Region",
        regions,
        default=regions
    )

    filtered_df = filtered_df[
        filtered_df["Region"].isin(selected_regions)
    ]

# -----------------------------------------------------
# State Filter
# -----------------------------------------------------

if "State" in df.columns:

    states = sorted(filtered_df["State"].dropna().unique())

    selected_states = st.sidebar.multiselect(
        "State",
        states,
        default=states
    )

    filtered_df = filtered_df[
        filtered_df["State"].isin(selected_states)
    ]

# -----------------------------------------------------
# Category Filter
# -----------------------------------------------------

if "Category" in df.columns:

    categories = sorted(
        filtered_df["Category"].dropna().unique()
    )

    selected_categories = st.sidebar.multiselect(
        "Category",
        categories,
        default=categories
    )

    filtered_df = filtered_df[
        filtered_df["Category"].isin(selected_categories)
    ]

# -----------------------------------------------------
# Segment Filter
# -----------------------------------------------------

if "Segment" in df.columns:

    segments = sorted(
        filtered_df["Segment"].dropna().unique()
    )

    selected_segments = st.sidebar.multiselect(
        "Segment",
        segments,
        default=segments
    )

    filtered_df = filtered_df[
        filtered_df["Segment"].isin(selected_segments)
    ]

# -----------------------------------------------------
# Payment Filter
# -----------------------------------------------------

if "Payment Mode" in filtered_df.columns:

    payments = sorted(
        filtered_df["Payment Mode"].dropna().unique()
    )

    selected_payment = st.sidebar.multiselect(
        "Payment Mode",
        payments,
        default=payments
    )

    filtered_df = filtered_df[
        filtered_df["Payment Mode"].isin(selected_payment)
    ]

# -----------------------------------------------------
# Date Filter
# -----------------------------------------------------

if "Order Date" in filtered_df.columns:

    min_date = filtered_df["Order Date"].min()

    max_date = filtered_df["Order Date"].max()

    dates = st.sidebar.date_input(
        "Date Range",
        [min_date, max_date]
    )

    if len(dates) == 2:

        start, end = dates

        filtered_df = filtered_df[
            (
                filtered_df["Order Date"]
                >= pd.to_datetime(start)
            )
            &
            (
                filtered_df["Order Date"]
                <= pd.to_datetime(end)
            )
        ]

# -----------------------------------------------------
# KPI Calculations
# -----------------------------------------------------

revenue = filtered_df["Sales"].sum()

profit = filtered_df["Profit"].sum()

orders = filtered_df["Order ID"].nunique()

customers = filtered_df["Customer ID"].nunique()

quantity = filtered_df["Quantity"].sum()

aov = revenue / orders if orders else 0

margin = (profit / revenue * 100) if revenue else 0

discount = (
    filtered_df["Discount"].mean()
    if "Discount" in filtered_df.columns
    else 0
)

shipping = (
    filtered_df["Shipping Cost"].sum()
    if "Shipping Cost" in filtered_df.columns
    else 0
)

# -----------------------------------------------------
# Growth %
# -----------------------------------------------------

growth = 0

if "Order Date" in filtered_df.columns:

    monthly = (
        filtered_df
        .groupby(
            pd.Grouper(
                key="Order Date",
                freq="M"
            )
        )["Sales"]
        .sum()
    )

    if len(monthly) >= 2:

        last = monthly.iloc[-1]

        previous = monthly.iloc[-2]

        if previous != 0:

            growth = (
                (last - previous)
                / previous
            ) * 100

# -----------------------------------------------------
# KPI Cards
# -----------------------------------------------------

st.subheader("Business KPIs")

row1 = st.columns(4)

row1[0].metric(
    "Revenue",
    f"₹ {revenue:,.0f}",
    f"{growth:.1f}%"
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

row2 = st.columns(4)

row2[0].metric(
    "Quantity",
    f"{quantity:,}"
)

row2[1].metric(
    "Average Order Value",
    f"₹ {aov:,.2f}"
)

row2[2].metric(
    "Profit Margin",
    f"{margin:.2f}%"
)

row2[3].metric(
    "Average Discount",
    f"{discount:.2f}%"
)

# -----------------------------------------------------
# Dataset Summary
# -----------------------------------------------------

st.divider()

st.subheader("Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Rows", len(filtered_df))

c2.metric("Columns", len(filtered_df.columns))

c3.metric(
    "Missing Values",
    int(filtered_df.isna().sum().sum())
)

memory = (
    filtered_df.memory_usage(deep=True)
    .sum()
    / 1024
    / 1024
)

c4.metric(
    "Memory",
    f"{memory:.2f} MB"
)

st.divider()
