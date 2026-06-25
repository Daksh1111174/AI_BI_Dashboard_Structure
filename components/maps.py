"""
===========================================================
Geographic Analytics
AI Business Intelligence Dashboard
===========================================================
"""

import streamlit as st
import pandas as pd
import plotly.express as px


def geographic_dashboard(df):

    required = [
        "Country",
        "State",
        "City",
        "Region",
        "Sales",
        "Profit",
        "Order ID"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        st.warning(
            f"Missing columns: {', '.join(missing)}"
        )
        return

    data = df.copy()

    st.header("🌍 Geographic Analytics")

    # =====================================================
    # Geographic KPIs
    # =====================================================

    total_regions = data["Region"].nunique()
    total_states = data["State"].nunique()
    total_cities = data["City"].nunique()
    total_countries = data["Country"].nunique()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Countries", total_countries)
    c2.metric("Regions", total_regions)
    c3.metric("States", total_states)
    c4.metric("Cities", total_cities)

    st.divider()

    # =====================================================
    # Region Sales
    # =====================================================

    st.subheader("🌎 Sales by Region")

    region = (
        data
        .groupby("Region")
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Orders=("Order ID", "count")
        )
        .reset_index()
        .sort_values("Sales", ascending=False)
    )

    fig = px.bar(
        region,
        x="Region",
        y="Sales",
        color="Sales",
        text_auto=".2s"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =====================================================
    # State Sales
    # =====================================================

    st.subheader("🏛 State-wise Sales")

    state = (
        data
        .groupby("State")
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum")
        )
        .reset_index()
        .sort_values("Sales", ascending=False)
    )

    fig = px.bar(
        state.head(20),
        x="Sales",
        y="State",
        orientation="h",
        color="Sales"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =====================================================
    # City Sales
    # =====================================================

    st.subheader("🏙 Top Cities")

    city = (
        data
        .groupby("City")
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum")
        )
        .reset_index()
        .sort_values("Sales", ascending=False)
    )

    fig = px.bar(
        city.head(20),
        x="Sales",
        y="City",
        orientation="h",
        color="Profit"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =====================================================
    # Profit by Region
    # =====================================================

    st.subheader("💰 Regional Profit")

    fig = px.pie(
        region,
        values="Profit",
        names="Region",
        hole=0.45
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =====================================================
    # Bubble Chart
    # =====================================================

    st.subheader("🫧 Sales vs Profit")

    fig = px.scatter(
        state,
        x="Sales",
        y="Profit",
        size="Sales",
        color="Profit",
        hover_name="State"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =====================================================
    # Treemap
    # =====================================================

    st.subheader("🌳 Geographic Treemap")

    fig = px.treemap(
        data,
        path=[
            "Country",
            "Region",
            "State",
            "City"
        ],
        values="Sales",
        color="Profit"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =====================================================
    # Geographic Summary
    # =====================================================

    st.subheader("📋 Geographic Summary")

    summary = (
        data
        .groupby(
            [
                "Country",
                "Region",
                "State"
            ]
        )
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Orders=("Order ID", "count")
        )
        .reset_index()
    )

    st.dataframe(
        summary,
        use_container_width=True
    )

    # =====================================================
    # Download
    # =====================================================

    csv = summary.to_csv(index=False)

    st.download_button(
        "📥 Download Geographic Report",
        csv,
        file_name="geographic_report.csv",
        mime="text/csv"
    )
