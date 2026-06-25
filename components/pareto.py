"""
===========================================================
Pareto Analysis (80/20 Rule)
AI Business Intelligence Dashboard

Author : Daksh Shah
===========================================================
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def pareto_dashboard(df):
    """
    Pareto Analysis Dashboard
    """

    required = ["Product Name", "Sales"]

    missing = [c for c in required if c not in df.columns]

    if missing:
        st.warning(f"Missing columns: {', '.join(missing)}")
        return

    st.header("🎯 Pareto Analysis (80/20 Rule)")

    # -------------------------------------------------------
    # Aggregate Sales
    # -------------------------------------------------------

    pareto = (
        df.groupby("Product Name")
        .agg(
            Sales=("Sales", "sum")
        )
        .reset_index()
        .sort_values("Sales", ascending=False)
    )

    pareto["Cumulative Sales"] = pareto["Sales"].cumsum()

    total_sales = pareto["Sales"].sum()

    pareto["Cumulative %"] = (
        pareto["Cumulative Sales"] /
        total_sales
    ) * 100

    # -------------------------------------------------------
    # Pareto Chart
    # -------------------------------------------------------

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=pareto["Product Name"],
            y=pareto["Sales"],
            name="Sales"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=pareto["Product Name"],
            y=pareto["Cumulative %"],
            mode="lines+markers",
            name="Cumulative %",
            yaxis="y2"
        )
    )

    fig.update_layout(

        title="Pareto Analysis",

        height=650,

        xaxis=dict(
            title="Products"
        ),

        yaxis=dict(
            title="Sales"
        ),

        yaxis2=dict(
            title="Cumulative %",
            overlaying="y",
            side="right",
            range=[0, 100]
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -------------------------------------------------------
    # Top 80%
    # -------------------------------------------------------

    top80 = pareto[
        pareto["Cumulative %"] <= 80
    ]

    st.subheader("🏆 Products Generating ~80% of Sales")

    st.dataframe(
        top80,
        use_container_width=True
    )

    # -------------------------------------------------------
    # KPI Cards
    # -------------------------------------------------------

    total_products = len(pareto)

    top_products = len(top80)

    contribution = (
        top80["Sales"].sum() /
        total_sales
    ) * 100

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Products",
        total_products
    )

    c2.metric(
        "Top Products",
        top_products
    )

    c3.metric(
        "Revenue Contribution",
        f"{contribution:.2f}%"
    )

    st.divider()

    # -------------------------------------------------------
    # AI Insights
    # -------------------------------------------------------

    st.subheader("🤖 AI Insights")

    st.success(
        f"{top_products} products generate "
        f"{contribution:.2f}% of total revenue."
    )

    st.info(
        "These products should receive priority "
        "for inventory planning and marketing."
    )

    if contribution > 85:

        st.warning(
            "Revenue is highly dependent on "
            "a small number of products."
        )

    elif contribution < 60:

        st.success(
            "Revenue is well diversified "
            "across products."
        )

    # -------------------------------------------------------
    # Recommendations
    # -------------------------------------------------------

    st.subheader("💡 Recommendations")

    recommendations = [

        "Maintain sufficient inventory for A-category products.",

        "Increase marketing for high-performing products.",

        "Review low-selling products for discontinuation.",

        "Bundle slow-moving products with best sellers.",

        "Monitor demand changes monthly."
    ]

    for item in recommendations:
        st.write("✅", item)

    # -------------------------------------------------------
    # Download
    # -------------------------------------------------------

    csv = pareto.to_csv(index=False)

    st.download_button(
        "📥 Download Pareto Report",
        csv,
        file_name="pareto_analysis.csv",
        mime="text/csv"
    )
