"""
===========================================================
Sankey Diagram
AI Business Intelligence Dashboard

Author : Daksh Shah
===========================================================
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def sankey_dashboard(df):
    """
    Region -> Category -> Sub-Category Sankey
    """

    required = [
        "Region",
        "Category",
        "Sub-Category",
        "Sales"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        st.warning(
            f"Missing columns: {', '.join(missing)}"
        )
        return

    st.header("📡 Sales Flow Analysis")

    # ---------------------------------------
    # Aggregate Sales
    # ---------------------------------------

    grouped = (
        df.groupby(
            [
                "Region",
                "Category",
                "Sub-Category"
            ]
        )["Sales"]
        .sum()
        .reset_index()
    )

    # ---------------------------------------
    # Create Node List
    # ---------------------------------------

    nodes = pd.unique(
        pd.concat(
            [
                grouped["Region"],
                grouped["Category"],
                grouped["Sub-Category"]
            ]
        )
    )

    node_map = {
        value: index
        for index, value in enumerate(nodes)
    }

    source = []
    target = []
    value = []

    # ---------------------------------------
    # Region -> Category
    # ---------------------------------------

    region_category = (
        grouped
        .groupby(
            ["Region", "Category"]
        )["Sales"]
        .sum()
        .reset_index()
    )

    for _, row in region_category.iterrows():

        source.append(
            node_map[row["Region"]]
        )

        target.append(
            node_map[row["Category"]]
        )

        value.append(
            row["Sales"]
        )

    # ---------------------------------------
    # Category -> SubCategory
    # ---------------------------------------

    category_sub = (
        grouped
        .groupby(
            [
                "Category",
                "Sub-Category"
            ]
        )["Sales"]
        .sum()
        .reset_index()
    )

    for _, row in category_sub.iterrows():

        source.append(
            node_map[row["Category"]]
        )

        target.append(
            node_map[row["Sub-Category"]]
        )

        value.append(
            row["Sales"]
        )

    # ---------------------------------------
    # Sankey
    # ---------------------------------------

    fig = go.Figure(

        go.Sankey(

            arrangement="snap",

            node=dict(

                pad=20,

                thickness=20,

                line=dict(
                    color="black",
                    width=0.5
                ),

                label=list(nodes)

            ),

            link=dict(

                source=source,

                target=target,

                value=value

            )

        )

    )

    fig.update_layout(

        title="Sales Flow",

        height=750

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ---------------------------------------
    # KPIs
    # ---------------------------------------

    st.subheader("Flow Statistics")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Regions",
        grouped["Region"].nunique()
    )

    c2.metric(
        "Categories",
        grouped["Category"].nunique()
    )

    c3.metric(
        "Sub-Categories",
        grouped["Sub-Category"].nunique()
    )

    # ---------------------------------------
    # Summary
    # ---------------------------------------

    st.subheader("Flow Summary")

    st.dataframe(
        grouped,
        use_container_width=True
    )

    # ---------------------------------------
    # Download
    # ---------------------------------------

    csv = grouped.to_csv(index=False)

    st.download_button(
        "📥 Download Flow Data",
        csv,
        file_name="sales_flow.csv",
        mime="text/csv"
    )

    # ---------------------------------------
    # Insights
    # ---------------------------------------

    st.subheader("🤖 AI Insights")

    top_region = (
        grouped.groupby("Region")["Sales"]
        .sum()
        .idxmax()
    )

    top_category = (
        grouped.groupby("Category")["Sales"]
        .sum()
        .idxmax()
    )

    top_sub = (
        grouped.groupby("Sub-Category")["Sales"]
        .sum()
        .idxmax()
    )

    st.success(
        f"Highest revenue region: **{top_region}**"
    )

    st.success(
        f"Highest revenue category: **{top_category}**"
    )

    st.success(
        f"Highest revenue sub-category: **{top_sub}**"
    )

    st.info(
        "Use this flow analysis to understand "
        "where revenue is concentrated across "
        "your business hierarchy."
    )
