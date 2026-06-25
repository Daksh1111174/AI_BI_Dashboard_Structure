"""
===========================================================
Heatmaps & Correlation Analysis
AI Business Intelligence Dashboard
===========================================================
"""

import streamlit as st
import pandas as pd
import plotly.express as px


def correlation_dashboard(df):

    st.header("🔥 Correlation & Heatmap Analysis")

    numeric = df.select_dtypes(include="number")

    if numeric.shape[1] < 2:
        st.warning("Not enough numeric columns.")
        return

    # =====================================================
    # Correlation Matrix
    # =====================================================

    st.subheader("📊 Correlation Matrix")

    corr = numeric.corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )

    fig.update_layout(height=650)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =====================================================
    # Missing Values Heatmap
    # =====================================================

    st.subheader("🧹 Missing Values")

    missing = (
        df.isna()
        .sum()
        .reset_index()
    )

    missing.columns = [
        "Column",
        "Missing Values"
    ]

    fig = px.bar(
        missing,
        x="Column",
        y="Missing Values",
        color="Missing Values",
        text_auto=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =====================================================
    # Correlation Table
    # =====================================================

    st.subheader("📋 Correlation Table")

    st.dataframe(
        corr.round(2),
        use_container_width=True
    )

    # =====================================================
    # Scatter Matrix
    # =====================================================

    st.subheader("🫧 Scatter Matrix")

    cols = numeric.columns[:5]

    fig = px.scatter_matrix(
        df,
        dimensions=cols
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =====================================================
    # Distribution
    # =====================================================

    st.subheader("📈 Numeric Distribution")

    column = st.selectbox(
        "Select Numeric Column",
        numeric.columns
    )

    fig = px.histogram(
        df,
        x=column,
        marginal="box",
        nbins=40
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =====================================================
    # Box Plot
    # =====================================================

    st.subheader("📦 Outlier Detection")

    fig = px.box(
        df,
        y=column
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =====================================================
    # Download Correlation
    # =====================================================

    csv = corr.to_csv()

    st.download_button(
        "📥 Download Correlation Matrix",
        csv,
        file_name="correlation_matrix.csv",
        mime="text/csv"
    )
