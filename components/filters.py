"""
===========================================================
Dashboard Filters
===========================================================
"""

import pandas as pd
import streamlit as st


def dashboard_filters(df):
    """
    Apply all dashboard filters and
    return filtered dataframe.
    """

    filtered_df = df.copy()

    st.sidebar.header("🔎 Dashboard Filters")

    # ----------------------------------------
    # Convert Date
    # ----------------------------------------

    if "Order Date" in filtered_df.columns:

        filtered_df["Order Date"] = pd.to_datetime(
            filtered_df["Order Date"],
            errors="coerce"
        )

    # ----------------------------------------
    # Date Range
    # ----------------------------------------

    if "Order Date" in filtered_df.columns:

        min_date = filtered_df["Order Date"].min()
        max_date = filtered_df["Order Date"].max()

        if pd.notna(min_date) and pd.notna(max_date):

            start_date, end_date = st.sidebar.date_input(
                "📅 Date Range",
                value=(min_date.date(), max_date.date())
            )

            filtered_df = filtered_df[
                (
                    filtered_df["Order Date"]
                    >= pd.to_datetime(start_date)
                )
                &
                (
                    filtered_df["Order Date"]
                    <= pd.to_datetime(end_date)
                )
            ]

    # ----------------------------------------
    # Region
    # ----------------------------------------

    if "Region" in filtered_df.columns:

        values = sorted(filtered_df["Region"].dropna().unique())

        selected = st.sidebar.multiselect(
            "🌍 Region",
            values,
            default=values
        )

        filtered_df = filtered_df[
            filtered_df["Region"].isin(selected)
        ]

    # ----------------------------------------
    # State
    # ----------------------------------------

    if "State" in filtered_df.columns:

        values = sorted(filtered_df["State"].dropna().unique())

        selected = st.sidebar.multiselect(
            "🏛 State",
            values,
            default=values
        )

        filtered_df = filtered_df[
            filtered_df["State"].isin(selected)
        ]

    # ----------------------------------------
    # City
    # ----------------------------------------

    if "City" in filtered_df.columns:

        values = sorted(filtered_df["City"].dropna().unique())

        selected = st.sidebar.multiselect(
            "🏙 City",
            values,
            default=values
        )

        filtered_df = filtered_df[
            filtered_df["City"].isin(selected)
        ]

    # ----------------------------------------
    # Category
    # ----------------------------------------

    if "Category" in filtered_df.columns:

        values = sorted(filtered_df["Category"].dropna().unique())

        selected = st.sidebar.multiselect(
            "📦 Category",
            values,
            default=values
        )

        filtered_df = filtered_df[
            filtered_df["Category"].isin(selected)
        ]

    # ----------------------------------------
    # Sub Category
    # ----------------------------------------

    if "Sub-Category" in filtered_df.columns:

        values = sorted(filtered_df["Sub-Category"].dropna().unique())

        selected = st.sidebar.multiselect(
            "📦 Sub-Category",
            values,
            default=values
        )

        filtered_df = filtered_df[
            filtered_df["Sub-Category"].isin(selected)
        ]

    # ----------------------------------------
    # Segment
    # ----------------------------------------

    if "Segment" in filtered_df.columns:

        values = sorted(filtered_df["Segment"].dropna().unique())

        selected = st.sidebar.multiselect(
            "👥 Segment",
            values,
            default=values
        )

        filtered_df = filtered_df[
            filtered_df["Segment"].isin(selected)
        ]

    # ----------------------------------------
    # Payment Mode
    # ----------------------------------------

    if "Payment Mode" in filtered_df.columns:

        values = sorted(filtered_df["Payment Mode"].dropna().unique())

        selected = st.sidebar.multiselect(
            "💳 Payment Mode",
            values,
            default=values
        )

        filtered_df = filtered_df[
            filtered_df["Payment Mode"].isin(selected)
        ]

    # ----------------------------------------
    # Product
    # ----------------------------------------

    if "Product Name" in filtered_df.columns:

        values = sorted(filtered_df["Product Name"].dropna().unique())

        selected = st.sidebar.multiselect(
            "📦 Product",
            values
        )

        if selected:

            filtered_df = filtered_df[
                filtered_df["Product Name"].isin(selected)
            ]

    # ----------------------------------------
    # Customer
    # ----------------------------------------

    if "Customer Name" in filtered_df.columns:

        values = sorted(filtered_df["Customer Name"].dropna().unique())

        selected = st.sidebar.multiselect(
            "👤 Customer",
            values
        )

        if selected:

            filtered_df = filtered_df[
                filtered_df["Customer Name"].isin(selected)
            ]

    # ----------------------------------------
    # Search
    # ----------------------------------------

    if "Product Name" in filtered_df.columns:

        keyword = st.sidebar.text_input(
            "🔍 Search Product"
        )

        if keyword:

            filtered_df = filtered_df[
                filtered_df["Product Name"]
                .str.contains(
                    keyword,
                    case=False,
                    na=False
                )
            ]

    # ----------------------------------------
    # Summary
    # ----------------------------------------

    st.sidebar.markdown("---")

    st.sidebar.success(
        f"{len(filtered_df):,} records selected"
    )

    return filtered_df
