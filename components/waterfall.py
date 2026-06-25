"""
===========================================================
Financial Waterfall
AI Business Intelligence Dashboard

Author : Daksh Shah
===========================================================
"""

import streamlit as st
import plotly.graph_objects as go


def waterfall_dashboard(df):
    """
    Financial Waterfall Dashboard
    """

    required = ["Sales", "Profit"]

    missing = [col for col in required if col not in df.columns]

    if missing:
        st.warning(f"Missing columns: {', '.join(missing)}")
        return

    st.header("📉 Financial Waterfall Analysis")

    # -----------------------------------------------------
    # Financial Values
    # -----------------------------------------------------

    revenue = df["Sales"].sum()

    discount = (
        df["Discount"].sum()
        if "Discount" in df.columns
        else 0
    )

    shipping = (
        df["Shipping Cost"].sum()
        if "Shipping Cost" in df.columns
        else 0
    )

    profit = df["Profit"].sum()

    # -----------------------------------------------------
    # KPI Cards
    # -----------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Revenue", f"₹ {revenue:,.0f}")
    c2.metric("Discount", f"₹ {discount:,.0f}")
    c3.metric("Shipping", f"₹ {shipping:,.0f}")
    c4.metric("Profit", f"₹ {profit:,.0f}")

    st.divider()

    # -----------------------------------------------------
    # Waterfall Chart
    # -----------------------------------------------------

    fig = go.Figure(
        go.Waterfall(
            orientation="v",

            measure=[
                "relative",
                "relative",
                "relative",
                "total"
            ],

            x=[
                "Revenue",
                "Discount",
                "Shipping",
                "Profit"
            ],

            y=[
                revenue,
                -discount,
                -shipping,
                profit
            ],

            connector={
                "line": {
                    "color": "gray"
                }
            }
        )
    )

    fig.update_layout(
        title="Revenue → Discount → Shipping → Profit",
        height=600
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -----------------------------------------------------
    # Financial Breakdown
    # -----------------------------------------------------

    st.subheader("💰 Financial Breakdown")

    margin = (
        (profit / revenue) * 100
        if revenue else 0
    )

    discount_pct = (
        (discount / revenue) * 100
        if revenue else 0
    )

    shipping_pct = (
        (shipping / revenue) * 100
        if revenue else 0
    )

    summary = {
        "Metric": [
            "Revenue",
            "Discount",
            "Shipping Cost",
            "Profit",
            "Profit Margin %",
            "Discount %",
            "Shipping %"
        ],
        "Value": [
            revenue,
            discount,
            shipping,
            profit,
            round(margin, 2),
            round(discount_pct, 2),
            round(shipping_pct, 2)
        ]
    }

    st.dataframe(
        summary,
        use_container_width=True
    )

    # -----------------------------------------------------
    # Financial Insights
    # -----------------------------------------------------

    st.subheader("🤖 Financial Insights")

    if margin >= 20:
        st.success(
            f"Excellent profit margin ({margin:.2f}%)."
        )

    elif margin >= 10:
        st.info(
            f"Healthy profit margin ({margin:.2f}%)."
        )

    else:
        st.warning(
            f"Low profit margin ({margin:.2f}%)."
        )

    if discount_pct > 15:
        st.warning(
            f"Discounts consume {discount_pct:.2f}% of revenue."
        )

    if shipping_pct > 10:
        st.warning(
            f"Shipping cost is {shipping_pct:.2f}% of revenue."
        )

    if profit > 0:
        st.success(
            "Business is profitable."
        )
    else:
        st.error(
            "Business is currently operating at a loss."
        )

    # -----------------------------------------------------
    # Download
    # -----------------------------------------------------

    csv = (
        "Metric,Value\n"
        f"Revenue,{revenue}\n"
        f"Discount,{discount}\n"
        f"Shipping,{shipping}\n"
        f"Profit,{profit}\n"
        f"Profit Margin,{margin:.2f}\n"
    )

    st.download_button(
        "📥 Download Financial Summary",
        csv,
        file_name="financial_summary.csv",
        mime="text/csv"
    )
