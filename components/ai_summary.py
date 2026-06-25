"""
===========================================================
AI Business Summary
AI Business Intelligence Dashboard
===========================================================
"""

import streamlit as st
import pandas as pd


def ai_summary(df):

    st.header("🤖 AI Business Insights")

    revenue = df["Sales"].sum()

    profit = df["Profit"].sum()

    orders = df["Order ID"].nunique()

    customers = df["Customer ID"].nunique()

    margin = (profit / revenue) * 100 if revenue else 0

    # ---------------------------------------------

    region = (
        df.groupby("Region")["Profit"]
        .sum()
        .sort_values(ascending=False)
    )

    category = (
        df.groupby("Category")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    product = (
        df.groupby("Product Name")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    # ---------------------------------------------

    best_region = region.index[0]
    worst_region = region.index[-1]

    best_category = category.index[0]
    worst_category = category.index[-1]

    best_product = product.index[0]
    worst_product = product.index[-1]

    # =============================================
    # Executive Summary
    # =============================================

    st.subheader("📄 Executive Summary")

    st.info(f"""
Revenue generated is **₹ {revenue:,.0f}**.

Total profit is **₹ {profit:,.0f}**.

Profit Margin is **{margin:.2f}%**.

The business served **{customers:,}** customers across **{orders:,}** orders.
""")

    # =============================================
    # Insights
    # =============================================

    st.subheader("📈 Key Insights")

    insights = [

        f"🏆 {best_region} generated the highest profit.",

        f"📦 {best_category} contributed the highest sales.",

        f"⭐ Best selling product is {best_product}.",

        f"⚠ Lowest selling product is {worst_product}.",

        f"📉 {worst_region} has the lowest profitability."

    ]

    for item in insights:

        st.success(item)

    # =============================================
    # Risk Alerts
    # =============================================

    st.subheader("🚨 Risk Alerts")

    if margin < 10:

        st.error(
            "Low profit margin detected."
        )

    if "Discount" in df.columns:

        avg_discount = df["Discount"].mean()

        if avg_discount > 20:

            st.warning(
                "Average discount is too high."
            )

    if profit < 0:

        st.error(
            "Business is operating at a loss."
        )

    # =============================================
    # Opportunities
    # =============================================

    st.subheader("💡 Opportunities")

    opportunities = [

        "Increase inventory of best-selling products.",

        "Expand operations in high-profit regions.",

        "Launch promotions for slow-moving products.",

        "Focus marketing on premium categories.",

        "Improve customer retention."

    ]

    for item in opportunities:

        st.write("✅", item)

    # =============================================
    # Recommendation Engine
    # =============================================

    st.subheader("🧠 AI Recommendations")

    recommendations = []

    if margin < 10:

        recommendations.append(
            "Increase pricing or reduce operational costs."
        )

    if avg_discount > 20:

        recommendations.append(
            "Reduce excessive discount campaigns."
        )

    recommendations.extend([

        "Increase marketing budget for best-selling categories.",

        "Improve stock planning for top products.",

        "Target loyal customers with rewards."

    ])

    for rec in recommendations:

        st.info(rec)

    # =============================================
    # Business Health
    # =============================================

    st.subheader("🏥 Business Health")

    if margin >= 20:

        st.success("Excellent financial health.")

    elif margin >= 10:

        st.info("Business is healthy.")

    else:

        st.warning("Business requires attention.")

    # =============================================
    # Executive Report
    # =============================================

    report = f"""
EXECUTIVE SUMMARY

Revenue : ₹ {revenue:,.0f}

Profit : ₹ {profit:,.0f}

Margin : {margin:.2f}%

Best Region : {best_region}

Best Category : {best_category}

Best Product : {best_product}

Recommendations

1. Increase inventory.

2. Reduce discounts.

3. Improve marketing.

4. Expand profitable regions.

5. Improve customer retention.

"""

    st.download_button(

        "📥 Download Executive Report",

        report,

        file_name="executive_summary.txt"

    )
