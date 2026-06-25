"""
=========================================================
Executive Dashboard
AI Business Intelligence Dashboard

Author : Daksh Shah
=========================================================
"""

import streamlit as st

from components.filters import dashboard_filters
from components.kpi_cards import show_kpis

from components.revenue_charts import revenue_dashboard
from components.category_charts import category_dashboard
from components.customer_charts import customer_dashboard
from components.product_charts import product_dashboard

from components.maps import geographic_dashboard
from components.heatmaps import correlation_dashboard
from components.waterfall import waterfall_dashboard
from components.pareto import pareto_dashboard
from components.sankey import sankey_dashboard

from components.ai_summary import ai_summary
from components.exports import export_dashboard

# -------------------------------------------------------
# Page Config
# -------------------------------------------------------

st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Executive Dashboard")

# -------------------------------------------------------
# Check Dataset
# -------------------------------------------------------

if "dataset" not in st.session_state:
    st.warning("Upload a dataset from the Home page.")
    st.stop()

df = st.session_state.dataset.copy()

# -------------------------------------------------------
# Apply Filters
# -------------------------------------------------------

filtered_df = dashboard_filters(df)

# -------------------------------------------------------
# KPI Cards
# -------------------------------------------------------

show_kpis(filtered_df)

st.divider()

# -------------------------------------------------------
# Revenue Analytics
# -------------------------------------------------------

revenue_dashboard(filtered_df)

st.divider()

# -------------------------------------------------------
# Category Analytics
# -------------------------------------------------------

category_dashboard(filtered_df)

st.divider()

# -------------------------------------------------------
# Customer Analytics
# -------------------------------------------------------

customer_dashboard(filtered_df)

st.divider()

# -------------------------------------------------------
# Product Analytics
# -------------------------------------------------------

product_dashboard(filtered_df)

st.divider()

# -------------------------------------------------------
# Maps
# -------------------------------------------------------

geographic_dashboard(filtered_df)

st.divider()

# -------------------------------------------------------
# Correlation
# -------------------------------------------------------

correlation_dashboard(filtered_df)

st.divider()

# -------------------------------------------------------
# Waterfall
# -------------------------------------------------------

waterfall_dashboard(filtered_df)

st.divider()

# -------------------------------------------------------
# Pareto
# -------------------------------------------------------

pareto_dashboard(filtered_df)

st.divider()

# -------------------------------------------------------
# Sankey
# -------------------------------------------------------

sankey_dashboard(filtered_df)

st.divider()

# -------------------------------------------------------
# AI Summary
# -------------------------------------------------------

ai_summary(filtered_df)

st.divider()

# -------------------------------------------------------
# Export
# -------------------------------------------------------

export_dashboard(filtered_df)
