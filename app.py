import streamlit as st
from config import *

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# Load CSS
# -------------------------------------------------

try:
    with open("assets/style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )
except:
    pass

# -------------------------------------------------
# Sidebar
# -------------------------------------------------

st.sidebar.image(
    "assets/logo.png",
    use_container_width=True
)

st.sidebar.title(APP_NAME)

st.sidebar.markdown("---")

st.sidebar.success("Version " + VERSION)

st.sidebar.info(
"""
Built using

✔ Streamlit

✔ Plotly

✔ Machine Learning

✔ AI Analytics
"""
)

# -------------------------------------------------
# Main Page
# -------------------------------------------------

st.title("📊 AI Business Intelligence Dashboard")

st.write(
"""
Welcome to the AI-powered Business Intelligence Dashboard.

This application provides:

- 📈 Sales Analytics
- 👥 Customer Analytics
- 📦 Product Analytics
- 🤖 AI Insights
- 🔮 Sales Forecasting
- 📄 Automated Reports
"""
)

st.markdown("---")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Sales",
    "₹0",
    "0%"
)

col2.metric(
    "Customers",
    "0",
    "0%"
)

col3.metric(
    "Orders",
    "0",
    "0%"
)

st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel File",
    type=SUPPORTED_FILES
)

if uploaded_file:

    st.success("Dataset Uploaded Successfully!")

    st.write("Filename:", uploaded_file.name)

    st.info(
        "Use the pages in the sidebar to begin analysis."
    )

else:

    st.warning("Please upload a dataset.")
