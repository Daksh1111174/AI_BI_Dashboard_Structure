import streamlit as st
from config import *

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)

st.title("📊 AI Business Intelligence Dashboard")

st.markdown("""
Welcome to the AI Business Intelligence Dashboard.

Use the sidebar to navigate through:

- 🏠 Home
- 📊 Dashboard
- 💰 Sales Analytics
- 👥 Customer Analytics
- 📦 Product Analytics
- 📈 Forecasting
- 🤖 AI Insights
- 📄 Report Generator
- 📉 Model Performance
- ⚙ Settings
""")

st.success("Select a page from the sidebar.")
