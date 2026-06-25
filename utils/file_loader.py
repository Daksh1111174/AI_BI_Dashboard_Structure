"""
===========================================================
File Loader Utility
AI Business Intelligence Dashboard

Author : Daksh Shah
===========================================================
"""

from pathlib import Path

import pandas as pd
import streamlit as st


SUPPORTED_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".xls",
    ".parquet",
    ".json"
}


@st.cache_data(show_spinner=False)
def load_dataset(uploaded_file):
    """
    Load uploaded dataset into a pandas DataFrame.
    """

    if uploaded_file is None:
        return None

    extension = Path(uploaded_file.name).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    if extension == ".csv":
        return pd.read_csv(uploaded_file)

    if extension in [".xlsx", ".xls"]:
        return pd.read_excel(uploaded_file)

    if extension == ".parquet":
        return pd.read_parquet(uploaded_file)

    if extension == ".json":
        return pd.read_json(uploaded_file)

    raise ValueError("Unable to load file.")
