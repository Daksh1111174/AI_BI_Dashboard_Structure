"""
===========================================================
Data Preprocessing
===========================================================
"""

import pandas as pd


def preprocess(df):

    data = df.copy()

    # Remove duplicate rows
    data = data.drop_duplicates()

    # Remove completely empty rows
    data = data.dropna(how="all")

    # Convert dates
    if "Order Date" in data.columns:
        data["Order Date"] = pd.to_datetime(
            data["Order Date"],
            errors="coerce"
        )

    if "Ship Date" in data.columns:
        data["Ship Date"] = pd.to_datetime(
            data["Ship Date"],
            errors="coerce"
        )

    # Fill numeric NaNs
    numeric = data.select_dtypes(include="number").columns

    data[numeric] = data[numeric].fillna(0)

    # Fill text NaNs
    text = data.select_dtypes(include="object").columns

    data[text] = data[text].fillna("Unknown")

    return data
