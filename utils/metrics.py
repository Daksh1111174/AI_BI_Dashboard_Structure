"""
===========================================================
Business Metrics
===========================================================
"""

import pandas as pd


def calculate_metrics(df):

    metrics = {}

    metrics["Revenue"] = df["Sales"].sum()

    metrics["Profit"] = df["Profit"].sum()

    metrics["Orders"] = df["Order ID"].nunique()

    metrics["Customers"] = df["Customer ID"].nunique()

    metrics["Quantity"] = (
        df["Quantity"].sum()
        if "Quantity" in df.columns
        else 0
    )

    metrics["Average Order Value"] = (
        metrics["Revenue"] / metrics["Orders"]
        if metrics["Orders"]
        else 0
    )

    metrics["Profit Margin"] = (
        (metrics["Profit"] / metrics["Revenue"]) * 100
        if metrics["Revenue"]
        else 0
    )

    return metrics
