"""
===========================================================
Dataset Validation
===========================================================
"""

REQUIRED_COLUMNS = [
    "Order ID",
    "Order Date",
    "Customer ID",
    "Product Name",
    "Category",
    "Sales",
    "Profit"
]


def validate(df):

    missing = []

    for col in REQUIRED_COLUMNS:

        if col not in df.columns:
            missing.append(col)

    return missing
