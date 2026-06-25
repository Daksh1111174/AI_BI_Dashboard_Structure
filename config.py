"""
==========================================================
AI Business Intelligence Dashboard
Configuration File
Author : Daksh Shah
Version : 1.0.0
==========================================================
"""

from pathlib import Path

# ==========================================================
# Project Information
# ==========================================================

APP_NAME = "AI Business Intelligence Dashboard"
APP_VERSION = "1.0.0"
AUTHOR = "Daksh Shah"

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

ASSETS_DIR = BASE_DIR / "assets"
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

UPLOAD_DIR = BASE_DIR / "uploads"
REPORT_DIR = BASE_DIR / "reports"
MODEL_DIR = BASE_DIR / "models"

NOTEBOOK_DIR = BASE_DIR / "notebooks"
DOCS_DIR = BASE_DIR / "docs"

# Automatically create folders

DIRECTORIES = [
    ASSETS_DIR,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    UPLOAD_DIR,
    REPORT_DIR,
    MODEL_DIR,
    NOTEBOOK_DIR,
    DOCS_DIR
]

for directory in DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Streamlit Settings
# ==========================================================

PAGE_TITLE = APP_NAME
PAGE_ICON = "📊"
LAYOUT = "wide"

# ==========================================================
# Upload Settings
# ==========================================================

SUPPORTED_FILE_TYPES = [
    "csv",
    "xlsx",
    "xls"
]

MAX_UPLOAD_SIZE_MB = 200

# ==========================================================
# Currency
# ==========================================================

DEFAULT_CURRENCY = "₹"

# ==========================================================
# Theme
# ==========================================================

PRIMARY_COLOR = "#2563EB"
SECONDARY_COLOR = "#14B8A6"
SUCCESS_COLOR = "#16A34A"
WARNING_COLOR = "#F59E0B"
ERROR_COLOR = "#DC2626"

PLOTLY_THEME = "plotly"

# ==========================================================
# Dashboard Defaults
# ==========================================================

DEFAULT_FORECAST_DAYS = 30

DEFAULT_TOP_PRODUCTS = 10

DEFAULT_CHART_HEIGHT = 450

# ==========================================================
# Dataset Columns
# ==========================================================

REQUIRED_COLUMNS = [
    "Order ID",
    "Order Date",
    "Customer ID",
    "Customer Name",
    "Country",
    "State",
    "City",
    "Region",
    "Category",
    "Sub-Category",
    "Product Name",
    "Sales",
    "Quantity",
    "Discount",
    "Profit"
]

OPTIONAL_COLUMNS = [
    "Shipping Cost",
    "Payment Mode",
    "Ship Date",
    "Segment"
]

# ==========================================================
# Machine Learning
# ==========================================================

FORECAST_MODELS = [
    "Prophet",
    "Random Forest",
    "Linear Regression",
    "XGBoost"
]

# ==========================================================
# Cache
# ==========================================================

CACHE_TIME = 3600

# ==========================================================
# Export Formats
# ==========================================================

EXPORT_FORMATS = [
    "CSV",
    "Excel",
    "PDF"
]
