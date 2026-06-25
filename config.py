"""
Configuration file for AI Business Intelligence Dashboard
"""

from pathlib import Path

# ==============================
# Project Information
# ==============================

APP_NAME = "AI Business Intelligence Dashboard"
VERSION = "1.0.0"
AUTHOR = "Daksh Shah"

# ==============================
# Directory Paths
# ==============================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"
REPORT_DIR = BASE_DIR / "reports"
MODEL_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

# Create directories if missing
for folder in [DATA_DIR, UPLOAD_DIR, REPORT_DIR, MODEL_DIR]:
    folder.mkdir(exist_ok=True)

# ==============================
# Dashboard Settings
# ==============================

PAGE_TITLE = APP_NAME
PAGE_ICON = "📊"
LAYOUT = "wide"

# ==============================
# Upload Settings
# ==============================

SUPPORTED_FILES = [
    "csv",
    "xlsx",
    "xls"
]

MAX_UPLOAD_SIZE_MB = 200

# ==============================
# Currency
# ==============================

DEFAULT_CURRENCY = "₹"

# ==============================
# Charts
# ==============================

DEFAULT_THEME = "plotly"

# ==============================
# Forecasting
# ==============================

FORECAST_DAYS = 30

# ==============================
# Colors
# ==============================

PRIMARY_COLOR = "#2563EB"
SECONDARY_COLOR = "#14B8A6"
SUCCESS_COLOR = "#16A34A"
WARNING_COLOR = "#F59E0B"
DANGER_COLOR = "#DC2626"
