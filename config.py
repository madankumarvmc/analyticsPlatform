#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration file for Warehouse Analysis Tool
Contains all constants, paths, and settings used across the application.
"""

import os
from pathlib import Path

# =============================================================================
# ENVIRONMENT DETECTION
# =============================================================================

def is_streamlit_cloud():
    """Detect if running on Streamlit Cloud."""
    return (
        os.environ.get('STREAMLIT_SHARING_MODE') == '1' or
        os.environ.get('STREAMLIT_SERVER_HEADLESS') == 'true' or
        'streamlit' in os.environ.get('HOME', '').lower() or
        '/mount/src/' in os.getcwd()
    )

def is_local_development():
    """Detect if running in local development."""
    return not is_streamlit_cloud()

# =============================================================================
# DATA PATHS AND FILES
# =============================================================================

# Main data file location - environment dependent
if is_streamlit_cloud():
    # On Streamlit Cloud, no default file path (must use uploaded files)
    DATA_FILE_PATH = None
else:
    # Local development path
    DATA_FILE_PATH = "/Users/MKSBX/Documents/Analytics Tool/TestData.xlsx"

# Sheet names in the Excel file
ORDER_DATA_SHEET = "OrderData"
SKU_MASTER_SHEET = "SkuMaster"

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

# Output file names
EXCEL_OUTPUT_FILE = "Order_Profiles.xlsx"

# Report directories - environment dependent
if is_streamlit_cloud():
    # On Streamlit Cloud, use temp directory for outputs
    import tempfile
    TEMP_DIR = Path(tempfile.gettempdir())
    REPORT_DIR = TEMP_DIR / "warehouse_analysis_reports"
else:
    # Local development
    REPORT_DIR = Path("report")

CHARTS_DIR = REPORT_DIR / "charts"
ASSETS_DIR = REPORT_DIR / "assets"

# Output files
HTML_FILE = REPORT_DIR / "Order_Profiles_Analysis.html"
METADATA_FILE = REPORT_DIR / "metadata.json"
CACHE_FILE = REPORT_DIR / "llm_cache.json"

# Ensure directories exist
REPORT_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)
ASSETS_DIR.mkdir(exist_ok=True)

# =============================================================================
# ANALYSIS PARAMETERS
# =============================================================================

# ABC Classification thresholds (cumulative percentages)
ABC_THRESHOLDS = {
    'A_THRESHOLD': 70.0,  # < 70% = A
    'B_THRESHOLD': 90.0   # 70-90% = B, >90% = C
}

# FMS Classification thresholds (cumulative percentages)
FMS_THRESHOLDS = {
    'F_THRESHOLD': 70.0,  # < 70% = Fast
    'M_THRESHOLD': 90.0   # 70-90% = Medium, >90% = Slow
}

# Percentile calculations
PERCENTILE_LEVELS = [95, 90, 85]  # Percentiles to calculate
PERCENTILE_LABELS = ["Max", "95%ile", "90%ile", "85%ile", "Average"]

# Metrics for aggregation (excluding Date)
AGGREGATION_METRICS = [
    "Distinct_Customers",
    "Distinct_Shipments", 
    "Distinct_Orders",
    "Distinct_SKUs",
    "Qty_Ordered_Cases",
    "Qty_Ordered_Eaches",
    "Total_Case_Equiv",
    "Total_Pallet_Equiv"
]

# =============================================================================
# LLM/GEMINI CONFIGURATION
# =============================================================================

# LLM Settings
USE_GEMINI = True  # Set False to skip LLM calls
GEMINI_API_KEY = "AIzaSyD3-HabX9Oc2Q_0R-wywpRk8QZ03Z7HHds"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
GEMINI_MODEL = "gemini-2.0-flash"
LLM_TIMEOUT = 30  # seconds

# =============================================================================
# CHART CONFIGURATION
# =============================================================================

# Chart settings
CHART_DPI = 150
CHART_CONFIGS = {
    'date_line_chart': {'figsize': (10, 4), 'title': 'Total Case Equivalent by Date'},
    'date_customers_chart': {'figsize': (10, 3), 'title': 'Distinct Customers by Date'},
    'percentile_chart': {'figsize': (8, 3), 'title': 'Percentiles for Total Case Equivalent (per day)'},
    'sku_pareto_chart': {'figsize': (10, 4), 'title': 'SKU Pareto', 'top_n': 50},
    'abc_volume_chart': {'figsize': (8, 4), 'title': 'Volume by ABC and FMS (stacked)'},
    'abc_heatmap_chart': {'figsize': (6, 4), 'title': 'ABC x FMS Volume % (row-wise)'}
}

# Chart file names
CHART_FILES = {
    'date_total_case_equiv': 'date_total_case_equiv.png',
    'date_distinct_customers': 'date_distinct_customers.png',
    'percentile_total_case_equiv': 'percentile_total_case_equiv.png',
    'sku_pareto': 'sku_pareto.png',
    'abc_volume_stacked': 'abc_volume_stacked.png',
    'abc_fms_heatmap': 'abc_fms_heatmap.png'
}

# =============================================================================
# REPORT CONFIGURATION
# =============================================================================

# HTML Report settings
TOP_N_TABLE_ROWS = 50  # Show top N rows in HTML tables
OPEN_AFTER_BUILD = False  # Auto-open HTML in browser

# Excel sheet names
EXCEL_SHEETS = {
    'date_summary': 'Date Order Summary',
    'sku_summary': 'SKU Order Summary', 
    'percentile': 'Order Profile(Percentile)',
    'sku_abc_fms': 'SKU_Profile_ABC_FMS',
    'abc_fms_summary': 'ABC_FMS_Summary'
}

# =============================================================================
# COLUMN MAPPINGS
# =============================================================================

# Expected columns in order data
ORDER_DATA_COLUMNS = {
    'date': 'Date',
    'shipment_no': 'Shipment No.',
    'order_no': 'Order No.',
    'sku_code': 'Sku Code',
    'qty_cases': 'Qty in Cases',
    'qty_eaches': 'Qty in Eaches'
}

# Expected columns in SKU master
SKU_MASTER_COLUMNS = {
    'sku_code': 'Sku Code',
    'category': 'Category',
    'case_config': 'Case Config',
    'pallet_fit': 'Pallet Fit'
}

# Volume column candidates for SKU analysis
VOLUME_COLUMN_CANDIDATES = [
    "Order_Volume_CE", 
    "Order_Volume", 
    "Total_Case_Equiv", 
    "Case_Equivalent"
]

# Required columns for ABC-FMS analysis
REQUIRED_ABC_FMS_COLUMNS = {
    "Sku Code", 
    "Total_Order_Lines", 
    "Total_Case_Equiv", 
    "ABC", 
    "FMS"
}

# =============================================================================
# LLM PROMPT CONFIGURATION
# =============================================================================

LLM_PROMPTS = {
    'cover': {
        'instruction': 'Produce a 3-sentence executive summary that highlights dataset scope and top-level operational implications (2 bullets).'
    },
    'date_profile': {
        'instruction': 'Write a 4-sentence description of demand patterns and 3 operational recommendations (staffing, pallet allocation, short-term buffer).'
    },
    'percentiles': {
        'instruction': 'Provide a short interpretation (3 sentences) of these percentiles for capacity planning and peak provisioning.'
    },
    'sku_profile': {
        'instruction': 'Summarize the Pareto characteristics (3 sentences) and one inventory slotting recommendation.'
    },
    'abc_fms': {
        'instruction': 'Provide a short analysis of distribution and 3 prioritized recommendations for slotting & replenishment cadence.'
    }
}

# =============================================================================
# VALIDATION SETTINGS
# =============================================================================

# Data validation thresholds
MIN_ROWS_REQUIRED = 1
MIN_SKUS_REQUIRED = 1
MIN_DATES_REQUIRED = 1

# =============================================================================
# DEBUGGING AND LOGGING
# =============================================================================

# Debug settings
DEBUG_MODE = False
VERBOSE_OUTPUT = True

# Success messages
SUCCESS_MESSAGES = {
    'analysis_complete': '✅ Analysis complete! Results written to {}',
    'html_report_built': '✅ HTML report built: {}',
    'charts_generated': ' - Charts directory: {}',
    'view_instructions': ' - If you want to view: open {} in a browser or run: cd {}; python -m http.server 8000'
}