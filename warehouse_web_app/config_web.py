#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Web Application Configuration for Warehouse Analysis Tool
Contains web-specific settings and configurations.
"""

import os
from pathlib import Path
import streamlit as st

# =============================================================================
# WEB APPLICATION SETTINGS
# =============================================================================

# App configuration
APP_CONFIG = {
    'title': "Warehouse Analysis Tool",
    'icon': "🏭",
    'layout': "wide",
    'sidebar_state': "expanded",
    'version': "2.0.0",
    'debug_mode': False
}

# Legacy support
APP_TITLE = APP_CONFIG['title']
APP_ICON = APP_CONFIG['icon']
APP_LAYOUT = APP_CONFIG['layout']

# Debug mode helper
def is_debug_mode():
    """Check if debug mode is enabled."""
    return APP_CONFIG.get('debug_mode', False)

def enable_debug_mode():
    """Enable debug mode (for development/troubleshooting)."""
    APP_CONFIG['debug_mode'] = True

def disable_debug_mode():
    """Disable debug mode (for production)."""
    APP_CONFIG['debug_mode'] = False
APP_SIDEBAR_STATE = APP_CONFIG['sidebar_state']

# File upload settings
MAX_FILE_SIZE_MB = 100  # Maximum file size in MB
ALLOWED_EXTENSIONS = ['xlsx', 'xls']
UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Session timeout (in seconds)
SESSION_TIMEOUT = 3600  # 1 hour

# =============================================================================
# DEFAULT ANALYSIS PARAMETERS
# =============================================================================

# Default ABC thresholds
DEFAULT_ABC_THRESHOLDS = {
    'A_THRESHOLD': 70.0,
    'B_THRESHOLD': 90.0
}

# Default FMS thresholds
DEFAULT_FMS_THRESHOLDS = {
    'F_THRESHOLD': 70.0,
    'M_THRESHOLD': 90.0
}

# Default percentile levels
DEFAULT_PERCENTILES = [95, 90, 85]

# Default output options
DEFAULT_OUTPUT_OPTIONS = {
    'generate_charts': True,
    'generate_llm_summaries': True,
    'generate_html_report': True,
    'generate_excel_export': True,
    'generate_word_report': False  # Optional due to dependency requirements
}

# Default FTE (Full-Time Equivalent) calculation parameters
DEFAULT_FTE_PARAMETERS = {
    'touch_time_per_unit': 0.17,          # minutes per unit (10 seconds default)
    'avg_walk_distance_per_pallet': 30.0, # meters per pallet
    'walk_speed': 60.0,                    # meters per minute (3.6 km/h)
    'shift_hours': 8.0,                    # hours per shift
    'efficiency': 0.8,                     # 80% effective time
    'enable_fte_calculation': True         # enable FTE calculation
}

# =============================================================================
# UI COMPONENT SETTINGS
# =============================================================================

# Slider configurations
SLIDER_CONFIG = {
    'abc_a': {
        'min_value': 50.0,
        'max_value': 90.0,
        'step': 1.0,
        'help': "SKUs contributing up to this % of total volume are classified as 'A'"
    },
    'abc_b': {
        'min_value': 70.0,  # Will be dynamic based on A threshold
        'max_value': 95.0,
        'step': 1.0,
        'help': "SKUs contributing between A and this % are classified as 'B'"
    },
    'fms_f': {
        'min_value': 50.0,
        'max_value': 90.0,
        'step': 1.0,
        'help': "SKUs contributing up to this % of total order lines are 'Fast'"
    },
    'fms_m': {
        'min_value': 70.0,  # Will be dynamic based on F threshold
        'max_value': 95.0,
        'step': 1.0,
        'help': "SKUs contributing between Fast and this % are 'Medium'"
    },
    # FTE calculation parameter controls
    'fte_touch_time': {
        'min_value': 0.05,   # 3 seconds
        'max_value': 0.5,    # 30 seconds
        'step': 0.01,
        'help': "Time in minutes required to pick one unit (case or each)"
    },
    'fte_walk_distance': {
        'min_value': 10.0,   # 10 meters
        'max_value': 100.0,  # 100 meters
        'step': 5.0,
        'help': "Average walking distance in meters per pallet equivalent"
    },
    'fte_walk_speed': {
        'min_value': 30.0,   # 1.8 km/h (slow)
        'max_value': 100.0,  # 6 km/h (fast)
        'step': 5.0,
        'help': "Walking speed in meters per minute"
    },
    'fte_shift_hours': {
        'min_value': 4.0,    # 4 hours
        'max_value': 12.0,   # 12 hours
        'step': 0.5,
        'help': "Number of working hours per shift"
    },
    'fte_efficiency': {
        'min_value': 0.5,    # 50%
        'max_value': 0.95,   # 95%
        'step': 0.05,
        'help': "Worker efficiency factor (effective work time ratio)"
    }
}

# Chart settings for web display
WEB_CHART_CONFIG = {
    'figure_size': (12, 6),
    'dpi': 100,
    'style': 'seaborn-v0_8',
    'color_palette': 'tab10'
}

# Table display settings
TABLE_CONFIG = {
    'max_rows_preview': 50,
    'pagination_size': 25,
    'show_index': False,
    'use_container_width': True
}

# =============================================================================
# VALIDATION SETTINGS
# =============================================================================

# Required columns for data validation
REQUIRED_ORDER_COLUMNS = [
    'Date', 'Shipment No.', 'Order No.', 'Sku Code', 
    'Qty in Cases', 'Qty in Eaches'
]

REQUIRED_SKU_COLUMNS = [
    'Sku Code', 'Category', 'Case Config', 'Pallet Fit'
]

# Data validation thresholds
MIN_ROWS_REQUIRED = 1
MIN_SKUS_REQUIRED = 1
MIN_DATES_REQUIRED = 1

# =============================================================================
# STYLING AND THEMES
# =============================================================================

# Custom CSS styles
CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 3rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #e3f2fd, #bbdefb);
        border-radius: 10px;
        border: 2px solid #1976d2;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ddd;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.375rem;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.375rem;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    
    .warning-message {
        background: #fff3cd;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.375rem;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
    
    .info-box {
        background: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.375rem;
        border: 1px solid #bee5eb;
        margin: 1rem 0;
    }
    
    .parameter-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    
    .results-container {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .download-button {
        background: #28a745;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.375rem;
        text-decoration: none;
        display: inline-block;
        margin: 0.25rem;
        border: none;
        cursor: pointer;
    }
    
    .download-button:hover {
        background: #218838;
        color: white;
        text-decoration: none;
    }
    
    .sidebar-section {
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #eee;
    }
    
    .progress-container {
        margin: 1rem 0;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
    
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.9rem;
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid #eee;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
"""

# =============================================================================
# HELP TEXT AND DOCUMENTATION
# =============================================================================

HELP_TEXT = {
    'file_upload': """
    **Upload your warehouse data Excel file** with the following sheets:
    - **OrderData**: Contains order transactions with dates, quantities, and SKU codes
    - **SkuMaster**: Contains SKU details including case configurations and pallet fits
    """,
    
    'abc_classification': """
    **ABC Classification** categorizes SKUs based on their volume contribution:
    - **A Items**: High-value items (typically top 20% of SKUs contributing 70-80% of volume)
    - **B Items**: Medium-value items (typically next 30% of SKUs contributing 15-20% of volume)
    - **C Items**: Low-value items (typically remaining 50% of SKUs contributing 5-10% of volume)
    """,
    
    'fms_classification': """
    **FMS Classification** categorizes SKUs based on movement frequency:
    - **Fast Moving**: High-frequency items that move regularly
    - **Medium Moving**: Moderate-frequency items
    - **Slow Moving**: Low-frequency items that move infrequently
    """,
    
    'analysis_outputs': """
    **Available Output Options:**
    - **Charts**: Visual representations of data patterns and distributions
    - **AI Insights**: Machine learning-generated insights and recommendations
    - **HTML Report**: Comprehensive web-based report with embedded visualizations
    - **Excel Export**: Detailed spreadsheet with all analysis results
    """,
    
    'fte_calculation': """
    **FTE (Full-Time Equivalent) Calculation** estimates daily workforce requirements:
    - **Touch Time**: Time required to physically pick each unit (cases + eaches)
    - **Walk Distance**: Average distance traveled per pallet equivalent
    - **Walk Speed**: Worker walking speed in the warehouse
    - **Shift Hours**: Standard working hours per shift
    - **Efficiency**: Worker effectiveness factor (accounts for breaks, setup time, etc.)
    
    **Formula**: FTE = (Touch Time + Walk Time) ÷ (Shift Hours × Efficiency)
    """
}

# =============================================================================
# ERROR MESSAGES
# =============================================================================

ERROR_MESSAGES = {
    'file_not_found': "📁 Please upload an Excel file to begin analysis.",
    'invalid_file_type': "❌ Invalid file type. Please upload an Excel file (.xlsx or .xls).",
    'file_too_large': f"❌ File size exceeds {MAX_FILE_SIZE_MB}MB limit.",
    'missing_sheets': "❌ Required sheets (OrderData, SkuMaster) not found in the uploaded file.",
    'missing_columns': "❌ Required columns are missing from the data sheets.",
    'empty_data': "❌ The uploaded file contains no data to analyze.",
    'analysis_failed': "❌ Analysis failed. Please check your data and try again.",
    'invalid_thresholds': "❌ Invalid threshold values. Please check your parameter settings."
}

SUCCESS_MESSAGES = {
    'file_uploaded': "✅ File uploaded successfully!",
    'analysis_complete': "✅ Analysis completed successfully!",
    'report_generated': "✅ Report generated successfully!",
    'data_validated': "✅ Data validation passed!"
}

# =============================================================================
# PERFORMANCE SETTINGS
# =============================================================================

# Caching settings
CACHE_TTL = 3600  # 1 hour in seconds
MAX_CACHE_ENTRIES = 100

# Memory management
MAX_MEMORY_USAGE_MB = 1024  # 1GB

# Progress tracking
PROGRESS_STEPS = {
    'data_loading': 20,
    'data_validation': 30,
    'order_analysis': 50,
    'sku_analysis': 70,
    'cross_tabulation': 85,
    'report_generation': 100
}

# =============================================================================
# FUTURE ANALYSIS MODULES (PLACEHOLDERS)
# =============================================================================

FUTURE_MODULES = {
    'manpower_analysis': {
        'title': '👥 Manpower Analysis',
        'description': 'Calculate optimal staffing requirements based on volume patterns',
        'status': 'coming_soon'
    },
    'slotting_analysis': {
        'title': '📦 Slotting Analysis',
        'description': 'Optimize warehouse layout and storage allocation',
        'status': 'coming_soon'
    },
    'storage_optimization': {
        'title': '🏗️ Storage Optimization',
        'description': 'Analyze space utilization and efficiency metrics',
        'status': 'coming_soon'
    }
}