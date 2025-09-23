#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Settings Page for Warehouse Analysis Web App
Application configuration, user preferences, and system settings.
"""

import streamlit as st
import pandas as pd
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config_web import (
    CUSTOM_CSS, DEFAULT_ABC_THRESHOLDS, DEFAULT_FMS_THRESHOLDS, 
    DEFAULT_PERCENTILES, DEFAULT_OUTPUT_OPTIONS
)
from components.header import create_simple_header

# Configure page
st.set_page_config(
    page_title="Settings - Warehouse Tool",
    page_icon="⚙️",
    layout="wide"
)

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def main():
    """Main function for Settings page."""
    
    # Create page header with logo
    create_simple_header(
        title="⚙️ Application Settings & Configuration",
        logo_path=None  # Uses default logo
    )
    
    # Initialize session state for settings
    if 'user_settings' not in st.session_state:
        st.session_state.user_settings = {
            'default_abc_thresholds': DEFAULT_ABC_THRESHOLDS.copy(),
            'default_fms_thresholds': DEFAULT_FMS_THRESHOLDS.copy(),
            'default_percentiles': DEFAULT_PERCENTILES.copy(),
            'default_output_options': DEFAULT_OUTPUT_OPTIONS.copy(),
            'ui_preferences': {
                'theme': 'light',
                'auto_run_analysis': False,
                'show_advanced_options': False,
                'default_chart_style': 'default'
            },
            'data_preferences': {
                'auto_save_parameters': True,
                'cache_analysis_results': True,
                'max_file_size_mb': 50
            }
        }
    
    # Create tabs for different setting categories
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔧 Default Parameters", 
        "🎨 User Interface", 
        "💾 Data & Performance", 
        "📊 Export & Reports",
        "🔄 Import/Export Settings"
    ])
    
    with tab1:
        configure_default_parameters()
    
    with tab2:
        configure_ui_preferences()
    
    with tab3:
        configure_data_preferences()
    
    with tab4:
        configure_export_preferences()
    
    with tab5:
        import_export_settings()
    
    # Save settings button
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("💾 Save All Settings", type="primary", use_container_width=True):
            save_settings()
            st.success("✅ Settings saved successfully!")
            st.balloons()


def configure_default_parameters():
    """Configure default analysis parameters."""
    st.header("🔧 Default Analysis Parameters")
    st.markdown("Set default values that will be pre-populated when starting new analyses.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔤 ABC Classification Defaults")
        
        abc_a_default = st.slider(
            "Default A Threshold (%)",
            min_value=50.0,
            max_value=90.0,
            value=st.session_state.user_settings['default_abc_thresholds']['A_THRESHOLD'],
            step=1.0,
            help="Default A classification threshold for new analyses",
            key="settings_abc_a"
        )
        
        abc_b_default = st.slider(
            "Default B Threshold (%)",
            min_value=abc_a_default + 1,
            max_value=95.0,
            value=max(st.session_state.user_settings['default_abc_thresholds']['B_THRESHOLD'], abc_a_default + 1),
            step=1.0,
            help="Default B classification threshold for new analyses",
            key="settings_abc_b"
        )
        
        # Update session state
        st.session_state.user_settings['default_abc_thresholds'] = {
            'A_THRESHOLD': abc_a_default,
            'B_THRESHOLD': abc_b_default
        }
    
    with col2:
        st.subheader("⚡ FMS Classification Defaults")
        
        fms_f_default = st.slider(
            "Default Fast Threshold (%)",
            min_value=50.0,
            max_value=90.0,
            value=st.session_state.user_settings['default_fms_thresholds']['F_THRESHOLD'],
            step=1.0,
            help="Default Fast moving threshold for new analyses",
            key="settings_fms_f"
        )
        
        fms_m_default = st.slider(
            "Default Medium Threshold (%)",
            min_value=fms_f_default + 1,
            max_value=95.0,
            value=max(st.session_state.user_settings['default_fms_thresholds']['M_THRESHOLD'], fms_f_default + 1),
            step=1.0,
            help="Default Medium moving threshold for new analyses",
            key="settings_fms_m"
        )
        
        # Update session state
        st.session_state.user_settings['default_fms_thresholds'] = {
            'F_THRESHOLD': fms_f_default,
            'M_THRESHOLD': fms_m_default
        }
    
    # Percentile defaults
    st.subheader("📊 Default Percentile Calculations")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pct_95 = st.checkbox("95th Percentile", value=95 in DEFAULT_PERCENTILES, key="settings_pct_95")
    with col2:
        pct_90 = st.checkbox("90th Percentile", value=90 in DEFAULT_PERCENTILES, key="settings_pct_90")
    with col3:
        pct_85 = st.checkbox("85th Percentile", value=85 in DEFAULT_PERCENTILES, key="settings_pct_85")
    with col4:
        pct_80 = st.checkbox("80th Percentile", value=80 in DEFAULT_PERCENTILES, key="settings_pct_80")
    
    # Custom percentiles
    custom_percentiles = st.text_input(
        "Additional Default Percentiles (comma-separated)",
        value=", ".join([str(p) for p in DEFAULT_PERCENTILES if p not in [95, 90, 85, 80]]),
        help="Enter additional percentile values to calculate by default",
        key="settings_custom_percentiles"
    )
    
    # Update percentiles in session state
    default_percentiles = []
    if pct_95: default_percentiles.append(95)
    if pct_90: default_percentiles.append(90)
    if pct_85: default_percentiles.append(85)
    if pct_80: default_percentiles.append(80)
    
    if custom_percentiles:
        try:
            custom = [float(p.strip()) for p in custom_percentiles.split(',') if p.strip()]
            default_percentiles.extend([p for p in custom if 0 < p < 100])
        except ValueError:
            st.warning("Invalid percentile values in custom percentiles field.")
    
    st.session_state.user_settings['default_percentiles'] = sorted(list(set(default_percentiles)), reverse=True)


def configure_ui_preferences():
    """Configure user interface preferences."""
    st.header("🎨 User Interface Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎨 Appearance")
        
        # Theme selection (future feature)
        theme = st.selectbox(
            "Theme",
            options=["Light", "Dark", "Auto"],
            index=0,
            disabled=True,
            help="Theme selection will be available in a future update",
            key="settings_theme"
        )
        
        # Chart style
        chart_style = st.selectbox(
            "Default Chart Style",
            options=["default", "seaborn", "ggplot", "bmh"],
            index=0,
            help="Default styling for generated charts",
            key="settings_chart_style"
        )
        
        # Chart resolution
        chart_dpi = st.slider(
            "Chart Resolution (DPI)",
            min_value=72,
            max_value=300,
            value=150,
            step=6,
            help="Higher values create sharper images but larger file sizes",
            key="settings_chart_dpi"
        )
    
    with col2:
        st.subheader("🔄 Behavior")
        
        auto_run = st.checkbox(
            "Auto-run Analysis",
            value=st.session_state.user_settings['ui_preferences']['auto_run_analysis'],
            help="Automatically run analysis when file is uploaded and parameters are valid",
            key="settings_auto_run"
        )
        
        show_advanced = st.checkbox(
            "Show Advanced Options by Default",
            value=st.session_state.user_settings['ui_preferences']['show_advanced_options'],
            help="Expand advanced option sections by default",
            key="settings_show_advanced"
        )
        
        # Table display options
        max_rows = st.number_input(
            "Max Table Rows to Display",
            min_value=10,
            max_value=1000,
            value=50,
            step=10,
            help="Maximum number of rows to show in data tables",
            key="settings_max_rows"
        )
    
    # Update UI preferences in session state
    st.session_state.user_settings['ui_preferences'].update({
        'theme': theme.lower(),
        'auto_run_analysis': auto_run,
        'show_advanced_options': show_advanced,
        'default_chart_style': chart_style,
        'chart_dpi': chart_dpi,
        'max_table_rows': max_rows
    })


def configure_data_preferences():
    """Configure data and performance preferences."""
    st.header("💾 Data & Performance Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 File Handling")
        
        max_file_size = st.slider(
            "Maximum File Size (MB)",
            min_value=10,
            max_value=200,
            value=st.session_state.user_settings['data_preferences']['max_file_size_mb'],
            step=10,
            help="Maximum allowed size for uploaded Excel files",
            key="settings_max_file_size"
        )
        
        auto_save_params = st.checkbox(
            "Auto-save Parameters",
            value=st.session_state.user_settings['data_preferences']['auto_save_parameters'],
            help="Automatically save analysis parameters for reuse",
            key="settings_auto_save_params"
        )
        
        # Data validation strictness
        validation_strict = st.selectbox(
            "Data Validation Strictness",
            options=["Strict", "Moderate", "Lenient"],
            index=1,
            help="How strictly to validate uploaded data files",
            key="settings_validation_strict"
        )
    
    with col2:
        st.subheader("⚡ Performance")
        
        cache_results = st.checkbox(
            "Cache Analysis Results",
            value=st.session_state.user_settings['data_preferences']['cache_analysis_results'],
            help="Cache results to speed up repeated analysis of same data",
            key="settings_cache_results"
        )
        
        # Memory management
        memory_limit = st.selectbox(
            "Memory Usage Limit",
            options=["Conservative", "Moderate", "Aggressive"],
            index=1,
            help="How much system memory to use for analysis",
            key="settings_memory_limit"
        )
        
        # Parallel processing
        enable_parallel = st.checkbox(
            "Enable Parallel Processing",
            value=True,
            help="Use multiple CPU cores for faster analysis",
            key="settings_enable_parallel"
        )
    
    # Update data preferences in session state
    st.session_state.user_settings['data_preferences'].update({
        'max_file_size_mb': max_file_size,
        'auto_save_parameters': auto_save_params,
        'cache_analysis_results': cache_results,
        'validation_strictness': validation_strict.lower(),
        'memory_limit': memory_limit.lower(),
        'enable_parallel_processing': enable_parallel
    })


def configure_export_preferences():
    """Configure export and report preferences."""
    st.header("📊 Export & Report Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Default Output Options")
        
        # Default enabled outputs
        default_charts = st.checkbox(
            "Generate Charts by Default",
            value=st.session_state.user_settings['default_output_options']['generate_charts'],
            key="settings_default_charts"
        )
        
        default_excel = st.checkbox(
            "Generate Excel Export by Default",
            value=st.session_state.user_settings['default_output_options']['generate_excel_export'],
            key="settings_default_excel"
        )
        
        default_html = st.checkbox(
            "Generate HTML Report by Default",
            value=st.session_state.user_settings['default_output_options']['generate_html_report'],
            key="settings_default_html"
        )
        
        default_llm = st.checkbox(
            "Generate AI Insights by Default",
            value=st.session_state.user_settings['default_output_options']['generate_llm_summaries'],
            key="settings_default_llm"
        )
    
    with col2:
        st.subheader("🎯 Report Preferences")
        
        # Report naming convention
        report_naming = st.selectbox(
            "Report Naming Convention",
            options=["timestamp", "custom", "sequential"],
            index=0,
            help="How to name generated report files",
            key="settings_report_naming"
        )
        
        # Include raw data in exports
        include_raw_data = st.checkbox(
            "Include Raw Data in Excel Exports",
            value=True,
            help="Include original data sheets in Excel exports",
            key="settings_include_raw_data"
        )
        
        # Compress exports
        compress_exports = st.checkbox(
            "Compress Export Files",
            value=False,
            help="Create ZIP archives for large exports",
            key="settings_compress_exports"
        )
        
        # Auto-download after generation
        auto_download = st.checkbox(
            "Auto-download Generated Reports",
            value=True,
            help="Automatically prompt download when reports are ready",
            key="settings_auto_download"
        )
    
    # Update export preferences
    st.session_state.user_settings['default_output_options'].update({
        'generate_charts': default_charts,
        'generate_excel_export': default_excel,
        'generate_html_report': default_html,
        'generate_llm_summaries': default_llm,
        'report_naming_convention': report_naming,
        'include_raw_data': include_raw_data,
        'compress_exports': compress_exports,
        'auto_download': auto_download
    })


def import_export_settings():
    """Handle import/export of settings."""
    st.header("🔄 Import/Export Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Export Settings")
        st.markdown("Download your current settings as a JSON file for backup or sharing.")
        
        if st.button("📋 Preview Current Settings", use_container_width=True):
            st.json(st.session_state.user_settings)
        
        # Export settings
        settings_json = json.dumps(st.session_state.user_settings, indent=2, default=str)
        
        st.download_button(
            "💾 Download Settings",
            data=settings_json,
            file_name="warehouse_analysis_settings.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        st.subheader("📥 Import Settings")
        st.markdown("Upload a previously saved settings file to restore your configuration.")
        
        uploaded_settings = st.file_uploader(
            "Choose settings file",
            type=['json'],
            help="Select a JSON settings file to import",
            key="import_settings_file"
        )
        
        if uploaded_settings is not None:
            try:
                imported_settings = json.loads(uploaded_settings.read())
                
                st.success("✅ Settings file loaded successfully!")
                
                # Preview imported settings
                with st.expander("📋 Preview Imported Settings"):
                    st.json(imported_settings)
                
                if st.button("🔄 Apply Imported Settings", type="primary", use_container_width=True):
                    # Validate and apply settings
                    if validate_imported_settings(imported_settings):
                        st.session_state.user_settings = imported_settings
                        st.success("✅ Settings imported and applied successfully!")
                        st.experimental_rerun()
                    else:
                        st.error("❌ Invalid settings file format.")
                        
            except json.JSONDecodeError:
                st.error("❌ Invalid JSON file. Please check the file format.")
            except Exception as e:
                st.error(f"❌ Error reading settings file: {str(e)}")
    
    # Reset to defaults
    st.divider()
    st.subheader("🔄 Reset Settings")
    st.warning("This will reset all settings to their default values.")
    
    if st.button("🔄 Reset All Settings to Defaults", use_container_width=True):
        reset_to_defaults()
        st.success("✅ Settings reset to defaults!")
        st.experimental_rerun()


def validate_imported_settings(settings: Dict[str, Any]) -> bool:
    """Validate imported settings structure."""
    required_keys = [
        'default_abc_thresholds',
        'default_fms_thresholds', 
        'default_percentiles',
        'default_output_options',
        'ui_preferences',
        'data_preferences'
    ]
    
    return all(key in settings for key in required_keys)


def reset_to_defaults():
    """Reset all settings to default values."""
    st.session_state.user_settings = {
        'default_abc_thresholds': DEFAULT_ABC_THRESHOLDS.copy(),
        'default_fms_thresholds': DEFAULT_FMS_THRESHOLDS.copy(),
        'default_percentiles': DEFAULT_PERCENTILES.copy(),
        'default_output_options': DEFAULT_OUTPUT_OPTIONS.copy(),
        'ui_preferences': {
            'theme': 'light',
            'auto_run_analysis': False,
            'show_advanced_options': False,
            'default_chart_style': 'default',
            'chart_dpi': 150,
            'max_table_rows': 50
        },
        'data_preferences': {
            'auto_save_parameters': True,
            'cache_analysis_results': True,
            'max_file_size_mb': 50,
            'validation_strictness': 'moderate',
            'memory_limit': 'moderate',
            'enable_parallel_processing': True
        }
    }


def save_settings():
    """Save current settings to persistent storage."""
    # In a real application, this would save to a database or config file
    # For now, we just update the session state
    st.session_state.settings_saved = True
    
    # You could add file-based persistence here
    # with open('user_settings.json', 'w') as f:
    #     json.dump(st.session_state.user_settings, f, indent=2, default=str)


if __name__ == "__main__":
    main()