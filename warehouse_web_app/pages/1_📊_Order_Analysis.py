#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Order Analysis Page for Warehouse Analysis Web App
Main page for order profiling and ABC-FMS analysis.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import components
from components.file_upload import create_file_upload_section
from components.parameter_controls import create_parameter_controls, display_parameter_summary
from components.results_display import create_results_display_section
from components.header import create_simple_header
from web_utils.analysis_integration import run_web_analysis, display_analysis_status
from config_web import CUSTOM_CSS

# Configure page
st.set_page_config(
    page_title="Order Analysis - Warehouse Tool",
    page_icon="📊",
    layout="wide"
)

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def main():
    """Main function for Order Analysis page."""
    
    # Create page header with logo
    create_simple_header(
        title="📊 Order Analysis & ABC-FMS Classification",
        logo_path=None  # Uses default logo
    )
    
    # Initialize session state
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'analysis_outputs' not in st.session_state:
        st.session_state.analysis_outputs = None
    if 'current_phase' not in st.session_state:
        st.session_state.current_phase = 'configuration'  # 'configuration' or 'results'
    
    # Determine current phase and update based on analysis status
    if st.session_state.analysis_complete and st.session_state.analysis_results:
        st.session_state.current_phase = 'results'
    elif not st.session_state.analysis_complete:
        # Only reset to configuration if we're not in the middle of analysis
        if st.session_state.current_phase == 'results':
            st.session_state.current_phase = 'configuration'
    
    # Phase navigation
    if st.session_state.current_phase == 'results':
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("← Back to Configuration", use_container_width=True, key="back_to_config_order"):
                # Reset analysis state and switch to configuration
                st.session_state.current_phase = 'configuration'
                st.session_state.analysis_complete = False
                st.session_state.analysis_results = None
                st.session_state.analysis_outputs = None
                st.rerun()
    
    # Phase-based full-width layout
    if st.session_state.current_phase == 'configuration':
        # CONFIGURATION PHASE - Full Width
        st.header("🔧 Order Analysis Configuration")
        
        # Welcome message for this specific page - full width
        st.markdown("""
        <div class="info-box">
            <h3>📊 Order Analysis & ABC-FMS Classification</h3>
            <p>This module provides comprehensive analysis of your warehouse order data including ABC classification, FMS analysis, order patterns, and cross-tabulation insights.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # File upload section - full width
        uploaded_file, is_valid = create_file_upload_section()
        
        if uploaded_file and is_valid:
            st.success("✅ File validated successfully!")
            
            # Parameter controls - full width
            st.divider()
            parameters = create_parameter_controls()
            
            # Analysis button - centered and prominent
            st.divider()
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                run_analysis = st.button(
                    "🚀 Run Order Analysis",
                    type="primary",
                    use_container_width=True,
                    disabled=not parameters.get('is_valid', False),
                    help="Start the warehouse order analysis with current parameters"
                )
            
            if run_analysis:
                # Run analysis with real integration
                with st.spinner("Running warehouse order analysis..."):
                    # Run actual analysis using the integration module
                    result = run_web_analysis(uploaded_file, parameters)
                    
                    # Store results in session state
                    st.session_state.analysis_results = result.get('analysis_results', {})
                    st.session_state.analysis_outputs = result.get('outputs', {})
                    st.session_state.analysis_complete = result.get('success', False)
                    
                    # Switch to results phase if successful
                    if result.get('success', False):
                        st.session_state.current_phase = 'results'
                
                # Display analysis status
                display_analysis_status(result)
                
                # Auto-refresh to switch to results view if successful
                if result.get('success', False):
                    st.rerun()
        
        elif uploaded_file and not is_valid:
            st.error("❌ Please fix the file validation errors before proceeding.")
        
        else:
            # Detailed welcome message for order analysis - full width
            st.markdown("""
            <div class="info-box">
                <h3>🎯 Welcome to Order Analysis</h3>
                <p>This module provides comprehensive analysis of your warehouse order data:</p>
                
                <h4>📊 What You'll Get:</h4>
                <ul>
                    <li><strong>ABC Classification:</strong> Categorize SKUs by volume contribution</li>
                    <li><strong>FMS Analysis:</strong> Classify by movement frequency (Fast/Medium/Slow)</li>
                    <li><strong>Order Patterns:</strong> Identify demand trends and seasonal patterns</li>
                    <li><strong>Percentile Analysis:</strong> Understand capacity planning requirements</li>
                    <li><strong>Cross-tabulation:</strong> Combined ABC×FMS insights</li>
                </ul>
                
                <h4>🔧 How It Works:</h4>
                <ol>
                    <li><strong>Upload Data:</strong> Excel file with OrderData and SkuMaster sheets</li>
                    <li><strong>Configure Parameters:</strong> Set your ABC/FMS thresholds</li>
                    <li><strong>Run Analysis:</strong> Process your data with custom settings</li>
                    <li><strong>Review Results:</strong> Interactive dashboards and downloadable reports</li>
                </ol>
                
                <p><strong>📁 Get started by uploading your warehouse data file below</strong></p>
            </div>
            """, unsafe_allow_html=True)
    
    elif st.session_state.current_phase == 'results':
        # RESULTS PHASE - Full Width
        st.header("📊 Order Analysis Results")
        
        # Display results if analysis is complete
        if st.session_state.analysis_complete and st.session_state.analysis_results:
            create_results_display_section(
                st.session_state.analysis_results,
                st.session_state.analysis_outputs
            )
        else:
            st.error("No analysis results available. Please return to configuration and run analysis.")
            if st.button("← Return to Configuration"):
                st.session_state.current_phase = 'configuration'
                st.rerun()


if __name__ == "__main__":
    main()