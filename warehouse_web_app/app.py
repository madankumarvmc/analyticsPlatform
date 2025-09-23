#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Warehouse Analysis Web Application
Streamlit-based web interface for the modular warehouse analysis tool.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import logging
import traceback
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import web app modules
try:
    from config_web import CUSTOM_CSS, APP_CONFIG, ERROR_MESSAGES
    from web_utils.session_manager import SessionManager
    from web_utils.error_handler import ErrorHandler
    from components.header import create_header
    from components.file_upload import create_file_upload_section
    from components.parameter_controls import create_parameter_controls
    from components.results_display import create_results_display_section
    from web_utils.analysis_integration import run_web_analysis, display_analysis_status
except ImportError as e:
    st.error(f"Failed to import web app modules: {e}")
    st.stop()

# Configure page
st.set_page_config(
    page_title="Warehouse Analysis Tool",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session manager and error handler
session_manager = SessionManager()
error_handler = ErrorHandler()

# Apply custom CSS from config
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def main():
    """Main application function."""
    
    # Create header with logo
    create_header(
        title="Warehouse Analysis Tool",
        subtitle="Advanced Analytics for Warehouse Operations",
        logo_path=None,  # Will use default SVG logo
        show_navigation=True
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
            if st.button("← Back to Configuration", use_container_width=True, key="back_to_config_main"):
                # Reset analysis state and switch to configuration
                st.session_state.current_phase = 'configuration'
                st.session_state.analysis_complete = False
                st.session_state.analysis_results = None
                st.session_state.analysis_outputs = None
                st.rerun()
    
    # Phase-based full-width layout
    if st.session_state.current_phase == 'configuration':
        # CONFIGURATION PHASE - Full Width
        st.header("🔧 Configuration")
        
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
                    "🚀 Run Analysis",
                    type="primary",
                    use_container_width=True,
                    disabled=not parameters.get('is_valid', False),
                    help="Start the warehouse analysis with current parameters"
                )
            
            if run_analysis:
                # Run analysis with real integration
                with st.spinner("Running warehouse analysis..."):
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
            # Welcome screen - full width
            st.markdown("""
            <div class="info-box">
                <h3>🎯 Welcome to the Warehouse Analysis Tool</h3>
                <p>This powerful tool helps you analyze your warehouse operations with:</p>
                <ul>
                    <li><strong>ABC Classification:</strong> Categorize SKUs by volume contribution</li>
                    <li><strong>FMS Analysis:</strong> Classify by movement frequency</li>
                    <li><strong>Order Patterns:</strong> Understand demand trends and peaks</li>
                    <li><strong>AI Insights:</strong> Get intelligent recommendations</li>
                    <li><strong>Interactive Reports:</strong> Professional HTML and Excel outputs</li>
                </ul>
                <p><strong>📁 To get started:</strong> Upload your Excel file below</p>
                <p><strong>📋 Required sheets:</strong> OrderData and SkuMaster</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Sample data structure - full width
            with st.expander("📋 Expected Data Structure"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**OrderData Sheet should contain:**")
                    st.code("""
Date, Shipment No., Order No., Sku Code, Qty in Cases, Qty in Eaches
2025-02-01, SH001, ORD001, SKU123, 10, 24
2025-02-01, SH002, ORD002, SKU456, 5, 0
                    """)
                
                with col2:
                    st.write("**SkuMaster Sheet should contain:**")
                    st.code("""
Sku Code, Category, Case Config, Pallet Fit
SKU123, Electronics, 12, 48
SKU456, Furniture, 1, 20
                    """)
    
    elif st.session_state.current_phase == 'results':
        # RESULTS PHASE - Full Width
        st.header("📊 Analysis Results")
        
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
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem;">
        🏭 Warehouse Analysis Tool v2.0 
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()