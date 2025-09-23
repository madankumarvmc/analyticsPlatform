#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parameter Controls Component for Warehouse Analysis Web App
Handles ABC/FMS threshold sliders and other analysis parameters.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Tuple
from pathlib import Path
import sys

# Import web config
sys.path.append(str(Path(__file__).parent.parent))
from config_web import (
    DEFAULT_ABC_THRESHOLDS, DEFAULT_FMS_THRESHOLDS, DEFAULT_PERCENTILES,
    DEFAULT_OUTPUT_OPTIONS, DEFAULT_FTE_PARAMETERS, SLIDER_CONFIG, HELP_TEXT
)


class ParameterController:
    """Handles parameter controls and validation for warehouse analysis."""
    
    def __init__(self):
        self.parameters = {}
        self.output_options = {}
    
    def create_abc_controls(self) -> Dict[str, float]:
        """
        Create ABC classification threshold controls.
        
        Returns:
            Dictionary with ABC threshold values
        """
        st.subheader("🔤 ABC Classification")
        st.markdown(HELP_TEXT['abc_classification'])
        
        # ABC A threshold
        abc_a_threshold = st.slider(
            "A Threshold (%)",
            min_value=SLIDER_CONFIG['abc_a']['min_value'],
            max_value=SLIDER_CONFIG['abc_a']['max_value'],
            value=DEFAULT_ABC_THRESHOLDS['A_THRESHOLD'],
            step=SLIDER_CONFIG['abc_a']['step'],
            help=SLIDER_CONFIG['abc_a']['help'],
            key="abc_a_threshold"
        )
        
        # ABC B threshold (dynamic min value based on A threshold)
        abc_b_threshold = st.slider(
            "B Threshold (%)",
            min_value=abc_a_threshold + 1,
            max_value=SLIDER_CONFIG['abc_b']['max_value'],
            value=max(DEFAULT_ABC_THRESHOLDS['B_THRESHOLD'], abc_a_threshold + 1),
            step=SLIDER_CONFIG['abc_b']['step'],
            help=SLIDER_CONFIG['abc_b']['help'],
            key="abc_b_threshold"
        )
        
        # Display classification ranges
        self._display_abc_ranges(abc_a_threshold, abc_b_threshold)
        
        return {
            'A_THRESHOLD': abc_a_threshold,
            'B_THRESHOLD': abc_b_threshold
        }
    
    def create_fms_controls(self) -> Dict[str, float]:
        """
        Create FMS classification threshold controls.
        
        Returns:
            Dictionary with FMS threshold values
        """
        st.subheader("⚡ FMS Classification")
        st.markdown(HELP_TEXT['fms_classification'])
        
        # FMS Fast threshold
        fms_f_threshold = st.slider(
            "Fast Threshold (%)",
            min_value=SLIDER_CONFIG['fms_f']['min_value'],
            max_value=SLIDER_CONFIG['fms_f']['max_value'],
            value=DEFAULT_FMS_THRESHOLDS['F_THRESHOLD'],
            step=SLIDER_CONFIG['fms_f']['step'],
            help=SLIDER_CONFIG['fms_f']['help'],
            key="fms_f_threshold"
        )
        
        # FMS Medium threshold (dynamic min value based on Fast threshold)
        fms_m_threshold = st.slider(
            "Medium Threshold (%)",
            min_value=fms_f_threshold + 1,
            max_value=SLIDER_CONFIG['fms_m']['max_value'],
            value=max(DEFAULT_FMS_THRESHOLDS['M_THRESHOLD'], fms_f_threshold + 1),
            step=SLIDER_CONFIG['fms_m']['step'],
            help=SLIDER_CONFIG['fms_m']['help'],
            key="fms_m_threshold"
        )
        
        # Display classification ranges
        self._display_fms_ranges(fms_f_threshold, fms_m_threshold)
        
        return {
            'F_THRESHOLD': fms_f_threshold,
            'M_THRESHOLD': fms_m_threshold
        }
    
    def create_analysis_options(self) -> Dict[str, Any]:
        """
        Create analysis and output option controls.
        
        Returns:
            Dictionary with analysis options
        """
        st.subheader("⚙️ Analysis Options")
        
        # Percentile options
        st.write("**Percentile Calculations:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            calc_95th = st.checkbox("95th Percentile", value=True, key="calc_95th")
        with col2:
            calc_90th = st.checkbox("90th Percentile", value=True, key="calc_90th")
        with col3:
            calc_85th = st.checkbox("85th Percentile", value=True, key="calc_85th")
        
        # Custom percentiles
        custom_percentiles = st.text_input(
            "Additional Percentiles (comma-separated)",
            placeholder="e.g., 75, 80, 99",
            help="Enter additional percentile values to calculate",
            key="custom_percentiles"
        )
        
        # Parse custom percentiles
        percentiles = []
        if calc_95th:
            percentiles.append(95)
        if calc_90th:
            percentiles.append(90)
        if calc_85th:
            percentiles.append(85)
        
        if custom_percentiles:
            try:
                custom = [float(p.strip()) for p in custom_percentiles.split(',') if p.strip()]
                percentiles.extend([p for p in custom if 0 < p < 100])
            except ValueError:
                st.warning("Invalid percentile values. Please enter numbers between 0 and 100.")
        
        return {
            'percentiles': sorted(list(set(percentiles)), reverse=True),
            'custom_percentiles': custom_percentiles
        }
    
    def create_output_options(self) -> Dict[str, bool]:
        """
        Create output option controls.
        
        Returns:
            Dictionary with output options
        """
        st.subheader("📊 Output Options")
        st.markdown(HELP_TEXT['analysis_outputs'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            generate_charts = st.checkbox(
                "📈 Generate Charts",
                value=DEFAULT_OUTPUT_OPTIONS['generate_charts'],
                help="Create visualization charts and graphs",
                key="generate_charts"
            )
            
            generate_html_report = st.checkbox(
                "🌐 HTML Report",
                value=DEFAULT_OUTPUT_OPTIONS['generate_html_report'],
                help="Generate comprehensive HTML report",
                key="generate_html_report"
            )
        
        with col2:
            generate_llm_summaries = st.checkbox(
                "🤖 AI Insights",
                value=DEFAULT_OUTPUT_OPTIONS['generate_llm_summaries'],
                help="Generate AI-powered insights and recommendations",
                key="generate_llm_summaries"
            )
            
            generate_excel_export = st.checkbox(
                "📊 Excel Export",
                value=DEFAULT_OUTPUT_OPTIONS['generate_excel_export'],
                help="Export detailed results to Excel",
                key="generate_excel_export"
            )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            
            # Chart options
            st.write("**Chart Settings:**")
            chart_style = st.selectbox(
                "Chart Style",
                options=['default', 'seaborn', 'ggplot', 'bmh'],
                index=0,
                key="chart_style"
            )
            
            chart_dpi = st.slider(
                "Chart Resolution (DPI)",
                min_value=72,
                max_value=300,
                value=150,
                step=6,
                key="chart_dpi"
            )
            
            # Table options
            st.write("**Table Display:**")
            max_table_rows = st.number_input(
                "Max Rows in HTML Tables",
                min_value=10,
                max_value=1000,
                value=50,
                step=10,
                key="max_table_rows"
            )
            
            # Performance options
            st.write("**Performance:**")
            enable_caching = st.checkbox(
                "Enable Caching",
                value=True,
                help="Cache intermediate results for faster repeated analysis",
                key="enable_caching"
            )
        
        return {
            'generate_charts': generate_charts,
            'generate_llm_summaries': generate_llm_summaries,
            'generate_html_report': generate_html_report,
            'generate_excel_export': generate_excel_export,
            'chart_style': chart_style,
            'chart_dpi': chart_dpi,
            'max_table_rows': max_table_rows,
            'enable_caching': enable_caching
        }
    
    def create_fte_controls(self) -> Dict[str, Any]:
        """
        Create FTE (Full-Time Equivalent) calculation controls.
        
        Returns:
            Dictionary with FTE parameters
        """
        st.subheader("👥 FTE Calculation")
        st.markdown(HELP_TEXT['fte_calculation'])
        
        # Enable/disable FTE calculation
        enable_fte = st.checkbox(
            "Enable FTE Calculation",
            value=DEFAULT_FTE_PARAMETERS['enable_fte_calculation'],
            help="Calculate daily workforce requirements based on operational parameters",
            key="enable_fte_calculation"
        )
        
        if enable_fte:
            # Touch time parameter
            touch_time = st.slider(
                "Touch Time per Unit (minutes)",
                min_value=SLIDER_CONFIG['fte_touch_time']['min_value'],
                max_value=SLIDER_CONFIG['fte_touch_time']['max_value'],
                value=DEFAULT_FTE_PARAMETERS['touch_time_per_unit'],
                step=SLIDER_CONFIG['fte_touch_time']['step'],
                help=SLIDER_CONFIG['fte_touch_time']['help'],
                key="fte_touch_time"
            )
            
            # Walking parameters
            col1, col2 = st.columns(2)
            
            with col1:
                walk_distance = st.slider(
                    "Walk Distance per Pallet (meters)",
                    min_value=SLIDER_CONFIG['fte_walk_distance']['min_value'],
                    max_value=SLIDER_CONFIG['fte_walk_distance']['max_value'],
                    value=DEFAULT_FTE_PARAMETERS['avg_walk_distance_per_pallet'],
                    step=SLIDER_CONFIG['fte_walk_distance']['step'],
                    help=SLIDER_CONFIG['fte_walk_distance']['help'],
                    key="fte_walk_distance"
                )
            
            with col2:
                walk_speed = st.slider(
                    "Walk Speed (meters/minute)",
                    min_value=SLIDER_CONFIG['fte_walk_speed']['min_value'],
                    max_value=SLIDER_CONFIG['fte_walk_speed']['max_value'],
                    value=DEFAULT_FTE_PARAMETERS['walk_speed'],
                    step=SLIDER_CONFIG['fte_walk_speed']['step'],
                    help=SLIDER_CONFIG['fte_walk_speed']['help'],
                    key="fte_walk_speed"
                )
            
            # Shift and efficiency parameters
            col3, col4 = st.columns(2)
            
            with col3:
                shift_hours = st.slider(
                    "Shift Hours",
                    min_value=SLIDER_CONFIG['fte_shift_hours']['min_value'],
                    max_value=SLIDER_CONFIG['fte_shift_hours']['max_value'],
                    value=DEFAULT_FTE_PARAMETERS['shift_hours'],
                    step=SLIDER_CONFIG['fte_shift_hours']['step'],
                    help=SLIDER_CONFIG['fte_shift_hours']['help'],
                    key="fte_shift_hours"
                )
            
            with col4:
                efficiency = st.slider(
                    "Worker Efficiency (%)",
                    min_value=int(SLIDER_CONFIG['fte_efficiency']['min_value'] * 100),
                    max_value=int(SLIDER_CONFIG['fte_efficiency']['max_value'] * 100),
                    value=int(DEFAULT_FTE_PARAMETERS['efficiency'] * 100),
                    step=int(SLIDER_CONFIG['fte_efficiency']['step'] * 100),
                    help=SLIDER_CONFIG['fte_efficiency']['help'],
                    key="fte_efficiency"
                ) / 100.0  # Convert back to decimal
            
            # Display FTE calculation preview
            self._display_fte_preview(touch_time, walk_distance, walk_speed, shift_hours, efficiency)
            
        else:
            # Use default values when disabled
            touch_time = DEFAULT_FTE_PARAMETERS['touch_time_per_unit']
            walk_distance = DEFAULT_FTE_PARAMETERS['avg_walk_distance_per_pallet']
            walk_speed = DEFAULT_FTE_PARAMETERS['walk_speed']
            shift_hours = DEFAULT_FTE_PARAMETERS['shift_hours']
            efficiency = DEFAULT_FTE_PARAMETERS['efficiency']
        
        return {
            'enable_fte_calculation': enable_fte,
            'touch_time_per_unit': touch_time,
            'avg_walk_distance_per_pallet': walk_distance,
            'walk_speed': walk_speed,
            'shift_hours': shift_hours,
            'efficiency': efficiency
        }
    
    def _display_fte_preview(self, touch_time: float, walk_distance: float, 
                            walk_speed: float, shift_hours: float, efficiency: float) -> None:
        """Display FTE calculation preview with current parameters."""
        st.markdown("**FTE Calculation Preview:**")
        
        # Example calculation with sample values
        sample_touches = 1000  # Sample: 1000 touches
        sample_pallets = 50    # Sample: 50 pallet equivalents
        
        sample_touch_time = sample_touches * touch_time
        sample_walk_time = (sample_pallets * walk_distance) / walk_speed
        sample_total_time = sample_touch_time + sample_walk_time
        sample_fte = sample_total_time / (shift_hours * 60 * efficiency)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Touch Time (1000 units)", f"{sample_touch_time:.1f} min")
        with col2:
            st.metric("Walk Time (50 pallets)", f"{sample_walk_time:.1f} min")
        with col3:
            st.metric("FTE Required", f"{sample_fte:.1f} workers")
        
        st.caption("*Preview based on 1000 total touches and 50 pallet equivalents*")
    
    def _display_abc_ranges(self, a_threshold: float, b_threshold: float) -> None:
        """Display ABC classification ranges."""
        st.markdown("**Classification Ranges:**")
        st.markdown(f"""
        - **A Items**: 0% - {a_threshold}% of volume (High-value)
        - **B Items**: {a_threshold}% - {b_threshold}% of volume (Medium-value)
        - **C Items**: {b_threshold}% - 100% of volume (Low-value)
        """)
    
    def _display_fms_ranges(self, f_threshold: float, m_threshold: float) -> None:
        """Display FMS classification ranges."""
        st.markdown("**Classification Ranges:**")
        st.markdown(f"""
        - **Fast Moving**: 0% - {f_threshold}% of order lines (High-frequency)
        - **Medium Moving**: {f_threshold}% - {m_threshold}% of order lines (Medium-frequency)
        - **Slow Moving**: {m_threshold}% - 100% of order lines (Low-frequency)
        """)
    
    def create_parameter_summary(self, abc_thresholds: Dict[str, float], 
                                fms_thresholds: Dict[str, float],
                                analysis_options: Dict[str, Any],
                                fte_parameters: Dict[str, Any],
                                output_options: Dict[str, bool]) -> pd.DataFrame:
        """
        Create a summary table of all parameters.
        
        Args:
            abc_thresholds: ABC threshold values
            fms_thresholds: FMS threshold values
            analysis_options: Analysis options
            fte_parameters: FTE calculation parameters
            output_options: Output options
            
        Returns:
            DataFrame with parameter summary
        """
        # Create parameter summary
        params = []
        
        # ABC parameters
        params.append({
            'Category': 'ABC Classification',
            'Parameter': 'A Threshold',
            'Value': f"{abc_thresholds['A_THRESHOLD']}%",
            'Description': f"Volume ≤ {abc_thresholds['A_THRESHOLD']}% = A Items"
        })
        
        params.append({
            'Category': 'ABC Classification',
            'Parameter': 'B Threshold',
            'Value': f"{abc_thresholds['B_THRESHOLD']}%",
            'Description': f"Volume {abc_thresholds['A_THRESHOLD']}-{abc_thresholds['B_THRESHOLD']}% = B Items"
        })
        
        # FMS parameters
        params.append({
            'Category': 'FMS Classification',
            'Parameter': 'Fast Threshold',
            'Value': f"{fms_thresholds['F_THRESHOLD']}%",
            'Description': f"Lines ≤ {fms_thresholds['F_THRESHOLD']}% = Fast Moving"
        })
        
        params.append({
            'Category': 'FMS Classification',
            'Parameter': 'Medium Threshold',
            'Value': f"{fms_thresholds['M_THRESHOLD']}%",
            'Description': f"Lines {fms_thresholds['F_THRESHOLD']}-{fms_thresholds['M_THRESHOLD']}% = Medium Moving"
        })
        
        # Percentiles
        if analysis_options['percentiles']:
            params.append({
                'Category': 'Analysis Options',
                'Parameter': 'Percentiles',
                'Value': ', '.join([f"{p}%" for p in analysis_options['percentiles']]),
                'Description': 'Percentile calculations to perform'
            })
        
        # FTE parameters
        if fte_parameters.get('enable_fte_calculation', False):
            params.append({
                'Category': 'FTE Calculation',
                'Parameter': 'Touch Time',
                'Value': f"{fte_parameters['touch_time_per_unit']:.2f} min/unit",
                'Description': 'Time to pick one unit (case or each)'
            })
            
            params.append({
                'Category': 'FTE Calculation',
                'Parameter': 'Walk Distance',
                'Value': f"{fte_parameters['avg_walk_distance_per_pallet']:.0f} m/pallet",
                'Description': 'Average walking distance per pallet equivalent'
            })
            
            params.append({
                'Category': 'FTE Calculation',
                'Parameter': 'Work Efficiency',
                'Value': f"{fte_parameters['efficiency']*100:.0f}%",
                'Description': 'Worker efficiency factor'
            })
            
            params.append({
                'Category': 'FTE Calculation',
                'Parameter': 'Shift Hours',
                'Value': f"{fte_parameters['shift_hours']:.1f} hours",
                'Description': 'Working hours per shift'
            })
        
        # Output options
        enabled_outputs = [k.replace('generate_', '').replace('_', ' ').title() 
                          for k, v in output_options.items() 
                          if k.startswith('generate_') and v]
        
        if enabled_outputs:
            params.append({
                'Category': 'Output Options',
                'Parameter': 'Enabled Outputs',
                'Value': ', '.join(enabled_outputs),
                'Description': 'Reports and outputs to generate'
            })
        
        return pd.DataFrame(params)
    
    def validate_parameters(self, abc_thresholds: Dict[str, float], 
                           fms_thresholds: Dict[str, float]) -> Tuple[bool, list]:
        """
        Validate parameter values.
        
        Args:
            abc_thresholds: ABC threshold values
            fms_thresholds: FMS threshold values
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # ABC validation
        if abc_thresholds['A_THRESHOLD'] >= abc_thresholds['B_THRESHOLD']:
            errors.append("ABC A threshold must be less than B threshold")
        
        if abc_thresholds['B_THRESHOLD'] >= 100:
            errors.append("ABC B threshold must be less than 100%")
        
        # FMS validation
        if fms_thresholds['F_THRESHOLD'] >= fms_thresholds['M_THRESHOLD']:
            errors.append("FMS Fast threshold must be less than Medium threshold")
        
        if fms_thresholds['M_THRESHOLD'] >= 100:
            errors.append("FMS Medium threshold must be less than 100%")
        
        return len(errors) == 0, errors


def create_parameter_controls() -> Dict[str, Any]:
    """
    Create the complete parameter controls section.
    
    Returns:
        Dictionary with all parameter values
    """
    controller = ParameterController()
    
    # Create all controls
    abc_thresholds = controller.create_abc_controls()
    st.divider()
    
    fms_thresholds = controller.create_fms_controls()
    st.divider()
    
    analysis_options = controller.create_analysis_options()
    st.divider()
    
    fte_parameters = controller.create_fte_controls()
    st.divider()
    
    output_options = controller.create_output_options()
    
    # Validate parameters
    is_valid, errors = controller.validate_parameters(abc_thresholds, fms_thresholds)
    
    if not is_valid:
        st.error("Parameter Validation Errors:")
        for error in errors:
            st.error(f"• {error}")
    
    # Store in session state
    parameters = {
        'abc_thresholds': abc_thresholds,
        'fms_thresholds': fms_thresholds,
        'analysis_options': analysis_options,
        'fte_parameters': fte_parameters,
        'output_options': output_options,
        'is_valid': is_valid,
        'errors': errors
    }
    
    st.session_state.parameters = parameters
    
    return parameters


def display_parameter_summary():
    """Display a summary of current parameters."""
    if 'parameters' not in st.session_state:
        st.warning("No parameters configured yet.")
        return
    
    params = st.session_state.parameters
    
    st.subheader("📋 Parameter Summary")
    
    controller = ParameterController()
    summary_df = controller.create_parameter_summary(
        params['abc_thresholds'],
        params['fms_thresholds'],
        params['analysis_options'],
        params['fte_parameters'],
        params['output_options']
    )
    
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Export parameters button
    if st.button("💾 Export Parameters", help="Download parameter configuration as JSON"):
        import json
        param_json = json.dumps(params, indent=2, default=str)
        st.download_button(
            "Download Parameters",
            param_json,
            "warehouse_analysis_parameters.json",
            "application/json"
        )


# Demo component
def parameter_controls_demo():
    """Demo component for testing parameter controls."""
    st.header("⚙️ Parameter Controls Demo")
    
    # Create parameter controls
    parameters = create_parameter_controls()
    
    # Display summary
    st.divider()
    display_parameter_summary()
    
    # Show validation status
    if parameters['is_valid']:
        st.success("✅ All parameters are valid!")
    else:
        st.error("❌ Parameter validation failed!")


if __name__ == "__main__":
    # Run demo if executed directly
    parameter_controls_demo()