#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Results Display Component for Warehouse Analysis Web App
Handles display of analysis results with interactive tables and charts.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import sys
import base64

# Import web config
sys.path.append(str(Path(__file__).parent.parent))
from config_web import TABLE_CONFIG, CUSTOM_CSS


class ResultsDisplayManager:
    """Manages the display of analysis results in the web interface."""
    
    def __init__(self):
        self.results = {}
        self.charts = {}
    
    def _detect_data_structure(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect the data structure type and map appropriate columns.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dictionary with structure info and column mappings
        """
        structure_info = {
            'type': 'unknown',
            'columns': {
                'volume': None,
                'sku_count': None,
                'order_lines': None,
                'abc': None,
                'fms': None
            },
            'is_cross_tab': False,
            'is_raw_data': False,
            'available_columns': list(data.columns)
        }
        
        # Check for cross-tabulation summary format
        cross_tab_indicators = ['SKU_F', 'SKU_M', 'SKU_S', 'Volume_Total', 'Line_Total']
        if all(col in data.columns for col in cross_tab_indicators):
            structure_info.update({
                'type': 'cross_tabulation_summary',
                'is_cross_tab': True,
                'columns': {
                    'volume': 'Volume_Total',
                    'sku_count': 'SKU_Total',
                    'order_lines': 'Line_Total',
                    'abc': 'ABC',
                    'fms': None,  # FMS data is in separate columns (F/M/S)
                    'volume_f': 'Volume_F',
                    'volume_m': 'Volume_M', 
                    'volume_s': 'Volume_S',
                    'sku_f': 'SKU_F',
                    'sku_m': 'SKU_M',
                    'sku_s': 'SKU_S',
                    'line_f': 'Line_F',
                    'line_m': 'Line_M',
                    'line_s': 'Line_S'
                }
            })
        
        # Check for raw SKU data format
        elif 'Sku Code' in data.columns and 'FMS' in data.columns:
            structure_info.update({
                'type': 'raw_sku_data',
                'is_raw_data': True,
                'columns': {
                    'volume': 'Total_Case_Equiv' if 'Total_Case_Equiv' in data.columns else 'Total_Case_Equivalent',
                    'sku_count': 'Sku Code',
                    'order_lines': 'Total_Order_Lines',
                    'abc': 'ABC',
                    'fms': 'FMS'
                }
            })
        
        # Check for alternative raw data format
        elif 'Sku_Code' in data.columns and 'FMS' in data.columns:
            structure_info.update({
                'type': 'raw_sku_data_alt',
                'is_raw_data': True,
                'columns': {
                    'volume': 'Total_Case_Equivalent' if 'Total_Case_Equivalent' in data.columns else 'Total_Case_Equiv',
                    'sku_count': 'Sku_Code',
                    'order_lines': 'Total_Order_Lines',
                    'abc': 'ABC',
                    'fms': 'FMS'
                }
            })
        
        # Validate that mapped columns actually exist
        for key, col_name in structure_info['columns'].items():
            if col_name and col_name not in data.columns:
                structure_info['columns'][key] = None
        
        return structure_info
    
    def _compute_basic_stats_from_data(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute basic statistics from available analysis results data."""
        computed_stats = {}
        
        # Try to extract statistics from available dataframes
        try:
            # Check for date_order_summary data
            if 'date_order_summary' in analysis_results:
                date_data = analysis_results['date_order_summary']
                if isinstance(date_data, pd.DataFrame) and not date_data.empty:
                    # Check what columns are available
                    available_cols = list(date_data.columns)
                    logger = logging.getLogger(__name__)
                    logger.info(f"Date summary columns: {available_cols}")
                    
                    # Try multiple possible column names for volume
                    total_volume = 0
                    volume_columns = ['Total_Case_Equiv', 'Total_Case_Equivalent', 'Case_Equivalent', 'Qty_Ordered']
                    for vol_col in volume_columns:
                        if vol_col in date_data.columns:
                            total_volume = date_data[vol_col].sum()
                            logger.info(f"Found volume in column {vol_col}: {total_volume}")
                            break
                    
                    computed_stats.update({
                        'total_case_equivalent': total_volume,
                        'unique_dates': len(date_data),
                    })
            
            # Check for SKU profile data
            if 'sku_profile_abc_fms' in analysis_results:
                sku_data = analysis_results['sku_profile_abc_fms']
                if isinstance(sku_data, pd.DataFrame) and not sku_data.empty:
                    # Check what columns are available
                    available_cols = list(sku_data.columns)
                    logger = logging.getLogger(__name__)
                    logger.info(f"SKU profile columns: {available_cols}")
                    
                    # Try multiple possible column names for order lines
                    total_order_lines = 0
                    order_line_columns = ['Total_Order_Lines', 'Order_Lines', 'Lines', 'Qty_Ordered']
                    for line_col in order_line_columns:
                        if line_col in sku_data.columns:
                            total_order_lines = sku_data[line_col].sum()
                            logger.info(f"Found order lines in column {line_col}: {total_order_lines}")
                            break
                    
                    # Get total volume from SKU data as well
                    sku_total_volume = 0
                    sku_volume_columns = ['Total_Case_Equiv', 'Total_Case_Equivalent', 'Case_Equivalent']
                    for vol_col in sku_volume_columns:
                        if vol_col in sku_data.columns:
                            sku_total_volume = sku_data[vol_col].sum()
                            logger.info(f"Found SKU volume in column {vol_col}: {sku_total_volume}")
                            break
                    
                    computed_stats.update({
                        'unique_skus': len(sku_data),
                        'total_order_lines': total_order_lines
                    })
                    
                    # Use SKU volume if we didn't get it from date summary
                    if computed_stats.get('total_case_equivalent', 0) == 0 and sku_total_volume > 0:
                        computed_stats['total_case_equivalent'] = sku_total_volume
            
            # Log what we computed
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Final computed basic stats: {computed_stats}")
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error computing basic stats: {e}")
        
        return computed_stats
    
    def _display_results_header(self, analysis_results: Dict[str, Any]) -> None:
        """Display full-width results header with key metrics."""
        st.markdown("### 📊 Analysis Summary")
        
        # Debug: Log the structure of analysis_results
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Analysis results keys: {list(analysis_results.keys())}")
        
        # Get statistics from results with fallback logic
        order_stats = analysis_results.get('order_statistics', {})
        
        # If order_statistics is empty, try alternative locations
        if not order_stats:
            # Try to extract from other possible locations
            if 'statistics' in analysis_results:
                order_stats = analysis_results['statistics']
                logger.info("Found statistics in root level 'statistics' key")
            elif 'order_analysis' in analysis_results and isinstance(analysis_results['order_analysis'], dict):
                order_stats = analysis_results['order_analysis'].get('statistics', {})
                logger.info("Found statistics in 'order_analysis.statistics'")
            else:
                # Try to compute basic stats from available data
                logger.warning("No order statistics found, attempting to compute from available data")
                order_stats = self._compute_basic_stats_from_data(analysis_results)
        
        logger.info(f"Order stats keys: {list(order_stats.keys()) if order_stats else 'None'}")
        logger.info(f"Order stats values: {order_stats if order_stats else 'Empty'}")
        
        # FORCE computed statistics if we have data but metrics are zero
        # This is a temporary fix to ensure users see correct values
        computed_backup = self._compute_basic_stats_from_data(analysis_results)
        if computed_backup:
            logger.info(f"Computed backup stats: {computed_backup}")
            # Check if order_stats has all zero values and override
            if order_stats:
                key_metrics = ['total_order_lines', 'unique_skus', 'unique_dates', 'total_case_equivalent']
                all_zeros = all(order_stats.get(key, 0) == 0 for key in key_metrics)
                if all_zeros:
                    logger.warning("Order stats found but all values are zero, using computed stats")
                    order_stats.update(computed_backup)
                    logger.info(f"Updated order stats with computed values: {order_stats}")
                else:
                    # Check individual metrics and update any that are zero but computed has values
                    for key in key_metrics:
                        if order_stats.get(key, 0) == 0 and computed_backup.get(key, 0) > 0:
                            logger.info(f"Overriding zero value for {key}: {order_stats.get(key)} -> {computed_backup[key]}")
                            order_stats[key] = computed_backup[key]
            else:
                # No order_stats found at all, use computed
                order_stats = computed_backup
                logger.info(f"No order_stats found, using computed stats: {order_stats}")
        
        sku_stats = analysis_results.get('sku_statistics', {})
        
        # Add debug info for development (can be removed later)
        try:
            with st.expander("🔧 Debug Info", expanded=False):
                st.write("**Analysis Results Structure:**")
                st.json({
                    "available_keys": list(analysis_results.keys()),
                    "order_stats_found": bool(order_stats),
                    "order_stats_keys": list(order_stats.keys()) if order_stats else [],
                    "order_stats_values": {k: v for k, v in order_stats.items()} if order_stats else "Empty",
                    "final_order_stats_for_display": {
                        "total_order_lines": order_stats.get('total_order_lines', 0),
                        "unique_skus": order_stats.get('unique_skus', 0),
                        "unique_dates": order_stats.get('unique_dates', 0),
                        "total_case_equivalent": order_stats.get('total_case_equivalent', 0)
                    } if order_stats else "No stats",
                    "data_types": {k: str(type(v)) for k, v in analysis_results.items()},
                    "total_analysis_results_keys": len(analysis_results),
                    "session_state_check": {
                        "analysis_complete": st.session_state.get('analysis_complete', 'Not found'),
                        "analysis_results_keys": list(st.session_state.get('analysis_results', {}).keys()),
                        "analysis_outputs_keys": list(st.session_state.get('analysis_outputs', {}).keys())
                    }
                })
                
                # Show some sample data if available
                if 'date_order_summary' in analysis_results:
                    date_data = analysis_results['date_order_summary']
                    if isinstance(date_data, pd.DataFrame) and not date_data.empty:
                        st.write("**Sample from date_order_summary:**")
                        st.dataframe(date_data.head(3))
                    else:
                        st.write("**date_order_summary:** Empty or not a DataFrame")
                        
                if 'sku_profile_abc_fms' in analysis_results:
                    sku_data = analysis_results['sku_profile_abc_fms']
                    if isinstance(sku_data, pd.DataFrame) and not sku_data.empty:
                        st.write("**Sample from sku_profile_abc_fms:**")
                        st.dataframe(sku_data.head(3))
                        
                        # Show quick calculations
                        if 'Total_Case_Equiv' in sku_data.columns:
                            quick_total = sku_data['Total_Case_Equiv'].sum()
                            st.write(f"**Quick calc - Total Case Equiv from SKU data:** {quick_total:,.2f}")
                        
                        if 'Total_Order_Lines' in sku_data.columns:
                            quick_lines = sku_data['Total_Order_Lines'].sum()
                            st.write(f"**Quick calc - Total Order Lines from SKU data:** {quick_lines:,}")
                    else:
                        st.write("**sku_profile_abc_fms:** Empty or not a DataFrame")
        except Exception as e:
            st.write(f"Debug info error: {e}")
        
        # Create metrics columns - full width with better spacing
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total Order Lines", 
                f"{order_stats.get('total_order_lines', 0):,}",
                help="Total number of order lines processed"
            )
        
        with col2:
            st.metric(
                "Unique SKUs", 
                f"{order_stats.get('unique_skus', 0):,}",
                help="Number of distinct SKUs in the dataset"
            )
        
        with col3:
            st.metric(
                "Date Range", 
                f"{order_stats.get('unique_dates', 0)} days",
                help="Number of days covered in the analysis"
            )
        
        with col4:
            st.metric(
                "Total Volume", 
                f"{order_stats.get('total_case_equivalent', 0):,.0f}",
                help="Total case equivalent volume"
            )
        
        with col5:
            st.metric(
                "Avg Daily Volume", 
                f"{order_stats.get('total_case_equivalent', 0) / max(order_stats.get('unique_dates', 1), 1):,.0f}",
                help="Average daily case equivalent volume"
            )
        
        st.markdown("---")
    
    def _display_overview_fullwidth(self, analysis_results: Dict[str, Any]) -> None:
        """Display overview optimized for full-width layout."""
        # ABC-FMS Distribution in side-by-side charts for full width usage
        if 'abc_fms_summary' in analysis_results:
            abc_fms_data = analysis_results['abc_fms_summary']
            
            if isinstance(abc_fms_data, pd.DataFrame):
                # Detect data structure
                structure_info = self._detect_data_structure(abc_fms_data)
                
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🔤 ABC Classification Distribution")
                    
                    if structure_info['is_cross_tab']:
                        # Handle cross-tabulation summary data
                        volume_col = structure_info['columns']['volume']
                        sku_col = structure_info['columns']['sku_count']
                        
                        if volume_col and sku_col and volume_col in abc_fms_data.columns:
                            abc_summary = abc_fms_data[['ABC', volume_col, sku_col]].copy()
                            abc_summary = abc_summary.rename(columns={sku_col: 'sku_count'})
                            
                            fig = px.pie(
                                abc_summary, 
                                values=volume_col, 
                                names='ABC',
                                title="Volume Distribution by ABC Class",
                                color_discrete_map={'A': '#ff6b6b', 'B': '#4ecdc4', 'C': '#45b7d1'}
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error(f"Missing required columns for cross-tab data: {volume_col}, {sku_col}")
                    
                    elif structure_info['is_raw_data']:
                        # Handle raw SKU data
                        volume_col = structure_info['columns']['volume']
                        sku_col = structure_info['columns']['sku_count']
                        
                        if volume_col and sku_col:
                            abc_summary = abc_fms_data.groupby('ABC').agg({
                                volume_col: 'sum',
                                sku_col: 'count'
                            }).reset_index()
                            
                            fig = px.pie(
                                abc_summary, 
                                values=volume_col, 
                                names='ABC',
                                title="Volume Distribution by ABC Class",
                                color_discrete_map={'A': '#ff6b6b', 'B': '#4ecdc4', 'C': '#45b7d1'}
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error(f"Missing required columns for raw data: {volume_col}, {sku_col}")
                    
                    else:
                        # Fallback: try to show basic ABC distribution
                        if 'ABC' in abc_fms_data.columns:
                            abc_counts = abc_fms_data['ABC'].value_counts()
                            st.bar_chart(abc_counts)
                            st.warning("Using basic ABC count fallback")
                        else:
                            st.error("Cannot create ABC chart - ABC column not found")
            
            with col2:
                st.subheader("⚡ FMS Classification Distribution")
                
                if structure_info['is_cross_tab']:
                    # Handle cross-tabulation data with F/M/S columns
                    line_f = structure_info['columns'].get('line_f')
                    line_m = structure_info['columns'].get('line_m') 
                    line_s = structure_info['columns'].get('line_s')
                    
                    if line_f and line_m and line_s and all(col in abc_fms_data.columns for col in [line_f, line_m, line_s]):
                        # Use order lines for FMS distribution
                        fms_totals = {
                            'Fast': abc_fms_data[line_f].sum(),
                            'Medium': abc_fms_data[line_m].sum(),
                            'Slow': abc_fms_data[line_s].sum()
                        }
                        
                        fms_df = pd.DataFrame(list(fms_totals.items()), columns=['FMS', 'Value'])
                        
                        fig = px.pie(
                            fms_df, 
                            values='Value', 
                            names='FMS',
                            title="Order Lines Distribution by FMS Class",
                            color_discrete_map={'Fast': '#ff9ff3', 'Medium': '#54a0ff', 'Slow': '#5f27cd'}
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Fallback to SKU counts if line data not available
                        sku_f = structure_info['columns'].get('sku_f')
                        sku_m = structure_info['columns'].get('sku_m')
                        sku_s = structure_info['columns'].get('sku_s')
                        
                        if sku_f and sku_m and sku_s and all(col in abc_fms_data.columns for col in [sku_f, sku_m, sku_s]):
                            fms_totals = {
                                'Fast': abc_fms_data[sku_f].sum(),
                                'Medium': abc_fms_data[sku_m].sum(),
                                'Slow': abc_fms_data[sku_s].sum()
                            }
                            
                            fms_df = pd.DataFrame(list(fms_totals.items()), columns=['FMS', 'Value'])
                            
                            fig = px.pie(
                                fms_df, 
                                values='Value', 
                                names='FMS',
                                title="SKU Count Distribution by FMS Class",
                                color_discrete_map={'Fast': '#ff9ff3', 'Medium': '#54a0ff', 'Slow': '#5f27cd'}
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("Cannot create FMS chart - missing F/M/S columns")
                            
                elif structure_info['is_raw_data']:
                    # Handle traditional raw data with FMS column
                    order_lines_col = structure_info['columns']['order_lines']
                    fms_col = structure_info['columns']['fms']
                    
                    if order_lines_col and fms_col:
                        fms_summary = abc_fms_data.groupby(fms_col).agg({
                            order_lines_col: 'sum'
                        }).reset_index()
                        
                        fig = px.pie(
                            fms_summary, 
                            values=order_lines_col, 
                            names=fms_col,
                            title="Order Lines Distribution by FMS Class"
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Missing required columns for FMS chart: {order_lines_col}, {fms_col}")
                        
                else:
                    st.warning("FMS distribution data not available in current format")
        
        # Full-width ABC-FMS Cross-tabulation
        st.subheader("📊 ABC-FMS Cross-Tabulation Matrix")
        if 'abc_fms_summary' in analysis_results:
            abc_fms_data = analysis_results['abc_fms_summary']
            if isinstance(abc_fms_data, pd.DataFrame):
                structure_info = self._detect_data_structure(abc_fms_data)
                
                if structure_info['is_cross_tab']:
                    # For cross-tabulation data, create a visual matrix
                    volume_cols = ['Volume_F', 'Volume_M', 'Volume_S']
                    if all(col in abc_fms_data.columns for col in volume_cols):
                        # Create matrix from existing cross-tab data
                        matrix_data = abc_fms_data[['ABC'] + volume_cols].set_index('ABC')
                        matrix_data.columns = ['Fast', 'Medium', 'Slow']
                        
                        # Create heatmap
                        fig = px.imshow(
                            matrix_data.values,
                            x=matrix_data.columns,
                            y=matrix_data.index,
                            aspect="auto",
                            title="Volume Distribution: ABC vs FMS Classification",
                            labels=dict(x="FMS Classification", y="ABC Classification", color="Volume")
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show the actual cross-tab table
                        st.dataframe(matrix_data, use_container_width=True)
                    else:
                        st.warning("Cross-tabulation matrix data not available")
                        
                elif structure_info['is_raw_data']:
                    # For raw data, create traditional crosstab
                    volume_col = structure_info['columns']['volume']
                    abc_col = structure_info['columns']['abc']
                    fms_col = structure_info['columns']['fms']
                    
                    if volume_col and abc_col and fms_col:
                        cross_tab = pd.crosstab(
                            abc_fms_data[abc_col], 
                            abc_fms_data[fms_col], 
                            values=abc_fms_data[volume_col], 
                            aggfunc='sum',
                            fill_value=0
                        )
                        
                        # Create heatmap
                        fig = px.imshow(
                            cross_tab.values,
                            x=cross_tab.columns,
                            y=cross_tab.index,
                            aspect="auto",
                            title="Volume Distribution: ABC vs FMS Classification",
                            labels=dict(x="FMS Classification", y="ABC Classification", color="Volume")
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show the actual cross-tab table
                        st.dataframe(cross_tab, use_container_width=True)
                    else:
                        st.error(f"Cannot create cross-tab: missing columns {volume_col}, {abc_col}, {fms_col}")
                        
                else:
                    st.warning("Cross-tabulation not available for this data format")
    
    def _display_date_analysis_fullwidth(self, analysis_results: Dict[str, Any]) -> None:
        """Display date analysis optimized for full-width layout."""
        if 'date_order_summary' in analysis_results:
            date_data = analysis_results['date_order_summary']
            
            if isinstance(date_data, pd.DataFrame) and not date_data.empty:
                st.subheader("📅 Daily Order Trends")
                
                # Detect volume column
                volume_col = None
                if 'Total_Case_Equiv' in date_data.columns:
                    volume_col = 'Total_Case_Equiv'
                elif 'Total_Case_Equivalent' in date_data.columns:
                    volume_col = 'Total_Case_Equivalent'
                elif 'Volume_Total' in date_data.columns:
                    volume_col = 'Volume_Total'
                else:
                    # Search for volume-like column
                    for col in date_data.columns:
                        if 'volume' in col.lower() or ('case' in col.lower() and 'equiv' in col.lower()):
                            volume_col = col
                            break
                
                if volume_col:
                    # Full-width time series chart
                    fig = px.line(
                        date_data, 
                        x='Date', 
                        y=volume_col,
                        title="Daily Case Equivalent Volume Trend",
                        labels={volume_col: 'Case Equivalent Volume', 'Date': 'Date'}
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Side-by-side metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Peak analysis
                        st.subheader("📈 Peak Analysis")
                        peak_day = date_data.loc[date_data[volume_col].idxmax()]
                        st.metric("Peak Volume Day", peak_day['Date'].strftime('%Y-%m-%d'))
                        st.metric("Peak Volume", f"{peak_day[volume_col]:,.0f} cases")
                        
                    with col2:
                        # Statistical summary
                        st.subheader("📊 Volume Statistics")
                        st.metric("Average Daily Volume", f"{date_data[volume_col].mean():,.0f}")
                        st.metric("Standard Deviation", f"{date_data[volume_col].std():,.0f}")
                else:
                    st.error(f"No volume column found in date data. Available columns: {list(date_data.columns)}")
                
                # FTE Analysis Section (if FTE columns are present)
                self._display_fte_analysis(date_data)
                
                # Full-width data table
                st.subheader("📋 Daily Data")
                st.dataframe(date_data, use_container_width=True)
    
    def _display_fte_analysis(self, date_data: pd.DataFrame) -> None:
        """Display FTE (Full-Time Equivalent) analysis if FTE columns are present."""
        # Check if FTE columns exist
        fte_columns = ['FTE_Required', 'Total_Touches', 'Touch_Time_Min', 'Walk_Time_Min', 'Total_Time_Min']
        has_fte_data = any(col in date_data.columns for col in fte_columns)
        
        if not has_fte_data:
            return
            
        st.subheader("👥 Workforce Planning (FTE Analysis)")
        
        if 'FTE_Required' in date_data.columns:
            # Create FTE trend chart
            fig = px.line(
                date_data, 
                x='Date', 
                y='FTE_Required',
                title="Daily FTE (Full-Time Equivalent) Requirements",
                labels={'FTE_Required': 'Workers Required', 'Date': 'Date'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # FTE metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_fte = date_data['FTE_Required'].mean()
                st.metric("Average FTE Required", f"{avg_fte:.1f}")
                
            with col2:
                max_fte = date_data['FTE_Required'].max()
                max_fte_date = date_data.loc[date_data['FTE_Required'].idxmax(), 'Date']
                st.metric("Peak FTE Required", f"{max_fte:.0f}")
                st.caption(f"on {max_fte_date.strftime('%Y-%m-%d')}")
                
            with col3:
                min_fte = date_data['FTE_Required'].min()
                min_fte_date = date_data.loc[date_data['FTE_Required'].idxmin(), 'Date']
                st.metric("Minimum FTE Required", f"{min_fte:.0f}")
                st.caption(f"on {min_fte_date.strftime('%Y-%m-%d')}")
                
            with col4:
                fte_std = date_data['FTE_Required'].std()
                st.metric("FTE Variability", f"{fte_std:.1f}")
                st.caption("Standard deviation")
        
        # Detailed FTE breakdown if time components are available
        if all(col in date_data.columns for col in ['Touch_Time_Min', 'Walk_Time_Min']):
            st.subheader("⏱️ Time Breakdown Analysis")
            
            # Time breakdown chart
            time_data = date_data[['Date', 'Touch_Time_Min', 'Walk_Time_Min']].copy()
            time_data_melted = time_data.melt(
                id_vars=['Date'], 
                value_vars=['Touch_Time_Min', 'Walk_Time_Min'],
                var_name='Time_Type', 
                value_name='Minutes'
            )
            
            fig = px.bar(
                time_data_melted, 
                x='Date', 
                y='Minutes', 
                color='Time_Type',
                title="Daily Time Breakdown: Touch Time vs Walk Time",
                labels={'Minutes': 'Time (Minutes)', 'Time_Type': 'Activity Type'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Time statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Touch Time Statistics**")
                st.metric("Average Daily Touch Time", f"{date_data['Touch_Time_Min'].mean():.0f} min")
                st.metric("Peak Touch Time", f"{date_data['Touch_Time_Min'].max():.0f} min")
                
            with col2:
                st.markdown("**Walk Time Statistics**")
                st.metric("Average Daily Walk Time", f"{date_data['Walk_Time_Min'].mean():.0f} min")
                st.metric("Peak Walk Time", f"{date_data['Walk_Time_Min'].max():.0f} min")
        
        # Total touches analysis if available
        if 'Total_Touches' in date_data.columns:
            st.subheader("🔢 Daily Touch Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Touches trend
                fig = px.line(
                    date_data, 
                    x='Date', 
                    y='Total_Touches',
                    title="Daily Total Touches (Picks)",
                    labels={'Total_Touches': 'Total Touches', 'Date': 'Date'}
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                # Touches statistics
                st.markdown("**Touch Activity Metrics**")
                avg_touches = date_data['Total_Touches'].mean()
                max_touches = date_data['Total_Touches'].max()
                st.metric("Average Daily Touches", f"{avg_touches:,.0f}")
                st.metric("Peak Daily Touches", f"{max_touches:,.0f}")
                
                # Calculate productivity if possible
                if 'FTE_Required' in date_data.columns:
                    # Touches per FTE
                    date_data_clean = date_data[date_data['FTE_Required'] > 0].copy()
                    if not date_data_clean.empty:
                        touches_per_fte = date_data_clean['Total_Touches'] / date_data_clean['FTE_Required']
                        avg_productivity = touches_per_fte.mean()
                        st.metric("Avg Touches per FTE", f"{avg_productivity:,.0f}")
        
        # FTE insights summary
        if 'FTE_Required' in date_data.columns:
            st.markdown("---")
            st.markdown("### 💡 FTE Insights")
            
            total_fte_days = date_data['FTE_Required'].sum()
            avg_fte = date_data['FTE_Required'].mean()
            peak_fte = date_data['FTE_Required'].max()
            
            insights = []
            insights.append(f"📊 **Total FTE-Days Required**: {total_fte_days:.0f} worker-days across all analyzed dates")
            insights.append(f"⚖️ **Staffing Recommendation**: Plan for {avg_fte:.1f} workers on average, with capacity to scale to {peak_fte:.0f} during peak periods")
            
            if fte_std > avg_fte * 0.3:  # High variability
                insights.append(f"⚠️ **High Variability**: FTE requirements vary significantly (σ={fte_std:.1f}). Consider flexible staffing or workforce balancing strategies")
            
            if 'Total_Touches' in date_data.columns and avg_productivity:
                insights.append(f"🎯 **Productivity Target**: Workers should handle approximately {avg_productivity:,.0f} touches per day to meet efficiency targets")
            
            for insight in insights:
                st.markdown(insight)
    
    def _display_sku_analysis_fullwidth(self, analysis_results: Dict[str, Any]) -> None:
        """Display SKU analysis optimized for full-width layout."""
        if 'sku_profile_abc_fms' in analysis_results:
            sku_profile = analysis_results['sku_profile_abc_fms']
            
            if isinstance(sku_profile, pd.DataFrame) and not sku_profile.empty:
                st.subheader("🏷️ SKU Analysis")
                
                # Detect data structure
                structure_info = self._detect_data_structure(sku_profile)
                
                # Get appropriate column names
                volume_col = structure_info['columns']['volume']
                sku_col = structure_info['columns']['sku_count']
                lines_col = structure_info['columns']['order_lines']
                abc_col = structure_info['columns']['abc']
                fms_col = structure_info['columns']['fms']
                
                if volume_col and sku_col:
                    # Top SKUs chart - full width
                    if volume_col in sku_profile.columns:
                        top_skus = sku_profile.nlargest(20, volume_col)
                        
                        fig = go.Figure(go.Bar(
                            y=top_skus[sku_col] if sku_col in top_skus.columns else range(len(top_skus)),
                            x=top_skus[volume_col],
                            orientation='h',
                            marker_color='lightgreen',
                            text=top_skus[volume_col].round(0),
                            textposition='auto'
                        ))
                        
                        fig.update_layout(
                            title="Top 20 SKUs by Volume",
                            xaxis_title="Volume",
                            yaxis_title="SKU" if sku_col in top_skus.columns else "Rank",
                            height=600,
                            yaxis={'categoryorder': 'total ascending'}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # ABC-FMS scatter plot - full width (only for raw data)
                    if structure_info['is_raw_data'] and lines_col and abc_col and fms_col:
                        st.subheader("🔤 ABC-FMS Classification Scatter")
                        
                        # Check if we have the 2D-Classification column
                        hover_cols = [sku_col]
                        if '2D-Classification' in sku_profile.columns:
                            hover_cols.append('2D-Classification')
                        
                        fig = px.scatter(
                            sku_profile.head(100),  # Top 100 for readability
                            x=lines_col,
                            y=volume_col,
                            color=abc_col,
                            symbol=fms_col,
                            hover_data=hover_cols,
                            title="SKU Classification Scatter Plot (Top 100 SKUs)",
                            labels={
                                lines_col: 'Order Lines (Movement Frequency)',
                                volume_col: 'Volume'
                            }
                        )
                        
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Full-width data table with filters
                    st.subheader("📋 SKU Profile Data")
                    
                    if structure_info['is_raw_data'] and abc_col and fms_col:
                        # Filter controls for raw data
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            abc_options = sorted(sku_profile[abc_col].unique()) if abc_col in sku_profile.columns else ['A', 'B', 'C']
                            abc_filter = st.multiselect(
                                "Filter by ABC",
                                options=abc_options,
                                default=abc_options,
                                key="abc_filter_fullwidth"
                            )
                        
                        with col2:
                            fms_options = sorted(sku_profile[fms_col].unique()) if fms_col in sku_profile.columns else ['F', 'M', 'S']
                            fms_filter = st.multiselect(
                                "Filter by FMS",
                                options=fms_options,
                                default=fms_options,
                                key="fms_filter_fullwidth"
                            )
                        
                        with col3:
                            # Handle small datasets gracefully
                            min_skus = max(1, min(10, len(sku_profile)))
                            max_skus = max(min_skus + 1, min(100, len(sku_profile)))
                            default_skus = max(min_skus, min(50, len(sku_profile)))
                            
                            top_n = st.slider(
                                "Show Top N SKUs",
                                min_value=min_skus,
                                max_value=max_skus,
                                value=default_skus,
                                key="top_n_fullwidth"
                            )
                        
                        # Apply filters
                        filtered_data = sku_profile[
                            (sku_profile[abc_col].isin(abc_filter)) & 
                            (sku_profile[fms_col].isin(fms_filter))
                        ].head(top_n)
                        
                        # Display filtered data
                        st.dataframe(filtered_data, use_container_width=True)
                    else:
                        # For cross-tab data, just show the data
                        st.dataframe(sku_profile, use_container_width=True)
                        
                else:
                    st.error(f"Cannot display SKU analysis - missing required columns. Found: {structure_info['columns']}")
    
    def _display_abc_fms_analysis_fullwidth(self, analysis_results: Dict[str, Any]) -> None:
        """Display ABC-FMS analysis optimized for full-width layout."""
        if 'abc_fms_summary' in analysis_results:
            abc_fms_data = analysis_results['abc_fms_summary']
            
            if isinstance(abc_fms_data, pd.DataFrame) and not abc_fms_data.empty:
                st.subheader("🔤 ABC-FMS Classification Analysis")
                
                # Detect data structure
                structure_info = self._detect_data_structure(abc_fms_data)
                
                if structure_info['is_cross_tab']:
                    # Handle cross-tabulation summary data
                    
                    # Full-width cross-tabulation heatmap using existing matrix
                    volume_cols = ['Volume_F', 'Volume_M', 'Volume_S']
                    if all(col in abc_fms_data.columns for col in volume_cols):
                        matrix_data = abc_fms_data[['ABC'] + volume_cols].set_index('ABC')
                        matrix_data.columns = ['Fast', 'Medium', 'Slow']
                        
                        fig = px.imshow(
                            matrix_data.values,
                            x=matrix_data.columns,
                            y=matrix_data.index,
                            aspect="auto",
                            title="ABC vs FMS Classification Heatmap",
                            labels=dict(x="FMS Classification", y="ABC Classification", color="Volume")
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Side-by-side summary tables
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 ABC Summary")
                        abc_cols = ['ABC', 'Volume_Total', 'SKU_Total', 'Line_Total']
                        if all(col in abc_fms_data.columns for col in abc_cols):
                            abc_summary = abc_fms_data[abc_cols].copy()
                            st.dataframe(abc_summary, use_container_width=True)
                        else:
                            st.warning("ABC summary data not available")
                    
                    with col2:
                        st.subheader("⚡ FMS Summary")
                        # Calculate FMS totals from F/M/S columns
                        if all(col in abc_fms_data.columns for col in ['Volume_F', 'Volume_M', 'Volume_S', 'SKU_F', 'SKU_M', 'SKU_S', 'Line_F', 'Line_M', 'Line_S']):
                            fms_summary = pd.DataFrame({
                                'FMS': ['Fast', 'Medium', 'Slow'],
                                'Total_Volume': [
                                    abc_fms_data['Volume_F'].sum(),
                                    abc_fms_data['Volume_M'].sum(),
                                    abc_fms_data['Volume_S'].sum()
                                ],
                                'Total_SKUs': [
                                    abc_fms_data['SKU_F'].sum(),
                                    abc_fms_data['SKU_M'].sum(),
                                    abc_fms_data['SKU_S'].sum()
                                ],
                                'Total_Lines': [
                                    abc_fms_data['Line_F'].sum(),
                                    abc_fms_data['Line_M'].sum(),
                                    abc_fms_data['Line_S'].sum()
                                ]
                            })
                            st.dataframe(fms_summary, use_container_width=True)
                        else:
                            st.warning("FMS summary data not available")
                
                elif structure_info['is_raw_data']:
                    # Handle raw SKU data format
                    volume_col = structure_info['columns']['volume']
                    sku_col = structure_info['columns']['sku_count']
                    lines_col = structure_info['columns']['order_lines']
                    abc_col = structure_info['columns']['abc']
                    fms_col = structure_info['columns']['fms']
                    
                    if all(col for col in [volume_col, abc_col, fms_col]):
                        # Create cross-tabulation from raw data
                        cross_tab = pd.crosstab(
                            abc_fms_data[abc_col], 
                            abc_fms_data[fms_col], 
                            values=abc_fms_data[volume_col], 
                            aggfunc='sum',
                            fill_value=0
                        )
                        
                        fig = px.imshow(
                            cross_tab.values,
                            x=cross_tab.columns,
                            y=cross_tab.index,
                            aspect="auto",
                            title="ABC vs FMS Classification Heatmap",
                            labels=dict(x="FMS Classification", y="ABC Classification", color="Volume")
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Summary tables for raw data
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📊 ABC Summary")
                            abc_summary = abc_fms_data.groupby(abc_col).agg({
                                volume_col: 'sum',
                                sku_col: 'count',
                                lines_col: 'sum'
                            }).reset_index()
                            st.dataframe(abc_summary, use_container_width=True)
                        
                        with col2:
                            st.subheader("⚡ FMS Summary")
                            fms_summary = abc_fms_data.groupby(fms_col).agg({
                                volume_col: 'sum',
                                sku_col: 'count', 
                                lines_col: 'sum'
                            }).reset_index()
                            st.dataframe(fms_summary, use_container_width=True)
                    else:
                        st.error("Cannot create ABC-FMS analysis - missing required columns")
                
                # Full-width detailed classification table
                st.subheader("📋 Detailed ABC-FMS Classification")
                st.dataframe(abc_fms_data, use_container_width=True)
    
    def _display_charts_fullwidth(self, outputs: Dict[str, Any]) -> None:
        """Display charts optimized for full-width layout."""
        st.subheader("📈 Interactive Charts")
        
        if outputs and 'charts' in outputs:
            charts = outputs['charts']
            st.success("Charts generated successfully!")
            
            # Charts are displayed below with actual images
        else:
            st.info("Charts will be displayed here when chart generation is enabled in the analysis configuration.")
            
        # Placeholder for chart display
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background-color: #f0f2f6; border-radius: 10px;">
            <h3>📈 Chart Gallery</h3>
            <p>Interactive charts will be displayed here when analysis is run with chart generation enabled.</p>
            <p>Charts include: ABC Distribution, FMS Classification, Order Trends, SKU Performance, and more.</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _display_reports_fullwidth(self, outputs: Dict[str, Any]) -> None:
        """Display reports and downloads optimized for full-width layout."""
        st.subheader("📄 Reports & Downloads")
        
        if outputs:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'excel_file' in outputs:
                    excel_path = outputs['excel_file']
                    st.success("✅ Excel Report Generated")
                    
                    try:
                        # Read the actual Excel file
                        with open(excel_path, 'rb') as f:
                            excel_data = f.read()
                        
                        st.download_button(
                            "📊 Download Excel Report",
                            data=excel_data,
                            file_name="warehouse_analysis.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                        
                        # Show file info
                        import os
                        file_size = len(excel_data) / 1024  # KB
                        st.caption(f"File size: {file_size:.1f} KB")
                        
                    except Exception as e:
                        st.error("Unable to prepare Excel report for download. Please try running the analysis again.")
                else:
                    st.info("Excel report available when enabled in analysis configuration")
            
            with col2:
                if 'html_report' in outputs:
                    html_path = outputs['html_report']
                    st.success("✅ HTML Report Generated")
                    
                    try:
                        # Read the actual HTML file
                        with open(html_path, 'r', encoding='utf-8') as f:
                            html_data = f.read()
                        
                        st.download_button(
                            "🌐 Download HTML Report",
                            data=html_data,
                            file_name="warehouse_analysis.html",
                            mime="text/html",
                            use_container_width=True
                        )
                        
                        # Show file info
                        file_size = len(html_data.encode('utf-8')) / 1024  # KB
                        st.caption(f"File size: {file_size:.1f} KB")
                        
                    except Exception as e:
                        st.error("Unable to prepare HTML report for download. Please try running the analysis again.")
                else:
                    st.info("HTML report available when enabled in analysis configuration")
            
            with col3:
                # Word Report Generation
                st.markdown("### 📝 MS Word Report")
                if st.button("🚀 Generate Word Report", use_container_width=True, type="primary"):
                    try:
                        with st.spinner("Generating professional Word document..."):
                            # Check python-docx availability first
                            try:
                                import docx
                                st.info("📦 python-docx package detected")
                            except ImportError:
                                st.error("❌ python-docx package not found. This package is required for Word document generation.")
                                st.info("💡 If you're running this locally, install with: `pip install python-docx`")
                                st.info("🌐 If you're on Streamlit Cloud, the package should be installed automatically from requirements.txt")
                                return
                            
                            # Import the Word report generator with path fixing
                            try:
                                # Add parent directory to Python path for deployment compatibility
                                import sys
                                from pathlib import Path
                                parent_dir = Path(__file__).parent.parent.parent
                                if str(parent_dir) not in sys.path:
                                    sys.path.insert(0, str(parent_dir))
                                    st.info(f"🔧 Added parent directory to path: {parent_dir}")
                                
                                from warehouse_analysis_modular.reporting.word_report import generate_word_report
                                st.info("🔧 Word report generator loaded successfully")
                            except ImportError as e:
                                st.error(f"❌ Could not import Word report generator: {str(e)}")
                                st.info("🔍 Module path issue detected. Checking alternative import methods...")
                                
                                # Try alternative import approach and show deployment structure
                                try:
                                    # Investigate the deployment file structure
                                    import os
                                    current_dir = Path(__file__).parent.parent
                                    parent_dir = current_dir.parent
                                    
                                    st.info(f"🔍 Current directory: {current_dir}")
                                    st.info(f"🔍 Parent directory: {parent_dir}")
                                    st.info(f"🔍 Working directory: {os.getcwd()}")
                                    
                                    # Show directory contents
                                    try:
                                        if parent_dir.exists():
                                            parent_contents = [f.name for f in parent_dir.iterdir()]
                                            st.info(f"🔍 Parent directory contents: {parent_contents}")
                                        else:
                                            st.info("🔍 Parent directory does not exist")
                                    except:
                                        st.info("🔍 Cannot read parent directory contents")
                                    
                                    # Check for the warehouse_analysis_modular package
                                    word_report_path = parent_dir / "warehouse_analysis_modular" / "reporting" / "word_report.py"
                                    st.info(f"🔍 Looking for word_report at: {word_report_path}")
                                    st.info(f"🔍 Word report exists: {word_report_path.exists()}")
                                    
                                    # Also check current working directory
                                    cwd_word_report = Path(os.getcwd()) / "warehouse_analysis_modular" / "reporting" / "word_report.py"
                                    st.info(f"🔍 Also checking in CWD: {cwd_word_report}")
                                    st.info(f"🔍 CWD word report exists: {cwd_word_report.exists()}")
                                    
                                    if word_report_path.exists():
                                        # Add to path and retry
                                        sys.path.insert(0, str(parent_dir))
                                        from warehouse_analysis_modular.reporting.word_report import generate_word_report
                                        st.success("✅ Word report generator loaded via alternative path")
                                    elif cwd_word_report.exists():
                                        # Try from working directory
                                        sys.path.insert(0, os.getcwd())
                                        from warehouse_analysis_modular.reporting.word_report import generate_word_report
                                        st.success("✅ Word report generator loaded from working directory")
                                    else:
                                        st.error("❌ Word report module files not found in any expected location")
                                        st.info("💡 This indicates the warehouse_analysis_modular package was not deployed to Streamlit Cloud")
                                        st.info("🔧 The package structure may need to be reorganized for cloud deployment")
                                        return
                                        
                                except Exception as alt_e:
                                    st.error(f"❌ Alternative import failed: {str(alt_e)}")
                                    st.info("💡 Word document generation requires the warehouse_analysis_modular package")
                                    return
                            
                            # Get analysis results from session state
                            analysis_results = st.session_state.get('analysis_results', {})
                            
                            if not analysis_results:
                                st.error("❌ No analysis results available for Word report generation.")
                                st.info("💡 Please run the analysis first to generate data for the report")
                                return
                            
                            st.info("📊 Analysis results found, starting Word document generation...")
                            
                            # Generate Word report
                            try:
                                word_path = generate_word_report(analysis_results)
                                st.info(f"📄 Word document generated at: {word_path}")
                                
                                # Verify file exists
                                if not word_path.exists():
                                    st.error("❌ Word document was not created successfully")
                                    return
                                
                                # Store in outputs for download
                                if 'word_report' not in outputs:
                                    outputs['word_report'] = str(word_path)
                                
                                st.success("✅ Word Report Generated Successfully!")
                                st.info(f"📁 File size: {word_path.stat().st_size / 1024:.1f} KB")
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"❌ Error during Word document generation: {str(e)}")
                                st.error(f"🐛 Error type: {type(e).__name__}")
                                import traceback
                                st.error(f"🔍 Full error details: {traceback.format_exc()}")
                                return
                                
                    except Exception as e:
                        st.error(f"❌ Unexpected error in Word report generation: {str(e)}")
                        st.error(f"🐛 Error type: {type(e).__name__}")
                        import traceback
                        st.error(f"🔍 Full error details: {traceback.format_exc()}")
                
                # Download Word report if available
                if 'word_report' in outputs:
                    word_path = outputs['word_report']
                    try:
                        with open(word_path, 'rb') as f:
                            word_data = f.read()
                        
                        st.download_button(
                            "📝 Download Word Report",
                            data=word_data,
                            file_name="warehouse_analysis_report.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True
                        )
                        
                        # Show file info
                        file_size = len(word_data) / 1024  # KB
                        st.caption(f"File size: {file_size:.1f} KB")
                        
                    except Exception as e:
                        st.error("Unable to prepare Word report for download.")
                
                # AI Insights in expandable section
                if 'llm_summaries' in outputs:
                    with st.expander("🤖 View AI Insights"):
                        summaries = outputs['llm_summaries']
                        if isinstance(summaries, dict):
                            for key, summary in summaries.items():
                                st.markdown(f"**{key}:**")
                                st.write(summary)
                else:
                    st.info("AI insights available when enabled")
        
        else:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; background-color: #f0f2f6; border-radius: 10px;">
                <h3>📄 Report Center</h3>
                <p>Generated reports and downloads will appear here after analysis.</p>
                <ul style="text-align: left; display: inline-block;">
                    <li>📊 Excel reports with detailed data tables</li>
                    <li>🌐 HTML reports with interactive charts</li>
                    <li>🤖 AI-powered insights and recommendations</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    def display_analysis_results(self, analysis_results: Dict[str, Any], 
                                outputs: Dict[str, Any] = None) -> None:
        """
        Display comprehensive analysis results optimized for full-width layout.
        
        Args:
            analysis_results: Dictionary containing analysis data
            outputs: Dictionary containing generated outputs (charts, reports, etc.)
        """
        if not analysis_results:
            st.warning("No analysis results to display.")
            return
        
        # Full-width results header with key metrics
        self._display_results_header(analysis_results)
        
        # Create tabs for different result categories - optimized for full width
        # Check if advanced analysis is available
        has_advanced = any(key in analysis_results for key in [
            'advanced_order_analysis', 'picking_analysis', 'enhanced_abc_fms_analysis'
        ])
        
        if has_advanced:
            tabs = st.tabs([
                "📊 Overview", 
                "📅 Date Analysis", 
                "🏷️ SKU Analysis", 
                "🔤 ABC-FMS Classification",
                "🔄 Multi-Metric Correlation",
                "📦 Case vs Piece Analysis",
                "📋 2D Classification Matrix", 
                "📈 Interactive Charts",
                "📄 Reports & Downloads"
            ])
        else:
            tabs = st.tabs([
                "📊 Overview", 
                "📅 Date Analysis", 
                "🏷️ SKU Analysis", 
                "🔤 ABC-FMS Classification",
                "📈 Interactive Charts",
                "📄 Reports & Downloads"
            ])
        
        with tabs[0]:
            self._display_overview_fullwidth(analysis_results)
        
        with tabs[1]:
            self._display_date_analysis_fullwidth(analysis_results)
        
        with tabs[2]:
            self._display_sku_analysis_fullwidth(analysis_results)
        
        with tabs[3]:
            self._display_abc_fms_analysis_fullwidth(analysis_results)
        
        if has_advanced:
            with tabs[4]:
                self._display_multi_metric_correlation(analysis_results)
            
            with tabs[5]:
                self._display_case_piece_analysis(analysis_results)
            
            with tabs[6]:
                self._display_2d_classification_matrix(analysis_results)
            
            with tabs[7]:
                self._display_charts(outputs)
            
            with tabs[8]:
                self._display_reports_fullwidth(outputs)
        else:
            with tabs[4]:
                self._display_charts(outputs)
            
            with tabs[5]:
                self._display_reports_fullwidth(outputs)
    
    def _display_overview(self, results: Dict[str, Any]) -> None:
        """Display high-level overview of results."""
        st.header("📊 Analysis Overview")
        
        # Key metrics
        order_stats = results.get('order_statistics', {})
        sku_stats = results.get('sku_statistics', {})
        
        if order_stats:
            st.subheader("📈 Key Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Order Lines",
                    f"{order_stats.get('total_order_lines', 0):,}",
                    help="Total number of order lines processed"
                )
            
            with col2:
                st.metric(
                    "Unique SKUs",
                    f"{order_stats.get('unique_skus', 0):,}",
                    help="Number of unique SKU codes"
                )
            
            with col3:
                st.metric(
                    "Date Range",
                    f"{order_stats.get('unique_dates', 0)} days",
                    help="Number of unique dates in the analysis period"
                )
            
            with col4:
                total_ce = order_stats.get('total_case_equivalent', 0)
                st.metric(
                    "Total Case Equivalent",
                    f"{total_ce:,.0f}",
                    help="Total volume in case equivalent units"
                )
        
        # Date range info
        date_range = order_stats.get('date_range', {})
        if date_range:
            st.subheader("📅 Analysis Period")
            col1, col2 = st.columns(2)
            
            with col1:
                start_date = date_range.get('start', 'N/A')
                if hasattr(start_date, 'strftime'):
                    start_date = start_date.strftime('%Y-%m-%d')
                st.info(f"**Start Date:** {start_date}")
            
            with col2:
                end_date = date_range.get('end', 'N/A')
                if hasattr(end_date, 'strftime'):
                    end_date = end_date.strftime('%Y-%m-%d')
                st.info(f"**End Date:** {end_date}")
        
        # ABC-FMS Distribution
        if sku_stats:
            st.subheader("🔤 Classification Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                abc_dist = sku_stats.get('abc_distribution', {})
                if abc_dist:
                    abc_df = pd.DataFrame([
                        {'Classification': k, 'Count': v} 
                        for k, v in abc_dist.items()
                    ])
                    
                    fig = px.pie(
                        abc_df, 
                        values='Count', 
                        names='Classification',
                        title="ABC Classification Distribution",
                        color_discrete_map={'A': '#ff7f0e', 'B': '#2ca02c', 'C': '#d62728'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fms_dist = sku_stats.get('fms_distribution', {})
                if fms_dist:
                    fms_df = pd.DataFrame([
                        {'Classification': k, 'Count': v} 
                        for k, v in fms_dist.items()
                    ])
                    
                    fig = px.pie(
                        fms_df, 
                        values='Count', 
                        names='Classification',
                        title="FMS Classification Distribution",
                        color_discrete_map={'F': '#1f77b4', 'M': '#ff7f0e', 'S': '#2ca02c'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    def _display_date_analysis(self, results: Dict[str, Any]) -> None:
        """Display date-wise analysis results."""
        st.header("📅 Date Analysis")
        
        # Date order summary
        date_summary = results.get('date_order_summary')
        if date_summary is not None and not date_summary.empty:
            
            st.subheader("📊 Daily Trends")
            
            # Interactive time series chart
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Total Case Equivalent by Date', 'Daily Order Count'),
                vertical_spacing=0.1
            )
            
            # Case equivalent trend
            fig.add_trace(
                go.Scatter(
                    x=date_summary['Date'],
                    y=date_summary['Total_Case_Equiv'],
                    mode='lines+markers',
                    name='Case Equivalent',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=4)
                ),
                row=1, col=1
            )
            
            # Order count trend
            fig.add_trace(
                go.Scatter(
                    x=date_summary['Date'],
                    y=date_summary['Distinct_Orders'],
                    mode='lines+markers',
                    name='Order Count',
                    line=dict(color='#ff7f0e', width=2),
                    marker=dict(size=4)
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                height=600,
                showlegend=True,
                title_text="Daily Order Patterns"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.subheader("📈 Summary Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_daily = date_summary['Total_Case_Equiv'].mean()
                st.metric("Avg Daily Volume", f"{avg_daily:,.0f}")
            
            with col2:
                max_daily = date_summary['Total_Case_Equiv'].max()
                st.metric("Peak Daily Volume", f"{max_daily:,.0f}")
            
            with col3:
                std_daily = date_summary['Total_Case_Equiv'].std()
                cv = std_daily / avg_daily if avg_daily > 0 else 0
                st.metric("Coefficient of Variation", f"{cv:.2f}")
            
            with col4:
                total_days = len(date_summary)
                st.metric("Total Days", f"{total_days}")
            
            # Data table
            st.subheader("📋 Date Summary Table")
            
            # Format the dataframe for display
            display_df = date_summary.copy()
            if 'Date' in display_df.columns:
                display_df['Date'] = pd.to_datetime(display_df['Date']).dt.strftime('%Y-%m-%d')
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("No date analysis data available.")
        
        # Percentile analysis
        percentile_profile = results.get('percentile_profile')
        if percentile_profile is not None and not percentile_profile.empty:
            st.subheader("📊 Percentile Analysis")
            
            # Create horizontal bar chart for percentiles
            fig = go.Figure(go.Bar(
                y=percentile_profile['Percentile'],
                x=percentile_profile['Total_Case_Equiv'],
                orientation='h',
                marker_color='lightblue',
                text=percentile_profile['Total_Case_Equiv'].round(0),
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Volume Percentiles for Capacity Planning",
                xaxis_title="Case Equivalent",
                yaxis_title="Percentile",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(percentile_profile, use_container_width=True, hide_index=True)
    
    def _display_sku_analysis(self, results: Dict[str, Any]) -> None:
        """Display SKU analysis results."""
        st.header("🏷️ SKU Analysis")
        
        sku_profile = results.get('sku_profile_abc_fms')
        if sku_profile is not None and not sku_profile.empty:
            
            # Top SKUs by volume
            st.subheader("🏆 Top SKUs by Volume")
            
            top_skus = sku_profile.nlargest(20, 'Total_Case_Equiv')
            
            # Horizontal bar chart
            fig = go.Figure(go.Bar(
                y=top_skus['Sku Code'],
                x=top_skus['Total_Case_Equiv'],
                orientation='h',
                marker_color='lightgreen',
                text=top_skus['Total_Case_Equiv'].round(0),
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Top 20 SKUs by Case Equivalent Volume",
                xaxis_title="Case Equivalent",
                yaxis_title="SKU Code",
                height=600,
                yaxis={'categoryorder': 'total ascending'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # ABC-FMS scatter plot
            st.subheader("🔤 ABC-FMS Classification Scatter")
            
            fig = px.scatter(
                sku_profile.head(100),  # Limit to top 100 for readability
                x='Total_Order_Lines',
                y='Total_Case_Equiv',
                color='ABC',
                symbol='FMS',
                hover_data=['Sku Code', '2D-Classification'],
                title="SKU Classification (Top 100 SKUs)",
                labels={
                    'Total_Order_Lines': 'Order Lines (Movement Frequency)',
                    'Total_Case_Equiv': 'Case Equivalent (Volume)'
                }
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # SKU profile table with filtering
            st.subheader("📋 SKU Profile Data")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                abc_filter = st.multiselect(
                    "Filter by ABC",
                    options=['A', 'B', 'C'],
                    default=['A', 'B', 'C'],
                    key="sku_abc_filter"
                )
            
            with col2:
                fms_filter = st.multiselect(
                    "Filter by FMS",
                    options=['F', 'M', 'S'],
                    default=['F', 'M', 'S'],
                    key="sku_fms_filter"
                )
            
            with col3:
                top_n = st.number_input(
                    "Show top N SKUs",
                    min_value=10,
                    max_value=len(sku_profile),
                    value=min(50, len(sku_profile)),
                    step=10,
                    key="sku_top_n"
                )
            
            # Apply filters
            filtered_df = sku_profile[
                (sku_profile['ABC'].isin(abc_filter)) &
                (sku_profile['FMS'].isin(fms_filter))
            ].head(top_n)
            
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)
            
        else:
            st.warning("No SKU analysis data available.")
    
    def _display_abc_fms_analysis(self, results: Dict[str, Any]) -> None:
        """Display ABC-FMS cross-tabulation analysis."""
        st.header("🔤 ABC-FMS Analysis")
        
        abc_fms_summary = results.get('abc_fms_summary')
        if abc_fms_summary is not None and not abc_fms_summary.empty:
            
            # Cross-tabulation heatmap
            st.subheader("🌡️ ABC-FMS Volume Heatmap")
            
            # Prepare data for heatmap (exclude Grand Total row)
            heatmap_data = abc_fms_summary[abc_fms_summary['ABC'] != 'Grand Total'].copy()
            
            # Create pivot table for volume percentages
            volume_matrix = heatmap_data.pivot_table(
                index='ABC',
                values=['Volume_F_pct', 'Volume_M_pct', 'Volume_S_pct'],
                aggfunc='first'
            )
            
            # Reshape for heatmap
            volume_heatmap = pd.DataFrame({
                'F': volume_matrix['Volume_F_pct'],
                'M': volume_matrix['Volume_M_pct'],
                'S': volume_matrix['Volume_S_pct']
            })
            
            fig = px.imshow(
                volume_heatmap.values,
                x=['F', 'M', 'S'],
                y=volume_heatmap.index,
                color_continuous_scale='Blues',
                aspect='auto',
                title="Volume Distribution (% of Total)"
            )
            
            # Add text annotations
            for i, row in enumerate(volume_heatmap.index):
                for j, col in enumerate(['F', 'M', 'S']):
                    value = volume_heatmap.loc[row, col]
                    fig.add_annotation(
                        x=j, y=i,
                        text=f"{value:.1f}%",
                        showarrow=False,
                        font=dict(color="white" if value > 50 else "black")
                    )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.subheader("📊 Classification Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**SKU Count Distribution:**")
                sku_data = heatmap_data[['ABC', 'SKU_F', 'SKU_M', 'SKU_S']].set_index('ABC')
                st.dataframe(sku_data, use_container_width=True)
            
            with col2:
                st.write("**Volume Distribution:**")
                vol_data = heatmap_data[['ABC', 'Volume_F', 'Volume_M', 'Volume_S']].set_index('ABC')
                st.dataframe(vol_data.round(2), use_container_width=True)
            
            # Complete summary table
            st.subheader("📋 Complete ABC-FMS Summary")
            st.dataframe(abc_fms_summary, use_container_width=True, hide_index=True)
            
        else:
            st.warning("No ABC-FMS analysis data available.")
    
    def _display_charts(self, outputs: Dict[str, Any]) -> None:
        """Display generated charts."""
        st.header("📈 Generated Charts")
        
        charts = outputs.get('charts', {}) if outputs else {}
        
        if charts:
            st.success(f"Charts generated successfully! Found {len(charts)} chart(s).")
            # Display each chart
            for chart_name, chart_path in charts.items():
                if chart_path and Path(chart_path).exists():
                    st.subheader(f"📊 {chart_name.replace('_', ' ').title()}")
                    
                    try:
                        # Display image
                        st.image(chart_path, use_container_width=True)
                        
                        # Download button
                        with open(chart_path, "rb") as file:
                            st.download_button(
                                label=f"Download {chart_name}.png",
                                data=file.read(),
                                file_name=f"{chart_name}.png",
                                mime="image/png"
                            )
                            
                    except Exception as e:
                        st.error(f"Could not display chart {chart_name}: {str(e)}")
                else:
                    st.warning(f"Chart file not found: {chart_name}")
        else:
            st.info("No charts were generated. Enable chart generation in the analysis options.")
    
    def _display_reports(self, outputs: Dict[str, Any]) -> None:
        """Display generated reports and download options."""
        st.header("📄 Generated Reports")
        
        if not outputs:
            st.info("No reports were generated.")
            return
        
        # Excel report
        excel_file = outputs.get('excel_file')
        if excel_file and Path(excel_file).exists():
            st.subheader("📊 Excel Report")
            
            try:
                with open(excel_file, "rb") as file:
                    st.download_button(
                        label="📥 Download Excel Report",
                        data=file.read(),
                        file_name="warehouse_analysis_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                st.success("Excel report is ready for download!")
                
            except Exception as e:
                st.error(f"Could not prepare Excel download: {str(e)}")
        
        # HTML report
        html_report = outputs.get('html_report')
        if html_report and Path(html_report).exists():
            st.subheader("🌐 HTML Report")
            
            try:
                with open(html_report, "rb") as file:
                    st.download_button(
                        label="📥 Download HTML Report",
                        data=file.read(),
                        file_name="warehouse_analysis_report.html",
                        mime="text/html"
                    )
                
                st.success("HTML report is ready for download!")
                
                # Option to view HTML content (with caution)
                if st.checkbox("Preview HTML Report", help="Show HTML report content in iframe"):
                    try:
                        with open(html_report, 'r', encoding='utf-8') as file:
                            html_content = file.read()
                        
                        # Display in iframe (limited functionality)
                        st.components.v1.html(html_content[:50000], height=600, scrolling=True)
                        
                    except Exception as e:
                        st.error(f"Could not display HTML preview: {str(e)}")
                        
            except Exception as e:
                st.error(f"Could not prepare HTML download: {str(e)}")
        
        # LLM summaries
        llm_summaries = outputs.get('llm_summaries', {})
        if llm_summaries:
            st.subheader("🤖 AI-Generated Insights")
            
            for section, summary in llm_summaries.items():
                if summary:
                    with st.expander(f"📝 {section.replace('_', ' ').title()} Insights"):
                        st.markdown(summary)

    def _display_multi_metric_correlation(self, analysis_results: Dict[str, Any]) -> None:
        """Display multi-metric correlation analysis."""
        st.header("🔄 Multi-Metric Correlation Analysis")
        
        advanced_order = analysis_results.get('advanced_order_analysis', {})
        if not advanced_order:
            st.info("Multi-metric correlation analysis not available. This feature requires advanced analysis to be enabled.")
            return
        
        correlation_analysis = advanced_order.get('correlation_analysis', {})
        if correlation_analysis:
            st.subheader("📊 Key Correlations")
            
            key_correlations = correlation_analysis.get('key_correlations', {})
            if key_correlations:
                # Display correlation metrics in a nice format
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    volume_lines = key_correlations.get('volume_lines', 0)
                    st.metric("Volume ↔ Lines", f"{volume_lines:.3f}")
                
                with col2:
                    volume_customers = key_correlations.get('volume_customers', 0)
                    st.metric("Volume ↔ Customers", f"{volume_customers:.3f}")
                
                with col3:
                    lines_customers = key_correlations.get('lines_customers', 0)
                    st.metric("Lines ↔ Customers", f"{lines_customers:.3f}")
            
            # Display daily metrics if available
            daily_metrics = advanced_order.get('daily_metrics', pd.DataFrame())
            if not daily_metrics.empty and len(daily_metrics) > 1:
                st.subheader("📈 Daily Multi-Metric Trends")
                
                # Create multi-metric time series chart
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Volume & Lines Over Time', 'Customer Activity'),
                    specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
                    vertical_spacing=0.15
                )
                
                # Plot volume and lines
                if 'Total_Case_Equiv' in daily_metrics.columns:
                    fig.add_trace(
                        go.Scatter(x=daily_metrics['Date'], y=daily_metrics['Total_Case_Equiv'], 
                                name="Case Equivalent", line=dict(color='blue')), 
                        row=1, col=1
                    )
                
                if 'Total_Lines' in daily_metrics.columns:
                    fig.add_trace(
                        go.Scatter(x=daily_metrics['Date'], y=daily_metrics['Total_Lines'], 
                                name="Total Lines", line=dict(color='red'), yaxis='y2'), 
                        row=1, col=1
                    )
                
                # Plot customers
                if 'Distinct_Customers' in daily_metrics.columns:
                    fig.add_trace(
                        go.Scatter(x=daily_metrics['Date'], y=daily_metrics['Distinct_Customers'], 
                                name="Distinct Customers", line=dict(color='green')), 
                        row=2, col=1
                    )
                
                fig.update_layout(height=600, showlegend=True)
                fig.update_yaxes(title_text="Case Equivalent", row=1, col=1)
                fig.update_yaxes(title_text="Lines", secondary_y=True, row=1, col=1)
                fig.update_yaxes(title_text="Customers", row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Enhanced percentile analysis
        enhanced_percentiles = advanced_order.get('enhanced_percentile_analysis', {})
        if enhanced_percentiles:
            st.subheader("📊 Enhanced Capacity Planning")
            
            percentile_data = enhanced_percentiles.get('percentile_breakdown', {})
            if percentile_data:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Volume Percentiles**")
                    volume_percentiles = {k: v for k, v in percentile_data.items() if 'volume' in k.lower()}
                    if volume_percentiles:
                        df_vol = pd.DataFrame(list(volume_percentiles.items()), columns=['Percentile', 'Value'])
                        st.dataframe(df_vol, use_container_width=True)
                
                with col2:
                    st.write("**Line Count Percentiles**") 
                    line_percentiles = {k: v for k, v in percentile_data.items() if 'line' in k.lower()}
                    if line_percentiles:
                        df_lines = pd.DataFrame(list(line_percentiles.items()), columns=['Percentile', 'Value'])
                        st.dataframe(df_lines, use_container_width=True)

    def _display_case_piece_analysis(self, analysis_results: Dict[str, Any]) -> None:
        """Display case vs piece picking analysis."""
        st.header("📦 Case vs Piece Picking Analysis")
        
        picking_analysis = analysis_results.get('picking_analysis', {})
        if not picking_analysis:
            st.info("Case vs piece picking analysis not available. This feature requires advanced analysis to be enabled.")
            return
        
        # Picking breakdown
        picking_breakdown = picking_analysis.get('picking_breakdown', {})
        if picking_breakdown:
            st.subheader("📊 Picking Method Distribution")
            
            # Create DataFrame for visualization
            breakdown_data = []
            for picking_type, data in picking_breakdown.items():
                breakdown_data.append({
                    'Picking Type': picking_type,
                    'Order Count': data.get('count', 0),
                    'Percentage': data.get('percentage', 0)
                })
            
            if breakdown_data:
                df_breakdown = pd.DataFrame(breakdown_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart
                    fig_pie = px.pie(df_breakdown, values='Order Count', names='Picking Type', 
                                   title="Picking Method Distribution")
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Bar chart
                    fig_bar = px.bar(df_breakdown, x='Picking Type', y='Order Count',
                                   title="Orders by Picking Method")
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Summary table
                st.subheader("📋 Detailed Breakdown")
                st.dataframe(df_breakdown, use_container_width=True)
        
        # Category analysis
        category_analysis = picking_analysis.get('category_analysis', {})
        if category_analysis:
            st.subheader("🏷️ Picking by Category")
            
            category_data = []
            for category, data in category_analysis.items():
                category_data.append({
                    'Category': category,
                    'Case Only': data.get('Case_Only', 0),
                    'Piece Only': data.get('Piece_Only', 0),
                    'Mixed': data.get('Mixed', 0)
                })
            
            if category_data:
                df_category = pd.DataFrame(category_data)
                
                # Stacked bar chart
                fig_stacked = go.Figure()
                
                fig_stacked.add_trace(go.Bar(
                    name='Case Only',
                    x=df_category['Category'],
                    y=df_category['Case Only'],
                    marker_color='blue'
                ))
                
                fig_stacked.add_trace(go.Bar(
                    name='Piece Only', 
                    x=df_category['Category'],
                    y=df_category['Piece Only'],
                    marker_color='red'
                ))
                
                fig_stacked.add_trace(go.Bar(
                    name='Mixed',
                    x=df_category['Category'], 
                    y=df_category['Mixed'],
                    marker_color='green'
                ))
                
                fig_stacked.update_layout(
                    barmode='stack',
                    title='Picking Methods by Category',
                    xaxis_title='Category',
                    yaxis_title='Order Count'
                )
                
                st.plotly_chart(fig_stacked, use_container_width=True)
                st.dataframe(df_category, use_container_width=True)

    def _display_2d_classification_matrix(self, analysis_results: Dict[str, Any]) -> None:
        """Display 2D ABC-FMS classification matrix."""
        st.header("📋 2D Classification Matrix")
        
        enhanced_abc = analysis_results.get('enhanced_abc_fms_analysis', {})
        if not enhanced_abc:
            st.info("2D classification matrix analysis not available. This feature requires advanced analysis to be enabled.")
            return
        
        # Classification matrix
        classification_matrix = enhanced_abc.get('classification_matrix', {})
        if classification_matrix:
            st.subheader("🔤 ABC × FMS Cross-Classification")
            
            # SKU count matrix
            sku_count_matrix = classification_matrix.get('sku_count_matrix', pd.DataFrame())
            if not sku_count_matrix.empty:
                st.write("**SKU Count Distribution**")
                
                # Create heatmap
                fig_heatmap = px.imshow(sku_count_matrix.iloc[:-1, :-1],  # Exclude totals
                                     text_auto=True,
                                     aspect="auto",
                                     title="SKU Count by ABC-FMS Classification")
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Show full table including totals
                st.dataframe(sku_count_matrix, use_container_width=True)
            
            # Volume matrix
            volume_matrix = classification_matrix.get('volume_matrix', pd.DataFrame())
            if not volume_matrix.empty:
                st.subheader("📊 Volume Distribution Matrix")
                
                # Create volume heatmap
                fig_vol_heatmap = px.imshow(volume_matrix.iloc[:-1, :-1],  # Exclude totals
                                         text_auto=True,
                                         aspect="auto", 
                                         title="Volume by ABC-FMS Classification")
                st.plotly_chart(fig_vol_heatmap, use_container_width=True)
                
                st.dataframe(volume_matrix, use_container_width=True)
        
        # Enhanced insights
        enhanced_insights = enhanced_abc.get('enhanced_insights', {})
        if enhanced_insights:
            st.subheader("🔍 Classification Insights")
            
            # Segmentation analysis
            segmentation = enhanced_insights.get('segmentation_analysis', {})
            if segmentation:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    high_value = segmentation.get('high_value_segments', [])
                    st.metric("High Value Segments", len(high_value))
                    if high_value:
                        st.write("**High Value:**")
                        for segment in high_value[:3]:  # Show top 3
                            st.write(f"• {segment}")
                
                with col2:
                    operational_complex = segmentation.get('operational_complexity', {})
                    complex_count = len([k for k, v in operational_complex.items() if v > 0.7])
                    st.metric("High Complexity", complex_count)
                
                with col3:
                    efficiency_score = enhanced_insights.get('classification_effectiveness', {})
                    eff_score = efficiency_score.get('overall_score', 0)
                    st.metric("Effectiveness Score", f"{eff_score:.2f}")


def create_results_display_section(analysis_results: Dict[str, Any], 
                                  outputs: Dict[str, Any] = None) -> None:
    """
    Create the complete results display section.
    
    Args:
        analysis_results: Dictionary containing analysis data
        outputs: Dictionary containing generated outputs
    """
    display_manager = ResultsDisplayManager()
    display_manager.display_analysis_results(analysis_results, outputs)


# Demo function
def results_display_demo():
    """Demo function for testing results display."""
    st.header("📊 Results Display Demo")
    
    # Create sample data
    sample_results = {
        'order_statistics': {
            'total_order_lines': 15000,
            'unique_skus': 250,
            'unique_dates': 30,
            'total_case_equivalent': 50000
        },
        'date_order_summary': pd.DataFrame({
            'Date': pd.date_range('2025-01-01', periods=30),
            'Total_Case_Equiv': np.random.normal(1500, 300, 30),
            'Distinct_Orders': np.random.poisson(50, 30)
        }),
        'sku_profile_abc_fms': pd.DataFrame({
            'Sku Code': [f'SKU{i:03d}' for i in range(50)],
            'ABC': np.random.choice(['A', 'B', 'C'], 50, p=[0.2, 0.3, 0.5]),
            'FMS': np.random.choice(['F', 'M', 'S'], 50, p=[0.3, 0.4, 0.3]),
            'Total_Case_Equiv': np.random.exponential(1000, 50),
            'Total_Order_Lines': np.random.poisson(100, 50)
        })
    }
    
    # Display results
    create_results_display_section(sample_results)


if __name__ == "__main__":
    results_display_demo()