#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross-Tabulation Analysis Module

Handles ABC×FMS cross-tabulation analysis and summary generation.
Extracted from the original Warehouse Analysis (2).py file.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional

# Import from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from warehouse_analysis_modular.utils.helpers import (
    validate_dataframe, normalize_abc_fms_values, 
    safe_division, setup_logging
)

logger = setup_logging()


class CrossTabulationAnalyzer:
    """
    Handles comprehensive ABC×FMS cross-tabulation analysis for strategic SKU insights.
    
    This analyzer creates detailed cross-tabulation matrices that reveal the relationship
    between volume-based ABC classification and movement frequency-based FMS classification.
    It provides multi-dimensional analysis across SKU counts, volumes, and order lines to
    support strategic inventory and warehouse management decisions.
    
    Key Capabilities:
    - Multi-dimensional Cross-tabulation: SKU count, volume, and order line matrices
    - Percentage Distribution Analysis: Relative distribution across categories
    - Comprehensive Summary Tables: Integrated ABC×FMS overview
    - Strategic Insights: Dominant categories and distribution patterns
    - Matrix Visualization: Heatmap-ready data structures
    
    Analysis Dimensions:
    - SKU Count Matrix: Number of SKUs in each ABC×FMS combination
    - Volume Matrix: Total volume distribution across categories
    - Order Lines Matrix: Order line frequency distribution
    - Percentage Matrices: Relative distribution analysis
    
    Input Requirements:
    - Complete SKU profile with ABC and FMS classifications
    - Required columns: Sku Code, Total_Order_Lines, Total_Case_Equiv, ABC, FMS
    - Data should come from SkuAnalyzer output
    
    Business Applications:
    - Strategic inventory segmentation and planning
    - Warehouse layout and slotting optimization
    - Resource allocation and capacity planning
    - Performance benchmarking and KPI development
    - Vendor and supplier relationship management
    
    Strategic Insights:
    - High-value, fast-moving SKUs (A×F): Premium slotting locations
    - High-value, slow-moving SKUs (A×S): Special handling considerations
    - Low-value, fast-moving SKUs (C×F): Efficiency optimization opportunities
    - Distribution patterns: Overall inventory balance analysis
    
    Example:
        # Initialize with SKU profile from SkuAnalyzer
        sku_profile = sku_analyzer.run_full_analysis()['sku_profile_abc_fms']
        analyzer = CrossTabulationAnalyzer(sku_profile)
        
        # Run cross-tabulation analysis
        results = analyzer.run_full_analysis()
        
        # Access comprehensive summary
        abc_fms_summary = results['abc_fms_summary']
        
        # Get insights for strategic decisions
        insights = results['insights']
        dominant_category = insights['dominant_categories']['highest_volume_category']
    """
    
    def __init__(self, sku_profile_abc_fms: pd.DataFrame):
        """
        Initialize the CrossTabulationAnalyzer with SKU profile data.
        
        Args:
            sku_profile_abc_fms: DataFrame with complete SKU profile including ABC and FMS classifications
        """
        self.sku_profile = sku_profile_abc_fms.copy()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._validate_and_prepare_data()
    
    def _validate_and_prepare_data(self):
        """Validate and prepare the SKU profile data."""
        required_columns = {"Sku Code", "Total_Order_Lines", "Total_Case_Equiv", "ABC", "FMS"}
        
        validate_dataframe(
            self.sku_profile,
            list(required_columns),
            min_rows=1,
            name="SKU profile"
        )
        
        # Normalize ABC and FMS values
        self.sku_profile = normalize_abc_fms_values(self.sku_profile)
        
        self.logger.info(f"Initialized cross-tabulation analyzer with {len(self.sku_profile)} SKUs")
    
    def create_sku_count_crosstab(self) -> pd.DataFrame:
        """
        Create cross-tabulation of SKU counts by ABC and FMS.
        
        Returns:
            DataFrame with SKU count cross-tabulation
        """
        self.logger.info("Creating SKU count cross-tabulation")
        
        # Create crosstab of SKU counts
        sku_count_crosstab = pd.crosstab(
            self.sku_profile["ABC"],
            self.sku_profile["FMS"],
            margins=False
        ).reindex(index=["A", "B", "C"], columns=["F", "M", "S"], fill_value=0)
        
        # Add row totals
        sku_count_crosstab["SKU_Total"] = sku_count_crosstab.sum(axis=1)
        
        return sku_count_crosstab
    
    def create_volume_crosstab(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create cross-tabulation of volume by ABC and FMS.
        
        Returns:
            Tuple of (volume_crosstab, volume_percentage)
        """
        self.logger.info("Creating volume cross-tabulation")
        
        # Create volume crosstab
        volume_crosstab = (
            self.sku_profile
            .groupby(["ABC", "FMS"], sort=False)["Total_Case_Equiv"]
            .sum()
            .unstack(fill_value=0)
        ).reindex(index=["A", "B", "C"], columns=["F", "M", "S"], fill_value=0)
        
        # Add row totals
        volume_crosstab["Volume_Total"] = volume_crosstab.sum(axis=1)
        
        # Calculate grand total for percentage calculation
        grand_total_volume = volume_crosstab[["F", "M", "S"]].to_numpy().sum()
        
        # Calculate percentages based on grand total
        volume_pct = safe_division(
            volume_crosstab[["F", "M", "S"]] * 100,
            grand_total_volume,
            fill_value=0
        ).round(0)
        
        volume_pct.columns = ["Volume_F_pct", "Volume_M_pct", "Volume_S_pct"]
        
        return volume_crosstab, volume_pct
    
    def create_lines_crosstab(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create cross-tabulation of order lines by ABC and FMS.
        
        Returns:
            Tuple of (lines_crosstab, lines_percentage)
        """
        self.logger.info("Creating order lines cross-tabulation")
        
        # Create lines crosstab
        lines_crosstab = (
            self.sku_profile
            .groupby(["ABC", "FMS"], sort=False)["Total_Order_Lines"]
            .sum()
            .unstack(fill_value=0)
        ).reindex(index=["A", "B", "C"], columns=["F", "M", "S"], fill_value=0)
        
        # Add row totals
        lines_crosstab["Line_Total"] = lines_crosstab.sum(axis=1)
        
        # Calculate grand total for percentage calculation
        grand_total_lines = lines_crosstab[["F", "M", "S"]].to_numpy().sum()
        
        # Calculate percentages based on grand total
        lines_pct = safe_division(
            lines_crosstab[["F", "M", "S"]] * 100,
            grand_total_lines,
            fill_value=0
        ).round(0)
        
        lines_pct.columns = ["Line_F_pct", "Line_M_pct", "Line_S_pct"]
        
        return lines_crosstab, lines_pct
    
    def create_abc_fms_summary(self) -> pd.DataFrame:
        """
        Create comprehensive ABC×FMS summary table.
        
        Returns:
            DataFrame with complete cross-tabulation summary
        """
        self.logger.info("Creating comprehensive ABC×FMS summary")
        
        # Get all cross-tabulations
        sku_count_crosstab = self.create_sku_count_crosstab()
        volume_crosstab, volume_pct = self.create_volume_crosstab()
        lines_crosstab, lines_pct = self.create_lines_crosstab()
        
        # Create summary rows for A, B, C
        abc_rows = ["A", "B", "C"]
        summary_rows = []
        
        for abc in abc_rows:
            row = {
                "ABC": abc,
                # SKU counts
                "SKU_F": int(sku_count_crosstab.at[abc, "F"]) if "F" in sku_count_crosstab.columns else 0,
                "SKU_M": int(sku_count_crosstab.at[abc, "M"]) if "M" in sku_count_crosstab.columns else 0,
                "SKU_S": int(sku_count_crosstab.at[abc, "S"]) if "S" in sku_count_crosstab.columns else 0,
                "SKU_Total": int(sku_count_crosstab.at[abc, "SKU_Total"]),
                # Volume absolute
                "Volume_F": float(volume_crosstab.at[abc, "F"]) if "F" in volume_crosstab.columns else 0.0,
                "Volume_M": float(volume_crosstab.at[abc, "M"]) if "M" in volume_crosstab.columns else 0.0,
                "Volume_S": float(volume_crosstab.at[abc, "S"]) if "S" in volume_crosstab.columns else 0.0,
                "Volume_Total": float(volume_crosstab.at[abc, "Volume_Total"]),
                # Volume percentages
                "Volume_F_pct": float(volume_pct.at[abc, "Volume_F_pct"]),
                "Volume_M_pct": float(volume_pct.at[abc, "Volume_M_pct"]),
                "Volume_S_pct": float(volume_pct.at[abc, "Volume_S_pct"]),
                # Lines absolute
                "Line_F": float(lines_crosstab.at[abc, "F"]) if "F" in lines_crosstab.columns else 0.0,
                "Line_M": float(lines_crosstab.at[abc, "M"]) if "M" in lines_crosstab.columns else 0.0,
                "Line_S": float(lines_crosstab.at[abc, "S"]) if "S" in lines_crosstab.columns else 0.0,
                "Line_Total": float(lines_crosstab.at[abc, "Line_Total"]),
                # Lines percentages
                "Line_F_pct": float(lines_pct.at[abc, "Line_F_pct"]),
                "Line_M_pct": float(lines_pct.at[abc, "Line_M_pct"]),
                "Line_S_pct": float(lines_pct.at[abc, "Line_S_pct"]),
            }
            summary_rows.append(row)
        
        abc_fms_summary = pd.DataFrame(summary_rows)
        
        # Add grand total row
        grand_total_row = self._create_grand_total_row(
            sku_count_crosstab, volume_crosstab, lines_crosstab
        )
        abc_fms_summary = pd.concat([abc_fms_summary, pd.DataFrame([grand_total_row])], ignore_index=True, sort=False)
        
        # Format and round numeric columns
        self._format_summary_table(abc_fms_summary)
        
        self.logger.info("ABC×FMS summary created successfully")
        return abc_fms_summary
    
    def _create_grand_total_row(self, sku_count_crosstab: pd.DataFrame,
                               volume_crosstab: pd.DataFrame,
                               lines_crosstab: pd.DataFrame) -> Dict:
        """
        Create grand total row for the summary table.
        
        Args:
            sku_count_crosstab: SKU count cross-tabulation
            volume_crosstab: Volume cross-tabulation
            lines_crosstab: Lines cross-tabulation
            
        Returns:
            Dictionary representing the grand total row
        """
        volume_total_sum = volume_crosstab["Volume_Total"].sum()
        lines_total_sum = lines_crosstab["Line_Total"].sum()
        
        grand_total = {
            "ABC": "Grand Total",
            "SKU_F": int(sku_count_crosstab[["F", "M", "S"]].sum().get("F", 0)),
            "SKU_M": int(sku_count_crosstab[["F", "M", "S"]].sum().get("M", 0)),
            "SKU_S": int(sku_count_crosstab[["F", "M", "S"]].sum().get("S", 0)),
            "SKU_Total": int(sku_count_crosstab["SKU_Total"].sum()),
            "Volume_F": float(volume_crosstab[["F", "M", "S"]].sum().get("F", 0.0)),
            "Volume_M": float(volume_crosstab[["F", "M", "S"]].sum().get("M", 0.0)),
            "Volume_S": float(volume_crosstab[["F", "M", "S"]].sum().get("S", 0.0)),
            "Volume_Total": float(volume_total_sum),
            # For grand row, percentage is percent of grand total
            "Volume_F_pct": round(safe_division(
                volume_crosstab[["F", "M", "S"]].sum().get("F", 0.0) * 100,
                volume_total_sum,
                fill_value=0
            ), 2),
            "Volume_M_pct": round(safe_division(
                volume_crosstab[["F", "M", "S"]].sum().get("M", 0.0) * 100,
                volume_total_sum,
                fill_value=0
            ), 2),
            "Volume_S_pct": round(safe_division(
                volume_crosstab[["F", "M", "S"]].sum().get("S", 0.0) * 100,
                volume_total_sum,
                fill_value=0
            ), 2),
            "Line_F": float(lines_crosstab[["F", "M", "S"]].sum().get("F", 0.0)),
            "Line_M": float(lines_crosstab[["F", "M", "S"]].sum().get("M", 0.0)),
            "Line_S": float(lines_crosstab[["F", "M", "S"]].sum().get("S", 0.0)),
            "Line_Total": float(lines_total_sum),
            "Line_F_pct": round(safe_division(
                lines_crosstab[["F", "M", "S"]].sum().get("F", 0.0) * 100,
                lines_total_sum,
                fill_value=0
            ), 2),
            "Line_M_pct": round(safe_division(
                lines_crosstab[["F", "M", "S"]].sum().get("M", 0.0) * 100,
                lines_total_sum,
                fill_value=0
            ), 2),
            "Line_S_pct": round(safe_division(
                lines_crosstab[["F", "M", "S"]].sum().get("S", 0.0) * 100,
                lines_total_sum,
                fill_value=0
            ), 2),
        }
        
        return grand_total
    
    def _format_summary_table(self, abc_fms_summary: pd.DataFrame):
        """
        Format the summary table with appropriate data types and rounding.
        
        Args:
            abc_fms_summary: Summary table to format (modified in place)
        """
        # Define column groups for formatting
        pct_cols = ["Volume_F_pct", "Volume_M_pct", "Volume_S_pct", "Line_F_pct", "Line_M_pct", "Line_S_pct"]
        amt_cols = ["Volume_F", "Volume_M", "Volume_S", "Volume_Total", "Line_F", "Line_M", "Line_S", "Line_Total"]
        sku_cols = ["SKU_F", "SKU_M", "SKU_S", "SKU_Total"]
        
        # Round percentage columns
        abc_fms_summary[pct_cols] = abc_fms_summary[pct_cols].fillna(0).astype(float).round(2)
        
        # Round amount columns
        abc_fms_summary[amt_cols] = abc_fms_summary[amt_cols].fillna(0).astype(float).round(2)
        
        # Ensure integer SKU counts
        abc_fms_summary[sku_cols] = abc_fms_summary[sku_cols].fillna(0).astype(int)
    
    def get_cross_tabulation_insights(self, abc_fms_summary: pd.DataFrame) -> Dict:
        """
        Generate insights from the cross-tabulation analysis.
        
        Args:
            abc_fms_summary: ABC×FMS summary table
            
        Returns:
            Dictionary with key insights
        """
        # Exclude grand total row for analysis
        analysis_data = abc_fms_summary[abc_fms_summary['ABC'] != 'Grand Total'].copy()
        
        insights = {
            'dominant_categories': {
                'highest_volume_category': analysis_data.loc[analysis_data['Volume_Total'].idxmax(), 'ABC'],
                'highest_lines_category': analysis_data.loc[analysis_data['Line_Total'].idxmax(), 'ABC'],
                'most_skus_category': analysis_data.loc[analysis_data['SKU_Total'].idxmax(), 'ABC']
            },
            'fast_moving_analysis': {
                'a_fast_volume_pct': analysis_data[analysis_data['ABC'] == 'A']['Volume_F_pct'].iloc[0] if len(analysis_data[analysis_data['ABC'] == 'A']) > 0 else 0,
                'b_fast_volume_pct': analysis_data[analysis_data['ABC'] == 'B']['Volume_F_pct'].iloc[0] if len(analysis_data[analysis_data['ABC'] == 'B']) > 0 else 0,
                'c_fast_volume_pct': analysis_data[analysis_data['ABC'] == 'C']['Volume_F_pct'].iloc[0] if len(analysis_data[analysis_data['ABC'] == 'C']) > 0 else 0
            },
            'distribution_summary': {
                'total_volume': analysis_data['Volume_Total'].sum(),
                'total_lines': analysis_data['Line_Total'].sum(),
                'total_skus': analysis_data['SKU_Total'].sum(),
                'abc_volume_distribution': analysis_data.set_index('ABC')['Volume_Total'].to_dict(),
                'abc_lines_distribution': analysis_data.set_index('ABC')['Line_Total'].to_dict()
            }
        }
        
        return insights
    
    def run_full_analysis(self) -> Dict:
        """
        Run complete cross-tabulation analysis.
        
        Returns:
            Dictionary containing all analysis results
        """
        self.logger.info("Running full cross-tabulation analysis")
        
        # Generate comprehensive summary
        abc_fms_summary = self.create_abc_fms_summary()
        
        # Generate insights
        insights = self.get_cross_tabulation_insights(abc_fms_summary)
        
        # Get individual crosstabs for detailed analysis
        sku_count_crosstab = self.create_sku_count_crosstab()
        volume_crosstab, volume_pct = self.create_volume_crosstab()
        lines_crosstab, lines_pct = self.create_lines_crosstab()
        
        results = {
            'abc_fms_summary': abc_fms_summary,
            'sku_count_crosstab': sku_count_crosstab,
            'volume_crosstab': volume_crosstab,
            'volume_percentage': volume_pct,
            'lines_crosstab': lines_crosstab,
            'lines_percentage': lines_pct,
            'insights': insights
        }
        
        self.logger.info("Cross-tabulation analysis completed successfully")
        return results


def analyze_cross_tabulation(sku_profile_abc_fms: pd.DataFrame) -> Dict:
    """
    Convenience function to run cross-tabulation analysis.
    
    Args:
        sku_profile_abc_fms: DataFrame with complete SKU profile
        
    Returns:
        Dictionary containing analysis results
    """
    analyzer = CrossTabulationAnalyzer(sku_profile_abc_fms)
    return analyzer.run_full_analysis()