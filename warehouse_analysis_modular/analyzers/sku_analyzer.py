#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SKU Analysis Module

Handles SKU profiling and ABC-FMS classification analysis.
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

from config import ABC_THRESHOLDS, FMS_THRESHOLDS
from warehouse_analysis_modular.utils.helpers import (
    validate_dataframe, classify_abc, classify_fms, 
    safe_division, normalize_abc_fms_values, 
    create_2d_classification, format_numeric_columns,
    handle_infinite_values, setup_logging
)

logger = setup_logging()


class SkuAnalyzer:
    """
    Handles SKU-level analysis including ABC-FMS classification.
    """
    
    def __init__(self, enriched_data: pd.DataFrame):
        """
        Initialize the SkuAnalyzer with enriched order data.
        
        Args:
            enriched_data: DataFrame with enriched order data including calculated fields
        """
        self.enriched_data = enriched_data
        self.logger = logging.getLogger(self.__class__.__name__)
        self._validate_input_data()
    
    def _validate_input_data(self):
        """Validate that the input data has required columns."""
        required_columns = [
            'Sku Code', 'Order No.', 'Date', 'Case_Equivalent'
        ]
        validate_dataframe(
            self.enriched_data, 
            required_columns, 
            min_rows=1, 
            name="Enriched order data"
        )
    
    def aggregate_sku_metrics(self) -> pd.DataFrame:
        """
        Aggregate base metrics per SKU.
        
        Returns:
            DataFrame with SKU-level metrics
        """
        self.logger.info("Aggregating SKU-level metrics")
        
        # Total order lines per SKU
        sku_lines = (
            self.enriched_data.groupby("Sku Code", sort=False)
            .agg(Total_Order_Lines=("Order No.", "count"))
            .reset_index()
        )
        
        # Total case-equivalent volume per SKU
        sku_volume = (
            self.enriched_data.groupby("Sku Code", sort=False)
            .agg(Total_Case_Equiv=("Case_Equivalent", "sum"))
            .reset_index()
        )
        
        # Merge line and volume metrics
        sku_metrics = sku_lines.merge(sku_volume, on="Sku Code", how="outer").fillna(0)
        
        self.logger.info(f"Aggregated metrics for {len(sku_metrics)} SKUs")
        return sku_metrics
    
    def classify_fms(self, sku_metrics: pd.DataFrame) -> pd.DataFrame:
        """
        Perform FMS (Fast/Medium/Slow) classification based on order lines.
        
        Args:
            sku_metrics: DataFrame with aggregated SKU metrics
            
        Returns:
            DataFrame with FMS classification
        """
        self.logger.info("Performing FMS classification")
        
        # Calculate total lines for percentage calculation
        total_lines_all = sku_metrics["Total_Order_Lines"].sum()
        
        # Calculate percentage of total order lines
        sku_metrics["Pct_of_Total_Order_Lines"] = safe_division(
            sku_metrics["Total_Order_Lines"] * 100,
            total_lines_all,
            fill_value=0
        )
        
        # Sort SKUs by order lines (descending) for cumulative calculation
        sku_fms_sorted = sku_metrics.sort_values(
            by="Total_Order_Lines", 
            ascending=False
        ).reset_index(drop=True)
        
        # Calculate cumulative percentage of order lines
        sku_fms_sorted["Cumulative_Pct_Lines"] = sku_fms_sorted["Pct_of_Total_Order_Lines"].cumsum()
        
        # Apply FMS classification
        sku_fms_sorted["FMS"] = sku_fms_sorted["Cumulative_Pct_Lines"].apply(
            lambda x: classify_fms(
                x, 
                FMS_THRESHOLDS['F_THRESHOLD'], 
                FMS_THRESHOLDS['M_THRESHOLD']
            )
        )
        
        self.logger.info("FMS classification completed")
        return sku_fms_sorted
    
    def classify_abc(self, sku_metrics: pd.DataFrame) -> pd.DataFrame:
        """
        Perform ABC classification based on case-equivalent volume.
        
        Args:
            sku_metrics: DataFrame with aggregated SKU metrics
            
        Returns:
            DataFrame with ABC classification
        """
        self.logger.info("Performing ABC classification")
        
        # Calculate total volume for percentage calculation
        total_volume_all = sku_metrics["Total_Case_Equiv"].sum()
        
        # Calculate percentage of total case equivalent
        sku_metrics["Pct_of_Total_Case_Equiv"] = safe_division(
            sku_metrics["Total_Case_Equiv"] * 100,
            total_volume_all,
            fill_value=0
        )
        
        # Sort SKUs by volume (descending) for cumulative calculation
        sku_abc_sorted = sku_metrics.sort_values(
            by="Total_Case_Equiv", 
            ascending=False
        ).reset_index(drop=True)
        
        # Calculate cumulative percentage of volume
        sku_abc_sorted["Cumulative_Pct_Volume"] = sku_abc_sorted["Pct_of_Total_Case_Equiv"].cumsum()
        
        # Apply ABC classification
        sku_abc_sorted["ABC"] = sku_abc_sorted["Cumulative_Pct_Volume"].apply(
            lambda x: classify_abc(
                x,
                ABC_THRESHOLDS['A_THRESHOLD'],
                ABC_THRESHOLDS['B_THRESHOLD']
            )
        )
        
        self.logger.info("ABC classification completed")
        return sku_abc_sorted
    
    def calculate_movement_frequency(self) -> pd.DataFrame:
        """
        Calculate frequency-of-movement metrics for SKUs.
        
        Returns:
            DataFrame with movement frequency metrics
        """
        self.logger.info("Calculating movement frequency metrics")
        
        # Calculate distinct movement days per SKU
        distinct_days = (
            self.enriched_data.dropna(subset=["Date"])
            .groupby("Sku Code", sort=False)
            .agg(Distinct_Movement_Days=("Date", "nunique"))
            .reset_index()
        )
        
        # Calculate total unique days in the dataset
        total_unique_days = self.enriched_data["Date"].nunique()
        total_unique_days = int(total_unique_days) if not np.isnan(total_unique_days) else 0
        
        # Calculate movement frequency percentage
        distinct_days["FMS_Period_Pct"] = safe_division(
            distinct_days["Distinct_Movement_Days"] * 100,
            total_unique_days,
            fill_value=0
        )
        
        # Calculate orders per movement day
        order_lines_per_sku = (
            self.enriched_data.groupby("Sku Code", sort=False)
            .agg(Total_Order_Lines=("Order No.", "count"))
            .reset_index()
        )
        
        # Merge movement days with order lines
        movement_metrics = distinct_days.merge(order_lines_per_sku, on="Sku Code", how="left")
        
        movement_metrics["Orders_per_Movement_Day"] = safe_division(
            movement_metrics["Total_Order_Lines"],
            movement_metrics["Distinct_Movement_Days"],
            fill_value=0
        )
        
        self.logger.info("Movement frequency calculation completed")
        return movement_metrics
    
    def create_sku_profile_abc_fms(self) -> pd.DataFrame:
        """
        Create comprehensive SKU profile with ABC-FMS classification.
        
        Returns:
            DataFrame with complete SKU profile
        """
        self.logger.info("Creating comprehensive SKU profile")
        
        # Step 1: Aggregate base metrics
        sku_metrics = self.aggregate_sku_metrics()
        
        # Step 2: Perform FMS classification
        sku_fms_sorted = self.classify_fms(sku_metrics.copy())
        
        # Step 3: Perform ABC classification
        sku_abc_sorted = self.classify_abc(sku_metrics.copy())
        
        # Step 4: Create base profile with metrics
        sku_profile = sku_metrics[
            ["Sku Code", "Total_Order_Lines", "Total_Case_Equiv"]
        ].copy()
        
        # Add percentage calculations
        total_lines = sku_metrics["Total_Order_Lines"].sum()
        total_volume = sku_metrics["Total_Case_Equiv"].sum()
        
        sku_profile["Pct_of_Total_Order_Lines"] = safe_division(
            sku_profile["Total_Order_Lines"] * 100,
            total_lines,
            fill_value=0
        )
        
        sku_profile["Pct_of_Total_Case_Equiv"] = safe_division(
            sku_profile["Total_Case_Equiv"] * 100,
            total_volume,
            fill_value=0
        )
        
        # Step 5: Merge FMS results
        sku_profile = sku_profile.merge(
            sku_fms_sorted[["Sku Code", "Cumulative_Pct_Lines", "FMS"]],
            on="Sku Code",
            how="left"
        )
        
        # Step 6: Merge ABC results
        sku_profile = sku_profile.merge(
            sku_abc_sorted[["Sku Code", "Cumulative_Pct_Volume", "ABC"]],
            on="Sku Code",
            how="left"
        )
        
        # Step 7: Add movement frequency metrics
        movement_metrics = self.calculate_movement_frequency()
        sku_profile = sku_profile.merge(
            movement_metrics[["Sku Code", "Distinct_Movement_Days", "FMS_Period_Pct", "Orders_per_Movement_Day"]],
            on="Sku Code",
            how="left"
        )
        
        # Step 8: Handle missing values
        sku_profile["Distinct_Movement_Days"] = sku_profile["Distinct_Movement_Days"].fillna(0).astype(int)
        sku_profile["FMS_Period_Pct"] = sku_profile["FMS_Period_Pct"].fillna(0)
        sku_profile["Orders_per_Movement_Day"] = sku_profile["Orders_per_Movement_Day"].fillna(0)
        
        # Step 9: Normalize ABC and FMS values
        sku_profile = normalize_abc_fms_values(sku_profile)
        
        # Step 10: Create 2D classification
        sku_profile = create_2d_classification(sku_profile)
        
        # Step 11: Format numeric columns
        numeric_columns = [
            "Pct_of_Total_Order_Lines", "Cumulative_Pct_Lines",
            "Pct_of_Total_Case_Equiv", "Cumulative_Pct_Volume",
            "FMS_Period_Pct", "Orders_per_Movement_Day"
        ]
        sku_profile = format_numeric_columns(sku_profile, numeric_columns, 2)
        
        # Step 12: Ensure integer columns are properly typed
        sku_profile["Total_Order_Lines"] = sku_profile["Total_Order_Lines"].astype(int)
        sku_profile["Distinct_Movement_Days"] = sku_profile["Distinct_Movement_Days"].astype(int)
        
        # Step 13: Handle infinite values
        sku_profile = handle_infinite_values(sku_profile)
        
        # Step 14: Arrange columns for readability
        column_order = [
            "Sku Code",
            "Total_Order_Lines",
            "Pct_of_Total_Order_Lines",
            "Cumulative_Pct_Lines",
            "FMS",
            "Total_Case_Equiv",
            "Pct_of_Total_Case_Equiv",
            "Cumulative_Pct_Volume",
            "ABC",
            "2D-Classification",
            "Distinct_Movement_Days",
            "FMS_Period_Pct",
            "Orders_per_Movement_Day"
        ]
        
        sku_profile = sku_profile[column_order]
        
        self.logger.info(f"SKU profile created for {len(sku_profile)} SKUs")
        return sku_profile
    
    def get_sku_statistics(self, sku_profile: pd.DataFrame) -> Dict:
        """
        Get statistics about the SKU profile.
        
        Args:
            sku_profile: DataFrame with complete SKU profile
            
        Returns:
            Dictionary with SKU statistics
        """
        stats = {
            'total_skus': len(sku_profile),
            'abc_distribution': sku_profile['ABC'].value_counts().to_dict(),
            'fms_distribution': sku_profile['FMS'].value_counts().to_dict(),
            'classification_2d_distribution': sku_profile['2D-Classification'].value_counts().to_dict(),
            'top_skus_by_volume': sku_profile.nlargest(10, 'Total_Case_Equiv')[['Sku Code', 'Total_Case_Equiv']].to_dict('records'),
            'top_skus_by_lines': sku_profile.nlargest(10, 'Total_Order_Lines')[['Sku Code', 'Total_Order_Lines']].to_dict('records')
        }
        
        return stats
    
    def run_full_analysis(self) -> Dict[str, pd.DataFrame]:
        """
        Run complete SKU analysis.
        
        Returns:
            Dictionary containing all analysis results
        """
        self.logger.info("Running full SKU analysis")
        
        # Generate SKU profile
        sku_profile = self.create_sku_profile_abc_fms()
        
        # Get statistics
        statistics = self.get_sku_statistics(sku_profile)
        
        results = {
            'sku_profile_abc_fms': sku_profile,
            'statistics': statistics
        }
        
        self.logger.info("SKU analysis completed successfully")
        return results


def analyze_skus(enriched_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to run SKU analysis.
    
    Args:
        enriched_data: DataFrame with enriched order data
        
    Returns:
        Dictionary containing analysis results
    """
    analyzer = SkuAnalyzer(enriched_data)
    return analyzer.run_full_analysis()