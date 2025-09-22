#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Order Analysis Module

Handles date-wise order analysis, percentile calculations, and summary statistics.
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

from config import AGGREGATION_METRICS, PERCENTILE_LEVELS, PERCENTILE_LABELS
from warehouse_analysis_modular.utils.helpers import (
    validate_dataframe, calculate_percentiles, setup_logging
)

logger = setup_logging()


class OrderAnalyzer:
    """
    Handles order-level analysis including date summaries and percentile calculations.
    """
    
    def __init__(self, enriched_data: pd.DataFrame):
        """
        Initialize the OrderAnalyzer with enriched order data.
        
        Args:
            enriched_data: DataFrame with enriched order data including calculated fields
        """
        self.enriched_data = enriched_data
        self.logger = logging.getLogger(self.__class__.__name__)
        self._validate_input_data()
    
    def _validate_input_data(self):
        """Validate that the input data has required columns."""
        required_columns = [
            'Date', 'Order No.', 'Shipment No.', 'Sku Code',
            'Qty in Cases', 'Qty in Eaches', 'Case_Equivalent'
        ]
        validate_dataframe(
            self.enriched_data, 
            required_columns, 
            min_rows=1, 
            name="Enriched order data"
        )
    
    def generate_date_order_summary(self) -> pd.DataFrame:
        """
        Generate date-wise order summary.
        
        Returns:
            DataFrame with date-wise aggregated metrics
        """
        self.logger.info("Generating date-wise order summary")
        
        # Perform date-wise aggregation
        date_summary = self.enriched_data.groupby("Date").agg(
            Distinct_Customers=("Order No.", "nunique"),   # Using Order No. as proxy for customers
            Distinct_Shipments=("Shipment No.", "nunique"),
            Distinct_Orders=("Order No.", "nunique"),
            Distinct_SKUs=("Sku Code", "nunique"),
            Qty_Ordered_Cases=("Qty in Cases", "sum"),
            Qty_Ordered_Eaches=("Qty in Eaches", "sum"),
            Total_Case_Equiv=("Case_Equivalent", "sum"),
            Total_Pallet_Equiv=("Pallet_Equivalent", "sum")
        ).reset_index()
        
        # Fill NaN values with 0 for numeric columns
        numeric_columns = [
            'Qty_Ordered_Cases', 'Qty_Ordered_Eaches', 
            'Total_Case_Equiv', 'Total_Pallet_Equiv'
        ]
        for col in numeric_columns:
            if col in date_summary.columns:
                date_summary[col] = date_summary[col].fillna(0)
        
        self.logger.info(f"Generated summary for {len(date_summary)} dates")
        return date_summary
    
    def generate_sku_order_summary(self) -> pd.DataFrame:
        """
        Generate SKU-wise order summary.
        
        Returns:
            DataFrame with SKU-wise aggregated metrics
        """
        self.logger.info("Generating SKU-wise order summary")
        
        sku_summary = self.enriched_data.groupby("Sku Code").agg(
            Order_Lines=("Order No.", "count"),
            Order_Volume_CE=("Case_Equivalent", "sum")
        ).reset_index()
        
        # Fill NaN values
        sku_summary['Order_Volume_CE'] = sku_summary['Order_Volume_CE'].fillna(0)
        
        self.logger.info(f"Generated summary for {len(sku_summary)} SKUs")
        return sku_summary
    
    def generate_percentile_profile(self, date_summary: pd.DataFrame) -> pd.DataFrame:
        """
        Generate percentile profile from date summary.
        
        Args:
            date_summary: Date-wise summary DataFrame
            
        Returns:
            DataFrame with percentile analysis
        """
        self.logger.info("Generating percentile profile")
        
        # Initialize percentile profile with labels
        percentile_profile = pd.DataFrame({
            "Percentile": PERCENTILE_LABELS
        })
        
        # Calculate percentiles for each metric
        for col in AGGREGATION_METRICS:
            if col in date_summary.columns:
                percentile_values = [
                    date_summary[col].max(),                      # Max
                    np.percentile(date_summary[col], 95),         # 95th Percentile
                    np.percentile(date_summary[col], 90),         # 90th Percentile
                    np.percentile(date_summary[col], 85),         # 85th Percentile
                    date_summary[col].mean()                      # Average
                ]
                percentile_profile[col] = percentile_values
            else:
                self.logger.warning(f"Column {col} not found in date summary")
                percentile_profile[col] = [0] * len(PERCENTILE_LABELS)
        
        self.logger.info("Percentile profile generated successfully")
        return percentile_profile
    
    def get_order_statistics(self) -> Dict:
        """
        Get basic statistics about the order data.
        
        Returns:
            Dictionary with key statistics
        """
        stats = {
            'total_order_lines': len(self.enriched_data),
            'unique_dates': self.enriched_data['Date'].nunique(),
            'unique_skus': self.enriched_data['Sku Code'].nunique(),
            'unique_orders': self.enriched_data['Order No.'].nunique(),
            'unique_shipments': self.enriched_data['Shipment No.'].nunique(),
            'date_range': {
                'start': self.enriched_data['Date'].min(),
                'end': self.enriched_data['Date'].max()
            },
            'total_case_equivalent': self.enriched_data['Case_Equivalent'].sum(),
            'total_pallet_equivalent': self.enriched_data['Pallet_Equivalent'].sum()
        }
        
        return stats
    
    def analyze_demand_patterns(self, date_summary: pd.DataFrame) -> Dict:
        """
        Analyze demand patterns from date summary.
        
        Args:
            date_summary: Date-wise summary DataFrame
            
        Returns:
            Dictionary with demand pattern insights
        """
        self.logger.info("Analyzing demand patterns")
        
        # Calculate day-of-week patterns if possible
        date_summary['DayOfWeek'] = pd.to_datetime(date_summary['Date']).dt.day_name()
        
        patterns = {
            'peak_date': date_summary.loc[date_summary['Total_Case_Equiv'].idxmax(), 'Date'],
            'peak_volume': date_summary['Total_Case_Equiv'].max(),
            'average_daily_volume': date_summary['Total_Case_Equiv'].mean(),
            'volume_std': date_summary['Total_Case_Equiv'].std(),
            'volume_cv': date_summary['Total_Case_Equiv'].std() / date_summary['Total_Case_Equiv'].mean(),
            'day_of_week_patterns': date_summary.groupby('DayOfWeek')['Total_Case_Equiv'].mean().to_dict()
        }
        
        return patterns
    
    def run_full_analysis(self) -> Dict[str, pd.DataFrame]:
        """
        Run complete order analysis.
        
        Returns:
            Dictionary containing all analysis results
        """
        self.logger.info("Running full order analysis")
        
        # Generate all summaries
        date_summary = self.generate_date_order_summary()
        sku_summary = self.generate_sku_order_summary()
        percentile_profile = self.generate_percentile_profile(date_summary)
        
        # Get additional insights
        statistics = self.get_order_statistics()
        demand_patterns = self.analyze_demand_patterns(date_summary)
        
        results = {
            'date_order_summary': date_summary,
            'sku_order_summary': sku_summary,
            'percentile_profile': percentile_profile,
            'statistics': statistics,
            'demand_patterns': demand_patterns
        }
        
        self.logger.info("Order analysis completed successfully")
        return results


def analyze_orders(enriched_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to run order analysis.
    
    Args:
        enriched_data: DataFrame with enriched order data
        
    Returns:
        Dictionary containing analysis results
    """
    analyzer = OrderAnalyzer(enriched_data)
    return analyzer.run_full_analysis()