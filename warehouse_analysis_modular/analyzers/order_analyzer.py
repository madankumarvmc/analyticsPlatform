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
from typing import Dict, List, Optional, Any

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
    Handles comprehensive order-level analysis for warehouse operations.
    
    This analyzer processes enriched order data to provide insights into daily operations,
    SKU performance, volume patterns, and demand characteristics. It generates multiple
    analytical outputs including date-wise summaries, SKU-level aggregations, percentile
    calculations, and demand pattern analysis.
    
    Key Capabilities:
    - Daily order summaries with volume, customer, and SKU metrics
    - SKU-level performance analysis
    - Percentile calculations for volume distribution
    - Demand pattern identification (peaks, seasonality, day-of-week patterns)
    - Comprehensive statistical summaries
    
    Input Requirements:
    - Enriched order data with calculated fields (Case_Equivalent, Pallet_Equivalent)
    - Required columns: Date, Order No., Shipment No., Sku Code, quantities
    
    Output Applications:
    - Operations planning and capacity management
    - Performance monitoring and KPI tracking
    - Demand forecasting and pattern analysis
    - Resource allocation optimization
    
    Example:
        # Initialize with enriched order data
        analyzer = OrderAnalyzer(enriched_order_data)
        
        # Run complete analysis
        results = analyzer.run_full_analysis()
        
        # Access specific results
        daily_summary = results['date_order_summary']
        sku_performance = results['sku_order_summary']
        demand_insights = results['demand_patterns']
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
    
    def calculate_fte_required(self, df: pd.DataFrame, 
                              touch_time_per_unit: float = 0.17,
                              avg_walk_distance_per_pallet: float = 30,
                              walk_speed: float = 60,
                              shift_hours: float = 8,
                              efficiency: float = 0.8) -> pd.DataFrame:
        """
        Calculate FTE (Full-Time Equivalent) requirements for each day.
        
        This method adds FTE calculation columns to the date_order_summary dataframe
        based on operational parameters for touch time, walking time, and efficiency.
        
        Args:
            df: Date order summary DataFrame
            touch_time_per_unit: Time in minutes to pick one unit (default 0.17 = 10 seconds)
            avg_walk_distance_per_pallet: Average walking distance in meters per pallet (default 30m)
            walk_speed: Walking speed in meters per minute (default 60 m/min = 3.6 km/h)
            shift_hours: Working hours per shift (default 8 hours)
            efficiency: Worker efficiency factor (default 0.8 = 80% effective time)
            
        Returns:
            DataFrame with added FTE calculation columns:
            - Total_Touches: Total picking touches (cases + eaches)
            - Touch_Time_Min: Total touch time in minutes
            - Walk_Time_Min: Total walking time in minutes  
            - Total_Time_Min: Total time required in minutes
            - FTE_Required: Number of full-time workers needed (rounded)
        
        Formula:
            - Total Touches = Qty_Ordered_Cases + Qty_Ordered_Eaches
            - Touch Time = Total Touches × touch_time_per_unit
            - Walk Time = (Total_Pallet_Equiv × avg_walk_distance_per_pallet) ÷ walk_speed
            - Total Time = Touch Time + Walk Time
            - FTE Required = Total Time ÷ (shift_hours × 60 × efficiency)
        """
        self.logger.info("Calculating FTE requirements")
        
        df = df.copy()
        
        # Step 1: Calculate total touches (cases + eaches both count as one pick action)
        df["Total_Touches"] = df["Qty_Ordered_Cases"].fillna(0) + df["Qty_Ordered_Eaches"].fillna(0)
        
        # Step 2: Calculate touch time in minutes
        df["Touch_Time_Min"] = df["Total_Touches"] * touch_time_per_unit
        
        # Step 3: Calculate walking time in minutes
        df["Walk_Time_Min"] = (
            df["Total_Pallet_Equiv"].fillna(0) * avg_walk_distance_per_pallet / walk_speed
        )
        
        # Step 4: Calculate total time required
        df["Total_Time_Min"] = df["Touch_Time_Min"] + df["Walk_Time_Min"]
        
        # Step 5: Calculate FTE required (rounded to nearest whole worker)
        df["FTE_Required"] = (
            df["Total_Time_Min"] / (shift_hours * 60 * efficiency)
        ).round(0).astype(int)
        
        self.logger.info(f"FTE calculation completed for {len(df)} days")
        return df
    
    def run_full_analysis(self, fte_parameters: Dict[str, Any] = None) -> Dict[str, pd.DataFrame]:
        """
        Run complete order analysis pipeline.
        
        Executes all order analysis steps including date summaries, SKU summaries,
        percentile calculations, statistical analysis, and demand pattern identification.
        Optionally includes FTE (Full-Time Equivalent) workforce calculations.
        
        Args:
            fte_parameters: Optional dictionary with FTE calculation parameters:
                - enable_fte_calculation: bool (default False)
                - touch_time_per_unit: float (minutes per unit)
                - avg_walk_distance_per_pallet: float (meters per pallet)
                - walk_speed: float (meters per minute)
                - shift_hours: float (hours per shift)
                - efficiency: float (worker efficiency factor)
        
        Returns:
            Dictionary containing all analysis results:
            {
                'date_order_summary': pd.DataFrame,     # Daily aggregations (with optional FTE)
                'sku_order_summary': pd.DataFrame,      # SKU-level summaries
                'percentile_profile': pd.DataFrame,     # Percentile calculations
                'statistics': dict,                     # Summary statistics
                'demand_patterns': dict                 # Demand pattern insights
            }
            
        Example:
            analyzer = OrderAnalyzer(enriched_data)
            
            # Basic analysis
            results = analyzer.run_full_analysis()
            
            # Analysis with FTE calculation
            fte_params = {
                'enable_fte_calculation': True,
                'touch_time_per_unit': 0.15,
                'shift_hours': 8,
                'efficiency': 0.85
            }
            results_with_fte = analyzer.run_full_analysis(fte_params)
            daily_summary = results_with_fte['date_order_summary']  # Includes FTE columns
        """
        self.logger.info("Running full order analysis")
        
        # Generate all summaries
        date_summary = self.generate_date_order_summary()
        
        # Apply FTE calculation if requested
        if fte_parameters and fte_parameters.get('enable_fte_calculation', False):
            self.logger.info("Applying FTE calculation to date summary")
            date_summary = self.calculate_fte_required(
                date_summary,
                touch_time_per_unit=fte_parameters.get('touch_time_per_unit', 0.17),
                avg_walk_distance_per_pallet=fte_parameters.get('avg_walk_distance_per_pallet', 30),
                walk_speed=fte_parameters.get('walk_speed', 60),
                shift_hours=fte_parameters.get('shift_hours', 8),
                efficiency=fte_parameters.get('efficiency', 0.8)
            )
        
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