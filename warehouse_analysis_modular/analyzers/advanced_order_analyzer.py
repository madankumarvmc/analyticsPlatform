#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced Order Profile Analysis Module

Provides advanced multi-metric correlation analysis, enhanced percentile calculations,
and sophisticated order pattern analysis for warehouse optimization.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from scipy import stats

# Import from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from warehouse_analysis_modular.utils.helpers import (
    validate_dataframe, calculate_percentiles, setup_logging
)

logger = setup_logging()


class AdvancedOrderAnalyzer:
    """
    Advanced Order Profile Analyzer providing multi-metric correlation analysis
    and sophisticated operational insights.
    
    Features:
    - Multi-metric time series correlation (Lines, Customer, Volume, Shipment)  
    - Enhanced percentile analysis with capacity planning ratios
    - Peak-to-percentile calculations for operational planning
    - Multi-truck order detection and complexity analysis
    - Design capacity recommendations based on statistical analysis
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def analyze_advanced_order_profile(self, enriched_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive advanced order profile analysis.
        
        Args:
            enriched_data: DataFrame with enriched order data
            
        Returns:
            Dictionary containing all advanced analysis results
        """
        self.logger.info("Starting advanced order profile analysis")
        
        # Validate input data
        if not validate_dataframe(enriched_data, required_columns=[
            'Date', 'Order No.', 'Shipment No.', 'Sku Code', 'Case_Equivalent'
        ]):
            raise ValueError("Invalid input data for advanced order analysis")
        
        results = {}
        
        # 1. Multi-metric daily aggregation
        daily_metrics = self._calculate_daily_multi_metrics(enriched_data)
        results['daily_multi_metrics'] = daily_metrics
        
        # 2. Multi-metric correlation analysis
        correlation_analysis = self._analyze_metric_correlations(daily_metrics)
        results['correlation_analysis'] = correlation_analysis
        
        # 3. Enhanced percentile analysis with ratios
        percentile_analysis = self._calculate_enhanced_percentiles(daily_metrics)
        results['enhanced_percentile_analysis'] = percentile_analysis
        
        # 4. Peak-to-percentile ratio calculations
        peak_ratios = self._calculate_peak_ratios(daily_metrics)
        results['peak_ratios'] = peak_ratios
        
        # 5. Multi-truck order analysis
        multi_truck_analysis = self._analyze_multi_truck_orders(enriched_data)
        results['multi_truck_analysis'] = multi_truck_analysis
        
        # 6. Capacity planning recommendations
        capacity_recommendations = self._generate_capacity_recommendations(
            daily_metrics, percentile_analysis, peak_ratios
        )
        results['capacity_recommendations'] = capacity_recommendations
        
        # 7. Operational complexity scoring
        complexity_score = self._calculate_operational_complexity(
            daily_metrics, correlation_analysis, multi_truck_analysis
        )
        results['operational_complexity'] = complexity_score
        
        self.logger.info("Advanced order profile analysis completed")
        return results
    
    def _calculate_daily_multi_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily metrics for multi-metric analysis."""
        self.logger.info("Calculating daily multi-metrics")
        
        # Ensure Date column is datetime
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Daily aggregations
        daily_metrics = data.groupby('Date').agg({
            'Order No.': 'nunique',           # Unique Orders
            'Shipment No.': 'nunique',        # Unique Shipments  
            'Sku Code': 'nunique',            # Unique SKUs
            'Case_Equivalent': 'sum',         # Total Case Equivalent
            'Qty in Cases': 'sum',            # Total Cases
            'Qty in Eaches': 'sum'            # Total Eaches
        }).reset_index()
        
        # Rename columns for clarity
        daily_metrics.columns = [
            'Date', 'Unique_Orders', 'Unique_Shipments', 'Unique_SKUs', 
            'Total_Case_Equiv', 'Total_Cases', 'Total_Eaches'
        ]
        
        # Calculate additional metrics
        daily_metrics['Total_Lines'] = data.groupby('Date').size().values
        
        # Handle customer column with fallbacks
        if 'Customer' in data.columns:
            daily_metrics['Distinct_Customers'] = data.groupby('Date')['Customer'].nunique().values
        elif 'Order No.' in data.columns:
            # Use Order No. as proxy for customers if Customer column not available
            daily_metrics['Distinct_Customers'] = data.groupby('Date')['Order No.'].nunique().values
        else:
            daily_metrics['Distinct_Customers'] = daily_metrics['Unique_Orders']  # Fallback to orders
        
        # Calculate derived metrics
        daily_metrics['Lines_per_Order'] = daily_metrics['Total_Lines'] / daily_metrics['Unique_Orders']
        daily_metrics['SKUs_per_Order'] = daily_metrics['Unique_SKUs'] / daily_metrics['Unique_Orders'] 
        daily_metrics['Cases_per_Line'] = daily_metrics['Total_Cases'] / daily_metrics['Total_Lines']
        daily_metrics['Eaches_per_Line'] = daily_metrics['Total_Eaches'] / daily_metrics['Total_Lines']
        
        return daily_metrics
    
    def _analyze_metric_correlations(self, daily_metrics: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between different operational metrics."""
        self.logger.info("Analyzing metric correlations")
        
        # Key metrics for correlation analysis
        key_metrics = [
            'Total_Lines', 'Distinct_Customers', 'Unique_Shipments', 'Total_Case_Equiv'
        ]
        
        # Calculate correlation matrix
        correlation_matrix = daily_metrics[key_metrics].corr()
        
        # Calculate specific correlations of interest
        correlations = {}
        correlations['volume_lines'] = daily_metrics['Total_Case_Equiv'].corr(daily_metrics['Total_Lines'])
        correlations['volume_customers'] = daily_metrics['Total_Case_Equiv'].corr(daily_metrics['Distinct_Customers'])
        correlations['lines_customers'] = daily_metrics['Total_Lines'].corr(daily_metrics['Distinct_Customers'])
        correlations['shipments_orders'] = daily_metrics['Unique_Shipments'].corr(daily_metrics['Unique_Orders'])
        
        # Statistical significance testing with error handling
        correlations_with_p_values = {}
        correlation_pairs = {
            'volume_lines': ('Total_Case_Equiv', 'Total_Lines'),
            'volume_customers': ('Total_Case_Equiv', 'Distinct_Customers'),
            'lines_customers': ('Total_Lines', 'Distinct_Customers'),
            'shipments_orders': ('Unique_Shipments', 'Unique_Orders')
        }
        
        for key, (col1, col2) in correlation_pairs.items():
            try:
                # Check if both columns have variance (not constant)
                if (daily_metrics[col1].std() == 0 or daily_metrics[col2].std() == 0 or
                    len(daily_metrics[col1].unique()) == 1 or len(daily_metrics[col2].unique()) == 1):
                    # Handle constant data case
                    correlations_with_p_values[key] = {
                        'correlation': 0.0,
                        'p_value': 1.0,
                        'significance': 'not_significant',
                        'note': 'constant_data'
                    }
                else:
                    corr, p_value = stats.pearsonr(daily_metrics[col1], daily_metrics[col2])
                    
                    # Handle NaN results
                    if pd.isna(corr) or pd.isna(p_value):
                        correlations_with_p_values[key] = {
                            'correlation': 0.0,
                            'p_value': 1.0,
                            'significance': 'not_significant',
                            'note': 'calculation_error'
                        }
                    else:
                        correlations_with_p_values[key] = {
                            'correlation': float(corr),
                            'p_value': float(p_value),
                            'significance': 'significant' if p_value < 0.05 else 'not_significant'
                        }
            except Exception as e:
                # Fallback for any other errors
                self.logger.warning(f"Correlation calculation failed for {key}: {e}")
                correlations_with_p_values[key] = {
                    'correlation': 0.0,
                    'p_value': 1.0,
                    'significance': 'not_significant',
                    'note': 'error'
                }
        
        return {
            'correlation_matrix': correlation_matrix,
            'key_correlations': correlations,
            'statistical_analysis': correlations_with_p_values
        }
    
    def _calculate_enhanced_percentiles(self, daily_metrics: pd.DataFrame) -> Dict[str, Any]:
        """Calculate enhanced percentiles for capacity planning."""
        self.logger.info("Calculating enhanced percentiles")
        
        enhanced_percentiles = {}
        
        # Key metrics for percentile analysis
        metrics = ['Total_Case_Equiv', 'Total_Lines', 'Distinct_Customers', 'Unique_Shipments']
        
        # Extended percentile levels including the ones from screenshots
        percentile_levels = [25, 50, 75, 85, 90, 95, 100]  # 100 = Max
        
        for metric in metrics:
            if metric in daily_metrics.columns:
                percentiles = {}
                for p in percentile_levels:
                    if p == 100:
                        percentiles['Max'] = daily_metrics[metric].max()
                    else:
                        percentiles[f'{p}%ile'] = daily_metrics[metric].quantile(p/100)
                
                # Calculate additional statistical measures
                percentiles['Avg'] = daily_metrics[metric].mean()
                percentiles['Std'] = daily_metrics[metric].std()
                percentiles['CV'] = percentiles['Std'] / percentiles['Avg'] if percentiles['Avg'] > 0 else 0
                
                enhanced_percentiles[metric] = percentiles
        
        return enhanced_percentiles
    
    def _calculate_peak_ratios(self, daily_metrics: pd.DataFrame) -> Dict[str, Any]:
        """Calculate peak-to-percentile ratios for capacity planning."""
        self.logger.info("Calculating peak ratios")
        
        ratios = {}
        metrics = ['Total_Case_Equiv', 'Total_Lines', 'Distinct_Customers']
        
        for metric in metrics:
            if metric in daily_metrics.columns:
                max_val = daily_metrics[metric].max()
                avg_val = daily_metrics[metric].mean()
                p95_val = daily_metrics[metric].quantile(0.95)
                p90_val = daily_metrics[metric].quantile(0.90)
                p50_val = daily_metrics[metric].quantile(0.50)
                
                ratios[metric] = {
                    'peak_to_avg': max_val / avg_val if avg_val > 0 else 0,
                    'peak_to_95th': max_val / p95_val if p95_val > 0 else 0,
                    'peak_to_90th': max_val / p90_val if p90_val > 0 else 0,
                    'p95_to_p50': p95_val / p50_val if p50_val > 0 else 0,
                    'p95_to_avg': p95_val / avg_val if avg_val > 0 else 0
                }
        
        return ratios
    
    def _analyze_multi_truck_orders(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze orders that span multiple trucks/shipments."""
        self.logger.info("Analyzing multi-truck orders")
        
        # Group by Order No. and Date to find orders with multiple shipments
        # Use the available data columns
        agg_dict = {
            'Shipment No.': 'nunique',
            'Case_Equivalent': 'sum'
        }
        
        order_shipment_analysis = data.groupby(['Date', 'Order No.']).agg(agg_dict).reset_index()
        
        # Add line count using size()
        line_counts = data.groupby(['Date', 'Order No.']).size().reset_index(name='Total_Lines')
        order_shipment_analysis = order_shipment_analysis.merge(line_counts, on=['Date', 'Order No.'], how='left')
        
        # Identify multi-truck orders (orders with multiple shipments)
        multi_truck_orders = order_shipment_analysis[
            order_shipment_analysis['Shipment No.'] > 1
        ]
        
        total_orders = len(order_shipment_analysis)
        multi_truck_count = len(multi_truck_orders)
        
        analysis = {
            'total_orders_analyzed': total_orders,
            'multi_truck_orders_count': multi_truck_count,
            'multi_truck_percentage': (multi_truck_count / total_orders * 100) if total_orders > 0 else 0,
            'avg_shipments_per_multi_truck_order': multi_truck_orders['Shipment No.'].mean() if multi_truck_count > 0 else 0,
            'max_shipments_per_order': order_shipment_analysis['Shipment No.'].max(),
            'multi_truck_volume_impact': multi_truck_orders['Case_Equivalent'].sum() / order_shipment_analysis['Case_Equivalent'].sum() * 100 if order_shipment_analysis['Case_Equivalent'].sum() > 0 else 0
        }
        
        return analysis
    
    def _generate_capacity_recommendations(self, daily_metrics: pd.DataFrame, 
                                         percentiles: Dict, peak_ratios: Dict) -> Dict[str, Any]:
        """Generate capacity planning recommendations based on analysis."""
        self.logger.info("Generating capacity recommendations")
        
        recommendations = {}
        
        # Volume-based capacity recommendations
        if 'Total_Case_Equiv' in percentiles and 'Total_Case_Equiv' in peak_ratios:
            volume_p95 = percentiles['Total_Case_Equiv']['95%ile']
            volume_max = percentiles['Total_Case_Equiv']['Max']
            volume_avg = percentiles['Total_Case_Equiv']['Avg']
            
            # Calculate buffer requirements
            peak_buffer = ((volume_max - volume_p95) / volume_p95 * 100) if volume_p95 > 0 else 0
            
            recommendations['volume_capacity'] = {
                'design_capacity_base': volume_p95,
                'recommended_buffer_percentage': max(15, peak_buffer),  # Minimum 15% buffer
                'total_design_capacity': volume_p95 * (1 + max(0.15, peak_buffer/100)),
                'utilization_at_avg': (volume_avg / volume_p95 * 100) if volume_p95 > 0 else 0
            }
        
        # Lines-based capacity recommendations  
        if 'Total_Lines' in percentiles:
            lines_p95 = percentiles['Total_Lines']['95%ile']
            lines_max = percentiles['Total_Lines']['Max']
            
            recommendations['lines_capacity'] = {
                'design_lines_capacity': lines_p95 * 1.15,  # 15% buffer
                'peak_lines_ratio': peak_ratios.get('Total_Lines', {}).get('peak_to_95th', 0)
            }
        
        # Operational recommendations
        recommendations['operational_insights'] = {
            'capacity_planning_principle': 'Design for 95th percentile with 15% buffer',
            'peak_management_strategy': 'Plan overflow capacity for peak-to-95th ratio > 1.2x',
            'monitoring_focus': 'Monitor 95th percentile trending for capacity adjustments'
        }
        
        return recommendations
    
    def _calculate_operational_complexity(self, daily_metrics: pd.DataFrame,
                                        correlations: Dict, multi_truck: Dict) -> Dict[str, Any]:
        """Calculate overall operational complexity score."""
        self.logger.info("Calculating operational complexity")
        
        complexity_factors = {}
        
        # Volume variability factor
        cv_volume = daily_metrics['Total_Case_Equiv'].std() / daily_metrics['Total_Case_Equiv'].mean()
        complexity_factors['volume_variability'] = min(cv_volume * 100, 100)  # Cap at 100
        
        # Multi-truck complexity
        complexity_factors['multi_truck_complexity'] = multi_truck.get('multi_truck_percentage', 0)
        
        # SKU complexity (daily SKU variation)
        cv_skus = daily_metrics['Unique_SKUs'].std() / daily_metrics['Unique_SKUs'].mean()
        complexity_factors['sku_complexity'] = min(cv_skus * 100, 50)  # Cap at 50
        
        # Correlation complexity (lower correlation = higher complexity)
        avg_correlation = np.mean([
            abs(correlations['key_correlations'].get('volume_lines', 0)),
            abs(correlations['key_correlations'].get('volume_customers', 0)),
            abs(correlations['key_correlations'].get('lines_customers', 0))
        ])
        complexity_factors['correlation_complexity'] = (1 - avg_correlation) * 100
        
        # Overall complexity score (weighted average)
        weights = {
            'volume_variability': 0.3,
            'multi_truck_complexity': 0.2,
            'sku_complexity': 0.2,
            'correlation_complexity': 0.3
        }
        
        overall_score = sum(
            complexity_factors[factor] * weights[factor] 
            for factor in complexity_factors
        )
        
        # Classify complexity level
        if overall_score < 25:
            complexity_level = 'Low'
        elif overall_score < 50:
            complexity_level = 'Medium'
        elif overall_score < 75:
            complexity_level = 'High'
        else:
            complexity_level = 'Very High'
        
        return {
            'complexity_factors': complexity_factors,
            'overall_complexity_score': overall_score,
            'complexity_level': complexity_level,
            'interpretation': self._interpret_complexity_score(overall_score)
        }
    
    def _interpret_complexity_score(self, score: float) -> str:
        """Provide interpretation of complexity score."""
        if score < 25:
            return "Operations show consistent patterns with low variability. Standard capacity planning approaches sufficient."
        elif score < 50:
            return "Moderate operational complexity. Consider flexible staffing and capacity buffers."
        elif score < 75:
            return "High operational complexity. Requires advanced planning, flexible resources, and demand management."
        else:
            return "Very high operational complexity. Critical need for sophisticated planning tools and dynamic resource allocation."