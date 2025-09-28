#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Manpower & FTE Analysis Module

Analyzes workforce requirements, staffing patterns, and provides FTE optimization
recommendations based on warehouse order data and operational metrics.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

# Import from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from warehouse_analysis_modular.utils.helpers import (
    validate_dataframe, setup_logging
)

logger = setup_logging()


class ManpowerAnalyzer:
    """
    Manpower & FTE Analysis providing workforce planning insights and
    staffing optimization recommendations.
    
    Features:
    - FTE requirements calculation based on volume patterns
    - Peak vs baseline staffing analysis
    - Category-wise labor allocation recommendations
    - Productivity metrics and benchmarking
    - Shift planning and overtime optimization
    - Cost analysis and budget planning
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Industry standard productivity benchmarks (cases per FTE per day)
        self.productivity_benchmarks = {
            'cases_per_fte_day': 150,  # Standard warehouse productivity
            'lines_per_fte_day': 80,   # Order lines per FTE per day
            'hours_per_shift': 8,      # Standard shift hours
            'shifts_per_day': 2,       # Standard warehouse operation
            'efficiency_factor': 0.85,  # Overall operational efficiency
            'peak_multiplier': 1.4     # Peak period staffing multiplier
        }
        
        # Category-specific labor coefficients (relative complexity)
        self.category_labor_coefficients = {
            'default': 1.0,
            'A': 0.8,  # High-volume, efficient handling
            'B': 1.0,  # Standard handling
            'C': 1.3   # Low-volume, more complex handling
        }
        
    def analyze_manpower_requirements(self, enriched_data: pd.DataFrame,
                                    date_summary: pd.DataFrame = None,
                                    sku_profile: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Perform comprehensive manpower and FTE analysis.
        
        Args:
            enriched_data: DataFrame with enriched order data
            date_summary: Optional date-wise summary data
            sku_profile: Optional SKU profile with ABC-FMS classification
            
        Returns:
            Dictionary containing comprehensive manpower analysis results
        """
        self.logger.info("Starting comprehensive manpower & FTE analysis")
        
        # Define required columns for manpower analysis (using actual column names from enriched data)
        required_columns = ['Date', 'Sku Code', 'Case_Equivalent', 'Total_Eaches']
        if not validate_dataframe(enriched_data, required_columns, min_rows=1, name='enriched_data'):
            raise ValueError("Invalid enriched_data provided for manpower analysis")
        
        results = {}
        
        try:
            # 1. Calculate base FTE requirements
            results['base_fte_analysis'] = self._calculate_base_fte_requirements(enriched_data)
            
            # 2. Analyze daily staffing patterns
            results['daily_staffing_analysis'] = self._analyze_daily_staffing_patterns(enriched_data)
            
            # 3. Peak vs baseline analysis
            results['peak_analysis'] = self._analyze_peak_staffing_requirements(enriched_data)
            
            # 4. Category-wise labor allocation
            if 'Category' in enriched_data.columns:
                results['category_labor_analysis'] = self._analyze_category_labor_requirements(enriched_data)
            
            # 5. Productivity analysis
            results['productivity_analysis'] = self._analyze_productivity_metrics(enriched_data)
            
            # 6. Shift planning recommendations
            results['shift_planning'] = self._calculate_shift_planning_recommendations(enriched_data)
            
            # 7. Cost analysis
            results['cost_analysis'] = self._calculate_labor_cost_analysis(results)
            
            # 8. Summary metrics for reporting
            results['summary_metrics'] = self._generate_summary_metrics(results)
            
            self.logger.info("Manpower analysis completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in manpower analysis: {str(e)}")
            raise
        
        return results
    
    def _calculate_base_fte_requirements(self, enriched_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate base FTE requirements based on volume and productivity benchmarks."""
        self.logger.info("Calculating base FTE requirements")
        
        # Daily volume aggregation
        daily_data = enriched_data.groupby('Date').agg({
            'Total_Eaches': 'sum',
            'Case_Equivalent': 'sum',
            'Order No.': 'count',
            'Shipment No.': 'nunique'
        }).reset_index()
        
        daily_data.columns = ['Date', 'Total_Eaches', 'Total_Cases', 'Total_Lines', 'Total_Shipments']
        
        # Calculate FTE requirements
        cases_benchmark = self.productivity_benchmarks['cases_per_fte_day']
        lines_benchmark = self.productivity_benchmarks['lines_per_fte_day']
        efficiency = self.productivity_benchmarks['efficiency_factor']
        
        daily_data['FTE_Required_Cases'] = np.ceil(
            daily_data['Total_Cases'] / (cases_benchmark * efficiency)
        )
        daily_data['FTE_Required_Lines'] = np.ceil(
            daily_data['Total_Lines'] / (lines_benchmark * efficiency)
        )
        
        # Use the higher requirement
        daily_data['FTE_Required_Total'] = np.maximum(
            daily_data['FTE_Required_Cases'], 
            daily_data['FTE_Required_Lines']
        )
        
        # Add utilization metrics
        daily_data['Utilization_Cases'] = (
            daily_data['Total_Cases'] / (daily_data['FTE_Required_Total'] * cases_benchmark * efficiency)
        ).round(3)
        
        daily_data['Utilization_Lines'] = (
            daily_data['Total_Lines'] / (daily_data['FTE_Required_Total'] * lines_benchmark * efficiency)  
        ).round(3)
        
        return daily_data
    
    def _analyze_daily_staffing_patterns(self, enriched_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze daily staffing patterns and variations."""
        self.logger.info("Analyzing daily staffing patterns")
        
        # Calculate daily metrics
        daily_fte = self._calculate_base_fte_requirements(enriched_data)
        
        patterns = {
            'avg_fte_required': daily_fte['FTE_Required_Total'].mean(),
            'min_fte_required': daily_fte['FTE_Required_Total'].min(),
            'max_fte_required': daily_fte['FTE_Required_Total'].max(),
            'std_fte_required': daily_fte['FTE_Required_Total'].std(),
            'percentile_50': daily_fte['FTE_Required_Total'].quantile(0.5),
            'percentile_75': daily_fte['FTE_Required_Total'].quantile(0.75),
            'percentile_85': daily_fte['FTE_Required_Total'].quantile(0.85),
            'percentile_95': daily_fte['FTE_Required_Total'].quantile(0.95),
        }
        
        # Add day-of-week analysis if date spans multiple weeks
        daily_fte['Date'] = pd.to_datetime(daily_fte['Date'])
        daily_fte['DayOfWeek'] = daily_fte['Date'].dt.day_name()
        
        dow_analysis = daily_fte.groupby('DayOfWeek')['FTE_Required_Total'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).round(2)
        
        patterns['day_of_week_analysis'] = dow_analysis
        patterns['daily_fte_data'] = daily_fte
        
        return patterns
    
    def _analyze_peak_staffing_requirements(self, enriched_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze peak vs baseline staffing requirements."""
        self.logger.info("Analyzing peak staffing requirements")
        
        daily_fte = self._calculate_base_fte_requirements(enriched_data)
        
        # Define peak threshold (85th percentile)
        peak_threshold = daily_fte['FTE_Required_Total'].quantile(0.85)
        baseline_threshold = daily_fte['FTE_Required_Total'].quantile(0.50)
        
        peak_days = daily_fte[daily_fte['FTE_Required_Total'] >= peak_threshold]
        baseline_days = daily_fte[daily_fte['FTE_Required_Total'] <= baseline_threshold]
        
        peak_analysis = {
            'baseline_fte': baseline_threshold,
            'peak_threshold_fte': peak_threshold,
            'peak_multiplier': peak_threshold / baseline_threshold if baseline_threshold > 0 else 1,
            'peak_days_count': len(peak_days),
            'peak_days_percentage': round(len(peak_days) / len(daily_fte) * 100, 1),
            'avg_peak_fte': peak_days['FTE_Required_Total'].mean() if len(peak_days) > 0 else 0,
            'max_peak_fte': peak_days['FTE_Required_Total'].max() if len(peak_days) > 0 else 0,
            'peak_days_data': peak_days[['Date', 'Total_Cases', 'Total_Lines', 'FTE_Required_Total']],
            'recommended_core_staff': int(baseline_threshold),
            'recommended_flex_capacity': int(peak_threshold - baseline_threshold) if peak_threshold > baseline_threshold else 0
        }
        
        return peak_analysis
    
    def _analyze_category_labor_requirements(self, enriched_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze labor requirements by product category."""
        self.logger.info("Analyzing category-wise labor requirements")
        
        if 'Category' not in enriched_data.columns:
            return {'error': 'Category column not found in data'}
        
        # Category-wise aggregation
        category_data = enriched_data.groupby('Category').agg({
            'Total_Eaches': 'sum',
            'Case_Equivalent': 'sum', 
            'Order No.': 'count',
            'Sku Code': 'nunique'
        }).reset_index()
        
        category_data.columns = ['Category', 'Total_Eaches', 'Total_Cases', 'Total_Lines', 'Unique_SKUs']
        
        # Apply category-specific labor coefficients
        category_data['Labor_Coefficient'] = category_data['Category'].map(
            lambda x: self.category_labor_coefficients.get(x, self.category_labor_coefficients['default'])
        )
        
        # Calculate adjusted FTE requirements
        base_cases_per_fte = self.productivity_benchmarks['cases_per_fte_day']
        base_lines_per_fte = self.productivity_benchmarks['lines_per_fte_day']
        
        # Get total days for averaging
        total_days = enriched_data['Date'].nunique()
        
        category_data['Daily_Cases'] = category_data['Total_Cases'] / total_days
        category_data['Daily_Lines'] = category_data['Total_Lines'] / total_days
        
        category_data['FTE_Cases_Required'] = np.ceil(
            category_data['Daily_Cases'] * category_data['Labor_Coefficient'] / base_cases_per_fte
        )
        category_data['FTE_Lines_Required'] = np.ceil(
            category_data['Daily_Lines'] * category_data['Labor_Coefficient'] / base_lines_per_fte
        )
        
        category_data['FTE_Total_Required'] = np.maximum(
            category_data['FTE_Cases_Required'],
            category_data['FTE_Lines_Required']
        )
        
        # Calculate percentages
        total_fte = category_data['FTE_Total_Required'].sum()
        if total_fte > 0:
            category_data['FTE_Percentage'] = (
                category_data['FTE_Total_Required'] / total_fte * 100
            ).round(1)
        else:
            category_data['FTE_Percentage'] = 0
        
        category_analysis = {
            'category_breakdown': category_data,
            'total_fte_required': total_fte,
            'most_labor_intensive': category_data.loc[
                category_data['FTE_Total_Required'].idxmax(), 'Category'
            ] if len(category_data) > 0 else None,
            'labor_distribution': category_data.set_index('Category')['FTE_Percentage'].to_dict()
        }
        
        return category_analysis
    
    def _analyze_productivity_metrics(self, enriched_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze productivity metrics and benchmarking."""
        self.logger.info("Analyzing productivity metrics")
        
        total_days = enriched_data['Date'].nunique()
        
        # Overall productivity metrics
        total_cases = enriched_data['Case_Equivalent'].sum()
        total_lines = len(enriched_data)
        total_eaches = enriched_data['Total_Eaches'].sum()
        total_skus = enriched_data['Sku Code'].nunique()
        
        # Daily averages
        daily_cases = total_cases / total_days
        daily_lines = total_lines / total_days
        
        # Benchmark comparisons
        benchmark_cases = self.productivity_benchmarks['cases_per_fte_day']
        benchmark_lines = self.productivity_benchmarks['lines_per_fte_day']
        
        productivity_metrics = {
            'actual_daily_cases': daily_cases,
            'actual_daily_lines': daily_lines,
            'benchmark_cases_per_fte': benchmark_cases,
            'benchmark_lines_per_fte': benchmark_lines,
            'cases_per_line_ratio': total_cases / total_lines if total_lines > 0 else 0,
            'eaches_per_case_ratio': total_eaches / total_cases if total_cases > 0 else 0,
            'lines_per_sku_ratio': total_lines / total_skus if total_skus > 0 else 0,
            'complexity_score': self._calculate_complexity_score(enriched_data)
        }
        
        # Performance vs benchmark
        productivity_metrics['performance_vs_benchmark'] = {
            'cases_efficiency': (benchmark_cases / daily_cases * 100) if daily_cases > 0 else 0,
            'lines_efficiency': (benchmark_lines / daily_lines * 100) if daily_lines > 0 else 0
        }
        
        return productivity_metrics
    
    def _calculate_complexity_score(self, enriched_data: pd.DataFrame) -> float:
        """Calculate operational complexity score based on order patterns."""
        # Factors contributing to complexity:
        # - SKU diversity
        # - Order size variation
        # - Case vs eaches mix
        
        try:
            sku_diversity = enriched_data['Sku Code'].nunique() / len(enriched_data)
            
            case_equiv_cv = (
                enriched_data['Case_Equivalent'].std() / 
                enriched_data['Case_Equivalent'].mean()
            ) if enriched_data['Case_Equivalent'].mean() > 0 else 0
            
            eaches_ratio = (
                enriched_data['Qty in Eaches'].fillna(0).sum() /
                enriched_data['Total_Eaches'].sum()
            ) if enriched_data['Total_Eaches'].sum() > 0 else 0
            
            # Weighted complexity score (0-10 scale)
            complexity_score = (
                sku_diversity * 3 + 
                min(case_equiv_cv, 1) * 4 +
                eaches_ratio * 3
            )
            
            return round(complexity_score, 2)
            
        except Exception as e:
            self.logger.warning(f"Could not calculate complexity score: {str(e)}")
            return 5.0  # Default medium complexity
    
    def _calculate_shift_planning_recommendations(self, enriched_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate shift planning and scheduling recommendations."""
        self.logger.info("Calculating shift planning recommendations")
        
        daily_fte = self._calculate_base_fte_requirements(enriched_data)
        
        # Basic shift recommendations
        avg_fte = daily_fte['FTE_Required_Total'].mean()
        max_fte = daily_fte['FTE_Required_Total'].max()
        
        # Standard 2-shift operation
        shift_1_fte = np.ceil(avg_fte * 0.6)  # Day shift (60%)
        shift_2_fte = np.ceil(avg_fte * 0.4)  # Evening shift (40%)
        
        # Peak period adjustments
        peak_shift_1 = np.ceil(max_fte * 0.6)
        peak_shift_2 = np.ceil(max_fte * 0.4)
        
        shift_planning = {
            'recommended_shifts': 2,
            'core_staffing': {
                'shift_1_day': int(shift_1_fte),
                'shift_2_evening': int(shift_2_fte),
                'total_core': int(shift_1_fte + shift_2_fte)
            },
            'peak_staffing': {
                'shift_1_day': int(peak_shift_1),
                'shift_2_evening': int(peak_shift_2), 
                'total_peak': int(peak_shift_1 + peak_shift_2)
            },
            'flexibility_requirements': {
                'flex_fte_needed': int(max_fte - avg_fte),
                'overtime_hours_per_week': ((max_fte - avg_fte) * 8 * 5),  # Assuming 5-day work week
                'temp_worker_days_per_month': ((max_fte - avg_fte) * 20)  # Assuming ~20 working days
            }
        }
        
        return shift_planning
    
    def _calculate_labor_cost_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate labor cost analysis and budget estimates."""
        self.logger.info("Calculating labor cost analysis")
        
        # Standard cost assumptions (can be configurable)
        hourly_rates = {
            'regular_fte': 18.50,  # Regular warehouse worker
            'overtime_premium': 27.75,  # 1.5x overtime rate
            'temp_worker': 16.00,   # Temporary worker rate
            'supervisor': 28.00     # Supervisor rate
        }
        
        # Get key metrics from previous analysis
        daily_patterns = results.get('daily_staffing_analysis', {})
        shift_planning = results.get('shift_planning', {})
        
        avg_fte = daily_patterns.get('avg_fte_required', 0)
        core_staff = shift_planning.get('core_staffing', {}).get('total_core', 0)
        flex_needs = shift_planning.get('flexibility_requirements', {})
        
        # Monthly cost calculations (assuming 22 working days)
        working_days_month = 22
        hours_per_day = 8
        
        monthly_costs = {
            'core_staff_regular': core_staff * hours_per_day * working_days_month * hourly_rates['regular_fte'],
            'overtime_costs': flex_needs.get('overtime_hours_per_week', 0) * 4.33 * hourly_rates['overtime_premium'],
            'temp_worker_costs': flex_needs.get('temp_worker_days_per_month', 0) * hours_per_day * hourly_rates['temp_worker'],
            'supervisor_costs': max(1, np.ceil(core_staff / 10)) * hours_per_day * working_days_month * hourly_rates['supervisor']
        }
        
        monthly_costs['total_monthly_labor'] = sum(monthly_costs.values())
        monthly_costs['annual_labor_budget'] = monthly_costs['total_monthly_labor'] * 12
        
        # Cost per unit metrics
        daily_patterns_data = daily_patterns.get('daily_fte_data')
        if daily_patterns_data is not None and len(daily_patterns_data) > 0:
            total_cases = daily_patterns_data['Total_Cases'].sum()
            total_lines = daily_patterns_data['Total_Lines'].sum()
            
            monthly_costs['cost_per_case'] = (
                monthly_costs['total_monthly_labor'] / (total_cases * working_days_month / len(daily_patterns_data))
            ) if total_cases > 0 else 0
            
            monthly_costs['cost_per_line'] = (
                monthly_costs['total_monthly_labor'] / (total_lines * working_days_month / len(daily_patterns_data))
            ) if total_lines > 0 else 0
        
        return monthly_costs
    
    def _generate_summary_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level summary metrics for reporting."""
        self.logger.info("Generating summary metrics")
        
        daily_patterns = results.get('daily_staffing_analysis', {})
        peak_analysis = results.get('peak_analysis', {})
        cost_analysis = results.get('cost_analysis', {})
        category_analysis = results.get('category_labor_analysis', {})
        
        summary = {
            'recommended_core_fte': peak_analysis.get('recommended_core_staff', 0),
            'peak_fte_requirement': daily_patterns.get('max_fte_required', 0),
            'average_fte_requirement': round(daily_patterns.get('avg_fte_required', 0), 1),
            'flex_capacity_needed': peak_analysis.get('recommended_flex_capacity', 0),
            'peak_days_percentage': peak_analysis.get('peak_days_percentage', 0),
            'monthly_labor_budget': cost_analysis.get('total_monthly_labor', 0),
            'cost_per_case': round(cost_analysis.get('cost_per_case', 0), 2),
            'total_categories_analyzed': len(category_analysis.get('category_breakdown', [])) if isinstance(category_analysis.get('category_breakdown'), pd.DataFrame) else 0,
            'most_labor_intensive_category': category_analysis.get('most_labor_intensive', 'N/A')
        }
        
        return summary
    
    def run_full_analysis(self, enriched_data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Run complete manpower analysis and return standardized results.
        
        Args:
            enriched_data: DataFrame with enriched warehouse order data
            **kwargs: Additional parameters for analysis customization
            
        Returns:
            Dictionary containing complete manpower analysis results
        """
        self.logger.info("Running full manpower & FTE analysis")
        
        return self.analyze_manpower_requirements(
            enriched_data,
            kwargs.get('date_summary'),
            kwargs.get('sku_profile')
        )