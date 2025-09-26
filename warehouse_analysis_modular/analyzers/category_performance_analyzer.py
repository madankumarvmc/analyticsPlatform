#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Category Performance Analysis

This analyzer provides three complementary distribution views of warehouse operations:
1. SKU % Distribution - Breadth analysis (how many SKUs exist in each category)
2. Cases % Distribution - Volume analysis (contribution by cases moved/picked)  
3. Lines % Distribution - Velocity analysis (frequency of ordering by category)

These three dimensions together provide strategic insights for warehouse slotting optimization.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any

# Import from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import utility functions
from warehouse_analysis_modular.utils.helpers import (
    validate_dataframe, safe_division, setup_logging
)

logger = setup_logging()


class CategoryPerformanceAnalyzer:
    """
    Category Performance Distribution Analyzer.
    
    This analyzer creates three complementary percentage distribution tables:
    - SKU % Distribution: Shows breadth of SKUs across categories and ABC-FMS classes
    - Cases % Distribution: Shows volume contribution by category and class
    - Lines % Distribution: Shows velocity (order frequency) by category and class
    
    These three views together provide strategic insights for:
    - Warehouse slotting optimization
    - Bin allocation strategies
    - Dock proximity planning
    - Picker efficiency improvements
    
    Example:
        analyzer = CategoryPerformanceAnalyzer(enriched_data)
        results = analyzer.run_full_analysis()
    """
    
    def __init__(self, input_data: pd.DataFrame):
        """
        Initialize the analyzer with SKU profile data.
        
        Args:
            input_data: DataFrame with SKU profile data containing:
                       - Category: SKU category codes
                       - ABC: ABC classification (A, B, C)
                       - FMS: FMS classification (Fast, Medium, Slow)
                       - Total_Case_Equiv: Total volume in cases
                       - Total_Order_Lines: Total order lines count
        
        Raises:
            ValueError: If input data doesn't meet requirements
        """
        self.input_data = input_data
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Validate input data first
        self._validate_input_data()
        
        # Initialize class variables
        self.analysis_results = {}
        
        self.logger.info(f"Initialized CategoryPerformanceAnalyzer with {len(self.input_data)} SKUs")
    
    def _validate_input_data(self):
        """
        Validate that the input data has required columns and structure.
        
        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        required_columns = [
            'Category',           # SKU category
            'ABC',               # ABC classification
            'FMS',               # FMS classification
            'Total_Case_Equiv',  # Volume data
            'Total_Order_Lines', # Order lines data
        ]
        
        # Validate using the helper function
        validate_dataframe(
            self.input_data, 
            required_columns, 
            min_rows=1, 
            name="CategoryPerformanceAnalyzer input data"
        )
        
        # Additional validation for ABC-FMS combinations
        if 'ABC' in self.input_data.columns and 'FMS' in self.input_data.columns:
            # Create ABC-FMS combined classification
            self.input_data = self.input_data.copy()
            self.input_data['ABC_FMS'] = self.input_data['ABC'].astype(str) + self.input_data['FMS'].astype(str)
    
    def analyze_sku_distribution(self) -> pd.DataFrame:
        """
        Analyze SKU count distribution across categories and ABC-FMS classes.
        
        This provides breadth analysis - showing how many SKUs exist in each category
        and their distribution across ABC-FMS classes. Critical for understanding
        which categories dominate the SKU base and require structured bin allocation.
        
        Returns:
            DataFrame: SKU count and percentage distribution by category and ABC-FMS class
        """
        self.logger.info("Analyzing SKU distribution")
        
        try:
            # Create crosstab of Category vs ABC-FMS with SKU counts
            sku_crosstab = pd.crosstab(
                self.input_data['Category'], 
                self.input_data['ABC_FMS'], 
                margins=True, 
                margins_name='Grand Total'
            )
            
            # Calculate total SKUs per category
            category_totals = sku_crosstab.drop('Grand Total', axis=1).sum(axis=1)
            sku_crosstab['#SKUs'] = category_totals
            
            # Calculate SKU percentages
            total_skus = category_totals.sum()
            sku_crosstab['SKU %'] = round(category_totals / total_skus * 100, 0).astype(int)
            
            # Sort by SKU count descending (excluding Grand Total)
            category_rows = sku_crosstab.index != 'Grand Total'
            sorted_categories = sku_crosstab.loc[category_rows].sort_values('#SKUs', ascending=False)
            grand_total_row = sku_crosstab.loc[['Grand Total']]
            
            sku_distribution = pd.concat([grand_total_row, sorted_categories])
            
            self.logger.info(f"SKU distribution analysis completed for {len(sorted_categories)} categories")
            return sku_distribution
            
        except Exception as e:
            self.logger.error(f"SKU distribution analysis failed: {str(e)}")
            raise
    
    def analyze_cases_distribution(self) -> pd.DataFrame:
        """
        Analyze volume distribution (cases) across categories and ABC-FMS classes.
        
        This provides volume analysis - showing which categories drive warehouse throughput.
        Critical for determining which categories should be placed near docks and 
        require priority slotting for efficiency.
        
        Returns:
            DataFrame: Volume percentage distribution by category and ABC-FMS class
        """
        self.logger.info("Analyzing cases distribution")
        
        try:
            # Create crosstab of Category vs ABC-FMS with volume sums
            cases_crosstab = pd.crosstab(
                self.input_data['Category'], 
                self.input_data['ABC_FMS'], 
                values=self.input_data['Total_Case_Equiv'],
                aggfunc='sum',
                margins=True, 
                margins_name='Grand Total'
            ).fillna(0)
            
            # Calculate total volume per category
            category_volumes = cases_crosstab.drop('Grand Total', axis=1).sum(axis=1)
            
            # Calculate volume percentages
            total_volume = category_volumes.sum()
            cases_percentages = cases_crosstab.div(total_volume) * 100
            
            # Round to appropriate precision
            cases_percentages = cases_percentages.round(0).astype(int)
            
            # Add total volume percentage column
            cases_percentages['% Cases'] = round(category_volumes / total_volume * 100, 0).astype(int)
            
            # Sort by volume percentage descending (excluding Grand Total)
            category_rows = cases_percentages.index != 'Grand Total'
            sorted_categories = cases_percentages.loc[category_rows].sort_values('% Cases', ascending=False)
            grand_total_row = cases_percentages.loc[['Grand Total']]
            
            cases_distribution = pd.concat([grand_total_row, sorted_categories])
            
            self.logger.info(f"Cases distribution analysis completed for {len(sorted_categories)} categories")
            return cases_distribution
            
        except Exception as e:
            self.logger.error(f"Cases distribution analysis failed: {str(e)}")
            raise
    
    def analyze_lines_distribution(self) -> pd.DataFrame:
        """
        Analyze order lines distribution (velocity) across categories and ABC-FMS classes.
        
        This provides velocity analysis - showing which categories are ordered most frequently.
        Critical for determining which SKUs should be placed for easy picker access 
        to reduce walking distance and picking time.
        
        Returns:
            DataFrame: Order lines percentage distribution by category and ABC-FMS class
        """
        self.logger.info("Analyzing lines distribution")
        
        try:
            # Create crosstab of Category vs ABC-FMS with order lines sums
            lines_crosstab = pd.crosstab(
                self.input_data['Category'], 
                self.input_data['ABC_FMS'], 
                values=self.input_data['Total_Order_Lines'],
                aggfunc='sum',
                margins=True, 
                margins_name='Grand Total'
            ).fillna(0)
            
            # Calculate total lines per category
            category_lines = lines_crosstab.drop('Grand Total', axis=1).sum(axis=1)
            
            # Calculate lines percentages
            total_lines = category_lines.sum()
            lines_percentages = lines_crosstab.div(total_lines) * 100
            
            # Round to appropriate precision
            lines_percentages = lines_percentages.round(0).astype(int)
            
            # Add total lines percentage column
            lines_percentages['% Lines'] = round(category_lines / total_lines * 100, 0).astype(int)
            
            # Sort by lines percentage descending (excluding Grand Total)
            category_rows = lines_percentages.index != 'Grand Total'
            sorted_categories = lines_percentages.loc[category_rows].sort_values('% Lines', ascending=False)
            grand_total_row = lines_percentages.loc[['Grand Total']]
            
            lines_distribution = pd.concat([grand_total_row, sorted_categories])
            
            self.logger.info(f"Lines distribution analysis completed for {len(sorted_categories)} categories")
            return lines_distribution
            
        except Exception as e:
            self.logger.error(f"Lines distribution analysis failed: {str(e)}")
            raise
    
    def generate_performance_summary(self, sku_dist: pd.DataFrame, 
                                   cases_dist: pd.DataFrame, 
                                   lines_dist: pd.DataFrame) -> pd.DataFrame:
        """
        Generate combined performance summary with rankings and priority scores.
        
        Args:
            sku_dist: SKU distribution DataFrame
            cases_dist: Cases distribution DataFrame  
            lines_dist: Lines distribution DataFrame
            
        Returns:
            DataFrame: Combined performance summary with rankings
        """
        self.logger.info("Generating performance summary")
        
        try:
            # Extract category-level summaries (exclude Grand Total)
            categories = [idx for idx in sku_dist.index if idx != 'Grand Total']
            
            summary_data = []
            for category in categories:
                if category in cases_dist.index and category in lines_dist.index:
                    sku_pct = sku_dist.loc[category, 'SKU %'] if 'SKU %' in sku_dist.columns else 0
                    cases_pct = cases_dist.loc[category, '% Cases'] if '% Cases' in cases_dist.columns else 0
                    lines_pct = lines_dist.loc[category, '% Lines'] if '% Lines' in lines_dist.columns else 0
                    sku_count = sku_dist.loc[category, '#SKUs'] if '#SKUs' in sku_dist.columns else 0
                    
                    # Calculate priority score (weighted combination)
                    # Volume and velocity are more important than SKU count for slotting
                    priority_score = (cases_pct * 0.5) + (lines_pct * 0.4) + (sku_pct * 0.1)
                    
                    summary_data.append({
                        'Category': category,
                        'SKU_Count': sku_count,
                        'SKU_Percent': sku_pct,
                        'Cases_Percent': cases_pct,
                        'Lines_Percent': lines_pct,
                        'Priority_Score': round(priority_score, 1),
                        'Slotting_Priority': 'High' if priority_score >= 15 else 'Medium' if priority_score >= 5 else 'Low'
                    })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df = summary_df.sort_values('Priority_Score', ascending=False)
            
            self.logger.info(f"Performance summary generated for {len(summary_df)} categories")
            return summary_df
            
        except Exception as e:
            self.logger.error(f"Performance summary generation failed: {str(e)}")
            return pd.DataFrame()
    
    def generate_slotting_insights(self, performance_summary: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate strategic slotting insights based on category performance.
        
        Args:
            performance_summary: Combined performance summary DataFrame
            
        Returns:
            Dictionary: Strategic insights and recommendations
        """
        self.logger.info("Generating slotting insights")
        
        try:
            if performance_summary.empty:
                return {}
            
            # Identify top performers
            high_priority = performance_summary[performance_summary['Slotting_Priority'] == 'High']
            medium_priority = performance_summary[performance_summary['Slotting_Priority'] == 'Medium']
            
            # Calculate key metrics
            total_high_volume = high_priority['Cases_Percent'].sum() if not high_priority.empty else 0
            total_high_velocity = high_priority['Lines_Percent'].sum() if not high_priority.empty else 0
            
            # ABC-FMS class analysis
            abc_fms_volume = {}
            abc_fms_velocity = {}
            
            # Analyze ABC-FMS contributions from the original data
            for abc_fms in self.input_data['ABC_FMS'].unique():
                abc_fms_data = self.input_data[self.input_data['ABC_FMS'] == abc_fms]
                vol_contribution = abc_fms_data['Total_Case_Equiv'].sum() / self.input_data['Total_Case_Equiv'].sum() * 100
                vel_contribution = abc_fms_data['Total_Order_Lines'].sum() / self.input_data['Total_Order_Lines'].sum() * 100
                abc_fms_volume[abc_fms] = round(vol_contribution, 1)
                abc_fms_velocity[abc_fms] = round(vel_contribution, 1)
            
            insights = {
                'high_priority_categories': high_priority['Category'].tolist() if not high_priority.empty else [],
                'medium_priority_categories': medium_priority['Category'].tolist() if not medium_priority.empty else [],
                'total_categories_analyzed': len(performance_summary),
                'high_priority_volume_share': round(total_high_volume, 1),
                'high_priority_velocity_share': round(total_high_velocity, 1),
                'abc_fms_volume_contribution': abc_fms_volume,
                'abc_fms_velocity_contribution': abc_fms_velocity,
                'slotting_recommendations': {
                    'dock_proximity': high_priority['Category'].head(3).tolist() if not high_priority.empty else [],
                    'structured_bins': medium_priority['Category'].head(5).tolist() if not medium_priority.empty else [],
                    'standard_storage': performance_summary[performance_summary['Slotting_Priority'] == 'Low']['Category'].tolist()
                },
                'key_findings': {
                    'top_volume_category': performance_summary.loc[0, 'Category'] if not performance_summary.empty else None,
                    'top_velocity_category': performance_summary.nlargest(1, 'Lines_Percent')['Category'].iloc[0] if not performance_summary.empty else None,
                    'critical_abc_fms_classes': [cls for cls, vol in abc_fms_volume.items() if vol >= 10]
                }
            }
            
            self.logger.info("Slotting insights generated successfully")
            return insights
            
        except Exception as e:
            self.logger.error(f"Slotting insights generation failed: {str(e)}")
            return {}
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics about the category performance analysis.
        
        Returns:
            Dictionary: Summary statistics
        """
        try:
            stats = {
                'total_skus': len(self.input_data),
                'total_categories': self.input_data['Category'].nunique(),
                'total_abc_fms_classes': self.input_data['ABC_FMS'].nunique(),
                'total_volume': self.input_data['Total_Case_Equiv'].sum(),
                'total_lines': self.input_data['Total_Order_Lines'].sum(),
                'category_list': sorted(self.input_data['Category'].unique()),
                'abc_fms_classes': sorted(self.input_data['ABC_FMS'].unique())
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Summary statistics calculation failed: {str(e)}")
            return {}
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run the complete category performance analysis pipeline.
        
        Returns:
            Dictionary containing all analysis results:
            {
                'sku_distribution': pd.DataFrame,      # SKU % by category/ABC-FMS
                'cases_distribution': pd.DataFrame,    # Cases % by category/ABC-FMS  
                'lines_distribution': pd.DataFrame,    # Lines % by category/ABC-FMS
                'performance_summary': pd.DataFrame,   # Combined rankings
                'slotting_insights': dict,            # Strategic recommendations
                'statistics': dict                    # Key metrics and totals
            }
        """
        self.logger.info("Starting full category performance analysis")
        
        try:
            # Step 1: Run three distribution analyses
            sku_distribution = self.analyze_sku_distribution()
            cases_distribution = self.analyze_cases_distribution()
            lines_distribution = self.analyze_lines_distribution()
            
            # Step 2: Generate performance summary
            performance_summary = self.generate_performance_summary(
                sku_distribution, cases_distribution, lines_distribution
            )
            
            # Step 3: Generate strategic insights
            slotting_insights = self.generate_slotting_insights(performance_summary)
            
            # Step 4: Get summary statistics
            statistics = self.get_summary_statistics()
            
            # Step 5: Prepare final results
            final_results = {
                'sku_distribution': sku_distribution,
                'cases_distribution': cases_distribution,
                'lines_distribution': lines_distribution,
                'performance_summary': performance_summary,
                'slotting_insights': slotting_insights,
                'statistics': statistics
            }
            
            # Store results for potential future use
            self.analysis_results = final_results
            
            self.logger.info("Category performance analysis completed successfully")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Full analysis failed: {str(e)}")
            
            # Return empty results structure on failure
            return {
                'sku_distribution': pd.DataFrame(),
                'cases_distribution': pd.DataFrame(),
                'lines_distribution': pd.DataFrame(),
                'performance_summary': pd.DataFrame(),
                'slotting_insights': {},
                'statistics': {}
            }


def analyze_category_performance(input_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Convenience function to run category performance analysis.
    
    Args:
        input_data: DataFrame with SKU profile data
        
    Returns:
        Dictionary containing analysis results
    """
    analyzer = CategoryPerformanceAnalyzer(input_data)
    return analyzer.run_full_analysis()


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage and basic testing.
    """
    print("CategoryPerformanceAnalyzer - Example Usage")
    
    # Create sample data for testing
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'Category': np.random.choice(['BI', 'SX', 'ND', 'CG', 'AT'], 100),
        'ABC': np.random.choice(['A', 'B', 'C'], 100),
        'FMS': np.random.choice(['F', 'M', 'S'], 100),
        'Total_Case_Equiv': np.random.randint(1, 1000, 100),
        'Total_Order_Lines': np.random.randint(1, 100, 100)
    })
    
    print(f"Created sample data with {len(sample_data)} SKUs")
    
    try:
        # Test the analyzer
        analyzer = CategoryPerformanceAnalyzer(sample_data)
        results = analyzer.run_full_analysis()
        
        print("\nAnalysis Results:")
        for key, value in results.items():
            if isinstance(value, pd.DataFrame):
                print(f"- {key}: DataFrame with {len(value)} rows")
            else:
                print(f"- {key}: {type(value).__name__}")
        
        print("\nTop 3 Categories by Priority:")
        if not results['performance_summary'].empty:
            top_3 = results['performance_summary'].head(3)
            for _, row in top_3.iterrows():
                print(f"  {row['Category']}: Priority Score {row['Priority_Score']}")
        
        print("\nAnalyzer test completed successfully!")
        
    except Exception as e:
        print(f"Error testing analyzer: {str(e)}")