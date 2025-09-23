#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
New Analyzer Template

This template provides a standardized structure for creating new warehouse analyzers.
Follow this pattern to ensure consistency and easy integration with the web interface.

Usage:
1. Copy this file: cp new_analyzer_template.py your_analyzer_name.py
2. Replace all instances of "NewAnalyzer" with your analyzer name
3. Replace placeholder methods with your analysis logic
4. Update the docstrings with your specific functionality
5. Add your analyzer to __init__.py exports
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any

# Import from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import configuration (customize as needed)
from config import ABC_THRESHOLDS, FMS_THRESHOLDS  # Example imports

# Import utility functions
from warehouse_analysis_modular.utils.helpers import (
    validate_dataframe, safe_division, setup_logging
)

logger = setup_logging()


class NewAnalyzerTemplate:
    """
    Template for creating new warehouse analyzers.
    
    This analyzer template demonstrates the standard structure and patterns
    that all analyzers should follow for consistency and integration.
    
    Replace this docstring with a description of your specific analysis:
    - What does this analyzer do?
    - What business questions does it answer?
    - What insights does it provide?
    
    Example:
        analyzer = NewAnalyzerTemplate(enriched_data)
        results = analyzer.run_full_analysis()
    """
    
    def __init__(self, input_data: pd.DataFrame):
        """
        Initialize the analyzer with input data.
        
        Args:
            input_data: DataFrame with the data to analyze
                       Must contain the required columns (see _validate_input_data)
        
        Raises:
            ValueError: If input data doesn't meet requirements
        """
        self.input_data = input_data
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Validate input data first
        self._validate_input_data()
        
        # Initialize any class variables
        self.analysis_results = {}
        
        self.logger.info(f"Initialized {self.__class__.__name__} with {len(self.input_data)} rows")
    
    def _validate_input_data(self):
        """
        Validate that the input data has required columns and structure.
        
        Customize this method based on your analyzer's requirements.
        
        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        # Define required columns for your analysis
        required_columns = [
            'Date',           # Example: Date column
            'Sku Code',       # Example: SKU identifier
            'Order No.',      # Example: Order identifier
            # Add your required columns here
        ]
        
        # Validate using the helper function
        validate_dataframe(
            self.input_data, 
            required_columns, 
            min_rows=1, 
            name=f"{self.__class__.__name__} input data"
        )
        
        # Add any custom validation logic here
        # Example: Check for specific data types, ranges, etc.
        
    def analyze_primary_metric(self) -> pd.DataFrame:
        """
        Perform your primary analysis.
        
        Replace this method with your main analysis logic.
        This is where you'll implement the core business logic for your analyzer.
        
        Returns:
            DataFrame: Results of your primary analysis
            
        Example:
            Returns a DataFrame with your calculated metrics, aggregations, 
            classifications, or other analysis results.
        """
        self.logger.info("Performing primary analysis")
        
        try:
            # Example analysis - replace with your logic
            primary_results = self.input_data.groupby('Sku Code').agg({
                'Order No.': 'count',  # Example aggregation
                # Add your aggregations here
            }).reset_index()
            
            # Example: Add calculated columns
            primary_results['calculated_metric'] = primary_results['Order No.'] * 2
            
            # Add any additional processing
            primary_results = self._add_classifications(primary_results)
            
            self.logger.info(f"Primary analysis completed with {len(primary_results)} results")
            return primary_results
            
        except Exception as e:
            self.logger.error(f"Primary analysis failed: {str(e)}")
            raise
    
    def analyze_secondary_metric(self) -> pd.DataFrame:
        """
        Perform secondary analysis (optional).
        
        Add additional analysis methods as needed for your use case.
        You can have multiple analysis methods that focus on different aspects.
        
        Returns:
            DataFrame: Results of secondary analysis
        """
        self.logger.info("Performing secondary analysis")
        
        try:
            # Example secondary analysis - replace with your logic
            secondary_results = self.input_data.groupby('Date').agg({
                'Sku Code': 'nunique',  # Example: unique SKUs per date
                # Add your aggregations here
            }).reset_index()
            
            self.logger.info(f"Secondary analysis completed with {len(secondary_results)} results")
            return secondary_results
            
        except Exception as e:
            self.logger.error(f"Secondary analysis failed: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame on failure
    
    def _add_classifications(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add classification logic (private helper method).
        
        This is an example of a helper method for adding classifications,
        categories, or other derived fields to your analysis results.
        
        Args:
            data: DataFrame to add classifications to
            
        Returns:
            DataFrame: Data with added classification columns
        """
        try:
            # Example classification logic - customize for your needs
            data = data.copy()
            
            # Example: Add a simple classification based on order count
            data['classification'] = data['Order No.'].apply(
                lambda x: 'High' if x > 10 else 'Medium' if x > 5 else 'Low'
            )
            
            # Add more classification logic as needed
            
            return data
            
        except Exception as e:
            self.logger.error(f"Classification failed: {str(e)}")
            return data
    
    def calculate_insights(self, analysis_results: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Calculate insights and summary statistics from analysis results.
        
        This method should generate key insights, statistics, and summaries
        that will be useful for reporting and visualization.
        
        Args:
            analysis_results: Dictionary containing analysis DataFrames
            
        Returns:
            Dictionary: Key insights and statistics
        """
        self.logger.info("Calculating insights")
        
        try:
            insights = {}
            
            # Example insights - customize for your analysis
            if 'primary_analysis' in analysis_results:
                primary_data = analysis_results['primary_analysis']
                
                insights.update({
                    'total_records': len(primary_data),
                    'unique_skus': primary_data['Sku Code'].nunique() if 'Sku Code' in primary_data.columns else 0,
                    'average_metric': primary_data['calculated_metric'].mean() if 'calculated_metric' in primary_data.columns else 0,
                    # Add your specific insights here
                })
            
            # Add insights from secondary analysis if available
            if 'secondary_analysis' in analysis_results:
                secondary_data = analysis_results['secondary_analysis']
                insights.update({
                    'time_periods': len(secondary_data),
                    # Add more insights
                })
            
            # Add any business-specific insights
            insights.update(self._calculate_business_insights(analysis_results))
            
            self.logger.info("Insights calculation completed")
            return insights
            
        except Exception as e:
            self.logger.error(f"Insights calculation failed: {str(e)}")
            return {}
    
    def _calculate_business_insights(self, analysis_results: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Calculate business-specific insights (private helper method).
        
        Override this method with your specific business logic and KPIs.
        
        Args:
            analysis_results: Dictionary containing analysis DataFrames
            
        Returns:
            Dictionary: Business-specific insights
        """
        # Add your business-specific insight calculations here
        business_insights = {
            'insight_example': 'This is where you add domain-specific insights',
            # Add your insights here
        }
        
        return business_insights
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics about the input data.
        
        This method provides basic statistics about the input data
        that can be useful for understanding data quality and coverage.
        
        Returns:
            Dictionary: Summary statistics
        """
        try:
            stats = {
                'total_rows': len(self.input_data),
                'date_range': {
                    'start': self.input_data['Date'].min() if 'Date' in self.input_data.columns else None,
                    'end': self.input_data['Date'].max() if 'Date' in self.input_data.columns else None,
                } if 'Date' in self.input_data.columns else None,
                'unique_skus': self.input_data['Sku Code'].nunique() if 'Sku Code' in self.input_data.columns else 0,
                # Add more summary statistics as needed
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Summary statistics calculation failed: {str(e)}")
            return {}
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline.
        
        This is the main method that orchestrates all analysis steps.
        All analyzers MUST implement this method with this exact signature.
        
        Returns:
            Dictionary containing all analysis results:
            {
                'primary_analysis': pd.DataFrame,
                'secondary_analysis': pd.DataFrame,  # Optional
                'insights': dict,
                'statistics': dict
            }
        """
        self.logger.info(f"Starting full analysis with {self.__class__.__name__}")
        
        try:
            # Step 1: Run primary analysis
            primary_results = self.analyze_primary_metric()
            
            # Step 2: Run secondary analysis (optional)
            secondary_results = self.analyze_secondary_metric()
            
            # Step 3: Compile results
            analysis_results = {
                'primary_analysis': primary_results,
            }
            
            # Add secondary results if they exist and aren't empty
            if not secondary_results.empty:
                analysis_results['secondary_analysis'] = secondary_results
            
            # Step 4: Calculate insights
            insights = self.calculate_insights(analysis_results)
            
            # Step 5: Get summary statistics
            statistics = self.get_summary_statistics()
            
            # Step 6: Prepare final results
            final_results = {
                **analysis_results,  # Include all analysis DataFrames
                'insights': insights,
                'statistics': statistics
            }
            
            # Store results for potential future use
            self.analysis_results = final_results
            
            self.logger.info(f"{self.__class__.__name__} analysis completed successfully")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Full analysis failed: {str(e)}")
            
            # Return empty results structure on failure
            return {
                'primary_analysis': pd.DataFrame(),
                'insights': {},
                'statistics': {}
            }


def analyze_with_template(input_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Convenience function to run the template analyzer.
    
    This is a helper function that follows the pattern used by other analyzers.
    Rename this function to match your analyzer (e.g., analyze_customers, analyze_performance).
    
    Args:
        input_data: DataFrame with enriched data
        
    Returns:
        Dictionary containing analysis results
    """
    analyzer = NewAnalyzerTemplate(input_data)
    return analyzer.run_full_analysis()


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage and basic testing.
    
    This section demonstrates how to use your analyzer and can serve as a basic test.
    """
    print("NewAnalyzerTemplate - Example Usage")
    
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'Date': pd.date_range('2025-01-01', periods=100),
        'Sku Code': [f'SKU{i%20:03d}' for i in range(100)],
        'Order No.': [f'ORD{i:04d}' for i in range(100)],
        'quantity': np.random.randint(1, 100, 100)
    })
    
    print(f"Created sample data with {len(sample_data)} rows")
    
    try:
        # Test the analyzer
        analyzer = NewAnalyzerTemplate(sample_data)
        results = analyzer.run_full_analysis()
        
        print("\nAnalysis Results:")
        for key, value in results.items():
            if isinstance(value, pd.DataFrame):
                print(f"- {key}: DataFrame with {len(value)} rows")
            else:
                print(f"- {key}: {type(value).__name__}")
        
        print("\nAnalyzer test completed successfully!")
        
    except Exception as e:
        print(f"Error testing analyzer: {str(e)}")


"""
CUSTOMIZATION CHECKLIST:

□ Rename class from "NewAnalyzerTemplate" to your analyzer name
□ Update class docstring with your analyzer's purpose
□ Modify required_columns in _validate_input_data()
□ Implement your analysis logic in analyze_primary_metric()
□ Add secondary analysis methods if needed
□ Customize classification logic in _add_classifications()
□ Add business-specific insights in _calculate_business_insights()
□ Update summary statistics in get_summary_statistics()
□ Rename convenience function (analyze_with_template)
□ Update example usage in __main__ section
□ Add your analyzer to __init__.py exports
□ Update web interface integration
□ Add unit tests for your analyzer
□ Update documentation

INTEGRATION STEPS:

1. Add to analyzers/__init__.py:
   from .your_analyzer import YourAnalyzer

2. Add to web_utils/analysis_integration.py:
   def _run_your_analysis(self, enriched_data):
       analyzer = YourAnalyzer(enriched_data)
       return analyzer.run_full_analysis()

3. Add to components/results_display.py:
   def _display_your_analysis(self, analysis_results):
       # Display your results

4. Test integration with web interface
"""