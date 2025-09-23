#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main Orchestration Script for Warehouse Analysis

This script coordinates the entire warehouse analysis workflow using the modular components.
It replaces the monolithic original script with a clean, organized approach.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import data loading
from data_loader import load_and_enrich_data

# Import analyzers
from warehouse_analysis_modular.analyzers import (
    OrderAnalyzer, SkuAnalyzer, CrossTabulationAnalyzer
)

# Import reporting modules
from warehouse_analysis_modular.reporting import (
    ChartGenerator, LLMIntegration, HTMLReportGenerator, ExcelExporter
)

# Import utilities
from warehouse_analysis_modular.utils.helpers import setup_logging

# Configure logging
logger = setup_logging()


class WarehouseAnalysisPipeline:
    """
    Main pipeline class that orchestrates the entire warehouse analysis workflow.
    """
    
    def __init__(self, 
                 generate_charts: bool = True,
                 generate_llm_summaries: bool = True,
                 generate_html_report: bool = True,
                 generate_excel_export: bool = True):
        """
        Initialize the analysis pipeline.
        
        Args:
            generate_charts: Whether to generate charts
            generate_llm_summaries: Whether to generate LLM summaries
            generate_html_report: Whether to generate HTML report
            generate_excel_export: Whether to export to Excel
        """
        self.generate_charts = generate_charts
        self.generate_llm_summaries = generate_llm_summaries
        self.generate_html_report = generate_html_report
        self.generate_excel_export = generate_excel_export
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Warehouse Analysis Pipeline initialized")
        
        # Initialize components
        self.chart_generator = ChartGenerator() if generate_charts else None
        self.llm_integration = LLMIntegration() if generate_llm_summaries else None
        self.html_generator = HTMLReportGenerator() if generate_html_report else None
        self.excel_exporter = ExcelExporter() if generate_excel_export else None
    
    def load_data(self) -> 'pd.DataFrame':
        """
        Load and enrich the warehouse data.
        
        Returns:
            Enriched DataFrame ready for analysis
        """
        self.logger.info("Loading and enriching data")
        try:
            enriched_data = load_and_enrich_data()  # Uses default file path from config
            self.logger.info(f"Successfully loaded {len(enriched_data)} records")
            return enriched_data
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def run_order_analysis(self, enriched_data: 'pd.DataFrame') -> Dict:
        """
        Run order-level analysis.
        
        Args:
            enriched_data: Enriched order data
            
        Returns:
            Dictionary containing order analysis results
        """
        self.logger.info("Running order analysis")
        try:
            analyzer = OrderAnalyzer(enriched_data)
            results = analyzer.run_full_analysis()
            self.logger.info("Order analysis completed successfully")
            return results
        except Exception as e:
            self.logger.error(f"Order analysis failed: {str(e)}")
            raise
    
    def run_sku_analysis(self, enriched_data: 'pd.DataFrame') -> Dict:
        """
        Run SKU-level analysis including ABC-FMS classification.
        
        Args:
            enriched_data: Enriched order data
            
        Returns:
            Dictionary containing SKU analysis results
        """
        self.logger.info("Running SKU analysis")
        try:
            analyzer = SkuAnalyzer(enriched_data)
            results = analyzer.run_full_analysis()
            self.logger.info("SKU analysis completed successfully")
            return results
        except Exception as e:
            self.logger.error(f"SKU analysis failed: {str(e)}")
            raise
    
    def run_cross_tabulation_analysis(self, sku_profile: 'pd.DataFrame') -> Dict:
        """
        Run cross-tabulation analysis on SKU profile data.
        
        Args:
            sku_profile: SKU profile DataFrame with ABC-FMS classifications
            
        Returns:
            Dictionary containing cross-tabulation results
        """
        self.logger.info("Running cross-tabulation analysis")
        try:
            analyzer = CrossTabulationAnalyzer(sku_profile)
            results = analyzer.run_full_analysis()
            self.logger.info("Cross-tabulation analysis completed successfully")
            return results
        except Exception as e:
            self.logger.error(f"Cross-tabulation analysis failed: {str(e)}")
            raise
    
    def combine_analysis_results(self, 
                                order_results: Dict,
                                sku_results: Dict,
                                cross_tab_results: Dict) -> Dict:
        """
        Combine all analysis results into a single dictionary.
        
        Args:
            order_results: Results from order analysis
            sku_results: Results from SKU analysis
            cross_tab_results: Results from cross-tabulation analysis
            
        Returns:
            Combined results dictionary
        """
        self.logger.info("Combining analysis results")
        
        combined_results = {}
        
        # Add order analysis results
        combined_results.update({
            'date_order_summary': order_results.get('date_order_summary'),
            'sku_order_summary': order_results.get('sku_order_summary'),
            'percentile_profile': order_results.get('percentile_profile'),
            'order_statistics': order_results.get('statistics'),
            'demand_patterns': order_results.get('demand_patterns')
        })
        
        # Add SKU analysis results
        combined_results.update({
            'sku_profile_abc_fms': sku_results.get('sku_profile_abc_fms'),
            'sku_statistics': sku_results.get('statistics')
        })
        
        # Add cross-tabulation results
        combined_results.update({
            'abc_fms_summary': cross_tab_results.get('abc_fms_summary'),
            'sku_count_crosstab': cross_tab_results.get('sku_count_crosstab'),
            'volume_crosstab': cross_tab_results.get('volume_crosstab'),
            'lines_crosstab': cross_tab_results.get('lines_crosstab'),
            'cross_tabulation_insights': cross_tab_results.get('insights')
        })
        
        # Remove None values
        combined_results = {k: v for k, v in combined_results.items() if v is not None}
        
        self.logger.info(f"Combined results contain {len(combined_results)} datasets")
        return combined_results
    
    def generate_charts_step(self, analysis_results: Dict) -> Dict[str, str]:
        """
        Generate all charts from analysis results.
        
        Args:
            analysis_results: Combined analysis results
            
        Returns:
            Dictionary mapping chart names to file paths
        """
        if not self.generate_charts or not self.chart_generator:
            self.logger.info("Chart generation disabled")
            return {}
        
        self.logger.info("Generating charts")
        try:
            chart_paths = self.chart_generator.generate_all_charts(analysis_results)
            self.logger.info(f"Generated {len(chart_paths)} charts")
            return chart_paths
        except Exception as e:
            self.logger.error(f"Chart generation failed: {str(e)}")
            # Return empty dict to allow pipeline to continue
            return {}
    
    def generate_llm_summaries_step(self, analysis_results: Dict) -> Dict[str, str]:
        """
        Generate LLM summaries for analysis results.
        
        Args:
            analysis_results: Combined analysis results
            
        Returns:
            Dictionary with generated summaries
        """
        if not self.generate_llm_summaries or not self.llm_integration:
            self.logger.info("LLM summary generation disabled")
            return {}
        
        self.logger.info("Generating LLM summaries")
        try:
            summaries = self.llm_integration.generate_all_summaries(analysis_results)
            self.logger.info(f"Generated {len(summaries)} LLM summaries")
            return summaries
        except Exception as e:
            self.logger.error(f"LLM summary generation failed: {str(e)}")
            # Return empty dict to allow pipeline to continue
            return {}
    
    def generate_html_report_step(self, 
                                 analysis_results: Dict,
                                 chart_paths: Dict[str, str],
                                 llm_summaries: Dict[str, str]) -> Optional[str]:
        """
        Generate HTML report.
        
        Args:
            analysis_results: Combined analysis results
            chart_paths: Dictionary of chart file paths
            llm_summaries: Dictionary of LLM summaries
            
        Returns:
            Path to generated HTML file or None if disabled/failed
        """
        if not self.generate_html_report or not self.html_generator:
            self.logger.info("HTML report generation disabled")
            return None
        
        self.logger.info("Generating HTML report")
        try:
            report_path = self.html_generator.generate_report(
                analysis_results, chart_paths, llm_summaries
            )
            self.logger.info(f"HTML report generated: {report_path}")
            return report_path
        except Exception as e:
            self.logger.error(f"HTML report generation failed: {str(e)}")
            return None
    
    def export_excel_step(self, analysis_results: Dict) -> Optional[str]:
        """
        Export results to Excel.
        
        Args:
            analysis_results: Combined analysis results
            
        Returns:
            Path to exported Excel file or None if disabled/failed
        """
        if not self.generate_excel_export or not self.excel_exporter:
            self.logger.info("Excel export disabled")
            return None
        
        self.logger.info("Exporting to Excel")
        try:
            excel_path = self.excel_exporter.export_to_excel(analysis_results)
            self.logger.info(f"Excel export completed: {excel_path}")
            return excel_path
        except Exception as e:
            self.logger.error(f"Excel export failed: {str(e)}")
            return None
    
    def run_full_analysis(self) -> Dict:
        """
        Run the complete warehouse analysis pipeline.
        
        Returns:
            Dictionary with all results and output paths
        """
        self.logger.info("Starting full warehouse analysis pipeline")
        
        try:
            # Step 1: Load and enrich data
            enriched_data = self.load_data()
            
            # Step 2: Run all analyses
            order_results = self.run_order_analysis(enriched_data)
            sku_results = self.run_sku_analysis(enriched_data)
            
            # Get SKU profile for cross-tabulation
            sku_profile = sku_results.get('sku_profile_abc_fms')
            if sku_profile is None:
                raise ValueError("SKU profile not found in analysis results")
            
            cross_tab_results = self.run_cross_tabulation_analysis(sku_profile)
            
            # Step 3: Combine results
            analysis_results = self.combine_analysis_results(
                order_results, sku_results, cross_tab_results
            )
            
            # Step 4: Generate charts
            chart_paths = self.generate_charts_step(analysis_results)
            
            # Step 5: Generate LLM summaries
            llm_summaries = self.generate_llm_summaries_step(analysis_results)
            
            # Step 6: Generate HTML report
            html_report_path = self.generate_html_report_step(
                analysis_results, chart_paths, llm_summaries
            )
            
            # Step 7: Export to Excel
            excel_path = self.export_excel_step(analysis_results)
            
            # Compile final results
            final_results = {
                'analysis_results': analysis_results,
                'chart_paths': chart_paths,
                'llm_summaries': llm_summaries,
                'html_report_path': html_report_path,
                'excel_path': excel_path,
                'pipeline_status': 'completed_successfully'
            }
            
            self.logger.info("Warehouse analysis pipeline completed successfully!")
            self._print_completion_summary(final_results)
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            return {
                'pipeline_status': 'failed',
                'error': str(e)
            }
    
    def _print_completion_summary(self, results: Dict):
        """Print a summary of the completed analysis."""
        print("\n" + "="*60)
        print("🏭 WAREHOUSE ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Analysis summary
        analysis_results = results.get('analysis_results', {})
        order_stats = analysis_results.get('order_statistics', {})
        
        print(f"📊 Analysis Summary:")
        print(f"   • Total order lines: {order_stats.get('total_order_lines', 'N/A'):,}")
        print(f"   • Unique SKUs: {order_stats.get('unique_skus', 'N/A'):,}")
        print(f"   • Date range: {order_stats.get('unique_dates', 'N/A')} days")
        print(f"   • Total case equivalent: {order_stats.get('total_case_equivalent', 0):,.0f}")
        
        # Output files
        print(f"\n📁 Generated Files:")
        if results.get('excel_path'):
            print(f"   • Excel: {results['excel_path']}")
        if results.get('html_report_path'):
            print(f"   • HTML Report: {results['html_report_path']}")
        
        # Charts
        chart_count = len([p for p in results.get('chart_paths', {}).values() if p])
        if chart_count > 0:
            print(f"   • Charts: {chart_count} files generated")
        
        # LLM summaries
        summary_count = len([s for s in results.get('llm_summaries', {}).values() if s])
        if summary_count > 0:
            print(f"   • LLM Summaries: {summary_count} sections")
        
        print("\n✅ All analysis tasks completed!")
        print("="*60)


def run_full_analysis(generate_charts: bool = True,
                     generate_llm_summaries: bool = True,
                     generate_html_report: bool = True,
                     generate_excel_export: bool = True) -> Dict:
    """
    Convenience function to run the full analysis pipeline.
    
    Args:
        generate_charts: Whether to generate charts
        generate_llm_summaries: Whether to generate LLM summaries
        generate_html_report: Whether to generate HTML report
        generate_excel_export: Whether to export to Excel
        
    Returns:
        Dictionary with all results and output paths
    """
    pipeline = WarehouseAnalysisPipeline(
        generate_charts=generate_charts,
        generate_llm_summaries=generate_llm_summaries,
        generate_html_report=generate_html_report,
        generate_excel_export=generate_excel_export
    )
    
    return pipeline.run_full_analysis()


def run_analysis_only() -> Dict:
    """
    Run only the core analysis without generating reports.
    
    Returns:
        Dictionary with analysis results
    """
    return run_full_analysis(
        generate_charts=False,
        generate_llm_summaries=False,
        generate_html_report=False,
        generate_excel_export=True  # Keep Excel export for data access
    )


def run_with_reports() -> Dict:
    """
    Run complete analysis with all reports and visualizations.
    
    Returns:
        Dictionary with all results
    """
    return run_full_analysis(
        generate_charts=True,
        generate_llm_summaries=True,
        generate_html_report=True,
        generate_excel_export=True
    )


if __name__ == "__main__":
    """
    Main entry point - runs the full analysis pipeline when script is executed directly.
    """
    print("🚀 Starting Warehouse Analysis (Modular Version)")
    print("="*50)
    
    try:
        results = run_full_analysis()
        
        if results.get('pipeline_status') == 'completed_successfully':
            print("\n🎉 Analysis pipeline completed successfully!")
        else:
            print(f"\n❌ Pipeline failed: {results.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        logger.error(f"Unexpected error in main: {str(e)}", exc_info=True)
        sys.exit(1)