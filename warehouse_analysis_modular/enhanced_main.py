#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced Main Orchestration Script for Advanced Warehouse Analysis

This script coordinates the entire advanced warehouse analysis workflow with 
new analysis modules including multi-metric correlation, picking analysis,
and enhanced ABC-FMS cross-classification.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import data loading
from data_loader import load_and_enrich_data

# Import original analyzers
from warehouse_analysis_modular.analyzers import (
    OrderAnalyzer, SkuAnalyzer, CrossTabulationAnalyzer
)

# Import new advanced analyzers
from warehouse_analysis_modular.analyzers.advanced_order_analyzer import AdvancedOrderAnalyzer
from warehouse_analysis_modular.analyzers.picking_analyzer import PickingAnalyzer
from warehouse_analysis_modular.analyzers.enhanced_abc_fms_analyzer import EnhancedABCFMSAnalyzer
from warehouse_analysis_modular.analyzers.category_performance_analyzer import CategoryPerformanceAnalyzer

# Import reporting modules
from warehouse_analysis_modular.reporting import (
    ChartGenerator, LLMIntegration, HTMLReportGenerator, ExcelExporter
)
from warehouse_analysis_modular.reporting.advanced_chart_generator import AdvancedChartGenerator
from warehouse_analysis_modular.reporting.word_report import WordReportGenerator

# Import utilities
from warehouse_analysis_modular.utils.helpers import setup_logging

# Configure logging
logger = setup_logging()


class EnhancedWarehouseAnalysisPipeline:
    """
    Enhanced pipeline class that orchestrates advanced warehouse analysis workflow
    with multi-metric correlation, picking analysis, and 2D classification matrix.
    """
    
    def __init__(self, 
                 generate_charts: bool = True,
                 generate_advanced_charts: bool = True,
                 generate_llm_summaries: bool = True,
                 generate_html_report: bool = True,
                 generate_word_report: bool = True,
                 generate_excel_export: bool = True,
                 run_advanced_analysis: bool = True):
        """
        Initialize the enhanced analysis pipeline.
        
        Args:
            generate_charts: Whether to generate basic charts
            generate_advanced_charts: Whether to generate advanced charts
            generate_llm_summaries: Whether to generate LLM summaries
            generate_html_report: Whether to generate HTML report
            generate_word_report: Whether to generate Word report
            generate_excel_export: Whether to export to Excel
            run_advanced_analysis: Whether to run advanced analysis modules
        """
        self.generate_charts = generate_charts
        self.generate_advanced_charts = generate_advanced_charts
        self.generate_llm_summaries = generate_llm_summaries
        self.generate_html_report = generate_html_report
        self.generate_word_report = generate_word_report
        self.generate_excel_export = generate_excel_export
        self.enable_advanced_analysis = run_advanced_analysis
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Enhanced Warehouse Analysis Pipeline initialized")
        
        # Initialize components
        self.chart_generator = ChartGenerator() if generate_charts else None
        self.advanced_chart_generator = AdvancedChartGenerator() if generate_advanced_charts else None
        self.llm_integration = LLMIntegration() if generate_llm_summaries else None
        self.html_generator = HTMLReportGenerator() if generate_html_report else None
        self.word_generator = WordReportGenerator() if generate_word_report else None
        self.excel_exporter = ExcelExporter() if generate_excel_export else None
        
        # Initialize advanced analyzers
        self.advanced_order_analyzer = AdvancedOrderAnalyzer() if run_advanced_analysis else None
        self.picking_analyzer = PickingAnalyzer() if run_advanced_analysis else None
        self.enhanced_abc_fms_analyzer = EnhancedABCFMSAnalyzer() if run_advanced_analysis else None
        self.use_category_performance_analyzer = run_advanced_analysis
    
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
    
    def run_basic_analysis(self, enriched_data: 'pd.DataFrame') -> Dict[str, Any]:
        """
        Run basic analysis modules (original functionality).
        
        Args:
            enriched_data: Enriched order data
            
        Returns:
            Dictionary containing basic analysis results
        """
        self.logger.info("Running basic analysis modules")
        
        results = {}
        
        try:
            # 1. Order Analysis
            self.logger.info("Running order analysis")
            order_analyzer = OrderAnalyzer(enriched_data)
            order_results = order_analyzer.run_full_analysis()
            results.update(order_results)
            
            # 2. SKU Analysis
            self.logger.info("Running SKU analysis")
            sku_analyzer = SkuAnalyzer(enriched_data)
            sku_results = sku_analyzer.run_full_analysis()
            results.update(sku_results)
            
            # 3. Cross-Tabulation Analysis
            self.logger.info("Running cross-tabulation analysis")
            if 'sku_profile_abc_fms' in sku_results:
                cross_tab_analyzer = CrossTabulationAnalyzer(sku_results['sku_profile_abc_fms'])
                cross_tab_results = cross_tab_analyzer.run_full_analysis()
                results.update(cross_tab_results)
            
            self.logger.info("Basic analysis completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in basic analysis: {str(e)}")
            raise
        
        return results
    
    def run_advanced_analysis(self, enriched_data: 'pd.DataFrame', 
                            basic_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run advanced analysis modules.
        
        Args:
            enriched_data: Enriched order data
            basic_results: Results from basic analysis
            
        Returns:
            Dictionary containing advanced analysis results
        """
        if not self.enable_advanced_analysis:
            self.logger.info("Advanced analysis disabled, skipping")
            return {}
        
        self.logger.info("Running advanced analysis modules")
        
        advanced_results = {}
        
        try:
            # 1. Advanced Order Profile Analysis
            if self.advanced_order_analyzer:
                self.logger.info("Running advanced order profile analysis")
                advanced_order_results = self.advanced_order_analyzer.analyze_advanced_order_profile(enriched_data)
                advanced_results['advanced_order_analysis'] = advanced_order_results
            
            # 2. Picking Methodology Analysis
            if self.picking_analyzer:
                self.logger.info("Running picking methodology analysis")
                picking_results = self.picking_analyzer.analyze_picking_patterns(enriched_data)
                advanced_results['picking_analysis'] = picking_results
            
            # 3. Enhanced ABC-FMS Analysis
            if self.enhanced_abc_fms_analyzer and 'sku_profile_abc_fms' in basic_results:
                self.logger.info("Running enhanced ABC-FMS analysis")
                enhanced_abc_fms_results = self.enhanced_abc_fms_analyzer.analyze_enhanced_abc_fms(
                    basic_results['sku_profile_abc_fms']
                )
                advanced_results['enhanced_abc_fms_analysis'] = enhanced_abc_fms_results
            
            # 4. Category Performance Analysis
            if self.use_category_performance_analyzer and 'sku_profile_abc_fms' in basic_results:
                self.logger.info("Running category performance analysis")
                
                # Prepare data with category information
                sku_profile = basic_results['sku_profile_abc_fms'].copy()
                
                # Merge category information from enriched data
                category_mapping = enriched_data.groupby('Sku Code')['Category'].first().reset_index()
                sku_profile_with_category = sku_profile.merge(category_mapping, on='Sku Code', how='left')
                
                # Create analyzer instance with enriched SKU profile data
                category_analyzer = CategoryPerformanceAnalyzer(sku_profile_with_category)
                category_performance_results = category_analyzer.run_full_analysis()
                advanced_results['category_performance_analysis'] = category_performance_results
            
            self.logger.info("Advanced analysis completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in advanced analysis: {str(e)}")
            # Don't raise - allow basic analysis to continue
            self.logger.warning("Continuing with basic analysis only")
        
        return advanced_results
    
    def _generate_advanced_charts(self, basic_results: Dict[str, Any], 
                                advanced_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate advanced charts.
        
        Args:
            basic_results: Results from basic analysis
            advanced_results: Results from advanced analysis
            
        Returns:
            Dictionary of chart file paths
        """
        if not self.generate_advanced_charts or not self.advanced_chart_generator:
            return {}
        
        self.logger.info("Generating advanced charts")
        
        chart_paths = {}
        
        try:
            # 1. Multi-metric time series chart
            if 'advanced_order_analysis' in advanced_results:
                daily_metrics = advanced_results['advanced_order_analysis'].get('daily_multi_metrics')
                if daily_metrics is not None:
                    chart_path = self.advanced_chart_generator.create_multi_metric_time_series(
                        daily_metrics, advanced_results['advanced_order_analysis']
                    )
                    chart_paths['multi_metric_time_series'] = chart_path
            
            # 2. ABC-FMS 2D Classification Heatmap
            if 'enhanced_abc_fms_analysis' in advanced_results:
                chart_path = self.advanced_chart_generator.create_abc_fms_2d_heatmap(
                    advanced_results['enhanced_abc_fms_analysis']
                )
                chart_paths['abc_fms_2d_heatmap'] = chart_path
            
            # 3. Picking Analysis Chart
            if 'picking_analysis' in advanced_results:
                chart_path = self.advanced_chart_generator.create_picking_analysis_chart(
                    advanced_results['picking_analysis']
                )
                chart_paths['picking_analysis'] = chart_path
            
            # 4. Correlation Matrix Chart
            if 'advanced_order_analysis' in advanced_results:
                correlation_data = advanced_results['advanced_order_analysis'].get('correlation_analysis')
                if correlation_data:
                    chart_path = self.advanced_chart_generator.create_correlation_matrix_chart(correlation_data)
                    chart_paths['correlation_matrix'] = chart_path
            
            # 5. Advanced Percentile Chart
            if 'advanced_order_analysis' in advanced_results:
                percentile_data = advanced_results['advanced_order_analysis'].get('enhanced_percentile_analysis')
                peak_data = advanced_results['advanced_order_analysis'].get('peak_ratios')
                if percentile_data and peak_data:
                    chart_path = self.advanced_chart_generator.create_advanced_percentile_chart(
                        percentile_data, peak_data
                    )
                    chart_paths['advanced_percentile'] = chart_path
            
            # 6. Enhanced Multi-Line Order Trend Chart (for Word reports)
            if 'date_order_summary' in basic_results:
                chart_path = self.advanced_chart_generator.create_enhanced_order_trend_chart(
                    basic_results['date_order_summary']
                )
                if chart_path:
                    chart_paths['enhanced_order_trend'] = chart_path
            
            # 7. SKU Profile 2D Classification Chart (for Word reports)
            # Combine basic and advanced results for comprehensive data
            combined_data = {**basic_results, **advanced_results}
            chart_path = self.advanced_chart_generator.create_sku_profile_2d_classification_chart(combined_data)
            if chart_path:
                chart_paths['sku_profile_2d_classification'] = chart_path
            
            self.logger.info(f"Generated {len(chart_paths)} advanced charts")
            
        except Exception as e:
            self.logger.error(f"Error generating advanced charts: {str(e)}")
        
        return chart_paths
    
    def run_full_analysis(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete enhanced analysis pipeline.
        
        Args:
            file_path: Optional path to data file (uses default if None)
            
        Returns:
            Dictionary containing all analysis results
        """
        self.logger.info("Starting enhanced warehouse analysis pipeline")
        
        # Load data
        enriched_data = self.load_data()
        
        # Run basic analysis
        basic_results = self.run_basic_analysis(enriched_data)
        
        # Run advanced analysis
        advanced_results = self.run_advanced_analysis(enriched_data, basic_results)
        
        # Combine results
        combined_results = {**basic_results, **advanced_results}
        
        # Generate basic charts
        if self.generate_charts and self.chart_generator:
            self.logger.info("Generating basic charts")
            try:
                self.chart_generator.generate_all_charts(combined_results)
            except Exception as e:
                self.logger.error(f"Error generating basic charts: {str(e)}")
        
        # Generate advanced charts
        advanced_chart_paths = {}
        if self.generate_advanced_charts and self.advanced_chart_generator:
            advanced_chart_paths = self._generate_advanced_charts(basic_results, advanced_results)
        if advanced_chart_paths:
            combined_results['advanced_chart_paths'] = advanced_chart_paths
        
        # Generate LLM summaries
        if self.generate_llm_summaries and self.llm_integration:
            self.logger.info("Generating LLM summaries")
            try:
                llm_summaries = self.llm_integration.generate_all_summaries(combined_results)
                combined_results['llm_summaries'] = llm_summaries
            except Exception as e:
                self.logger.error(f"Error generating LLM summaries: {str(e)}")
        
        # Generate HTML report
        if self.generate_html_report and self.html_generator:
            self.logger.info("Generating HTML report")
            try:
                html_path = self.html_generator.generate_html_report(combined_results)
                combined_results['html_report_path'] = html_path
            except Exception as e:
                self.logger.error(f"Error generating HTML report: {str(e)}")
        
        # Generate Word report
        if self.generate_word_report and self.word_generator:
            self.logger.info("Generating Word report")
            try:
                word_path = self.word_generator.generate_word_report(combined_results)
                combined_results['word_report_path'] = word_path
            except Exception as e:
                self.logger.error(f"Error generating Word report: {str(e)}")
        
        # Export to Excel
        if self.generate_excel_export and self.excel_exporter:
            self.logger.info("Exporting to Excel")
            try:
                excel_path = self.excel_exporter.export_to_excel(combined_results)
                combined_results['excel_export_path'] = excel_path
            except Exception as e:
                self.logger.error(f"Error exporting to Excel: {str(e)}")
        
        self.logger.info("Enhanced warehouse analysis pipeline completed successfully")
        return combined_results


def run_enhanced_analysis(file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to run the enhanced analysis pipeline.
    
    Args:
        file_path: Optional path to data file
        
    Returns:
        Dictionary containing all analysis results
    """
    pipeline = EnhancedWarehouseAnalysisPipeline()
    return pipeline.run_full_analysis(file_path)


if __name__ == "__main__":
    # Run the enhanced analysis pipeline
    try:
        results = run_enhanced_analysis()
        logger.info("Enhanced analysis completed successfully")
        
        # Print summary of results
        print("\n" + "="*60)
        print("ENHANCED WAREHOUSE ANALYSIS SUMMARY")
        print("="*60)
        
        # Basic analysis summary
        if 'order_statistics' in results:
            stats = results['order_statistics']
            print(f"📊 Data Overview:")
            print(f"   • {stats.get('unique_dates', 'N/A')} days analyzed")
            print(f"   • {stats.get('unique_skus', 'N/A')} unique SKUs")
            print(f"   • {stats.get('total_order_lines', 'N/A')} order lines")
            print(f"   • {stats.get('total_case_equivalent', 0):,.0f} case equivalents")
        
        # Advanced analysis summary
        if 'advanced_order_analysis' in results:
            print(f"\n🔍 Advanced Analysis:")
            adv_analysis = results['advanced_order_analysis']
            
            if 'peak_ratios' in adv_analysis:
                peak_ratios = adv_analysis['peak_ratios']
                for metric, ratios in peak_ratios.items():
                    if 'peak_to_avg' in ratios:
                        print(f"   • {metric}: Peak/Avg ratio = {ratios['peak_to_avg']:.1f}x")
            
            if 'operational_complexity' in adv_analysis:
                complexity = adv_analysis['operational_complexity']
                print(f"   • Operational Complexity: {complexity.get('complexity_level', 'N/A')}")
        
        # Reports generated
        print(f"\n📋 Reports Generated:")
        if 'html_report_path' in results:
            print(f"   • HTML Report: {results['html_report_path']}")
        if 'word_report_path' in results:
            print(f"   • Word Report: {results['word_report_path']}")
        if 'excel_export_path' in results:
            print(f"   • Excel Export: {results['excel_export_path']}")
        
        # Charts generated
        if 'advanced_chart_paths' in results:
            charts = results['advanced_chart_paths']
            print(f"   • {len(charts)} Advanced Charts Generated")
        
        print("="*60)
        
    except Exception as e:
        logger.error(f"Enhanced analysis failed: {str(e)}")
        sys.exit(1)