#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analysis Integration for Warehouse Analysis Web App
Integrates the web interface with the existing modular warehouse analysis backend.
"""

import streamlit as st
import pandas as pd
import tempfile
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
import traceback

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import existing modular analysis components
try:
    from warehouse_analysis_modular.analyzers import OrderAnalyzer, SkuAnalyzer, CrossTabulationAnalyzer
    from warehouse_analysis_modular.reporting import ChartGenerator, LLMIntegration, HTMLReportGenerator, ExcelExporter
    from warehouse_analysis_modular.utils.helpers import setup_logging
    # Import enhanced analysis pipeline
    from warehouse_analysis_modular.enhanced_main import EnhancedWarehouseAnalysisPipeline
    BACKEND_AVAILABLE = True
    ENHANCED_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Backend modules not available: {e}")
    BACKEND_AVAILABLE = False
    ENHANCED_AVAILABLE = False

# Import data loading functionality
try:
    from data_loader import load_order_data, load_sku_master, enrich_order_data, load_and_enrich_data
    DATA_LOADER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Data loader not available: {e}")
    DATA_LOADER_AVAILABLE = False

logger = setup_logging() if BACKEND_AVAILABLE else logging.getLogger(__name__)


class WebAnalysisIntegrator:
    """Integrates web interface with warehouse analysis backend."""
    
    def __init__(self):
        self.temp_files = []
        self.analysis_results = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def run_analysis_pipeline(self, uploaded_file, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline with web-uploaded file and custom parameters.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            parameters: Analysis parameters from web interface
            
        Returns:
            Dictionary with analysis results and metadata
        """
        if not BACKEND_AVAILABLE:
            return self._create_mock_results()
        
        try:
            # Step 1: Prepare data
            enriched_data = self._prepare_data_from_upload(uploaded_file)
            self.logger.info(f"Data preparation completed. Shape: {enriched_data.shape if enriched_data is not None else 'None'}")
            
            if enriched_data is None or enriched_data.empty:
                raise Exception("Data preparation resulted in empty dataset")
            
            # Step 2: Update configuration with web parameters
            self._update_analysis_config(parameters)
            
            # Step 3: Run enhanced analysis pipeline if available
            if ENHANCED_AVAILABLE:
                try:
                    combined_results = self._run_enhanced_analysis(enriched_data, parameters)
                    self.logger.info(f"Enhanced analysis completed, result keys: {list(combined_results.keys())}")
                except Exception as enhanced_error:
                    self.logger.error(f"Enhanced analysis failed: {enhanced_error}")
                    self.logger.info("Falling back to basic analysis pipeline")
                    # Fallback to basic analysis
                    order_results = self._run_order_analysis(enriched_data, parameters)
                    sku_results = self._run_sku_analysis(enriched_data, parameters)
                    cross_tab_results = self._run_cross_tabulation_analysis(
                        sku_results.get('sku_profile_abc_fms'), parameters
                    )
                    combined_results = self._combine_analysis_results(
                        order_results, sku_results, cross_tab_results
                    )
            else:
                # Fallback to basic analysis
                order_results = self._run_order_analysis(enriched_data, parameters)
                sku_results = self._run_sku_analysis(enriched_data, parameters)
                cross_tab_results = self._run_cross_tabulation_analysis(
                    sku_results.get('sku_profile_abc_fms'), parameters
                )
                combined_results = self._combine_analysis_results(
                    order_results, sku_results, cross_tab_results
                )
            
            # Step 4: Generate outputs
            outputs = self._generate_outputs(combined_results, parameters)
            
            # Step 5: Prepare final response
            return {
                'success': True,
                'analysis_results': combined_results,
                'outputs': outputs,
                'metadata': self._create_metadata(combined_results, parameters),
                'message': 'Analysis completed successfully with advanced features!' if ENHANCED_AVAILABLE else 'Analysis completed successfully!'
            }
            
        except Exception as e:
            self.logger.error(f"Analysis pipeline failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            # Try to provide some basic analysis even if the main pipeline fails
            try:
                self.logger.info("Attempting basic fallback analysis...")
                basic_order_results = self._run_order_analysis(enriched_data, parameters)
                basic_sku_results = self._run_sku_analysis(enriched_data, parameters)
                
                if basic_order_results or basic_sku_results:
                    fallback_results = self._combine_analysis_results(
                        basic_order_results, basic_sku_results, {}
                    )
                    return {
                        'success': True,
                        'analysis_results': fallback_results,
                        'outputs': {},
                        'metadata': self._create_metadata(fallback_results, parameters),
                        'message': f'Analysis completed with basic features (advanced analysis failed: {str(e)})'
                    }
            except Exception as fallback_error:
                self.logger.error(f"Fallback analysis also failed: {fallback_error}")
            
            return {
                'success': False,
                'error': str(e),
                'analysis_results': {},
                'outputs': {},
                'metadata': {},
                'message': f'Analysis failed: {str(e)}'
            }
        finally:
            self._cleanup_temp_files()
    
    def _prepare_data_from_upload(self, uploaded_file) -> pd.DataFrame:
        """Prepare data from uploaded file for analysis."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
            self.temp_files.append(tmp_path)
        
        try:
            # Load data using existing data loader if available
            if DATA_LOADER_AVAILABLE:
                # Use the new file_path parameter instead of modifying global config
                enriched_data = load_and_enrich_data(tmp_path)
                return enriched_data
            else:
                # Manual data loading
                return self._manual_data_loading(tmp_path)
                
        except Exception as e:
            raise Exception(f"Data preparation failed: {str(e)}")
    
    def _manual_data_loading(self, file_path: str) -> pd.DataFrame:
        """Manual data loading when data_loader is not available."""
        # Load order data
        order_df = pd.read_excel(file_path, sheet_name="OrderData")
        sku_df = pd.read_excel(file_path, sheet_name="SkuMaster")
        
        # Basic enrichment
        order_merged = order_df.merge(sku_df, on="Sku Code", how="left")
        
        # Calculate enrichment fields
        order_merged["Total_Eaches"] = (
            order_merged["Qty in Eaches"].fillna(0) +
            order_merged["Qty in Cases"].fillna(0) * order_merged["Case Config"]
        )
        
        order_merged["Case_Equivalent"] = order_merged["Total_Eaches"] / order_merged["Case Config"]
        order_merged["Pallet_Equivalent"] = order_merged["Case_Equivalent"] / order_merged["Pallet Fit"]
        
        # Handle infinite values
        order_merged["Case_Equivalent"] = order_merged["Case_Equivalent"].replace([float('inf'), -float('inf')], 0)
        order_merged["Pallet_Equivalent"] = order_merged["Pallet_Equivalent"].replace([float('inf'), -float('inf')], 0)
        
        return order_merged
    
    def _update_analysis_config(self, parameters: Dict[str, Any]):
        """Update analysis configuration with web parameters."""
        if not BACKEND_AVAILABLE:
            return
            
        try:
            import config
            
            # Update ABC thresholds
            if 'abc_thresholds' in parameters:
                config.ABC_THRESHOLDS.update(parameters['abc_thresholds'])
            
            # Update FMS thresholds
            if 'fms_thresholds' in parameters:
                config.FMS_THRESHOLDS.update(parameters['fms_thresholds'])
            
            # Update other parameters as needed
            if 'analysis_options' in parameters:
                analysis_opts = parameters['analysis_options']
                if 'percentiles' in analysis_opts:
                    config.PERCENTILE_LEVELS = analysis_opts['percentiles']
                    
        except Exception as e:
            self.logger.warning(f"Could not update config: {str(e)}")
    
    def _run_order_analysis(self, enriched_data: pd.DataFrame, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run order analysis with optional FTE parameters."""
        if not BACKEND_AVAILABLE:
            return {}
            
        analyzer = OrderAnalyzer(enriched_data)
        
        # Extract FTE parameters if provided
        fte_parameters = None
        if parameters and 'fte_parameters' in parameters:
            fte_parameters = parameters['fte_parameters']
            
        return analyzer.run_full_analysis(fte_parameters=fte_parameters)
    
    def _run_sku_analysis(self, enriched_data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run SKU analysis with custom parameters."""
        if not BACKEND_AVAILABLE:
            return {}
            
        analyzer = SkuAnalyzer(enriched_data)
        return analyzer.run_full_analysis()
    
    def _run_cross_tabulation_analysis(self, sku_profile: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run cross-tabulation analysis."""
        if not BACKEND_AVAILABLE or sku_profile is None:
            return {}
            
        analyzer = CrossTabulationAnalyzer(sku_profile)
        return analyzer.run_full_analysis()
    
    def _run_enhanced_analysis(self, enriched_data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run enhanced analysis pipeline with advanced features."""
        if not ENHANCED_AVAILABLE:
            return {}
        
        try:
            # Initialize enhanced pipeline with parameters
            enable_advanced = parameters.get('enable_advanced_features', True)
            generate_charts = parameters.get('generate_charts', True)
            generate_reports = parameters.get('generate_reports', True)
            
            pipeline = EnhancedWarehouseAnalysisPipeline(
                generate_charts=generate_charts,
                generate_advanced_charts=True,
                generate_llm_summaries=True,
                generate_html_report=True,
                generate_word_report=generate_reports,
                generate_excel_export=True,
                run_advanced_analysis=enable_advanced
            )
            
            # Run basic analysis first
            basic_results = pipeline.run_basic_analysis(enriched_data)
            
            # Run advanced analysis if enabled
            enhanced_results = {}
            if enable_advanced:
                enhanced_results = pipeline.run_advanced_analysis(enriched_data, basic_results)
            
            # Combine basic and enhanced results - ensure proper structure for web display
            combined_results = {**basic_results, **enhanced_results}
            
            # Ensure order_statistics is properly mapped from basic results
            if 'statistics' in basic_results and 'order_statistics' not in combined_results:
                combined_results['order_statistics'] = basic_results['statistics']
                self.logger.info("Mapped 'statistics' to 'order_statistics' for web display compatibility")
            elif 'statistics' in combined_results and 'order_statistics' not in combined_results:
                combined_results['order_statistics'] = combined_results['statistics']
                self.logger.info("Mapped root-level 'statistics' to 'order_statistics' for web display compatibility")
            
            # Generate charts and reports
            if generate_charts:
                # Use the enhanced pipeline's chart generation method
                try:
                    chart_results = pipeline.generate_advanced_charts(basic_results, enhanced_results)
                    combined_results['chart_paths'] = chart_results
                except Exception as e:
                    self.logger.warning(f"Chart generation failed: {e}")
            
            # Note: Report generation is handled within the pipeline automatically
            # based on the initialization parameters
            
            return combined_results
            
        except Exception as e:
            self.logger.error(f"Enhanced analysis failed: {str(e)}")
            # Fallback to basic analysis
            order_results = self._run_order_analysis(enriched_data, parameters)
            sku_results = self._run_sku_analysis(enriched_data, parameters)
            cross_tab_results = self._run_cross_tabulation_analysis(
                sku_results.get('sku_profile_abc_fms'), parameters
            )
            return self._combine_analysis_results(order_results, sku_results, cross_tab_results)
    
    def _combine_analysis_results(self, order_results: Dict, sku_results: Dict, cross_tab_results: Dict) -> Dict[str, Any]:
        """Combine all analysis results."""
        combined = {}
        
        # Add order analysis results
        if order_results:
            combined.update({
                'date_order_summary': order_results.get('date_order_summary'),
                'sku_order_summary': order_results.get('sku_order_summary'),
                'percentile_profile': order_results.get('percentile_profile'),
                'order_statistics': order_results.get('statistics'),
                'demand_patterns': order_results.get('demand_patterns')
            })
        
        # Add SKU analysis results
        if sku_results:
            combined.update({
                'sku_profile_abc_fms': sku_results.get('sku_profile_abc_fms'),
                'sku_statistics': sku_results.get('statistics')
            })
        
        # Add cross-tabulation results
        if cross_tab_results:
            combined.update({
                'abc_fms_summary': cross_tab_results.get('abc_fms_summary'),
                'cross_tabulation_insights': cross_tab_results.get('insights')
            })
        
        # Remove None values
        return {k: v for k, v in combined.items() if v is not None}
    
    def _generate_outputs(self, analysis_results: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate output files based on parameters."""
        outputs = {}
        output_options = parameters.get('output_options', {})
        
        try:
            # Generate charts
            if output_options.get('generate_charts', False) and BACKEND_AVAILABLE:
                chart_generator = ChartGenerator()
                chart_paths = chart_generator.generate_all_charts(analysis_results)
                outputs['charts'] = chart_paths
            
            # Generate Excel export
            if output_options.get('generate_excel_export', False) and BACKEND_AVAILABLE:
                excel_exporter = ExcelExporter()
                excel_path = excel_exporter.export_to_excel(analysis_results)
                outputs['excel_file'] = excel_path
            
            # Generate LLM summaries
            if output_options.get('generate_llm_summaries', False) and BACKEND_AVAILABLE:
                llm_integration = LLMIntegration()
                summaries = llm_integration.generate_all_summaries(analysis_results)
                outputs['llm_summaries'] = summaries
            
            # Generate HTML report
            if output_options.get('generate_html_report', False) and BACKEND_AVAILABLE:
                html_generator = HTMLReportGenerator()
                chart_paths = outputs.get('charts', {})
                summaries = outputs.get('llm_summaries', {})
                html_path = html_generator.generate_report(analysis_results, chart_paths, summaries)
                outputs['html_report'] = html_path
            
            # Generate Word report
            if output_options.get('generate_word_report', False) and BACKEND_AVAILABLE:
                try:
                    from warehouse_analysis_modular.reporting.word_report import generate_word_report
                    word_path = generate_word_report(analysis_results)
                    outputs['word_report'] = word_path
                except ImportError:
                    self.logger.warning("Word report generation requires python-docx package")
                except Exception as e:
                    self.logger.error(f"Word report generation failed: {str(e)}")
                
        except Exception as e:
            self.logger.error(f"Output generation failed: {str(e)}")
            outputs['generation_errors'] = str(e)
        
        return outputs
    
    def _create_metadata(self, analysis_results: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata about the analysis."""
        metadata = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'parameters_used': parameters,
            'backend_available': BACKEND_AVAILABLE,
            'data_loader_available': DATA_LOADER_AVAILABLE
        }
        
        # Add data statistics if available
        if 'order_statistics' in analysis_results:
            stats = analysis_results['order_statistics']
            metadata.update({
                'total_order_lines': stats.get('total_order_lines', 0),
                'unique_skus': stats.get('unique_skus', 0),
                'unique_dates': stats.get('unique_dates', 0),
                'total_case_equivalent': stats.get('total_case_equivalent', 0)
            })
        
        return metadata
    
    def _create_mock_results(self) -> Dict[str, Any]:
        """Create mock results when backend is not available."""
        mock_data = {
            'success': True,
            'analysis_results': {
                'date_order_summary': pd.DataFrame({
                    'Date': pd.date_range('2025-01-01', periods=10),
                    'Total_Case_Equiv': [1000 + i*100 for i in range(10)],
                    'Distinct_Orders': [50 + i*5 for i in range(10)]
                }),
                'sku_profile_abc_fms': pd.DataFrame({
                    'Sku Code': [f'SKU{i:03d}' for i in range(20)],
                    'ABC': ['A'] * 5 + ['B'] * 8 + ['C'] * 7,
                    'FMS': ['F'] * 8 + ['M'] * 7 + ['S'] * 5,
                    'Total_Case_Equiv': [1000 - i*40 for i in range(20)]
                })
            },
            'outputs': {
                'charts': {'demo_chart': 'mock_chart.png'},
                'llm_summaries': {'summary': 'This is a mock analysis summary.'}
            },
            'metadata': {
                'timestamp': pd.Timestamp.now().isoformat(),
                'backend_available': False,
                'total_order_lines': 10000,
                'unique_skus': 20
            },
            'message': 'Mock analysis completed (backend not available)'
        }
        
        return mock_data
    
    def _cleanup_temp_files(self):
        """Clean up temporary files."""
        for file_path in self.temp_files:
            try:
                os.unlink(file_path)
            except Exception as e:
                self.logger.warning(f"Could not delete temp file {file_path}: {e}")
        self.temp_files.clear()


def run_web_analysis(uploaded_file, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to run analysis from web interface.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        parameters: Analysis parameters from web interface
        
    Returns:
        Dictionary with analysis results
    """
    integrator = WebAnalysisIntegrator()
    return integrator.run_analysis_pipeline(uploaded_file, parameters)


def create_progress_callback():
    """Create a progress callback for Streamlit."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(step: str, percentage: int):
        progress_bar.progress(percentage)
        status_text.text(f'{step}... {percentage}%')
    
    return update_progress


def display_analysis_status(result: Dict[str, Any]):
    """Display analysis results status."""
    if result.get('success', False):
        st.success(result.get('message', 'Analysis completed successfully!'))
        
        # Display metadata
        metadata = result.get('metadata', {})
        if metadata:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Order Lines", f"{metadata.get('total_order_lines', 0):,}")
            
            with col2:
                st.metric("Unique SKUs", f"{metadata.get('unique_skus', 0):,}")
            
            with col3:
                st.metric("Date Range", f"{metadata.get('unique_dates', 0)} days")
            
            with col4:
                st.metric("Case Equivalent", f"{metadata.get('total_case_equivalent', 0):,.0f}")
        
    else:
        st.error(f"Analysis failed: {result.get('message', 'Unknown error')}")
        
        if 'error' in result:
            with st.expander("Error Details"):
                st.code(result['error'])


# Demo function
def analysis_integration_demo():
    """Demo function for testing analysis integration."""
    st.header("🔧 Analysis Integration Demo")
    
    st.info(f"""
    **Backend Status:**
    - Backend Available: {'✅' if BACKEND_AVAILABLE else '❌'}
    - Data Loader Available: {'✅' if DATA_LOADER_AVAILABLE else '❌'}
    """)
    
    if st.button("Test Mock Analysis"):
        integrator = WebAnalysisIntegrator()
        result = integrator._create_mock_results()
        
        st.json(result['metadata'])
        
        if 'analysis_results' in result:
            for key, value in result['analysis_results'].items():
                if isinstance(value, pd.DataFrame):
                    st.subheader(f"Sample {key}")
                    st.dataframe(value.head(), use_container_width=True)


if __name__ == "__main__":
    analysis_integration_demo()