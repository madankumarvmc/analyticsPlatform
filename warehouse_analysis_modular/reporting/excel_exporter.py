#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Excel Export Module

Handles Excel workbook creation and export for warehouse analysis results.
Extracted from the original Warehouse Analysis (2).py file.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Optional, List

# Import from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import EXCEL_OUTPUT_FILE, EXCEL_SHEETS, SUCCESS_MESSAGES
from warehouse_analysis_modular.utils.helpers import validate_dataframe, setup_logging

logger = setup_logging()


class ExcelExporter:
    """
    Handles export of analysis results to Excel workbooks.
    """
    
    def __init__(self, output_file: Optional[str] = None):
        """
        Initialize the Excel exporter.
        
        Args:
            output_file: Path for the output Excel file (defaults to config setting)
        """
        self.output_file = output_file or EXCEL_OUTPUT_FILE
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.logger.info(f"Excel exporter initialized with output: {self.output_file}")
    
    def prepare_dataframes(self, analysis_results: Dict) -> Dict[str, pd.DataFrame]:
        """
        Prepare and validate DataFrames for export.
        
        Args:
            analysis_results: Dictionary containing analysis results
            
        Returns:
            Dictionary of DataFrames ready for export
        """
        self.logger.info("Preparing DataFrames for Excel export")
        
        export_data = {}
        
        # Map analysis results to Excel sheet names
        sheet_mapping = {
            'date_order_summary': EXCEL_SHEETS['date_summary'],
            'sku_order_summary': EXCEL_SHEETS['sku_summary'],
            'percentile_profile': EXCEL_SHEETS['percentile'],
            'sku_profile_abc_fms': EXCEL_SHEETS['sku_abc_fms'],
            'abc_fms_summary': EXCEL_SHEETS['abc_fms_summary']
        }
        
        # Add advanced analysis sheet mappings
        advanced_mapping = {
            'advanced_order_analysis': 'Advanced Order Analysis',
            'picking_analysis': 'Picking Analysis',
            'enhanced_abc_fms_analysis': 'Enhanced ABC-FMS',
            'manpower_analysis': EXCEL_SHEETS['manpower_analysis']
        }
        
        for result_key, sheet_name in sheet_mapping.items():
            if result_key in analysis_results:
                df = analysis_results[result_key]
                if isinstance(df, pd.DataFrame) and not df.empty:
                    export_data[sheet_name] = df
                    self.logger.debug(f"Prepared {result_key} for sheet '{sheet_name}' ({len(df)} rows)")
                else:
                    self.logger.warning(f"Skipping {result_key} - invalid or empty DataFrame")
        
        # Handle special case where sku_order_summary might not exist
        if EXCEL_SHEETS['sku_summary'] not in export_data and 'sku_profile_abc_fms' in analysis_results:
            # Create a simplified SKU summary from the full profile
            sku_profile = analysis_results['sku_profile_abc_fms']
            if isinstance(sku_profile, pd.DataFrame) and not sku_profile.empty:
                sku_summary = sku_profile[['Sku Code', 'Total_Order_Lines', 'Total_Case_Equiv']].copy()
                sku_summary = sku_summary.rename(columns={
                    'Total_Order_Lines': 'Order_Lines',
                    'Total_Case_Equiv': 'Order_Volume_CE'
                })
                export_data[EXCEL_SHEETS['sku_summary']] = sku_summary
                self.logger.info("Created SKU summary from SKU profile data")
        
        # Process advanced analysis results
        self._process_advanced_analysis(analysis_results, advanced_mapping, export_data)
        
        self.logger.info(f"Prepared {len(export_data)} sheets for export")
        return export_data
    
    def format_dataframes(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Apply formatting to DataFrames before export.
        
        Args:
            dataframes: Dictionary of DataFrames to format
            
        Returns:
            Dictionary of formatted DataFrames
        """
        self.logger.info("Formatting DataFrames for Excel export")
        
        formatted_data = {}
        
        for sheet_name, df in dataframes.items():
            formatted_df = df.copy()
            
            # Apply sheet-specific formatting
            if sheet_name == EXCEL_SHEETS['date_summary']:
                formatted_df = self._format_date_summary(formatted_df)
            elif sheet_name == EXCEL_SHEETS['percentile']:
                formatted_df = self._format_percentile_profile(formatted_df)
            elif sheet_name == EXCEL_SHEETS['sku_abc_fms']:
                formatted_df = self._format_sku_profile(formatted_df)
            elif sheet_name == EXCEL_SHEETS['abc_fms_summary']:
                formatted_df = self._format_abc_fms_summary(formatted_df)
            elif sheet_name == EXCEL_SHEETS['sku_summary']:
                formatted_df = self._format_sku_summary(formatted_df)
            
            formatted_data[sheet_name] = formatted_df
        
        return formatted_data
    
    def _format_date_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format date summary DataFrame."""
        formatted_df = df.copy()
        
        # Ensure Date column is properly formatted
        if 'Date' in formatted_df.columns:
            formatted_df['Date'] = pd.to_datetime(formatted_df['Date']).dt.date
        
        # Round numeric columns
        numeric_cols = ['Total_Case_Equiv', 'Total_Pallet_Equiv']
        for col in numeric_cols:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].round(2)
        
        return formatted_df
    
    def _format_percentile_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format percentile profile DataFrame."""
        formatted_df = df.copy()
        
        # Round all numeric columns except the Percentile column
        for col in formatted_df.columns:
            if col != 'Percentile' and formatted_df[col].dtype in ['float64', 'float32']:
                formatted_df[col] = formatted_df[col].round(2)
        
        return formatted_df
    
    def _format_sku_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format SKU profile DataFrame."""
        formatted_df = df.copy()
        
        # Ensure integer columns are properly typed
        int_cols = ['Total_Order_Lines', 'Distinct_Movement_Days']
        for col in int_cols:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].fillna(0).astype(int)
        
        # Round percentage and rate columns
        float_cols = [
            'Pct_of_Total_Order_Lines', 'Cumulative_Pct_Lines',
            'Pct_of_Total_Case_Equiv', 'Cumulative_Pct_Volume',
            'FMS_Period_Pct', 'Orders_per_Movement_Day'
        ]
        for col in float_cols:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].round(2)
        
        # Round volume column
        if 'Total_Case_Equiv' in formatted_df.columns:
            formatted_df['Total_Case_Equiv'] = formatted_df['Total_Case_Equiv'].round(2)
        
        return formatted_df
    
    def _format_abc_fms_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format ABC-FMS summary DataFrame."""
        formatted_df = df.copy()
        
        # Ensure SKU count columns are integers
        sku_cols = ['SKU_F', 'SKU_M', 'SKU_S', 'SKU_Total']
        for col in sku_cols:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].fillna(0).astype(int)
        
        # Round percentage columns
        pct_cols = [col for col in formatted_df.columns if col.endswith('_pct')]
        for col in pct_cols:
            formatted_df[col] = formatted_df[col].round(2)
        
        # Round amount columns
        amt_cols = [col for col in formatted_df.columns if col.startswith(('Volume_', 'Line_')) and not col.endswith('_pct')]
        for col in amt_cols:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].round(2)
        
        return formatted_df
    
    def _format_sku_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format SKU summary DataFrame."""
        formatted_df = df.copy()
        
        # Ensure Order_Lines is integer
        if 'Order_Lines' in formatted_df.columns:
            formatted_df['Order_Lines'] = formatted_df['Order_Lines'].fillna(0).astype(int)
        
        # Round volume columns
        volume_cols = ['Order_Volume_CE', 'Total_Case_Equiv']
        for col in volume_cols:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].round(2)
        
        return formatted_df
    
    def _process_advanced_analysis(self, analysis_results: Dict, advanced_mapping: Dict, export_data: Dict):
        """Process advanced analysis results and add to export data."""
        
        # Process Advanced Order Analysis
        if 'advanced_order_analysis' in analysis_results:
            advanced_order = analysis_results['advanced_order_analysis']
            
            # Multi-metric correlations
            if 'correlation_analysis' in advanced_order:
                correlations = advanced_order['correlation_analysis'].get('key_correlations', {})
                if correlations:
                    corr_df = pd.DataFrame([
                        {'Metric Pair': 'Volume ↔ Lines', 'Correlation': correlations.get('volume_lines', 0)},
                        {'Metric Pair': 'Volume ↔ Customers', 'Correlation': correlations.get('volume_customers', 0)},
                        {'Metric Pair': 'Lines ↔ Customers', 'Correlation': correlations.get('lines_customers', 0)}
                    ])
                    export_data['Multi-Metric Correlations'] = corr_df
            
            # Daily metrics
            daily_metrics = advanced_order.get('daily_metrics', pd.DataFrame())
            if not daily_metrics.empty:
                export_data['Daily Multi-Metrics'] = daily_metrics
            
            # Enhanced percentiles
            if 'enhanced_percentile_analysis' in advanced_order:
                percentile_data = advanced_order['enhanced_percentile_analysis'].get('percentile_breakdown', {})
                if percentile_data:
                    percentile_df = pd.DataFrame([
                        {'Metric': key, 'Value': value} for key, value in percentile_data.items()
                    ])
                    export_data['Enhanced Percentiles'] = percentile_df
        
        # Process Picking Analysis
        if 'picking_analysis' in analysis_results:
            picking_analysis = analysis_results['picking_analysis']
            
            # Picking breakdown (check both old and new data structure)
            picking_breakdown = picking_analysis.get('picking_breakdown', {})
            if not picking_breakdown:
                # Try new data structure
                overall_patterns = picking_analysis.get('overall_picking_patterns', {})
                picking_summary = overall_patterns.get('picking_summary')
                if picking_summary is not None and not picking_summary.empty:
                    export_data['Picking Method Breakdown'] = picking_summary
            else:
                breakdown_data = []
                for picking_type, data in picking_breakdown.items():
                    breakdown_data.append({
                        'Picking Type': picking_type,
                        'Count': data.get('count', 0),
                        'Percentage': data.get('percentage', 0)
                    })
                if breakdown_data:
                    export_data['Picking Method Breakdown'] = pd.DataFrame(breakdown_data)
            
            # Category analysis (check both old and new data structure)
            category_analysis = picking_analysis.get('category_analysis', {})
            if not category_analysis:
                # Try new data structure with list format
                category_picking = picking_analysis.get('category_picking_analysis', {})
                if 'category_breakdown' in category_picking:
                    category_breakdown_list = category_picking['category_breakdown']
                    if isinstance(category_breakdown_list, list):
                        category_data = []
                        for category_item in category_breakdown_list:
                            if isinstance(category_item, dict):
                                category_data.append({
                                    'Category': category_item.get('category', 'Unknown'),
                                    'Total Lines': category_item.get('total_lines', 0),
                                    'Total Volume': category_item.get('total_volume', 0),
                                    'Piece Lines %': category_item.get('pcs_lines_percentage', 0),
                                    'Case Only %': category_item.get('case_only_lines_percentage', 0),
                                    'Operational Complexity': category_item.get('operational_complexity', 0)
                                })
                        if category_data:
                            export_data['Picking by Category'] = pd.DataFrame(category_data)
            elif category_analysis:
                # Old structure
                category_data = []
                for category, data in category_analysis.items():
                    category_data.append({
                        'Category': category,
                        'Case Only': data.get('Case_Only', 0),
                        'Piece Only': data.get('Piece_Only', 0),
                        'Mixed': data.get('Mixed', 0)
                    })
                if category_data:
                    export_data['Picking by Category'] = pd.DataFrame(category_data)
        
        # Process Enhanced ABC-FMS Analysis
        if 'enhanced_abc_fms_analysis' in analysis_results:
            enhanced_abc = analysis_results['enhanced_abc_fms_analysis']
            
            # Classification matrix (check both old and new data structure)
            matrix_data = enhanced_abc.get('classification_matrix')
            if not matrix_data:
                # Try new data structure
                matrix_data = enhanced_abc.get('classification_matrix_2d')
            
            if matrix_data:
                
                # SKU count matrix
                sku_count_matrix = matrix_data.get('sku_count_matrix', pd.DataFrame())
                if not sku_count_matrix.empty:
                    export_data['ABC-FMS SKU Count Matrix'] = sku_count_matrix
                
                # Volume matrix
                volume_matrix = matrix_data.get('volume_matrix', pd.DataFrame())
                if not volume_matrix.empty:
                    export_data['ABC-FMS Volume Matrix'] = volume_matrix
            
            # Enhanced insights
            if 'enhanced_insights' in enhanced_abc:
                insights = enhanced_abc['enhanced_insights']
                
                # Segmentation analysis
                segmentation = insights.get('segmentation_analysis', {})
                if segmentation:
                    seg_data = []
                    high_value = segmentation.get('high_value_segments', [])
                    for segment in high_value:
                        seg_data.append({'Segment': segment, 'Type': 'High Value'})
                    
                    if seg_data:
                        export_data['High Value Segments'] = pd.DataFrame(seg_data)
        
        # Process Category Performance Analysis
        if 'category_performance_analysis' in analysis_results:
            category_analysis = analysis_results['category_performance_analysis']
            
            # SKU Distribution Table
            if 'sku_distribution' in category_analysis:
                sku_dist = category_analysis['sku_distribution']
                if not sku_dist.empty:
                    export_data['Category SKU Distribution'] = sku_dist
            
            # Cases Distribution Table  
            if 'cases_distribution' in category_analysis:
                cases_dist = category_analysis['cases_distribution']
                if not cases_dist.empty:
                    export_data['Category Volume Distribution'] = cases_dist
            
            # Lines Distribution Table
            if 'lines_distribution' in category_analysis:
                lines_dist = category_analysis['lines_distribution']
                if not lines_dist.empty:
                    export_data['Category Lines Distribution'] = lines_dist
                    
            # Performance Summary
            if 'performance_summary' in category_analysis:
                perf_summary = category_analysis['performance_summary']
                if not perf_summary.empty:
                    export_data['Category Performance Summary'] = perf_summary
            
            # Slotting Recommendations
            if 'slotting_insights' in category_analysis:
                insights = category_analysis['slotting_insights']
                if insights and 'slotting_recommendations' in insights:
                    recommendations = insights['slotting_recommendations']
                    
                    # Create slotting recommendations worksheet
                    rec_data = []
                    
                    # High priority (dock proximity)
                    for category in recommendations.get('dock_proximity', []):
                        rec_data.append({
                            'Category': category,
                            'Recommendation': 'Dock Proximity',
                            'Priority': 'High',
                            'Reason': 'High volume and velocity - needs easy access'
                        })
                    
                    # Medium priority (structured bins)
                    for category in recommendations.get('structured_bins', []):
                        rec_data.append({
                            'Category': category,
                            'Recommendation': 'Structured Bins',
                            'Priority': 'Medium', 
                            'Reason': 'Moderate activity - needs organized storage'
                        })
                    
                    # Standard storage
                    for category in recommendations.get('standard_storage', []):
                        rec_data.append({
                            'Category': category,
                            'Recommendation': 'Standard Storage',
                            'Priority': 'Low',
                            'Reason': 'Low activity - standard placement acceptable'
                        })
                    
                    if rec_data:
                        export_data['Slotting Recommendations'] = pd.DataFrame(rec_data)
                
                # Key insights summary
                if 'key_findings' in insights:
                    findings = insights['key_findings']
                    insights_data = []
                    
                    if findings.get('top_volume_category'):
                        insights_data.append({
                            'Metric': 'Top Volume Category',
                            'Value': findings['top_volume_category'],
                            'Impact': 'Highest throughput contributor'
                        })
                    
                    if findings.get('top_velocity_category'): 
                        insights_data.append({
                            'Metric': 'Top Velocity Category',
                            'Value': findings['top_velocity_category'],
                            'Impact': 'Most frequently ordered'
                        })
                    
                    if findings.get('critical_abc_fms_classes'):
                        critical_classes = ', '.join(findings['critical_abc_fms_classes'])
                        insights_data.append({
                            'Metric': 'Critical ABC-FMS Classes',
                            'Value': critical_classes,
                            'Impact': 'Focus classes for efficiency optimization'
                        })
                    
                    if insights_data:
                        export_data['Category Key Insights'] = pd.DataFrame(insights_data)
        
        # Process Manpower Analysis
        if 'manpower_analysis' in analysis_results:
            manpower_analysis = analysis_results['manpower_analysis']
            
            # Daily FTE requirements
            daily_patterns = manpower_analysis.get('daily_staffing_analysis', {})
            daily_fte_data = daily_patterns.get('daily_fte_data')
            if daily_fte_data is not None and not daily_fte_data.empty:
                # Format daily FTE data for Excel
                fte_data = daily_fte_data[[
                    'Date', 'Total_Cases', 'Total_Lines', 'Total_Shipments',
                    'FTE_Required_Total', 'Utilization_Cases', 'Utilization_Lines'
                ]].copy()
                fte_data.columns = [
                    'Date', 'Daily_Cases', 'Daily_Lines', 'Daily_Shipments',
                    'FTE_Required', 'Cases_Utilization', 'Lines_Utilization'
                ]
                export_data['Daily FTE Requirements'] = fte_data
            
            # Manpower summary metrics
            summary_metrics = manpower_analysis.get('summary_metrics', {})
            peak_analysis = manpower_analysis.get('peak_analysis', {})
            cost_analysis = manpower_analysis.get('cost_analysis', {})
            
            if summary_metrics or peak_analysis or cost_analysis:
                summary_data = []
                
                # FTE metrics
                if summary_metrics:
                    summary_data.extend([
                        {'Metric': 'Recommended Core FTE', 'Value': summary_metrics.get('recommended_core_fte', 0)},
                        {'Metric': 'Peak FTE Requirement', 'Value': summary_metrics.get('peak_fte_requirement', 0)},
                        {'Metric': 'Average FTE Requirement', 'Value': summary_metrics.get('average_fte_requirement', 0)},
                        {'Metric': 'Flex Capacity Needed', 'Value': summary_metrics.get('flex_capacity_needed', 0)},
                        {'Metric': 'Peak Days (%)', 'Value': summary_metrics.get('peak_days_percentage', 0)}
                    ])
                
                # Cost metrics
                if cost_analysis:
                    summary_data.extend([
                        {'Metric': 'Monthly Labor Budget ($)', 'Value': round(cost_analysis.get('total_monthly_labor', 0), 0)},
                        {'Metric': 'Cost per Case ($)', 'Value': round(cost_analysis.get('cost_per_case', 0), 2)},
                        {'Metric': 'Annual Labor Budget ($)', 'Value': round(cost_analysis.get('annual_labor_budget', 0), 0)}
                    ])
                
                if summary_data:
                    export_data['Manpower Summary'] = pd.DataFrame(summary_data)
            
            # Category labor breakdown
            category_analysis = manpower_analysis.get('category_labor_analysis', {})
            category_breakdown = category_analysis.get('category_breakdown')
            if category_breakdown is not None and not category_breakdown.empty:
                category_data = category_breakdown[[
                    'Category', 'Daily_Cases', 'Daily_Lines', 'Labor_Coefficient',
                    'FTE_Total_Required', 'FTE_Percentage'
                ]].copy()
                category_data.columns = [
                    'Category', 'Avg_Daily_Cases', 'Avg_Daily_Lines', 'Labor_Complexity',
                    'FTE_Required', 'FTE_Percentage'
                ]
                export_data['Category Labor Allocation'] = category_data
            
            # Shift planning recommendations
            shift_planning = manpower_analysis.get('shift_planning', {})
            if shift_planning:
                shift_data = []
                
                core_staffing = shift_planning.get('core_staffing', {})
                peak_staffing = shift_planning.get('peak_staffing', {})
                flex_requirements = shift_planning.get('flexibility_requirements', {})
                
                if core_staffing:
                    shift_data.extend([
                        {'Planning_Type': 'Core Staffing', 'Shift_1_Day': core_staffing.get('shift_1_day', 0),
                         'Shift_2_Evening': core_staffing.get('shift_2_evening', 0), 'Total': core_staffing.get('total_core', 0)},
                    ])
                
                if peak_staffing:
                    shift_data.extend([
                        {'Planning_Type': 'Peak Staffing', 'Shift_1_Day': peak_staffing.get('shift_1_day', 0),
                         'Shift_2_Evening': peak_staffing.get('shift_2_evening', 0), 'Total': peak_staffing.get('total_peak', 0)},
                    ])
                
                if flex_requirements:
                    shift_data.extend([
                        {'Planning_Type': 'Flexibility Needs', 'Shift_1_Day': flex_requirements.get('flex_fte_needed', 0),
                         'Shift_2_Evening': flex_requirements.get('overtime_hours_per_week', 0), 'Total': flex_requirements.get('temp_worker_days_per_month', 0)}
                    ])
                
                if shift_data:
                    export_data['Shift Planning'] = pd.DataFrame(shift_data)

        # Process Enhanced Chart Data (following expansion framework)
        if 'chart_paths' in analysis_results:
            chart_paths = analysis_results['chart_paths']
            
            # Enhanced Order Trend Chart Data
            if 'enhanced_order_trend' in chart_paths:
                # If we have the date_order_summary, add it as chart data
                if 'date_order_summary' in analysis_results:
                    date_data = analysis_results['date_order_summary']
                    if not date_data.empty:
                        # Add enhanced trend chart data
                        trend_data = date_data.copy()
                        trend_data['Chart_Type'] = 'Enhanced Order Trend'
                        export_data['Enhanced Order Trend Data'] = trend_data
            
            # SKU Profile 2D Classification Chart Data  
            if 'sku_profile_2d_classification' in chart_paths:
                # Create chart data from ABC-FMS analysis
                if 'abc_fms_summary' in analysis_results:
                    abc_fms_data = analysis_results['abc_fms_summary']
                    if not abc_fms_data.empty:
                        # Create 2D classification data for Excel
                        classification_data = []
                        
                        for _, row in abc_fms_data.iterrows():
                            abc = row.get('ABC', '')
                            if abc and abc != 'Grand Total':
                                for fms in ['F', 'M', 'S']:
                                    sku_col = f'SKU_{fms}'
                                    vol_col = f'Volume_{fms}'
                                    line_col = f'Line_{fms}'
                                    
                                    if all(col in row for col in [sku_col, vol_col, line_col]):
                                        classification_data.append({
                                            'ABC_FMS_Class': f'{abc}{fms}',
                                            'SKU_Count': row[sku_col],
                                            'Volume_Total': row[vol_col],
                                            'Line_Total': row[line_col],
                                            'Chart_Type': 'SKU Profile 2D Classification'
                                        })
                        
                        if classification_data:
                            export_data['SKU 2D Classification Data'] = pd.DataFrame(classification_data)
        
        # Chart Generation Summary
        if 'chart_paths' in analysis_results:
            chart_paths = analysis_results['chart_paths']
            if chart_paths:
                chart_summary = []
                for chart_name, chart_path in chart_paths.items():
                    chart_summary.append({
                        'Chart_Name': chart_name,
                        'File_Path': chart_path,
                        'Chart_Type': 'Advanced Chart',
                        'Status': 'Generated'
                    })
                
                if chart_summary:
                    export_data['Generated Charts Summary'] = pd.DataFrame(chart_summary)
    
    def export_to_excel(self, analysis_results: Dict, 
                       include_formatting: bool = True) -> str:
        """
        Export analysis results to Excel workbook.
        
        Args:
            analysis_results: Dictionary containing analysis results
            include_formatting: Whether to apply formatting to the data
            
        Returns:
            Path to the exported Excel file
        """
        self.logger.info(f"Exporting analysis results to Excel: {self.output_file}")
        
        try:
            # Prepare DataFrames
            dataframes = self.prepare_dataframes(analysis_results)
            
            if not dataframes:
                raise ValueError("No valid DataFrames found for export")
            
            # Apply formatting if requested
            if include_formatting:
                dataframes = self.format_dataframes(dataframes)
            
            # Export to Excel
            with pd.ExcelWriter(self.output_file, engine='openpyxl') as writer:
                for sheet_name, df in dataframes.items():
                    self.logger.debug(f"Writing sheet '{sheet_name}' with {len(df)} rows")
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            self.logger.info(SUCCESS_MESSAGES['analysis_complete'].format(self.output_file))
            return str(self.output_file)
            
        except Exception as e:
            self.logger.error(f"Failed to export to Excel: {str(e)}")
            raise
    
    def export_individual_sheets(self, analysis_results: Dict, 
                                output_dir: Optional[str] = None) -> List[str]:
        """
        Export each analysis result to a separate Excel file.
        
        Args:
            analysis_results: Dictionary containing analysis results
            output_dir: Directory for output files (defaults to current directory)
            
        Returns:
            List of paths to exported files
        """
        self.logger.info("Exporting individual Excel sheets")
        
        if output_dir is None:
            output_dir = Path(self.output_file).parent
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = []
        dataframes = self.prepare_dataframes(analysis_results)
        
        for sheet_name, df in dataframes.items():
            # Create filename from sheet name
            safe_name = sheet_name.replace(' ', '_').replace('(', '').replace(')', '')
            filename = output_dir / f"{safe_name}.xlsx"
            
            try:
                df.to_excel(filename, index=False)
                exported_files.append(str(filename))
                self.logger.debug(f"Exported {sheet_name} to {filename}")
            except Exception as e:
                self.logger.warning(f"Failed to export {sheet_name}: {e}")
        
        self.logger.info(f"Exported {len(exported_files)} individual Excel files")
        return exported_files
    
    def validate_export_data(self, analysis_results: Dict) -> bool:
        """
        Validate that the analysis results contain exportable data.
        
        Args:
            analysis_results: Dictionary containing analysis results
            
        Returns:
            True if data is valid for export
        """
        dataframes = self.prepare_dataframes(analysis_results)
        
        if not dataframes:
            self.logger.error("No valid DataFrames found for export")
            return False
        
        # Check that we have at least some core data
        core_sheets = [EXCEL_SHEETS['date_summary'], EXCEL_SHEETS['sku_abc_fms']]
        has_core_data = any(sheet in dataframes for sheet in core_sheets)
        
        if not has_core_data:
            self.logger.error("Missing core analysis data for export")
            return False
        
        # Validate individual DataFrames
        for sheet_name, df in dataframes.items():
            try:
                validate_dataframe(df, [], min_rows=1, name=sheet_name)
            except ValueError as e:
                self.logger.error(f"Invalid DataFrame for {sheet_name}: {e}")
                return False
        
        self.logger.info("Export data validation passed")
        return True


def export_to_excel(analysis_results: Dict, 
                   output_file: Optional[str] = None,
                   include_formatting: bool = True) -> str:
    """
    Convenience function to export analysis results to Excel.
    
    Args:
        analysis_results: Dictionary containing analysis results
        output_file: Path for output file
        include_formatting: Whether to apply formatting
        
    Returns:
        Path to exported Excel file
    """
    exporter = ExcelExporter(output_file)
    return exporter.export_to_excel(analysis_results, include_formatting)