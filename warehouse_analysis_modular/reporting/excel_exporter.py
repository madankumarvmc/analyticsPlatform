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