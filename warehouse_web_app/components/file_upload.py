#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File Upload Component for Warehouse Analysis Web App
Handles Excel file upload, validation, and preview functionality.
"""

import streamlit as st
import pandas as pd
import tempfile
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging
from datetime import datetime, timedelta
import io

# Import web config
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config_web import (
    MAX_FILE_SIZE_MB, ALLOWED_EXTENSIONS, REQUIRED_ORDER_COLUMNS, 
    REQUIRED_SKU_COLUMNS, ERROR_MESSAGES, SUCCESS_MESSAGES,
    MIN_ROWS_REQUIRED, MIN_SKUS_REQUIRED
)

logger = logging.getLogger(__name__)


def create_excel_template() -> bytes:
    """
    Create an Excel template with sample data for OrderData and SkuMaster sheets.
    
    Returns:
        bytes: Excel file content as bytes
    """
    # Create sample OrderData
    base_date = datetime(2025, 2, 1)
    order_data = []
    
    # Sample SKU codes and categories
    sample_skus = [
        ('SKU001', 'BI'), ('SKU002', 'SX'), ('SKU003', 'ND'), ('SKU004', 'CG'), 
        ('SKU005', 'BI'), ('SKU006', 'AT'), ('SKU007', 'SX'), ('SKU008', 'TS'),
        ('SKU009', 'AG'), ('SKU010', 'DT')
    ]
    
    # Generate sample order data (50 rows)
    for i in range(50):
        sku_code, category = sample_skus[i % len(sample_skus)]
        order_data.append({
            'Date': (base_date + timedelta(days=i % 30)).strftime('%Y-%m-%d'),
            'Shipment No.': f'SH{1000 + i}',
            'Order No.': f'ORD{2000 + i}',
            'Sku Code': sku_code,
            'Qty in Cases': max(1, (i % 10) * 2),
            'Qty in Eaches': (i % 24) * 5
        })
    
    order_df = pd.DataFrame(order_data)
    
    # Create sample SkuMaster data
    sku_master_data = []
    for sku_code, category in sample_skus:
        sku_master_data.append({
            'Sku Code': sku_code,
            'Category': category,
            'Case Config': 12 if category in ['BI', 'SX'] else 24,  # Items per case
            'Pallet Fit': 48 if category in ['BI', 'SX'] else 64    # Cases per pallet
        })
    
    sku_df = pd.DataFrame(sku_master_data)
    
    # Create Excel file in memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Write OrderData sheet
        order_df.to_excel(writer, sheet_name='OrderData', index=False)
        
        # Write SkuMaster sheet
        sku_df.to_excel(writer, sheet_name='SkuMaster', index=False)
        
        # Format the sheets
        workbook = writer.book
        
        # Format OrderData sheet
        order_sheet = workbook['OrderData']
        for col in order_sheet.columns:
            max_length = max(len(str(cell.value)) for cell in col if cell.value is not None)
            order_sheet.column_dimensions[col[0].column_letter].width = min(max_length + 2, 20)
        
        # Format SkuMaster sheet
        sku_sheet = workbook['SkuMaster']
        for col in sku_sheet.columns:
            max_length = max(len(str(cell.value)) for cell in col if cell.value is not None)
            sku_sheet.column_dimensions[col[0].column_letter].width = min(max_length + 2, 20)
    
    output.seek(0)
    return output.getvalue()


class FileUploadValidator:
    """Handles file upload validation and preview functionality."""
    
    def __init__(self):
        self.uploaded_file = None
        self.validation_results = {}
        self.data_preview = {}
    
    def upload_file_component(self) -> Optional[Any]:
        """
        Create the file upload component with validation.
        
        Returns:
            Uploaded file object or None
        """
        st.subheader("📁 Data Upload")
        
        # Template download section
        st.info("📋 **First time using the tool?** Download our Excel template below to see the required data structure.")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("""
            **Template includes:**
            - **OrderData sheet**: Date, Shipment No., Order No., Sku Code, Qty in Cases, Qty in Eaches
            - **SkuMaster sheet**: Sku Code, Category, Case Config, Pallet Fit
            - **Sample data**: 50 sample order lines with 10 sample SKUs
            """)
        with col2:
            # Create template download button
            template_data = create_excel_template()
            st.download_button(
                label="📥 Download Template",
                data=template_data,
                file_name="warehouse_analysis_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Download Excel template with sample data structure",
                use_container_width=True,
                type="primary"
            )
        
        st.markdown("---")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose Excel file",
            type=ALLOWED_EXTENSIONS,
            help=f"Maximum file size: {MAX_FILE_SIZE_MB}MB. Required sheets: OrderData, SkuMaster",
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            # File size validation
            if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
                st.error(ERROR_MESSAGES['file_too_large'])
                return None
            
            # Store in session state
            st.session_state.uploaded_file = uploaded_file
            
            # Show file info
            self._display_file_info(uploaded_file)
            
            # Validate file structure
            validation_success = self._validate_file_structure(uploaded_file)
            
            if validation_success:
                st.success(SUCCESS_MESSAGES['file_uploaded'])
                
                # Show data preview
                self._display_data_preview()
                
                return uploaded_file
            else:
                return None
        
        return None
    
    def _display_file_info(self, uploaded_file) -> None:
        """Display information about the uploaded file."""
        file_size_mb = uploaded_file.size / (1024 * 1024)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Filename", uploaded_file.name)
        with col2:
            st.metric("📊 File Size", f"{file_size_mb:.2f} MB")
        with col3:
            st.metric("📋 Type", uploaded_file.type)
    
    def _validate_file_structure(self, uploaded_file) -> bool:
        """
        Validate the Excel file structure and required sheets.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Read Excel file and get sheet names
                excel_file = pd.ExcelFile(tmp_path)
                available_sheets = excel_file.sheet_names
                
                # Check for required sheets
                required_sheets = ['OrderData', 'SkuMaster']
                missing_sheets = [sheet for sheet in required_sheets if sheet not in available_sheets]
                
                if missing_sheets:
                    st.error(f"{ERROR_MESSAGES['missing_sheets']} Missing: {', '.join(missing_sheets)}")
                    st.info(f"Available sheets: {', '.join(available_sheets)}")
                    return False
                
                # Validate sheet contents
                validation_results = {}
                
                # Validate OrderData sheet
                order_validation = self._validate_order_data(excel_file)
                validation_results['OrderData'] = order_validation
                
                # Validate SkuMaster sheet
                sku_validation = self._validate_sku_master(excel_file)
                validation_results['SkuMaster'] = sku_validation
                
                # Store validation results
                self.validation_results = validation_results
                
                # Check if all validations passed
                all_valid = all(result['valid'] for result in validation_results.values())
                
                if all_valid:
                    st.success(SUCCESS_MESSAGES['data_validated'])
                    self._display_validation_summary(validation_results)
                else:
                    self._display_validation_errors(validation_results)
                
                return all_valid
                
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)
                
        except Exception as e:
            st.error(f"Error reading Excel file: {str(e)}")
            logger.error(f"File validation error: {str(e)}")
            return False
    
    def _validate_order_data(self, excel_file) -> Dict[str, Any]:
        """Validate OrderData sheet."""
        try:
            order_df = pd.read_excel(excel_file, sheet_name='OrderData')
            
            # Check for required columns
            missing_columns = [col for col in REQUIRED_ORDER_COLUMNS if col not in order_df.columns]
            
            validation_result = {
                'valid': len(missing_columns) == 0 and len(order_df) >= MIN_ROWS_REQUIRED,
                'row_count': len(order_df),
                'column_count': len(order_df.columns),
                'missing_columns': missing_columns,
                'available_columns': list(order_df.columns),
                'data_preview': order_df.head(3) if len(order_df) > 0 else pd.DataFrame()
            }
            
            # Store preview data
            self.data_preview['OrderData'] = order_df.head(10)
            
            return validation_result
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'row_count': 0,
                'column_count': 0,
                'missing_columns': REQUIRED_ORDER_COLUMNS,
                'available_columns': [],
                'data_preview': pd.DataFrame()
            }
    
    def _validate_sku_master(self, excel_file) -> Dict[str, Any]:
        """Validate SkuMaster sheet."""
        try:
            sku_df = pd.read_excel(excel_file, sheet_name='SkuMaster')
            
            # Check for required columns
            missing_columns = [col for col in REQUIRED_SKU_COLUMNS if col not in sku_df.columns]
            
            validation_result = {
                'valid': len(missing_columns) == 0 and len(sku_df) >= MIN_SKUS_REQUIRED,
                'row_count': len(sku_df),
                'column_count': len(sku_df.columns),
                'missing_columns': missing_columns,
                'available_columns': list(sku_df.columns),
                'data_preview': sku_df.head(3) if len(sku_df) > 0 else pd.DataFrame()
            }
            
            # Store preview data
            self.data_preview['SkuMaster'] = sku_df.head(10)
            
            return validation_result
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'row_count': 0,
                'column_count': 0,
                'missing_columns': REQUIRED_SKU_COLUMNS,
                'available_columns': [],
                'data_preview': pd.DataFrame()
            }
    
    def _display_validation_summary(self, validation_results: Dict[str, Any]) -> None:
        """Display validation summary for valid files."""
        st.subheader("📊 Data Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**OrderData Sheet:**")
            order_data = validation_results['OrderData']
            st.write(f"- Rows: {order_data['row_count']:,}")
            st.write(f"- Columns: {order_data['column_count']}")
            st.write(f"- Status: ✅ Valid")
        
        with col2:
            st.write("**SkuMaster Sheet:**")
            sku_data = validation_results['SkuMaster']
            st.write(f"- Rows: {sku_data['row_count']:,}")
            st.write(f"- Columns: {sku_data['column_count']}")
            st.write(f"- Status: ✅ Valid")
    
    def _display_validation_errors(self, validation_results: Dict[str, Any]) -> None:
        """Display validation errors for invalid files."""
        st.subheader("❌ Validation Errors")
        
        for sheet_name, result in validation_results.items():
            if not result['valid']:
                st.error(f"**{sheet_name} Sheet Issues:**")
                
                if 'error' in result:
                    st.write(f"- Error: {result['error']}")
                
                if result['missing_columns']:
                    st.write(f"- Missing columns: {', '.join(result['missing_columns'])}")
                
                if result['row_count'] < MIN_ROWS_REQUIRED:
                    st.write(f"- Insufficient data: {result['row_count']} rows (minimum: {MIN_ROWS_REQUIRED})")
                
                if result['available_columns']:
                    # Only show column info in debug mode
                    from config_web import is_debug_mode
                    if is_debug_mode():
                        st.write(f"- Available columns: {', '.join(result['available_columns'])}")
                    else:
                        st.write(f"- Found {len(result['available_columns'])} columns in {sheet_name} sheet")
    
    def _display_data_preview(self) -> None:
        """Display preview of the uploaded data."""
        if not self.data_preview:
            return
        
        st.subheader("👀 Data Preview")
        
        tab1, tab2 = st.tabs(["OrderData", "SkuMaster"])
        
        with tab1:
            if 'OrderData' in self.data_preview:
                st.write("**OrderData Sample (first 10 rows):**")
                st.dataframe(
                    self.data_preview['OrderData'], 
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.write("No OrderData preview available")
        
        with tab2:
            if 'SkuMaster' in self.data_preview:
                st.write("**SkuMaster Sample (first 10 rows):**")
                st.dataframe(
                    self.data_preview['SkuMaster'], 
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.write("No SkuMaster preview available")
    
    def get_validation_status(self) -> bool:
        """
        Check if the current file passed validation.
        
        Returns:
            True if file is valid, False otherwise
        """
        if not self.validation_results:
            return False
        
        return all(result['valid'] for result in self.validation_results.values())
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the uploaded data.
        
        Returns:
            Dictionary with data summary information
        """
        if not self.validation_results:
            return {}
        
        summary = {}
        for sheet_name, result in self.validation_results.items():
            if result['valid']:
                summary[sheet_name] = {
                    'rows': result['row_count'],
                    'columns': result['column_count']
                }
        
        return summary


def create_file_upload_section() -> Tuple[Optional[Any], bool]:
    """
    Create the complete file upload section with validation.
    
    Returns:
        Tuple of (uploaded_file, is_valid)
    """
    validator = FileUploadValidator()
    uploaded_file = validator.upload_file_component()
    is_valid = validator.get_validation_status()
    
    # Store validator in session state for access by other components
    st.session_state.file_validator = validator
    
    return uploaded_file, is_valid


# Example usage component
def file_upload_demo():
    """Demo component for testing file upload functionality."""
    st.header("📁 File Upload Demo")
    
    uploaded_file, is_valid = create_file_upload_section()
    
    if uploaded_file and is_valid:
        st.success("File is ready for analysis!")
        
        # Show data summary
        if hasattr(st.session_state, 'file_validator'):
            summary = st.session_state.file_validator.get_data_summary()
            st.json(summary)
    elif uploaded_file and not is_valid:
        st.error("Please fix the validation errors before proceeding.")
    else:
        st.info("Please upload an Excel file to get started.")


if __name__ == "__main__":
    # Run demo if executed directly
    file_upload_demo()