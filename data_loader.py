#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Loading and Enrichment Module for Warehouse Analysis Tool

This module handles:
- Loading order data and SKU master data from Excel
- Data validation and cleaning
- Calculating enrichment fields (case equivalents, pallet equivalents)
- Merging order data with SKU master data
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

from config import (
    DATA_FILE_PATH, ORDER_DATA_SHEET, SKU_MASTER_SHEET,
    ORDER_DATA_COLUMNS, SKU_MASTER_COLUMNS,
    MIN_ROWS_REQUIRED, MIN_SKUS_REQUIRED, MIN_DATES_REQUIRED
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_order_data() -> pd.DataFrame:
    """
    Load order data from Excel file.
    
    Returns:
        pd.DataFrame: Raw order data with columns:
                     Date, Shipment No., Order No., Sku Code, 
                     Qty in Cases, Qty in Eaches
    
    Raises:
        FileNotFoundError: If the data file doesn't exist
        ValueError: If required columns are missing
    """
    try:
        logger.info(f"Loading order data from {DATA_FILE_PATH}, sheet: {ORDER_DATA_SHEET}")
        
        order_df = pd.read_excel(DATA_FILE_PATH, sheet_name=ORDER_DATA_SHEET)
        
        # Validate required columns exist
        expected_cols = set(ORDER_DATA_COLUMNS.values())
        actual_cols = set(order_df.columns)
        missing_cols = expected_cols - actual_cols
        
        if missing_cols:
            raise ValueError(f"Missing required columns in order data: {missing_cols}")
        
        # Basic data validation
        if len(order_df) < MIN_ROWS_REQUIRED:
            raise ValueError(f"Order data has insufficient rows: {len(order_df)} < {MIN_ROWS_REQUIRED}")
        
        # Convert Date column to datetime if it's not already
        if ORDER_DATA_COLUMNS['date'] in order_df.columns:
            order_df[ORDER_DATA_COLUMNS['date']] = pd.to_datetime(order_df[ORDER_DATA_COLUMNS['date']])
        
        logger.info(f"Successfully loaded {len(order_df)} order records")
        return order_df
        
    except FileNotFoundError:
        logger.error(f"Data file not found: {DATA_FILE_PATH}")
        raise
    except Exception as e:
        logger.error(f"Error loading order data: {str(e)}")
        raise


def load_sku_master() -> pd.DataFrame:
    """
    Load SKU master data from Excel file.
    
    Returns:
        pd.DataFrame: SKU master data with columns:
                     Sku Code, Category, Case Config, Pallet Fit
    
    Raises:
        FileNotFoundError: If the data file doesn't exist
        ValueError: If required columns are missing
    """
    try:
        logger.info(f"Loading SKU master from {DATA_FILE_PATH}, sheet: {SKU_MASTER_SHEET}")
        
        sku_df = pd.read_excel(DATA_FILE_PATH, sheet_name=SKU_MASTER_SHEET)
        
        # Validate required columns exist
        expected_cols = set(SKU_MASTER_COLUMNS.values())
        actual_cols = set(sku_df.columns)
        missing_cols = expected_cols - actual_cols
        
        if missing_cols:
            raise ValueError(f"Missing required columns in SKU master: {missing_cols}")
        
        # Basic data validation
        if len(sku_df) < MIN_SKUS_REQUIRED:
            raise ValueError(f"SKU master has insufficient rows: {len(sku_df)} < {MIN_SKUS_REQUIRED}")
        
        # Validate numeric fields
        numeric_cols = [SKU_MASTER_COLUMNS['case_config'], SKU_MASTER_COLUMNS['pallet_fit']]
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(sku_df[col]):
                logger.warning(f"Converting {col} to numeric")
                sku_df[col] = pd.to_numeric(sku_df[col], errors='coerce')
        
        # Check for missing critical numeric values
        for col in numeric_cols:
            missing_count = sku_df[col].isna().sum()
            if missing_count > 0:
                logger.warning(f"Found {missing_count} missing values in {col}")
        
        logger.info(f"Successfully loaded {len(sku_df)} SKU records")
        return sku_df
        
    except FileNotFoundError:
        logger.error(f"Data file not found: {DATA_FILE_PATH}")
        raise
    except Exception as e:
        logger.error(f"Error loading SKU master: {str(e)}")
        raise


def calculate_enrichment_fields(order_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate enrichment fields for merged order data.
    
    Args:
        order_merged (pd.DataFrame): Merged order data with SKU master
    
    Returns:
        pd.DataFrame: Order data with additional calculated fields:
                     Total_Eaches, Case_Equivalent, Pallet_Equivalent
    """
    logger.info("Calculating enrichment fields")
    
    # Make a copy to avoid modifying original data
    enriched_df = order_merged.copy()
    
    # Get column names
    qty_eaches_col = ORDER_DATA_COLUMNS['qty_eaches']
    qty_cases_col = ORDER_DATA_COLUMNS['qty_cases']
    case_config_col = SKU_MASTER_COLUMNS['case_config']
    pallet_fit_col = SKU_MASTER_COLUMNS['pallet_fit']
    
    # Calculate total eaches for every order line
    enriched_df["Total_Eaches"] = (
        enriched_df[qty_eaches_col].fillna(0) +
        enriched_df[qty_cases_col].fillna(0) * enriched_df[case_config_col]
    )
    
    # Convert to case equivalent
    # Avoid division by zero
    case_config_safe = enriched_df[case_config_col].replace(0, np.nan)
    enriched_df["Case_Equivalent"] = enriched_df["Total_Eaches"] / case_config_safe
    
    # Convert to pallet equivalent
    # Avoid division by zero
    pallet_fit_safe = enriched_df[pallet_fit_col].replace(0, np.nan)
    enriched_df["Pallet_Equivalent"] = enriched_df["Case_Equivalent"] / pallet_fit_safe
    
    # Handle any infinite or very large values
    enriched_df["Case_Equivalent"] = enriched_df["Case_Equivalent"].replace([np.inf, -np.inf], np.nan)
    enriched_df["Pallet_Equivalent"] = enriched_df["Pallet_Equivalent"].replace([np.inf, -np.inf], np.nan)
    
    # Log summary statistics
    logger.info(f"Enrichment summary:")
    logger.info(f"  Total Eaches - Mean: {enriched_df['Total_Eaches'].mean():.2f}, Max: {enriched_df['Total_Eaches'].max():.2f}")
    logger.info(f"  Case Equivalent - Mean: {enriched_df['Case_Equivalent'].mean():.2f}, Max: {enriched_df['Case_Equivalent'].max():.2f}")
    logger.info(f"  Pallet Equivalent - Mean: {enriched_df['Pallet_Equivalent'].mean():.2f}, Max: {enriched_df['Pallet_Equivalent'].max():.2f}")
    
    return enriched_df


def enrich_order_data(order_df: pd.DataFrame, sku_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge order data with SKU master and calculate enrichment fields.
    
    Args:
        order_df (pd.DataFrame): Order data
        sku_df (pd.DataFrame): SKU master data
    
    Returns:
        pd.DataFrame: Enriched order data with calculated fields
    
    Raises:
        ValueError: If merge fails or results in insufficient data
    """
    logger.info("Enriching order data with SKU master")
    
    # Get SKU code column name
    sku_code_col = ORDER_DATA_COLUMNS['sku_code']
    
    # Merge on SKU code
    order_merged = order_df.merge(sku_df, on=sku_code_col, how="left")
    
    # Check for SKUs not found in master
    missing_skus = order_merged[sku_df.columns[1]].isna().sum()  # Check for NaN in first SKU master column
    if missing_skus > 0:
        logger.warning(f"Found {missing_skus} order lines with SKUs not in master data")
        
        # Log some examples of missing SKUs
        missing_sku_codes = order_merged[order_merged[sku_df.columns[1]].isna()][sku_code_col].unique()[:5]
        logger.warning(f"Example missing SKUs: {list(missing_sku_codes)}")
    
    # Calculate enrichment fields
    enriched_data = calculate_enrichment_fields(order_merged)
    
    # Final validation
    if len(enriched_data) == 0:
        raise ValueError("Enriched data is empty after merge")
    
    # Check date range
    date_col = ORDER_DATA_COLUMNS['date']
    if date_col in enriched_data.columns:
        unique_dates = enriched_data[date_col].nunique()
        if unique_dates < MIN_DATES_REQUIRED:
            raise ValueError(f"Insufficient date range: {unique_dates} < {MIN_DATES_REQUIRED}")
        
        min_date = enriched_data[date_col].min()
        max_date = enriched_data[date_col].max()
        logger.info(f"Date range: {min_date.date()} to {max_date.date()} ({unique_dates} unique dates)")
    
    logger.info(f"Successfully enriched {len(enriched_data)} order records")
    return enriched_data


def load_and_enrich_data() -> pd.DataFrame:
    """
    Main function to load and enrich all data.
    
    Returns:
        pd.DataFrame: Fully enriched order data ready for analysis
    
    Raises:
        Exception: If any step in the data loading/enrichment process fails
    """
    try:
        # Load raw data
        order_df = load_order_data()
        sku_df = load_sku_master()
        
        # Enrich data
        enriched_data = enrich_order_data(order_df, sku_df)
        
        logger.info("Data loading and enrichment completed successfully")
        return enriched_data
        
    except Exception as e:
        logger.error(f"Failed to load and enrich data: {str(e)}")
        raise


def validate_enriched_data(enriched_data: pd.DataFrame) -> bool:
    """
    Validate the enriched data for completeness and consistency.
    
    Args:
        enriched_data (pd.DataFrame): Enriched order data
    
    Returns:
        bool: True if validation passes
    
    Raises:
        ValueError: If validation fails
    """
    logger.info("Validating enriched data")
    
    # Check required columns exist
    required_cols = ['Total_Eaches', 'Case_Equivalent', 'Pallet_Equivalent']
    missing_cols = [col for col in required_cols if col not in enriched_data.columns]
    if missing_cols:
        raise ValueError(f"Missing enriched columns: {missing_cols}")
    
    # Check for completely null enriched columns
    for col in required_cols:
        if enriched_data[col].isna().all():
            raise ValueError(f"Enriched column {col} is completely null")
    
    # Check for negative values (shouldn't exist in quantities)
    for col in required_cols:
        negative_count = (enriched_data[col] < 0).sum()
        if negative_count > 0:
            logger.warning(f"Found {negative_count} negative values in {col}")
    
    # Summary statistics
    for col in required_cols:
        non_null_count = enriched_data[col].notna().sum()
        logger.info(f"{col}: {non_null_count} non-null values, mean: {enriched_data[col].mean():.2f}")
    
    logger.info("Data validation completed successfully")
    return True


if __name__ == "__main__":
    # Test the data loading functionality
    try:
        enriched_data = load_and_enrich_data()
        validate_enriched_data(enriched_data)
        print(f"✅ Successfully loaded and enriched {len(enriched_data)} records")
        print(f"📊 Data shape: {enriched_data.shape}")
        print(f"📅 Date range: {enriched_data['Date'].min().date()} to {enriched_data['Date'].max().date()}")
        print(f"🏷️  Unique SKUs: {enriched_data['Sku Code'].nunique()}")
    except Exception as e:
        print(f"❌ Data loading failed: {str(e)}")