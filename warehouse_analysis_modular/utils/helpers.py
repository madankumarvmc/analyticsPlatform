#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Helper Utilities for Warehouse Analysis

Contains common utility functions used across the warehouse analysis modules.
"""

import pandas as pd
import numpy as np
import logging
from typing import Union, Optional


def safe_division(numerator: Union[float, pd.Series], 
                  denominator: Union[float, pd.Series], 
                  fill_value: float = 0.0) -> Union[float, pd.Series]:
    """
    Perform division with safe handling of division by zero.
    
    Args:
        numerator: The numerator value(s)
        denominator: The denominator value(s)
        fill_value: Value to return when denominator is zero
    
    Returns:
        Result of division with fill_value for zero denominators
    """
    if isinstance(denominator, pd.Series):
        # Replace zeros with NaN temporarily for division
        safe_denom = denominator.replace(0, np.nan)
        result = numerator / safe_denom
        return result.fillna(fill_value)
    else:
        return numerator / denominator if denominator != 0 else fill_value


def classify_abc(cumulative_percentage: float, 
                 a_threshold: float = 70.0, 
                 b_threshold: float = 90.0) -> str:
    """
    Classify items into ABC categories based on cumulative percentage.
    
    Args:
        cumulative_percentage: Cumulative percentage value
        a_threshold: Threshold for A classification (default 70%)
        b_threshold: Threshold for B classification (default 90%)
    
    Returns:
        'A', 'B', or 'C' classification
    """
    if cumulative_percentage < a_threshold:
        return "A"
    elif cumulative_percentage <= b_threshold:
        return "B"
    else:
        return "C"


def classify_fms(cumulative_percentage: float,
                 f_threshold: float = 70.0,
                 m_threshold: float = 90.0) -> str:
    """
    Classify items into FMS (Fast/Medium/Slow) categories based on cumulative percentage.
    
    Args:
        cumulative_percentage: Cumulative percentage value
        f_threshold: Threshold for Fast classification (default 70%)
        m_threshold: Threshold for Medium classification (default 90%)
    
    Returns:
        'F', 'M', or 'S' classification
    """
    if cumulative_percentage < f_threshold:
        return "F"
    elif cumulative_percentage <= m_threshold:
        return "M"
    else:
        return "S"


def setup_logging(level: int = logging.INFO, 
                  format_string: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration for the warehouse analysis modules.
    
    Args:
        level: Logging level (default: INFO)
        format_string: Custom format string for log messages
    
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    return logging.getLogger(__name__)


def validate_dataframe(df: pd.DataFrame, 
                      required_columns: list, 
                      min_rows: int = 1,
                      name: str = "DataFrame") -> bool:
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        name: Name of the DataFrame for error messages
    
    Returns:
        True if validation passes
    
    Raises:
        ValueError: If validation fails
    """
    if df is None:
        raise ValueError(f"{name} is None")
    
    if df.empty:
        raise ValueError(f"{name} is empty")
    
    if len(df) < min_rows:
        raise ValueError(f"{name} has insufficient rows: {len(df)} < {min_rows}")
    
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"{name} missing required columns: {missing_cols}")
    
    return True


def calculate_percentiles(data: pd.Series, 
                         percentiles: list = [95, 90, 85]) -> dict:
    """
    Calculate percentiles for a data series.
    
    Args:
        data: Pandas Series containing numeric data
        percentiles: List of percentile values to calculate
    
    Returns:
        Dictionary with percentile labels and values
    """
    results = {
        "Max": data.max(),
        "Average": data.mean()
    }
    
    for p in percentiles:
        results[f"{p}%ile"] = np.percentile(data, p)
    
    return results


def format_numeric_columns(df: pd.DataFrame, 
                          columns: list, 
                          decimal_places: int = 2) -> pd.DataFrame:
    """
    Format numeric columns in a DataFrame to specified decimal places.
    
    Args:
        df: DataFrame to format
        columns: List of column names to format
        decimal_places: Number of decimal places
    
    Returns:
        DataFrame with formatted columns
    """
    df_formatted = df.copy()
    
    for col in columns:
        if col in df_formatted.columns:
            df_formatted[col] = df_formatted[col].round(decimal_places)
    
    return df_formatted


def normalize_abc_fms_values(df: pd.DataFrame, 
                           abc_col: str = "ABC", 
                           fms_col: str = "FMS") -> pd.DataFrame:
    """
    Normalize ABC and FMS values to ensure consistent formatting.
    
    Args:
        df: DataFrame containing ABC and FMS columns
        abc_col: Name of the ABC column
        fms_col: Name of the FMS column
    
    Returns:
        DataFrame with normalized ABC and FMS values
    """
    df_normalized = df.copy()
    
    if abc_col in df_normalized.columns:
        df_normalized[abc_col] = df_normalized[abc_col].astype(str).str.strip().str.upper()
    
    if fms_col in df_normalized.columns:
        df_normalized[fms_col] = df_normalized[fms_col].astype(str).str.strip().str.upper()
    
    return df_normalized


def create_2d_classification(df: pd.DataFrame,
                           abc_col: str = "ABC",
                           fms_col: str = "FMS",
                           output_col: str = "2D-Classification") -> pd.DataFrame:
    """
    Create 2D classification by combining ABC and FMS classifications.
    
    Args:
        df: DataFrame containing ABC and FMS columns
        abc_col: Name of the ABC column
        fms_col: Name of the FMS column
        output_col: Name for the output classification column
    
    Returns:
        DataFrame with added 2D classification column
    """
    df_with_2d = df.copy()
    df_with_2d[output_col] = df_with_2d[abc_col] + df_with_2d[fms_col]
    return df_with_2d


def handle_infinite_values(df: pd.DataFrame, 
                          columns: Optional[list] = None,
                          replacement_value: float = np.nan) -> pd.DataFrame:
    """
    Replace infinite values in specified columns with a replacement value.
    
    Args:
        df: DataFrame to process
        columns: List of columns to process (if None, processes all numeric columns)
        replacement_value: Value to replace infinites with
    
    Returns:
        DataFrame with infinite values replaced
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], replacement_value)
    
    return df_clean