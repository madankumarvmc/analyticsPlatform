#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chart Generation Module

Handles generation of matplotlib charts for warehouse analysis reporting.
Extracted from the original Warehouse Analysis (2).py file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import CHARTS_DIR, CHART_FILES, CHART_CONFIGS, CHART_DPI
from warehouse_analysis_modular.utils.helpers import validate_dataframe, setup_logging

logger = setup_logging()


class ChartGenerator:
    """
    Handles generation of various charts for warehouse analysis reporting.
    """
    
    def __init__(self, charts_dir: Optional[Path] = None, dpi: int = CHART_DPI):
        """
        Initialize the ChartGenerator.
        
        Args:
            charts_dir: Directory to save charts (defaults to config setting)
            dpi: DPI for saving charts
        """
        self.charts_dir = charts_dir or CHARTS_DIR
        self.dpi = dpi
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Ensure charts directory exists
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Chart generator initialized with output directory: {self.charts_dir}")
    
    def _save_figure(self, fig, filename: str) -> str:
        """
        Save matplotlib figure to file.
        
        Args:
            fig: Matplotlib figure object
            filename: Filename for the chart
            
        Returns:
            Full path to saved file
        """
        filepath = self.charts_dir / filename
        fig.savefig(str(filepath), bbox_inches="tight", dpi=self.dpi)
        plt.close(fig)
        return str(filepath)
    
    def chart_date_time_series(self, date_summary: pd.DataFrame) -> Tuple[str, str]:
        """
        Create line chart of Total_Case_Equiv over Date and bar chart of Distinct_Customers.
        
        Args:
            date_summary: DataFrame with date-wise summary data
            
        Returns:
            Tuple of (line_chart_path, bar_chart_path)
        """
        self.logger.info("Generating date time series charts")
        
        validate_dataframe(date_summary, ['Date', 'Total_Case_Equiv', 'Distinct_Customers'])
        
        # Prepare data
        df = date_summary.copy().sort_values("Date")
        
        # Line chart for Total_Case_Equiv
        config = CHART_CONFIGS['date_line_chart']
        fig, ax = plt.subplots(figsize=config['figsize'])
        ax.plot(df["Date"], df["Total_Case_Equiv"], marker="o", linewidth=1)
        ax.set_title(config['title'])
        ax.set_xlabel("Date")
        ax.set_ylabel("Total Case Equivalent")
        ax.grid(True, linestyle=':', linewidth=0.5)
        line_chart_path = self._save_figure(fig, CHART_FILES['date_total_case_equiv'])
        
        # Bar chart for Distinct_Customers
        config = CHART_CONFIGS['date_customers_chart']
        fig2, ax2 = plt.subplots(figsize=config['figsize'])
        ax2.bar(df["Date"].astype(str), df["Distinct_Customers"])
        ax2.set_title(config['title'])
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Distinct Customers")
        plt.xticks(rotation=45, ha="right")
        ax2.grid(False)
        bar_chart_path = self._save_figure(fig2, CHART_FILES['date_distinct_customers'])
        
        self.logger.info("Date time series charts generated successfully")
        return line_chart_path, bar_chart_path
    
    def chart_percentiles(self, percentile_profile: pd.DataFrame) -> str:
        """
        Create horizontal bar chart for percentiles.
        
        Args:
            percentile_profile: DataFrame with percentile data
            
        Returns:
            Path to saved chart
        """
        self.logger.info("Generating percentile chart")
        
        validate_dataframe(percentile_profile, ['Percentile', 'Total_Case_Equiv'])
        
        # Prepare data
        p = percentile_profile.set_index("Percentile")
        values = p["Total_Case_Equiv"]
        
        # Create horizontal bar chart
        config = CHART_CONFIGS['percentile_chart']
        fig, ax = plt.subplots(figsize=config['figsize'])
        ax.barh(values.index, values.values)
        ax.set_title(config['title'])
        ax.set_xlabel("Case Equivalent")
        
        chart_path = self._save_figure(fig, CHART_FILES['percentile_total_case_equiv'])
        
        self.logger.info("Percentile chart generated successfully")
        return chart_path
    
    def chart_sku_pareto(self, sku_data: pd.DataFrame, top_n: Optional[int] = None) -> Optional[str]:
        """
        Create SKU Pareto chart (bar + cumulative line).
        
        Args:
            sku_data: DataFrame with SKU data
            top_n: Number of top SKUs to show (defaults to config)
            
        Returns:
            Path to saved chart or None if data is insufficient
        """
        self.logger.info("Generating SKU Pareto chart")
        
        if sku_data is None or sku_data.empty:
            self.logger.warning("SKU data is empty, skipping Pareto chart")
            return None
        
        # Use config default if not specified
        if top_n is None:
            top_n = CHART_CONFIGS['sku_pareto_chart']['top_n']
        
        # Find volume column
        volume_col = None
        volume_candidates = ["Order_Volume_CE", "Order_Volume", "Case_Equivalent", "Total_Case_Equiv"]
        for col in volume_candidates:
            if col in sku_data.columns:
                volume_col = col
                break
        
        if volume_col is None:
            self.logger.warning("No volume column found for Pareto chart")
            return None
        
        # Prepare data
        df_sorted = sku_data.sort_values(volume_col, ascending=False).head(top_n)
        if len(df_sorted) == 0:
            self.logger.warning("No data available for Pareto chart")
            return None
        
        # Calculate cumulative percentage
        cum = df_sorted[volume_col].cumsum() / df_sorted[volume_col].sum() * 100
        
        # Create Pareto chart
        config = CHART_CONFIGS['sku_pareto_chart']
        fig, ax1 = plt.subplots(figsize=config['figsize'])
        
        # Bar chart
        ax1.bar(range(len(df_sorted)), df_sorted[volume_col])
        ax1.set_xlabel("Top SKUs (by volume)")
        ax1.set_ylabel("Volume")
        
        # Cumulative line
        ax2 = ax1.twinx()
        ax2.plot(range(len(df_sorted)), cum, marker="o")
        ax2.set_ylabel("Cumulative %")
        
        ax1.set_title(f"SKU Pareto (Top {len(df_sorted)})")
        
        chart_path = self._save_figure(fig, CHART_FILES['sku_pareto'])
        
        self.logger.info("SKU Pareto chart generated successfully")
        return chart_path
    
    def chart_abc_volume_stacked(self, sku_profile: pd.DataFrame) -> Optional[str]:
        """
        Create stacked bar chart of volume by ABC and FMS.
        
        Args:
            sku_profile: DataFrame with SKU profile including ABC, FMS, and volume data
            
        Returns:
            Path to saved chart or None if data is insufficient
        """
        self.logger.info("Generating ABC volume stacked chart")
        
        if sku_profile is None or sku_profile.empty:
            self.logger.warning("SKU profile data is empty, skipping ABC volume chart")
            return None
        
        required_columns = ["ABC", "FMS", "Total_Case_Equiv"]
        missing_cols = [col for col in required_columns if col not in sku_profile.columns]
        if missing_cols:
            self.logger.warning(f"Missing columns for ABC volume chart: {missing_cols}")
            return None
        
        # Create pivot table
        pivot = sku_profile.pivot_table(
            index="ABC", 
            columns="FMS", 
            values="Total_Case_Equiv", 
            aggfunc="sum", 
            fill_value=0
        )
        
        # Ensure proper ordering
        pivot = pivot.reindex(index=["A", "B", "C"], columns=["F", "M", "S"], fill_value=0)
        
        # Create stacked bar chart
        config = CHART_CONFIGS['abc_volume_chart']
        fig, ax = plt.subplots(figsize=config['figsize'])
        pivot.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title(config['title'])
        ax.set_ylabel("Total Case Equivalent")
        ax.set_xlabel("ABC Class")
        plt.xticks(rotation=0)
        
        chart_path = self._save_figure(fig, CHART_FILES['abc_volume_stacked'])
        
        self.logger.info("ABC volume stacked chart generated successfully")
        return chart_path
    
    def chart_abc_fms_heatmap(self, sku_profile: pd.DataFrame) -> Optional[str]:
        """
        Create heatmap of ABC×FMS volume percentages.
        
        Args:
            sku_profile: DataFrame with SKU profile including ABC, FMS, and volume data
            
        Returns:
            Path to saved chart or None if data is insufficient
        """
        self.logger.info("Generating ABC×FMS heatmap")
        
        if sku_profile is None or sku_profile.empty:
            self.logger.warning("SKU profile data is empty, skipping heatmap")
            return None
        
        required_columns = ["ABC", "FMS", "Total_Case_Equiv"]
        missing_cols = [col for col in required_columns if col not in sku_profile.columns]
        if missing_cols:
            self.logger.warning(f"Missing columns for ABC×FMS heatmap: {missing_cols}")
            return None
        
        # Create pivot table
        pivot = sku_profile.pivot_table(
            index="ABC", 
            columns="FMS", 
            values="Total_Case_Equiv", 
            aggfunc="sum", 
            fill_value=0
        )
        
        # Ensure proper ordering
        pivot = pivot.reindex(index=["A", "B", "C"], columns=["F", "M", "S"], fill_value=0)
        
        # Calculate row-wise percentages (each row sums to 100%)
        pivot_pct = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
        
        # Create heatmap with blue color scheme
        config = CHART_CONFIGS['abc_heatmap_chart']
        fig, ax = plt.subplots(figsize=config['figsize'])
        
        # Use blue color scheme as requested
        from matplotlib.colors import LinearSegmentedColormap
        colors = ['#f0f8ff', '#4682b4', '#191970']  # Light blue to dark blue
        blue_cmap = LinearSegmentedColormap.from_list('blue_custom', colors)
        
        cax = ax.imshow(pivot_pct.values, aspect='auto', cmap=blue_cmap)
        ax.set_xticks(range(len(pivot_pct.columns)))
        ax.set_xticklabels(pivot_pct.columns)
        ax.set_yticks(range(len(pivot_pct.index)))
        ax.set_yticklabels(pivot_pct.index)
        ax.set_title(config['title'])
        
        # Add value annotations with white text for better visibility on blue
        for (i, j), val in np.ndenumerate(pivot_pct.values):
            text_color = 'white' if val > 50 else 'black'
            ax.text(j, i, f"{val:.1f}%", ha='center', va='center', fontsize=9, 
                   color=text_color, fontweight='bold')
        
        # Add colorbar
        fig.colorbar(cax, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        
        chart_path = self._save_figure(fig, CHART_FILES['abc_fms_heatmap'])
        
        self.logger.info("ABC×FMS heatmap generated successfully")
        return chart_path
    
    def generate_all_charts(self, analysis_results: Dict) -> Dict[str, str]:
        """
        Generate all charts from analysis results.
        
        Args:
            analysis_results: Dictionary containing all analysis data
            
        Returns:
            Dictionary mapping chart names to file paths
        """
        self.logger.info("Generating all charts")
        
        chart_paths = {}
        
        try:
            # Date time series charts
            if 'date_order_summary' in analysis_results:
                line_path, bar_path = self.chart_date_time_series(analysis_results['date_order_summary'])
                chart_paths['date_line'] = line_path
                chart_paths['date_customers'] = bar_path
            
            # Percentile chart
            if 'percentile_profile' in analysis_results:
                chart_paths['percentile'] = self.chart_percentiles(analysis_results['percentile_profile'])
            
            # SKU Pareto chart
            if 'sku_order_summary' in analysis_results:
                pareto_path = self.chart_sku_pareto(analysis_results['sku_order_summary'])
                if pareto_path:
                    chart_paths['sku_pareto'] = pareto_path
            elif 'sku_profile_abc_fms' in analysis_results:
                pareto_path = self.chart_sku_pareto(analysis_results['sku_profile_abc_fms'])
                if pareto_path:
                    chart_paths['sku_pareto'] = pareto_path
            
            # ABC volume stacked chart
            if 'sku_profile_abc_fms' in analysis_results:
                abc_volume_path = self.chart_abc_volume_stacked(analysis_results['sku_profile_abc_fms'])
                if abc_volume_path:
                    chart_paths['abc_volume'] = abc_volume_path
            
            # ABC×FMS heatmap
            if 'sku_profile_abc_fms' in analysis_results:
                heatmap_path = self.chart_abc_fms_heatmap(analysis_results['sku_profile_abc_fms'])
                if heatmap_path:
                    chart_paths['abc_heatmap'] = heatmap_path
            
            self.logger.info(f"Generated {len(chart_paths)} charts successfully")
            
        except Exception as e:
            self.logger.error(f"Error generating charts: {str(e)}")
            raise
        
        return chart_paths
    
    def get_relative_chart_paths(self, chart_paths: Dict[str, str], 
                                relative_to: Optional[Path] = None) -> Dict[str, str]:
        """
        Convert absolute chart paths to relative paths.
        
        Args:
            chart_paths: Dictionary of chart paths
            relative_to: Base path for relative calculation
            
        Returns:
            Dictionary with relative paths
        """
        if relative_to is None:
            relative_to = Path.cwd()
        
        relative_paths = {}
        for name, path in chart_paths.items():
            if path:
                try:
                    rel_path = os.path.relpath(path, start=str(relative_to))
                    relative_paths[name] = rel_path
                except Exception:
                    relative_paths[name] = path
            else:
                relative_paths[name] = path
        
        return relative_paths


def generate_charts(analysis_results: Dict, charts_dir: Optional[Path] = None) -> Dict[str, str]:
    """
    Convenience function to generate all charts.
    
    Args:
        analysis_results: Dictionary containing analysis data
        charts_dir: Directory to save charts
        
    Returns:
        Dictionary mapping chart names to file paths
    """
    generator = ChartGenerator(charts_dir)
    return generator.generate_all_charts(analysis_results)