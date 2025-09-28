#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced Chart Generation Module

Provides advanced visualization capabilities including dual Y-axis charts,
correlation matrices, heatmaps, and multi-metric time series visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Import from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import CHARTS_DIR, CHART_DPI
from warehouse_analysis_modular.utils.helpers import setup_logging

logger = setup_logging()


class AdvancedChartGenerator:
    """
    Advanced Chart Generator providing sophisticated visualization capabilities
    for warehouse analysis including multi-metric correlations and heatmaps.
    
    Features:
    - Multi-metric time series with dual Y-axis
    - Cross-classification heatmaps
    - Advanced correlation matrices
    - Category-level picking analysis charts
    - Enhanced percentile capacity planning visualizations
    """
    
    def __init__(self, charts_dir: Optional[Path] = None, dpi: int = CHART_DPI):
        """Initialize the Advanced Chart Generator."""
        self.charts_dir = charts_dir or CHARTS_DIR
        self.dpi = dpi
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Ensure charts directory exists
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        self.logger.info(f"Advanced chart generator initialized with output directory: {self.charts_dir}")
    
    def create_multi_metric_time_series(self, daily_metrics: pd.DataFrame, 
                                       analysis_results: Dict) -> str:
        """
        Create multi-metric time series chart with dual Y-axis (matching screenshot 1).
        
        Args:
            daily_metrics: DataFrame with daily operational metrics
            analysis_results: Dictionary with correlation and other analysis results
            
        Returns:
            Path to saved chart file
        """
        self.logger.info("Creating multi-metric time series chart")
        
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # Ensure Date column is datetime
        if 'Date' not in daily_metrics.columns:
            self.logger.error("Date column not found in daily_metrics")
            return ""
            
        daily_metrics['Date'] = pd.to_datetime(daily_metrics['Date'])
        
        # Primary Y-axis (left) - Lines and Customers
        color1 = 'tab:blue'
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('# Lines / # Customer', color=color1, fontsize=12)
        
        # Plot Lines (blue)
        line1 = ax1.plot(daily_metrics['Date'], daily_metrics.get('Total_Lines', []), 
                        color='tab:blue', linewidth=2, label='#Lines', marker='o', markersize=2)
        
        # Plot Customers (gray)
        line2 = ax1.plot(daily_metrics['Date'], daily_metrics.get('Distinct_Customers', []), 
                        color='gray', linewidth=2, label='#Customer', marker='s', markersize=2)
        
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim(bottom=0)
        
        # Secondary Y-axis (right) - Shipments and Case Equivalents
        ax2 = ax1.twinx()
        color2 = 'tab:orange'
        ax2.set_ylabel('# Shipments / # Case Equivalents', color=color2, fontsize=12)
        
        # Plot Shipments (yellow)
        line3 = ax2.plot(daily_metrics['Date'], daily_metrics.get('Unique_Shipments', []), 
                        color='gold', linewidth=2, label='#Shipment', marker='^', markersize=2)
        
        # Plot Case Equivalents (orange/red)
        line4 = ax2.plot(daily_metrics['Date'], daily_metrics.get('Total_Case_Equiv', []), 
                        color='orangered', linewidth=2.5, label='# Case Equi.', marker='D', markersize=2)
        
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim(bottom=0)
        
        # Title and formatting
        plt.title('Order Profile', fontsize=16, fontweight='bold', pad=20)
        
        # Combined legend
        lines = line1 + line2 + line3 + line4
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(0, 0.95), ncol=4)
        
        # Format x-axis dates
        ax1.tick_params(axis='x', rotation=45)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add statistics table below chart (matching screenshot)
        if analysis_results.get('enhanced_percentile_analysis'):
            percentiles = analysis_results['enhanced_percentile_analysis']
            
            # Create statistics table data
            stats_data = []
            
            for metric in ['Total_Case_Equiv', 'Total_Lines', 'Distinct_Customers', 'Unique_Shipments']:
                if metric in percentiles:
                    pct_data = percentiles[metric]
                    stats_data.append([
                        metric.replace('_', ' ').replace('Total ', '# '),
                        int(pct_data.get('Max', 0)),
                        int(pct_data.get('95%ile', 0)),
                        int(pct_data.get('90%ile', 0)), 
                        int(pct_data.get('85%ile', 0)),
                        int(pct_data.get('Avg', 0))
                    ])
            
            if stats_data:
                # Add table below chart
                table_ax = fig.add_subplot(2, 1, 2)
                table_ax.axis('off')
                
                table = table_ax.table(cellText=stats_data,
                                     colLabels=['Metric', 'Max', '95%ile', '90%ile', '85%ile', 'Avg'],
                                     cellLoc='center', loc='center',
                                     colWidths=[0.2, 0.15, 0.15, 0.15, 0.15, 0.15])
                
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1.2, 1.5)
                
                # Style the table
                for i in range(len(stats_data) + 1):
                    for j in range(6):
                        cell = table[i, j]
                        if i == 0:  # Header row
                            cell.set_facecolor('#4CAF50')
                            cell.set_text_props(weight='bold', color='white')
                        else:
                            cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        plt.tight_layout()
        
        # Save chart
        filename = "order_profile_multi_metric.png"
        filepath = self.charts_dir / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Multi-metric time series chart saved: {filename}")
        return str(filepath)
    
    def create_abc_fms_2d_heatmap(self, enhanced_analysis: Dict) -> str:
        """
        Create 2D ABC-FMS classification heatmap (matching screenshot 2).
        
        Args:
            enhanced_analysis: Results from enhanced ABC-FMS analysis
            
        Returns:
            Path to saved chart file
        """
        self.logger.info("Creating ABC-FMS 2D classification heatmap")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        matrices = enhanced_analysis.get('classification_matrix_2d', {})
        
        # 1. SKU Count Matrix Heatmap
        if 'sku_count_matrix' in matrices:
            sku_matrix = matrices['sku_count_matrix']
            # Remove Grand Total for cleaner heatmap
            if 'Grand Total' in sku_matrix.index:
                sku_matrix_clean = sku_matrix.drop('Grand Total').drop('Grand Total', axis=1)
                
                sns.heatmap(sku_matrix_clean, annot=True, fmt='d', cmap='Blues', ax=ax1)
                ax1.set_title('SKU Count Distribution', fontsize=12, fontweight='bold')
                ax1.set_xlabel('FMS Classification')
                ax1.set_ylabel('ABC Classification')
        
        # 2. Volume Percentage Heatmap
        if 'volume_percentage_matrix' in matrices:
            vol_pct_matrix = matrices['volume_percentage_matrix']
            if 'Grand Total' in vol_pct_matrix.index:
                vol_pct_clean = vol_pct_matrix.drop('Grand Total').drop('Grand Total', axis=1)
                
                sns.heatmap(vol_pct_clean, annot=True, fmt='.0f', cmap='Oranges', ax=ax2)
                ax2.set_title('Volume Distribution (%)', fontsize=12, fontweight='bold')
                ax2.set_xlabel('FMS Classification')
                ax2.set_ylabel('ABC Classification')
        
        # 3. Lines Percentage Heatmap
        if 'lines_percentage_matrix' in matrices:
            lines_pct_matrix = matrices['lines_percentage_matrix']
            if 'Grand Total' in lines_pct_matrix.index:
                lines_pct_clean = lines_pct_matrix.drop('Grand Total').drop('Grand Total', axis=1)
                
                sns.heatmap(lines_pct_clean, annot=True, fmt='.0f', cmap='Greens', ax=ax3)
                ax3.set_title('Lines Distribution (%)', fontsize=12, fontweight='bold')
                ax3.set_xlabel('FMS Classification')
                ax3.set_ylabel('ABC Classification')
        
        # 4. Combined Analysis Visualization
        if 'advanced_segmentation' in enhanced_analysis:
            segments = enhanced_analysis['advanced_segmentation']['segment_analysis']
            
            # Create bubble chart showing segment relationships
            segment_data = []
            for segment, data in segments.items():
                segment_data.append({
                    'Segment': segment,
                    'SKU_Pct': data['sku_percentage'],
                    'Volume_Pct': data['volume_percentage'],
                    'Lines_Pct': data['lines_percentage']
                })
            
            if segment_data:
                df_segments = pd.DataFrame(segment_data)
                
                # Bubble chart
                scatter = ax4.scatter(df_segments['SKU_Pct'], df_segments['Volume_Pct'], 
                                    s=df_segments['Lines_Pct']*10, alpha=0.6, 
                                    c=range(len(df_segments)), cmap='viridis')
                
                # Annotate bubbles with segment names
                for i, row in df_segments.iterrows():
                    ax4.annotate(row['Segment'], 
                               (row['SKU_Pct'], row['Volume_Pct']),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=10, fontweight='bold')
                
                ax4.set_xlabel('SKU Percentage (%)')
                ax4.set_ylabel('Volume Percentage (%)')
                ax4.set_title('Segment Analysis\n(Bubble size = Lines %)', fontsize=12, fontweight='bold')
                ax4.grid(True, alpha=0.3)
        
        plt.suptitle('SKU Profile - 2D Classification Analysis', fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        
        # Save chart
        filename = "abc_fms_2d_classification.png"
        filepath = self.charts_dir / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"ABC-FMS 2D heatmap saved: {filename}")
        return str(filepath)
    
    def create_picking_analysis_chart(self, picking_analysis: Dict) -> str:
        """
        Create case vs piece picking analysis chart (matching screenshot 3).
        
        Args:
            picking_analysis: Results from picking methodology analysis
            
        Returns:
            Path to saved chart file
        """
        self.logger.info("Creating picking analysis chart")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Overall Picking Pattern Breakdown
        if 'overall_picking_patterns' in picking_analysis:
            overall = picking_analysis['overall_picking_patterns']
            if 'picking_summary' in overall:
                summary = overall['picking_summary']
                
                # Pie chart of picking types by lines
                ax1.pie(summary['Line_Percentage'], labels=summary['Picking_Type'], 
                       autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral', 'lightgreen'])
                ax1.set_title('Picking Methods by Lines (%)', fontsize=12, fontweight='bold')
        
        # 2. Category-level Picking Analysis
        if 'category_picking_analysis' in picking_analysis:
            category_data = picking_analysis['category_picking_analysis']['category_breakdown']
            
            if category_data:
                # Create category comparison chart
                categories = [cat['category'] for cat in category_data[:10]]  # Top 10 categories
                pcs_percentages = [cat['pcs_lines_percentage'] for cat in category_data[:10]]
                case_percentages = [cat['case_only_lines_percentage'] for cat in category_data[:10]]
                
                x = np.arange(len(categories))
                width = 0.35
                
                ax2.bar(x - width/2, case_percentages, width, label='Case Only %', color='steelblue')
                ax2.bar(x + width/2, pcs_percentages, width, label='PCS Lines %', color='orange')
                
                ax2.set_xlabel('Category')
                ax2.set_ylabel('Percentage of Lines')
                ax2.set_title('Case vs PCS Lines by Category', fontsize=12, fontweight='bold')
                ax2.set_xticks(x)
                ax2.set_xticklabels(categories, rotation=45, ha='right')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        # 3. Piece Picking Impact Analysis
        if 'piece_picking_impact' in picking_analysis:
            impact = picking_analysis['piece_picking_impact']
            
            # Create impact comparison
            categories = ['Lines Impact', 'Volume Impact']
            percentages = [impact.get('piece_lines_percentage', 0), 
                          impact.get('piece_volume_percentage', 0)]
            
            bars = ax3.bar(categories, percentages, color=['coral', 'lightblue'])
            ax3.set_ylabel('Percentage (%)')
            ax3.set_title('Piece Picking Impact Assessment', fontsize=12, fontweight='bold')
            
            # Add value labels on bars
            for bar, pct in zip(bars, percentages):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            ax3.grid(True, alpha=0.3)
        
        # 4. Picking Complexity Heatmap
        if 'category_picking_analysis' in picking_analysis:
            category_data = picking_analysis['category_picking_analysis']['category_breakdown']
            
            if category_data:
                # Create complexity matrix
                complexity_data = []
                for cat in category_data[:8]:  # Top 8 categories for visibility
                    complexity_data.append([
                        cat['pcs_lines_percentage'],
                        cat['pcs_only_lines_percentage'], 
                        cat['case_only_lines_percentage'],
                        cat['operational_complexity']
                    ])
                
                complexity_matrix = np.array(complexity_data)
                category_labels = [cat['category'] for cat in category_data[:8]]
                
                im = ax4.imshow(complexity_matrix, cmap='RdYlGn_r', aspect='auto')
                ax4.set_xticks(range(4))
                ax4.set_xticklabels(['PCS Lines %', 'PCS Only %', 'Case Only %', 'Complexity'], rotation=45, ha='right')
                ax4.set_yticks(range(len(category_labels)))
                ax4.set_yticklabels(category_labels)
                ax4.set_title('Category Picking Complexity', fontsize=12, fontweight='bold')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
                cbar.set_label('Intensity', rotation=270, labelpad=15)
        
        plt.suptitle('Order Profile - Case vs Piece Picking Analysis', fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        
        # Save chart
        filename = "picking_methodology_analysis.png" 
        filepath = self.charts_dir / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Picking analysis chart saved: {filename}")
        return str(filepath)
    
    def create_correlation_matrix_chart(self, correlation_analysis: Dict) -> str:
        """
        Create correlation matrix visualization for multi-metric analysis.
        
        Args:
            correlation_analysis: Results from correlation analysis
            
        Returns:
            Path to saved chart file
        """
        self.logger.info("Creating correlation matrix chart")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Correlation Matrix Heatmap
        if 'correlation_matrix' in correlation_analysis:
            corr_matrix = correlation_analysis['correlation_matrix']
            
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, ax=ax1)
            ax1.set_title('Multi-Metric Correlation Matrix', fontsize=12, fontweight='bold')
        
        # 2. Key Correlations Bar Chart
        if 'key_correlations' in correlation_analysis:
            correlations = correlation_analysis['key_correlations']
            
            corr_names = list(correlations.keys())
            corr_values = list(correlations.values())
            
            bars = ax2.barh(corr_names, corr_values, color=['green' if x > 0 else 'red' for x in corr_values])
            ax2.set_xlabel('Correlation Coefficient')
            ax2.set_title('Key Metric Correlations', fontsize=12, fontweight='bold')
            ax2.axvline(x=0, color='black', linewidth=0.5)
            ax2.grid(True, alpha=0.3)
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, corr_values)):
                ax2.text(val + (0.05 if val > 0 else -0.05), i, f'{val:.2f}',
                        va='center', ha='left' if val > 0 else 'right', fontweight='bold')
        
        plt.tight_layout()
        
        # Save chart
        filename = "correlation_matrix_analysis.png"
        filepath = self.charts_dir / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Correlation matrix chart saved: {filename}")
        return str(filepath)
    
    def create_advanced_percentile_chart(self, percentile_analysis: Dict, peak_ratios: Dict) -> str:
        """
        Create advanced percentile capacity planning chart.
        
        Args:
            percentile_analysis: Enhanced percentile analysis results
            peak_ratios: Peak ratio calculations
            
        Returns:
            Path to saved chart file
        """
        self.logger.info("Creating advanced percentile capacity planning chart")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        
        # 1. Volume Percentile Distribution
        if 'Total_Case_Equiv' in percentile_analysis:
            vol_pct = percentile_analysis['Total_Case_Equiv']
            percentiles = ['25%ile', '50%ile', '75%ile', '85%ile', '90%ile', '95%ile', 'Max']
            values = [vol_pct.get(p, 0) for p in percentiles]
            
            ax1.plot(percentiles, values, marker='o', linewidth=2, markersize=8, color='steelblue')
            ax1.fill_between(percentiles, values, alpha=0.3, color='steelblue')
            ax1.set_ylabel('Case Equivalents')
            ax1.set_title('Volume Capacity Planning Curve', fontsize=12, fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
        
        # 2. Peak Ratios Analysis
        if peak_ratios:
            metrics = list(peak_ratios.keys())
            peak_to_avg = [peak_ratios[m].get('peak_to_avg', 0) for m in metrics]
            peak_to_95th = [peak_ratios[m].get('peak_to_95th', 0) for m in metrics]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax2.bar(x - width/2, peak_to_avg, width, label='Peak/Avg', color='coral')
            ax2.bar(x + width/2, peak_to_95th, width, label='Peak/95th', color='lightblue')
            
            ax2.set_ylabel('Ratio')
            ax2.set_title('Peak Demand Ratios', fontsize=12, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels([m.replace('Total_', '') for m in metrics], rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Capacity Planning Zones
        if 'Total_Case_Equiv' in percentile_analysis:
            vol_pct = percentile_analysis['Total_Case_Equiv']
            
            zones = ['Normal Operations', 'Peak Season', 'Emergency Buffer']
            zone_values = [
                vol_pct.get('Avg', 0),
                vol_pct.get('95%ile', 0), 
                vol_pct.get('Max', 0)
            ]
            colors = ['green', 'orange', 'red']
            
            wedges, texts, autotexts = ax3.pie(zone_values, labels=zones, colors=colors,
                                              autopct='%1.1f%%', startangle=90)
            ax3.set_title('Capacity Planning Zones', fontsize=12, fontweight='bold')
        
        # 4. Coefficient of Variation Analysis
        cv_data = []
        cv_labels = []
        
        for metric, data in percentile_analysis.items():
            if 'CV' in data:
                cv_data.append(data['CV'])
                cv_labels.append(metric.replace('Total_', ''))
        
        if cv_data:
            bars = ax4.bar(cv_labels, cv_data, color='purple', alpha=0.7)
            ax4.set_ylabel('Coefficient of Variation')
            ax4.set_title('Demand Variability Analysis', fontsize=12, fontweight='bold')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
            
            # Add threshold line
            ax4.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='High Variability')
            ax4.legend()
        
        plt.suptitle('Advanced Capacity Planning Analysis', fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        
        # Save chart
        filename = "advanced_percentile_capacity.png"
        filepath = self.charts_dir / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Advanced percentile chart saved: {filename}")
        return str(filepath)
    
    def create_enhanced_order_trend_chart(self, date_summary: pd.DataFrame) -> str:
        """
        Create enhanced multi-line order trend chart showing multiple metrics in one visualization.
        
        Args:
            date_summary: DataFrame with daily summary data
            
        Returns:
            Path to saved chart file
        """
        self.logger.info("Creating enhanced multi-line order trend chart")
        
        if date_summary is None or date_summary.empty:
            self.logger.warning("❌ No date summary data available for enhanced trend chart")
            return ""
        
        # Debug information about the data received
        self.logger.info(f"📊 Creating enhanced order trend chart with data shape: {date_summary.shape}")
        self.logger.info(f"📊 Available columns: {list(date_summary.columns)}")
        self.logger.info(f"📊 Date range: {date_summary['Date'].min()} to {date_summary['Date'].max()}" if 'Date' in date_summary.columns else "No Date column")
        
        # Prepare data
        df = date_summary.copy().sort_values("Date")
        
        # Create figure with dual y-axis
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # Primary y-axis - Volume metrics (larger scale)
        color1 = '#ff7f0e'  # Orange for Case Equivalent
        ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Volume Metrics', color=color1, fontsize=12, fontweight='bold')
        
        # Plot Case Equivalent with primary axis
        if 'Total_Case_Equiv' in df.columns:
            line1 = ax1.plot(df['Date'], df['Total_Case_Equiv'], 
                           color=color1, linewidth=2.5, marker='o', markersize=4,
                           label='# Case Equiv.', alpha=0.8)
            ax1.tick_params(axis='y', labelcolor=color1)
        
        # Secondary y-axis - Count metrics (smaller scale)
        ax2 = ax1.twinx()
        color2 = '#1f77b4'  # Blue for Lines
        color3 = '#2ca02c'  # Green for Customers
        color4 = '#d62728'  # Red for Shipments
        
        ax2.set_ylabel('Count Metrics', color='black', fontsize=12, fontweight='bold')
        
        # Plot count metrics
        lines = []
        if 'Total_Lines' in df.columns:
            line2 = ax2.plot(df['Date'], df['Total_Lines'],
                           color=color2, linewidth=2, marker='s', markersize=3,
                           label='#Lines', alpha=0.8)
            lines.extend(line2)
        
        if 'Distinct_Customers' in df.columns:
            line3 = ax2.plot(df['Date'], df['Distinct_Customers'],
                           color=color3, linewidth=2, marker='^', markersize=3,
                           label='#Customer', alpha=0.8)
            lines.extend(line3)
        
        if 'Distinct_Shipments' in df.columns:
            line4 = ax2.plot(df['Date'], df['Distinct_Shipments'],
                           color=color4, linewidth=2, marker='d', markersize=3,
                           label='#Shipment', alpha=0.8)
            lines.extend(line4)
        
        # Styling
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.set_title('Order Profile - Trend\nMultiple Metrics Comparison', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Format x-axis dates
        ax1.tick_params(axis='x', rotation=45)
        
        # Combined legend
        all_lines = line1 + lines if 'Total_Case_Equiv' in df.columns else lines
        labels = [line.get_label() for line in all_lines]
        ax1.legend(all_lines, labels, loc='upper right', bbox_to_anchor=(1, 0.95))
        
        # Add summary statistics table
        stats_data = []
        for col in ['Total_Case_Equiv', 'Total_Lines', 'Distinct_Customers', 'Distinct_Shipments']:
            if col in df.columns:
                stats_data.append([
                    col.replace('Total_', '#').replace('Distinct_', '#'),
                    f"{df[col].max():.0f}",
                    f"{df[col].mean():.0f}",
                    f"{(df[col].max() / df[col].mean()):.1f}x" if df[col].mean() > 0 else "N/A"
                ])
        
        if stats_data:
            # Add table below the chart
            table = ax1.table(cellText=stats_data,
                            colLabels=['Metric', 'Max', 'Avg', 'Peak/Avg'],
                            cellLoc='center',
                            loc='lower center',
                            bbox=[0.15, -0.35, 0.7, 0.25])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.2)
            
            # Style table
            for i in range(len(stats_data) + 1):
                for j in range(4):
                    cell = table[(i, j)]
                    if i == 0:  # Header
                        cell.set_facecolor('#4CAF50')
                        cell.set_text_props(weight='bold', color='white')
                    else:
                        cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        plt.tight_layout()
        
        # Save chart
        filename = "enhanced_order_trend_profile.png"
        filepath = self.charts_dir / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Enhanced order trend chart saved: {filename}")
        return str(filepath)
    
    def create_sku_profile_2d_classification_chart(self, analysis_results: Dict) -> str:
        """
        Create SKU Profile 2D classification chart showing SKU%, Volume%, and Lines% relationships.
        
        Args:
            analysis_results: Dictionary containing analysis results with SKU profile data
            
        Returns:
            Path to saved chart file
        """
        self.logger.info("Creating SKU Profile 2D classification chart")
        
        # Extract data from analysis results
        abc_fms_summary = analysis_results.get('abc_fms_summary')
        
        # Debug information about available analysis results
        self.logger.info(f"📊 Available analysis result keys: {list(analysis_results.keys())}")
        
        if abc_fms_summary is None or abc_fms_summary.empty:
            self.logger.warning("❌ No ABC-FMS summary data available for 2D classification chart")
            if abc_fms_summary is None:
                self.logger.warning("abc_fms_summary is None")
            else:
                self.logger.warning(f"abc_fms_summary is empty, shape: {abc_fms_summary.shape}")
            return ""
        
        # Debug information about the ABC-FMS data received
        self.logger.info(f"📊 Creating SKU 2D classification chart with ABC-FMS data shape: {abc_fms_summary.shape}")
        self.logger.info(f"📊 ABC-FMS columns: {list(abc_fms_summary.columns)}")
        self.logger.info(f"📊 First few rows of ABC-FMS data:")
        self.logger.info(f"{abc_fms_summary.head()}")
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 10))
        
        # Main chart area for the 2D visualization
        ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
        
        # Side chart for the bar comparison
        ax_bars = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
        
        # Table area for data
        ax_table = plt.subplot2grid((3, 3), (2, 0), colspan=3)
        ax_table.axis('off')
        
        # Calculate percentages for 2D classification
        classification_data = []
        colors = {'AF': '#4CAF50', 'AM': '#8BC34A', 'AS': '#CDDC39',
                 'BF': '#FF9800', 'BM': '#FF5722', 'BS': '#F44336',
                 'CF': '#9C27B0', 'CM': '#673AB7', 'CS': '#3F51B5'}
        
        # Extract ABC-FMS combinations and calculate percentages
        total_skus = abc_fms_summary['SKU_Total'].sum() if 'SKU_Total' in abc_fms_summary.columns else 1
        total_volume = abc_fms_summary['Volume_Total'].sum() if 'Volume_Total' in abc_fms_summary.columns else 1
        total_lines = abc_fms_summary['Line_Total'].sum() if 'Line_Total' in abc_fms_summary.columns else 1
        
        # Process each ABC-FMS combination
        for _, row in abc_fms_summary.iterrows():
            if pd.isna(row.get('ABC', '')):
                continue
                
            abc = row.get('ABC', '')
            if abc == 'Grand Total':
                continue
                
            # Calculate percentages for each FMS category
            for fms in ['F', 'M', 'S']:
                sku_col = f'SKU_{fms}'
                vol_col = f'Volume_{fms}'
                line_col = f'Line_{fms}'
                
                if sku_col in row and vol_col in row and line_col in row:
                    sku_pct = (row[sku_col] / total_skus) * 100 if total_skus > 0 else 0
                    vol_pct = (row[vol_col] / total_volume) * 100 if total_volume > 0 else 0
                    line_pct = (row[line_col] / total_lines) * 100 if total_lines > 0 else 0
                    
                    if sku_pct > 0 or vol_pct > 0 or line_pct > 0:  # Only include non-zero entries
                        classification_data.append({
                            'Class': f'{abc}{fms}',
                            'SKU_Pct': sku_pct,
                            'Volume_Pct': vol_pct,
                            'Lines_Pct': line_pct,
                            'Color': colors.get(f'{abc}{fms}', '#808080')
                        })
        
        if not classification_data:
            self.logger.warning("No valid classification data for 2D chart")
            return ""
        
        df_class = pd.DataFrame(classification_data)
        
        # Main 2D visualization - Stacked bar with connected lines
        x_positions = range(len(df_class))
        
        # Create grouped bars for SKU%, Volume%, Lines%
        width = 0.25
        x1 = [x - width for x in x_positions]
        x2 = x_positions
        x3 = [x + width for x in x_positions]
        
        bars1 = ax_main.bar(x1, df_class['SKU_Pct'], width, label='SKU %', 
                           color='lightblue', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = ax_main.bar(x2, df_class['Volume_Pct'], width, label='Volume %', 
                           color='lightgreen', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars3 = ax_main.bar(x3, df_class['Lines_Pct'], width, label='Lines %', 
                           color='lightcoral', alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add connecting lines to show relationships
        for i in range(len(df_class)):
            ax_main.plot([x1[i], x2[i], x3[i]], 
                        [df_class.iloc[i]['SKU_Pct'], df_class.iloc[i]['Volume_Pct'], df_class.iloc[i]['Lines_Pct']], 
                        'o--', color='gray', alpha=0.6, linewidth=1, markersize=4)
        
        ax_main.set_xlabel('ABC-FMS Classification', fontsize=12, fontweight='bold')
        ax_main.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax_main.set_title('SKU Profile - 2D Class\nPercentage Distribution Analysis', fontsize=14, fontweight='bold')
        ax_main.set_xticks(x_positions)
        ax_main.set_xticklabels(df_class['Class'], rotation=45)
        ax_main.legend(loc='upper right')
        ax_main.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0.5:  # Only label bars with meaningful height
                    ax_main.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                               f'{height:.0f}%', ha='center', va='bottom', fontsize=8)
        
        # Side bar chart showing key insights
        # Find top performers
        top_volume = df_class.nlargest(3, 'Volume_Pct')
        top_sku = df_class.nlargest(3, 'SKU_Pct')
        
        insight_data = []
        if not top_volume.empty:
            insight_data.append(['Top Vol Class', top_volume.iloc[0]['Class'], f"{top_volume.iloc[0]['Volume_Pct']:.0f}%"])
        if not top_sku.empty:
            insight_data.append(['Top SKU Class', top_sku.iloc[0]['Class'], f"{top_sku.iloc[0]['SKU_Pct']:.0f}%"])
        
        # Calculate key ratios (e.g., AF class analysis)
        af_class = df_class[df_class['Class'] == 'AF']
        if not af_class.empty:
            af_row = af_class.iloc[0]
            insight_data.append(['AF Class SKU%', 'High Value/Fast', f"{af_row['SKU_Pct']:.0f}%"])
            insight_data.append(['AF Class Vol%', 'Critical Items', f"{af_row['Volume_Pct']:.0f}%"])
            insight_data.append(['AF Class Lines%', 'High Activity', f"{af_row['Lines_Pct']:.0f}%"])
        
        # Create insight visualization in side panel
        if insight_data:
            y_pos = range(len(insight_data))
            values = [float(row[2].replace('%', '')) for row in insight_data]
            labels = [f"{row[0]}\n{row[1]}" for row in insight_data]
            
            bars = ax_bars.barh(y_pos, values, color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#607D8B'][:len(values)])
            ax_bars.set_yticks(y_pos)
            ax_bars.set_yticklabels(labels, fontsize=9)
            ax_bars.set_xlabel('Percentage (%)', fontsize=10)
            ax_bars.set_title('Key Insights', fontsize=12, fontweight='bold')
            ax_bars.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, values)):
                ax_bars.text(val + 1, i, f'{val:.0f}%', va='center', ha='left', fontsize=9, fontweight='bold')
        
        # Data table
        table_data = []
        for _, row in df_class.head(8).iterrows():  # Show top 8 classes
            table_data.append([
                row['Class'],
                f"{row['SKU_Pct']:.1f}%",
                f"{row['Volume_Pct']:.1f}%", 
                f"{row['Lines_Pct']:.1f}%"
            ])
        
        if table_data:
            table = ax_table.table(cellText=table_data,
                                  colLabels=['ABC-FMS Class', 'SKU %', 'Volume %', 'Lines %'],
                                  cellLoc='center',
                                  loc='center',
                                  bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Style table
            for i in range(len(table_data) + 1):
                for j in range(4):
                    cell = table[(i, j)]
                    if i == 0:  # Header
                        cell.set_facecolor('#2196F3')
                        cell.set_text_props(weight='bold', color='white')
                    else:
                        cell.set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')
        
        plt.suptitle('SKU Profile 2D Classification Analysis\nStrategic Distribution Patterns', 
                     fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        
        # Save chart
        filename = "sku_profile_2d_classification.png"
        filepath = self.charts_dir / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"SKU Profile 2D classification chart saved: {filename}")
        return str(filepath)
    
    def create_fte_requirements_timeline_chart(self, manpower_analysis: Dict) -> str:
        """Create FTE requirements timeline chart showing daily staffing needs."""
        self.logger.info("Creating FTE requirements timeline chart")
        
        try:
            # Get daily staffing data
            daily_patterns = manpower_analysis.get('daily_staffing_analysis', {})
            daily_fte_data = daily_patterns.get('daily_fte_data')
            
            if daily_fte_data is None or daily_fte_data.empty:
                self.logger.warning("❌ No daily FTE data available for timeline chart")
                return ""
            
            self.logger.info(f"📊 Creating FTE timeline chart with data shape: {daily_fte_data.shape}")
            
            # Set up the plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            fig.suptitle('Daily FTE Requirements Analysis\nStaffing Timeline & Utilization Patterns', 
                        fontsize=16, fontweight='bold', y=0.95)
            
            # Prepare data
            dates = pd.to_datetime(daily_fte_data['Date'])
            fte_required = daily_fte_data['FTE_Required_Total']
            total_cases = daily_fte_data['Total_Cases']
            total_lines = daily_fte_data['Total_Lines']
            
            # Top chart: FTE Requirements Timeline
            ax1.fill_between(dates, fte_required, alpha=0.3, color='#2196F3', label='FTE Required')
            ax1.plot(dates, fte_required, color='#1976D2', linewidth=2.5, marker='o', markersize=4)
            
            # Add benchmark lines
            avg_fte = fte_required.mean()
            peak_threshold = fte_required.quantile(0.85)
            
            ax1.axhline(y=avg_fte, color='#4CAF50', linestyle='--', alpha=0.7, 
                       label=f'Average ({avg_fte:.1f} FTE)')
            ax1.axhline(y=peak_threshold, color='#FF9800', linestyle='--', alpha=0.7,
                       label=f'Peak Threshold ({peak_threshold:.1f} FTE)')
            
            # Highlight peak days
            peak_days = daily_fte_data[daily_fte_data['FTE_Required_Total'] >= peak_threshold]
            if not peak_days.empty:
                peak_dates = pd.to_datetime(peak_days['Date'])
                peak_fte = peak_days['FTE_Required_Total']
                ax1.scatter(peak_dates, peak_fte, color='#F44336', s=80, 
                           zorder=5, label='Peak Days', alpha=0.8)
            
            ax1.set_ylabel('FTE Required', fontsize=11, fontweight='bold')
            ax1.set_title('Daily FTE Requirements Timeline', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper right')
            ax1.tick_params(axis='x', rotation=45)
            
            # Bottom chart: Volume vs FTE Correlation
            ax2_twin = ax2.twinx()
            
            # Cases (bars)
            bars = ax2.bar(dates, total_cases, alpha=0.6, color='#607D8B', 
                          label='Daily Cases', width=0.8)
            
            # FTE line
            line = ax2_twin.plot(dates, fte_required, color='#FF5722', linewidth=3, 
                               marker='s', markersize=5, label='FTE Required')
            
            ax2.set_ylabel('Daily Cases', fontsize=11, fontweight='bold', color='#607D8B')
            ax2_twin.set_ylabel('FTE Required', fontsize=11, fontweight='bold', color='#FF5722')
            ax2.set_title('Volume vs FTE Correlation Analysis', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
            
            # Combine legends
            bars_legend = ax2.legend(loc='upper left')
            line_legend = ax2_twin.legend(loc='upper right')
            
            # Color coordinate axes
            ax2.tick_params(axis='y', colors='#607D8B')
            ax2_twin.tick_params(axis='y', colors='#FF5722')
            
            plt.tight_layout()
            
            # Save chart
            filename = "fte_requirements_timeline.png"
            filepath = self.charts_dir / filename
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            plt.close()
            
            self.logger.info(f"FTE requirements timeline chart saved: {filename}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error creating FTE timeline chart: {str(e)}")
            plt.close()
            return ""
    
    def create_workforce_planning_summary_chart(self, manpower_analysis: Dict) -> str:
        """Create workforce planning summary chart with key metrics and recommendations."""
        self.logger.info("Creating workforce planning summary chart")
        
        try:
            # Get analysis data
            summary_metrics = manpower_analysis.get('summary_metrics', {})
            peak_analysis = manpower_analysis.get('peak_analysis', {})
            cost_analysis = manpower_analysis.get('cost_analysis', {})
            shift_planning = manpower_analysis.get('shift_planning', {})
            
            if not summary_metrics:
                self.logger.warning("❌ No summary metrics available for workforce planning chart")
                return ""
            
            # Set up the plot with subplots
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # Main title
            fig.suptitle('Workforce Planning Dashboard\nComprehensive FTE Analysis & Recommendations', 
                        fontsize=18, fontweight='bold', y=0.95)
            
            # 1. FTE Requirements Gauge (top left)
            ax1 = fig.add_subplot(gs[0, 0])
            core_fte = summary_metrics.get('recommended_core_fte', 0)
            peak_fte = summary_metrics.get('peak_fte_requirement', 0)
            avg_fte = summary_metrics.get('average_fte_requirement', 0)
            
            fte_data = [core_fte, peak_fte - core_fte if peak_fte > core_fte else 0]
            fte_labels = ['Core Staff', 'Flex Capacity']
            colors = ['#4CAF50', '#FF9800']
            
            wedges, texts, autotexts = ax1.pie(fte_data, labels=fte_labels, autopct='%1.0f FTE',
                                              colors=colors, startangle=90, textprops={'fontsize': 10})
            ax1.set_title(f'FTE Distribution\nTotal Peak: {peak_fte:.0f} FTE', 
                         fontsize=12, fontweight='bold')
            
            # 2. Cost Breakdown (top center)
            ax2 = fig.add_subplot(gs[0, 1])
            if cost_analysis:
                monthly_budget = cost_analysis.get('total_monthly_labor', 0)
                cost_per_case = cost_analysis.get('cost_per_case', 0)
                
                # Create cost metrics display
                cost_metrics = [
                    f"Monthly Budget:\n${monthly_budget:,.0f}",
                    f"Cost per Case:\n${cost_per_case:.2f}",
                    f"Annual Budget:\n${monthly_budget * 12:,.0f}"
                ]
                
                for i, metric in enumerate(cost_metrics):
                    ax2.text(0.5, 0.8 - i * 0.3, metric, ha='center', va='center',
                            fontsize=11, fontweight='bold', 
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='#E3F2FD', alpha=0.8))
                
                ax2.set_xlim(0, 1)
                ax2.set_ylim(0, 1)
                ax2.axis('off')
                ax2.set_title('Labor Cost Analysis', fontsize=12, fontweight='bold')
            
            # 3. Peak vs Average Comparison (top right)
            ax3 = fig.add_subplot(gs[0, 2])
            fte_comparison = [avg_fte, peak_fte - avg_fte if peak_fte > avg_fte else 0]
            comparison_labels = ['Average Demand', 'Peak Surge']
            comparison_colors = ['#2196F3', '#F44336']
            
            bars = ax3.bar(comparison_labels, fte_comparison, color=comparison_colors, alpha=0.7)
            ax3.set_ylabel('FTE Required', fontweight='bold')
            ax3.set_title('Peak vs Average Staffing', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, fte_comparison):
                if value > 0:
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
            
            # 4. Category Labor Distribution (middle left)
            ax4 = fig.add_subplot(gs[1, 0])
            category_analysis = manpower_analysis.get('category_labor_analysis', {})
            category_breakdown = category_analysis.get('category_breakdown')
            
            if category_breakdown is not None and not category_breakdown.empty:
                categories = category_breakdown['Category']
                fte_percentages = category_breakdown['FTE_Percentage']
                
                bars = ax4.barh(categories, fte_percentages, color=plt.cm.Set3(range(len(categories))))
                ax4.set_xlabel('FTE Allocation (%)', fontweight='bold')
                ax4.set_title('Category Labor Distribution', fontsize=12, fontweight='bold')
                ax4.grid(True, alpha=0.3, axis='x')
                
                # Add percentage labels
                for bar, value in zip(bars, fte_percentages):
                    if value > 0:
                        ax4.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                                f'{value:.1f}%', ha='left', va='center', fontweight='bold')
            
            # 5. Shift Planning Recommendations (middle center & right)
            ax5 = fig.add_subplot(gs[1, 1:])
            if shift_planning:
                core_staffing = shift_planning.get('core_staffing', {})
                peak_staffing = shift_planning.get('peak_staffing', {})
                
                if core_staffing and peak_staffing:
                    shifts = ['Shift 1 (Day)', 'Shift 2 (Evening)']
                    core_values = [core_staffing.get('shift_1_day', 0), core_staffing.get('shift_2_evening', 0)]
                    peak_values = [peak_staffing.get('shift_1_day', 0), peak_staffing.get('shift_2_evening', 0)]
                    
                    x = np.arange(len(shifts))
                    width = 0.35
                    
                    bars1 = ax5.bar(x - width/2, core_values, width, label='Core Staffing', 
                                   color='#4CAF50', alpha=0.8)
                    bars2 = ax5.bar(x + width/2, peak_values, width, label='Peak Staffing',
                                   color='#FF5722', alpha=0.8)
                    
                    ax5.set_xlabel('Shifts', fontweight='bold')
                    ax5.set_ylabel('FTE Required', fontweight='bold')
                    ax5.set_title('Shift Planning Recommendations', fontsize=12, fontweight='bold')
                    ax5.set_xticks(x)
                    ax5.set_xticklabels(shifts)
                    ax5.legend()
                    ax5.grid(True, alpha=0.3, axis='y')
                    
                    # Add value labels
                    for bars in [bars1, bars2]:
                        for bar in bars:
                            height = bar.get_height()
                            if height > 0:
                                ax5.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                                        f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
            
            # 6. Key Recommendations (bottom)
            ax6 = fig.add_subplot(gs[2, :])
            
            # Generate recommendations text
            recommendations = []
            if peak_analysis:
                peak_days_pct = peak_analysis.get('peak_days_percentage', 0)
                recommendations.append(f"• Peak staffing needed {peak_days_pct:.1f}% of operating days")
            
            if summary_metrics:
                flex_capacity = summary_metrics.get('flex_capacity_needed', 0)
                if flex_capacity > 0:
                    recommendations.append(f"• Maintain {flex_capacity:.0f} FTE flexible capacity for peak periods")
            
            if cost_analysis:
                cost_per_case = cost_analysis.get('cost_per_case', 0)
                if cost_per_case > 0:
                    recommendations.append(f"• Current labor cost efficiency: ${cost_per_case:.2f} per case")
            
            recommendations.extend([
                "• Consider cross-training staff for operational flexibility",
                "• Implement workforce scheduling software for optimal allocation",
                "• Monitor productivity metrics against industry benchmarks"
            ])
            
            # Display recommendations
            rec_text = '\n'.join(recommendations)
            ax6.text(0.05, 0.95, 'Key Workforce Planning Recommendations:', 
                    transform=ax6.transAxes, fontsize=14, fontweight='bold')
            ax6.text(0.05, 0.05, rec_text, transform=ax6.transAxes, fontsize=11,
                    verticalalignment='bottom',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='#F5F5F5', alpha=0.9))
            ax6.axis('off')
            
            plt.tight_layout()
            
            # Save chart
            filename = "workforce_planning_dashboard.png"
            filepath = self.charts_dir / filename
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            plt.close()
            
            self.logger.info(f"Workforce planning dashboard chart saved: {filename}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error creating workforce planning chart: {str(e)}")
            plt.close()
            return ""