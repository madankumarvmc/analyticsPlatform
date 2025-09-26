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