#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced ABC-FMS Cross-Classification Analysis Module

Provides advanced 2D classification matrix analysis with detailed breakdowns
for volume, lines, and SKU count distribution patterns.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple

# Import from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from warehouse_analysis_modular.utils.helpers import (
    validate_dataframe, normalize_abc_fms_values, setup_logging
)

logger = setup_logging()


class EnhancedABCFMSAnalyzer:
    """
    Enhanced ABC-FMS Cross-Classification Analyzer providing advanced 2D matrix analysis
    and sophisticated operational insights matching industry analysis standards.
    
    Features:
    - 2D Classification Matrix with volume/lines/SKU breakdowns
    - Advanced segmentation insights (AF, BM, CS classes etc.)
    - Cross-classification effectiveness scoring
    - Market SKU level analysis capability
    - Detailed percentage breakdowns by dimension
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze_enhanced_abc_fms(self, sku_profile: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive enhanced ABC-FMS cross-classification analysis.
        
        Args:
            sku_profile: DataFrame with ABC and FMS classifications
            
        Returns:
            Dictionary containing all enhanced analysis results
        """
        self.logger.info("Starting enhanced ABC-FMS cross-classification analysis")
        
        # Validate input data
        required_columns = ["Sku Code", "ABC", "FMS", "Total_Case_Equiv", "Total_Order_Lines"]
        if not validate_dataframe(sku_profile, required_columns=required_columns):
            raise ValueError("Invalid input data for enhanced ABC-FMS analysis")
        
        results = {}
        
        # 1. 2D Classification Matrix Analysis
        classification_matrix = self._create_2d_classification_matrix(sku_profile)
        results['classification_matrix_2d'] = classification_matrix
        
        # 2. Advanced segmentation analysis
        segmentation_analysis = self._analyze_advanced_segmentation(sku_profile, classification_matrix)
        results['advanced_segmentation'] = segmentation_analysis
        
        # 3. Cross-classification effectiveness
        effectiveness_analysis = self._analyze_classification_effectiveness(sku_profile, classification_matrix)
        results['classification_effectiveness'] = effectiveness_analysis
        
        # 4. Detailed percentage breakdowns
        percentage_breakdowns = self._calculate_detailed_percentages(sku_profile, classification_matrix)
        results['percentage_breakdowns'] = percentage_breakdowns
        
        # 5. Strategic insights and recommendations
        strategic_insights = self._generate_strategic_insights(
            classification_matrix, segmentation_analysis, effectiveness_analysis
        )
        results['strategic_insights'] = strategic_insights
        
        # 6. Market SKU level analysis
        market_sku_analysis = self._analyze_market_sku_level(sku_profile)
        results['market_sku_analysis'] = market_sku_analysis
        
        self.logger.info("Enhanced ABC-FMS cross-classification analysis completed")
        return results
    
    def _create_2d_classification_matrix(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create comprehensive 2D classification matrix."""
        self.logger.info("Creating 2D classification matrix")
        
        # Normalize ABC and FMS values
        data = normalize_abc_fms_values(data)
        
        # SKU Count Matrix
        sku_count_matrix = pd.crosstab(
            data['ABC'], data['FMS'], 
            margins=True, margins_name='Grand Total'
        )
        
        # Volume Matrix (Case Equivalent)
        volume_matrix = pd.crosstab(
            data['ABC'], data['FMS'], 
            values=data['Total_Case_Equiv'], aggfunc='sum',
            margins=True, margins_name='Grand Total'
        ).fillna(0)
        
        # Lines Matrix
        lines_matrix = pd.crosstab(
            data['ABC'], data['FMS'],
            values=data['Total_Order_Lines'], aggfunc='sum',
            margins=True, margins_name='Grand Total'
        ).fillna(0)
        
        # Percentage matrices
        sku_pct_matrix = self._calculate_percentage_matrix(sku_count_matrix)
        volume_pct_matrix = self._calculate_percentage_matrix(volume_matrix)
        lines_pct_matrix = self._calculate_percentage_matrix(lines_matrix)
        
        # Combined matrix for detailed analysis
        combined_matrix = self._create_combined_matrix(
            sku_count_matrix, volume_matrix, lines_matrix,
            sku_pct_matrix, volume_pct_matrix, lines_pct_matrix
        )
        
        return {
            'sku_count_matrix': sku_count_matrix,
            'volume_matrix': volume_matrix,
            'lines_matrix': lines_matrix,
            'sku_percentage_matrix': sku_pct_matrix,
            'volume_percentage_matrix': volume_pct_matrix,
            'lines_percentage_matrix': lines_pct_matrix,
            'combined_matrix': combined_matrix
        }
    
    def _analyze_advanced_segmentation(self, data: pd.DataFrame, matrices: Dict) -> Dict[str, Any]:
        """Analyze advanced segmentation patterns."""
        self.logger.info("Analyzing advanced segmentation")
        
        # Create combined classification labels (e.g., AF, BM, CS)
        data['Combined_Class'] = data['ABC'] + data['FMS']
        
        # Analyze each segment
        segment_analysis = {}
        
        for segment in data['Combined_Class'].unique():
            if pd.isna(segment):
                continue
                
            segment_data = data[data['Combined_Class'] == segment]
            
            total_skus = len(data)
            total_volume = data['Total_Case_Equiv'].sum()
            total_lines = data['Total_Order_Lines'].sum()
            
            segment_analysis[segment] = {
                'sku_count': len(segment_data),
                'sku_percentage': round(len(segment_data) / total_skus * 100, 1) if total_skus > 0 else 0,
                'volume_contribution': segment_data['Total_Case_Equiv'].sum(),
                'volume_percentage': round(segment_data['Total_Case_Equiv'].sum() / total_volume * 100, 1) if total_volume > 0 else 0,
                'lines_contribution': segment_data['Total_Order_Lines'].sum(),
                'lines_percentage': round(segment_data['Total_Order_Lines'].sum() / total_lines * 100, 1) if total_lines > 0 else 0,
                'avg_volume_per_sku': round(segment_data['Total_Case_Equiv'].mean(), 1) if len(segment_data) > 0 else 0,
                'avg_lines_per_sku': round(segment_data['Total_Order_Lines'].mean(), 1) if len(segment_data) > 0 else 0
            }
        
        # Key insights from screenshot: "AF Class - 3% SKUs contribute to 69% by volume and 31% by Lines"
        key_insights = self._identify_key_segments(segment_analysis)
        
        return {
            'segment_analysis': segment_analysis,
            'key_insights': key_insights,
            'total_segments': len(segment_analysis)
        }
    
    def _analyze_classification_effectiveness(self, data: pd.DataFrame, matrices: Dict) -> Dict[str, Any]:
        """Analyze the effectiveness of the ABC-FMS classification."""
        self.logger.info("Analyzing classification effectiveness")
        
        effectiveness_metrics = {}
        
        # ABC Classification effectiveness
        abc_groups = data.groupby('ABC').agg({
            'Total_Case_Equiv': ['sum', 'mean', 'count']
        })
        abc_groups.columns = ['Total_Volume', 'Avg_Volume', 'SKU_Count']
        abc_groups = abc_groups.reset_index()
        
        total_volume = data['Total_Case_Equiv'].sum()
        abc_groups['Volume_Percentage'] = round(abc_groups['Total_Volume'] / total_volume * 100, 1) if total_volume > 0 else 0
        
        # Calculate ABC concentration (how well A items concentrate volume)
        a_volume_pct = abc_groups[abc_groups['ABC'] == 'A']['Volume_Percentage'].sum() if 'A' in abc_groups['ABC'].values else 0
        a_sku_pct = abc_groups[abc_groups['ABC'] == 'A']['SKU_Count'].sum() / abc_groups['SKU_Count'].sum() * 100 if abc_groups['SKU_Count'].sum() > 0 else 0
        
        effectiveness_metrics['abc_effectiveness'] = {
            'a_class_volume_concentration': a_volume_pct,
            'a_class_sku_percentage': a_sku_pct,
            'concentration_ratio': a_volume_pct / a_sku_pct if a_sku_pct > 0 else 0,
            'classification_score': 'Excellent' if a_volume_pct > 60 and a_sku_pct < 30 else 'Good' if a_volume_pct > 40 else 'Needs Improvement'
        }
        
        # FMS Classification effectiveness
        fms_groups = data.groupby('FMS').agg({
            'Total_Order_Lines': ['sum', 'mean', 'count']
        })
        fms_groups.columns = ['Total_Lines', 'Avg_Lines', 'SKU_Count']
        fms_groups = fms_groups.reset_index()
        
        total_lines = data['Total_Order_Lines'].sum()
        fms_groups['Lines_Percentage'] = round(fms_groups['Total_Lines'] / total_lines * 100, 1) if total_lines > 0 else 0
        
        # Fast-moving items concentration
        fast_lines_pct = fms_groups[fms_groups['FMS'] == 'Fast']['Lines_Percentage'].sum() if 'Fast' in fms_groups['FMS'].values else 0
        fast_sku_pct = fms_groups[fms_groups['FMS'] == 'Fast']['SKU_Count'].sum() / fms_groups['SKU_Count'].sum() * 100 if fms_groups['SKU_Count'].sum() > 0 else 0
        
        effectiveness_metrics['fms_effectiveness'] = {
            'fast_class_lines_concentration': fast_lines_pct,
            'fast_class_sku_percentage': fast_sku_pct,
            'movement_concentration_ratio': fast_lines_pct / fast_sku_pct if fast_sku_pct > 0 else 0
        }
        
        # Overall classification balance
        effectiveness_metrics['classification_balance'] = self._assess_classification_balance(data)
        
        return effectiveness_metrics
    
    def _calculate_detailed_percentages(self, data: pd.DataFrame, matrices: Dict) -> Dict[str, Any]:
        """Calculate detailed percentage breakdowns for all dimensions."""
        self.logger.info("Calculating detailed percentages")
        
        percentages = {}
        
        # ABC distribution percentages
        abc_dist = data['ABC'].value_counts(normalize=True).sort_index() * 100
        percentages['abc_distribution'] = {
            'A_percentage': round(abc_dist.get('A', 0), 1),
            'B_percentage': round(abc_dist.get('B', 0), 1),
            'C_percentage': round(abc_dist.get('C', 0), 1)
        }
        
        # FMS distribution percentages
        fms_dist = data['FMS'].value_counts(normalize=True).sort_index() * 100
        percentages['fms_distribution'] = {
            'Fast_percentage': round(fms_dist.get('Fast', 0), 1),
            'Medium_percentage': round(fms_dist.get('Medium', 0), 1),
            'Slow_percentage': round(fms_dist.get('Slow', 0), 1)
        }
        
        # Cross-classification percentages (matching screenshot format)
        total_skus = len(data)
        total_volume = data['Total_Case_Equiv'].sum()
        total_lines = data['Total_Order_Lines'].sum()
        
        cross_percentages = []
        
        for abc in ['A', 'B', 'C']:
            for fms in ['Fast', 'Medium', 'Slow']:
                subset = data[(data['ABC'] == abc) & (data['FMS'] == fms)]
                
                if len(subset) > 0:
                    cross_percentages.append({
                        'ABC': abc,
                        'FMS': fms,
                        'Combined_Class': f"{abc}{fms}",
                        'SKU_Count': len(subset),
                        'SKU_Percentage': round(len(subset) / total_skus * 100, 0) if total_skus > 0 else 0,
                        'Volume_Contribution': subset['Total_Case_Equiv'].sum(),
                        'Volume_Percentage': round(subset['Total_Case_Equiv'].sum() / total_volume * 100, 0) if total_volume > 0 else 0,
                        'Lines_Contribution': subset['Total_Order_Lines'].sum(),
                        'Lines_Percentage': round(subset['Total_Order_Lines'].sum() / total_lines * 100, 0) if total_lines > 0 else 0
                    })
        
        percentages['cross_classification_detail'] = pd.DataFrame(cross_percentages)
        
        return percentages
    
    def _generate_strategic_insights(self, matrices: Dict, segmentation: Dict, effectiveness: Dict) -> Dict[str, Any]:
        """Generate strategic insights for warehouse optimization."""
        self.logger.info("Generating strategic insights")
        
        insights = {
            'high_impact_segments': [],
            'optimization_opportunities': [],
            'strategic_recommendations': []
        }
        
        # Identify high-impact segments (high volume contribution, low SKU percentage)
        for segment, data in segmentation['segment_analysis'].items():
            if data['volume_percentage'] > 20 and data['sku_percentage'] < 10:
                insights['high_impact_segments'].append({
                    'segment': segment,
                    'volume_contribution': data['volume_percentage'],
                    'sku_percentage': data['sku_percentage'],
                    'impact_ratio': data['volume_percentage'] / data['sku_percentage'] if data['sku_percentage'] > 0 else 0
                })
        
        # Identify optimization opportunities
        for segment, data in segmentation['segment_analysis'].items():
            # Low volume, high SKU count = potential rationalization opportunity
            if data['volume_percentage'] < 5 and data['sku_percentage'] > 20:
                insights['optimization_opportunities'].append({
                    'type': 'SKU_Rationalization',
                    'segment': segment,
                    'description': f"{segment} class has {data['sku_percentage']}% SKUs contributing only {data['volume_percentage']}% volume"
                })
            
            # High lines percentage vs volume = potential picking optimization
            if data['lines_percentage'] > data['volume_percentage'] * 1.5:
                insights['optimization_opportunities'].append({
                    'type': 'Picking_Optimization',
                    'segment': segment,
                    'description': f"{segment} class generates {data['lines_percentage']}% lines but only {data['volume_percentage']}% volume"
                })
        
        # Strategic recommendations based on analysis
        abc_score = effectiveness['abc_effectiveness']['classification_score']
        if abc_score == 'Needs Improvement':
            insights['strategic_recommendations'].append({
                'priority': 'High',
                'recommendation': 'Re-calibrate ABC classification thresholds',
                'rationale': 'Current ABC classification shows poor volume concentration'
            })
        
        insights['strategic_recommendations'].append({
            'priority': 'Medium',
            'recommendation': 'Implement segment-specific slotting strategies',
            'rationale': 'Optimize warehouse layout based on ABC-FMS cross-classification patterns'
        })
        
        return insights
    
    def _analyze_market_sku_level(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze at market SKU level as suggested in screenshot."""
        self.logger.info("Analyzing market SKU level patterns")
        
        # This analysis would ideally be done at a more granular level
        # For now, we'll provide framework and insights based on current data
        
        market_analysis = {
            'recommendation': 'Analysis should ideally be done at Market SKU level for more granular insights',
            'current_level_insights': {},
            'suggested_enhancements': []
        }
        
        # Analyze current granularity
        sku_variety = data['Sku Code'].nunique()
        avg_volume_per_sku = data.groupby('Sku Code')['Total_Case_Equiv'].sum().mean()
        volume_variance = data.groupby('Sku Code')['Total_Case_Equiv'].sum().var()
        
        market_analysis['current_level_insights'] = {
            'total_skus_analyzed': sku_variety,
            'avg_volume_per_sku': round(avg_volume_per_sku, 1) if avg_volume_per_sku else 0,
            'volume_variance_across_skus': round(volume_variance, 1) if volume_variance else 0,
            'analysis_granularity': 'SKU Level'
        }
        
        # Suggest enhancements for market-level analysis
        market_analysis['suggested_enhancements'] = [
            'Include market/region dimension in SKU classification',
            'Analyze seasonal patterns at market-SKU level',
            'Consider market-specific demand patterns in FMS classification',
            'Implement market-based inventory allocation strategies'
        ]
        
        return market_analysis
    
    def _calculate_percentage_matrix(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """Calculate percentage matrix from absolute values."""
        total = matrix.loc['Grand Total', 'Grand Total'] if 'Grand Total' in matrix.index and 'Grand Total' in matrix.columns else matrix.sum().sum()
        return round(matrix / total * 100, 1) if total > 0 else matrix * 0
    
    def _create_combined_matrix(self, sku_matrix, volume_matrix, lines_matrix, 
                               sku_pct, volume_pct, lines_pct) -> pd.DataFrame:
        """Create combined matrix for comprehensive analysis."""
        combined_data = []
        
        for abc in ['A', 'B', 'C']:
            for fms in ['Fast', 'Medium', 'Slow']:
                if abc in sku_matrix.index and fms in sku_matrix.columns:
                    combined_data.append({
                        'ABC': abc,
                        'FMS': fms,
                        'Combined_Class': f"{abc}{fms}",
                        'SKU_Count': sku_matrix.loc[abc, fms] if abc in sku_matrix.index and fms in sku_matrix.columns else 0,
                        'SKU_Percentage': sku_pct.loc[abc, fms] if abc in sku_pct.index and fms in sku_pct.columns else 0,
                        'Volume_Sum': volume_matrix.loc[abc, fms] if abc in volume_matrix.index and fms in volume_matrix.columns else 0,
                        'Volume_Percentage': volume_pct.loc[abc, fms] if abc in volume_pct.index and fms in volume_pct.columns else 0,
                        'Lines_Sum': lines_matrix.loc[abc, fms] if abc in lines_matrix.index and fms in lines_matrix.columns else 0,
                        'Lines_Percentage': lines_pct.loc[abc, fms] if abc in lines_pct.index and fms in lines_pct.columns else 0
                    })
        
        return pd.DataFrame(combined_data)
    
    def _identify_key_segments(self, segment_analysis: Dict) -> Dict[str, Any]:
        """Identify key segments based on contribution patterns."""
        key_insights = {}
        
        # Find segments with disproportionate impact (screenshot insights)
        for segment, data in segment_analysis.items():
            volume_to_sku_ratio = data['volume_percentage'] / data['sku_percentage'] if data['sku_percentage'] > 0 else 0
            lines_to_sku_ratio = data['lines_percentage'] / data['sku_percentage'] if data['sku_percentage'] > 0 else 0
            
            # High-impact segments (like AF class in screenshot: 3% SKUs → 69% volume)
            if volume_to_sku_ratio > 10:  # Volume contribution much higher than SKU percentage
                key_insights[f'{segment}_high_impact'] = {
                    'segment': segment,
                    'pattern': 'High volume concentration',
                    'sku_percentage': data['sku_percentage'],
                    'volume_percentage': data['volume_percentage'],
                    'lines_percentage': data['lines_percentage'],
                    'impact_description': f"{data['sku_percentage']}% SKUs contribute to {data['volume_percentage']}% volume and {data['lines_percentage']}% lines"
                }
            
            # Low-impact segments (like CS class: 72% SKUs → only 3% volume)
            if volume_to_sku_ratio < 0.2 and data['sku_percentage'] > 10:  # Many SKUs, little volume
                key_insights[f'{segment}_low_impact'] = {
                    'segment': segment,
                    'pattern': 'Low volume concentration',
                    'sku_percentage': data['sku_percentage'],
                    'volume_percentage': data['volume_percentage'],
                    'lines_percentage': data['lines_percentage'],
                    'impact_description': f"{data['sku_percentage']}% SKUs contribute to only {data['volume_percentage']}% volume and {data['lines_percentage']}% lines"
                }
        
        return key_insights
    
    def _assess_classification_balance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess the balance and effectiveness of the classification system."""
        balance_metrics = {}
        
        # ABC balance assessment
        abc_counts = data['ABC'].value_counts(normalize=True).sort_index() * 100
        abc_balance_score = 1 - np.std([abc_counts.get('A', 0), abc_counts.get('B', 0), abc_counts.get('C', 0)]) / 33.33
        
        balance_metrics['abc_balance'] = {
            'balance_score': round(abc_balance_score, 3),
            'distribution_evenness': 'Balanced' if abc_balance_score > 0.7 else 'Skewed',
            'a_class_percentage': round(abc_counts.get('A', 0), 1),
            'b_class_percentage': round(abc_counts.get('B', 0), 1),
            'c_class_percentage': round(abc_counts.get('C', 0), 1)
        }
        
        # FMS balance assessment
        fms_counts = data['FMS'].value_counts(normalize=True).sort_index() * 100
        fms_balance_score = 1 - np.std([fms_counts.get('Fast', 0), fms_counts.get('Medium', 0), fms_counts.get('Slow', 0)]) / 33.33
        
        balance_metrics['fms_balance'] = {
            'balance_score': round(fms_balance_score, 3),
            'distribution_evenness': 'Balanced' if fms_balance_score > 0.7 else 'Skewed',
            'fast_percentage': round(fms_counts.get('Fast', 0), 1),
            'medium_percentage': round(fms_counts.get('Medium', 0), 1),
            'slow_percentage': round(fms_counts.get('Slow', 0), 1)
        }
        
        return balance_metrics