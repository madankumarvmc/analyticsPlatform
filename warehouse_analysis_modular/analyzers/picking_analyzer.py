#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Case vs Piece Picking Analysis Module

Analyzes picking methodology patterns, operational complexity by category,
and provides optimization recommendations for warehouse picking operations.
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
    validate_dataframe, setup_logging
)

logger = setup_logging()


class PickingAnalyzer:
    """
    Picking Methodology Analyzer providing case vs piece picking analysis
    and operational optimization insights.
    
    Features:
    - Case vs Piece picking pattern analysis by category
    - Operational complexity assessment by picking method
    - Category-level picking optimization recommendations
    - Piece picking impact assessment (% by lines vs % by volume)
    - Eaches-only lines identification and analysis
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def analyze_picking_patterns(self, enriched_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive picking methodology analysis.
        
        Args:
            enriched_data: DataFrame with enriched order data including categories
            
        Returns:
            Dictionary containing all picking analysis results
        """
        self.logger.info("Starting picking patterns analysis")
        
        # Validate input data
        if not validate_dataframe(enriched_data, required_columns=[
            'Sku Code', 'Qty in Cases', 'Qty in Eaches', 'Case_Equivalent'
        ]):
            raise ValueError("Invalid input data for picking analysis")
        
        results = {}
        
        # 1. Overall picking methodology breakdown
        overall_picking = self._analyze_overall_picking_patterns(enriched_data)
        results['overall_picking_patterns'] = overall_picking
        
        # 2. Category-level picking analysis
        category_picking = self._analyze_category_picking_patterns(enriched_data)
        results['category_picking_analysis'] = category_picking
        
        # 3. Piece picking impact assessment
        piece_impact = self._calculate_piece_picking_impact(enriched_data)
        results['piece_picking_impact'] = piece_impact
        
        # 4. Eaches-only lines analysis
        eaches_only = self._analyze_eaches_only_lines(enriched_data)
        results['eaches_only_analysis'] = eaches_only
        
        # 5. Picking complexity scoring
        complexity_score = self._calculate_picking_complexity(enriched_data, category_picking)
        results['picking_complexity'] = complexity_score
        
        # 6. Optimization recommendations
        recommendations = self._generate_picking_recommendations(
            overall_picking, category_picking, piece_impact, eaches_only
        )
        results['picking_recommendations'] = recommendations
        
        self.logger.info("Picking patterns analysis completed")
        return results
    
    def _analyze_overall_picking_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze overall picking patterns across all orders."""
        self.logger.info("Analyzing overall picking patterns")
        
        # Classify each line by picking type
        data['Picking_Type'] = data.apply(self._classify_picking_type, axis=1)
        
        # Overall statistics
        total_lines = len(data)
        total_volume = data['Case_Equivalent'].sum()
        
        picking_summary = data.groupby('Picking_Type').agg({
            'Case_Equivalent': ['count', 'sum'],
            'Qty in Cases': 'sum',
            'Qty in Eaches': 'sum'
        })
        
        # Flatten column names
        picking_summary.columns = ['Line_Count', 'Volume_Sum', 'Total_Cases', 'Total_Eaches']
        picking_summary = picking_summary.reset_index()
        
        # Round numeric columns safely
        for col in ['Volume_Sum', 'Total_Cases', 'Total_Eaches']:
            if col in picking_summary.columns:
                picking_summary[col] = picking_summary[col].round(2)
        
        # Calculate percentages
        if total_lines > 0:
            picking_summary['Line_Percentage'] = (picking_summary['Line_Count'] / total_lines * 100).round(1)
        else:
            picking_summary['Line_Percentage'] = 0.0
            
        if total_volume > 0:
            picking_summary['Volume_Percentage'] = (picking_summary['Volume_Sum'] / total_volume * 100).round(1)
        else:
            picking_summary['Volume_Percentage'] = 0.0
        
        # Key insights
        case_only_lines_pct = picking_summary[picking_summary['Picking_Type'] == 'Case_Only']['Line_Percentage'].sum()
        piece_only_lines_pct = picking_summary[picking_summary['Picking_Type'] == 'Piece_Only']['Line_Percentage'].sum()
        mixed_lines_pct = picking_summary[picking_summary['Picking_Type'] == 'Mixed']['Line_Percentage'].sum()
        
        piece_volume_pct = picking_summary[
            picking_summary['Picking_Type'].isin(['Piece_Only', 'Mixed'])
        ]['Volume_Percentage'].sum()
        
        return {
            'picking_summary': picking_summary,
            'total_lines': total_lines,
            'total_volume': total_volume,
            'key_metrics': {
                'case_only_lines_percentage': case_only_lines_pct,
                'piece_only_lines_percentage': piece_only_lines_pct,
                'mixed_lines_percentage': mixed_lines_pct,
                'piece_picking_volume_impact': piece_volume_pct
            }
        }
    
    def _analyze_category_picking_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze picking patterns by category."""
        self.logger.info("Analyzing category-level picking patterns")
        
        if 'Category' not in data.columns:
            # If no category column, try to identify from other columns or create generic categories
            if 'Sku Code' in data.columns:
                # Create basic categories from SKU patterns (this is a fallback)
                data['Category'] = data['Sku Code'].str[:2]  # Use first 2 characters as category
            else:
                data['Category'] = 'General'
        
        # Ensure Picking_Type is available
        if 'Picking_Type' not in data.columns:
            data['Picking_Type'] = data.apply(self._classify_picking_type, axis=1)
        
        # Category-level analysis
        category_analysis = []
        
        for category in data['Category'].unique():
            if pd.isna(category):
                continue
                
            category_data = data[data['Category'] == category]
            total_lines_cat = len(category_data)
            total_volume_cat = category_data['Case_Equivalent'].sum()
            
            # Picking type breakdown for this category
            picking_breakdown = category_data.groupby('Picking_Type').agg({
                'Case_Equivalent': ['count', 'sum']
            })
            
            # Flatten columns
            picking_breakdown.columns = ['Line_Count', 'Volume_Sum']
            picking_breakdown = picking_breakdown.reset_index()
            
            # Calculate percentages
            if total_lines_cat > 0:
                picking_breakdown['Line_Percentage'] = (picking_breakdown['Line_Count'] / total_lines_cat * 100).round(1)
            else:
                picking_breakdown['Line_Percentage'] = 0.0
                
            if total_volume_cat > 0:
                picking_breakdown['Volume_Percentage'] = (picking_breakdown['Volume_Sum'] / total_volume_cat * 100).round(1)
            else:
                picking_breakdown['Volume_Percentage'] = 0.0
            
            # Calculate key metrics for this category
            piece_only_lines = picking_breakdown[picking_breakdown['Picking_Type'] == 'Piece_Only']['Line_Count'].sum()
            case_only_lines = picking_breakdown[picking_breakdown['Picking_Type'] == 'Case_Only']['Line_Count'].sum()
            
            # PCS lines analysis (lines with any eaches quantity)
            pcs_lines = len(category_data[category_data['Qty in Eaches'] > 0])
            pcs_only_lines = len(category_data[
                (category_data['Qty in Eaches'] > 0) & (category_data['Qty in Cases'] == 0)
            ])
            
            category_summary = {
                'category': category,
                'total_lines': total_lines_cat,
                'total_volume': total_volume_cat,
                'picking_breakdown': picking_breakdown,
                'pcs_lines_count': pcs_lines,
                'pcs_lines_percentage': round(pcs_lines / total_lines_cat * 100, 1) if total_lines_cat > 0 else 0,
                'pcs_only_lines_count': pcs_only_lines,
                'pcs_only_lines_percentage': round(pcs_only_lines / total_lines_cat * 100, 1) if total_lines_cat > 0 else 0,
                'case_only_lines_percentage': round(case_only_lines / total_lines_cat * 100, 1) if total_lines_cat > 0 else 0,
                'operational_complexity': self._calculate_category_complexity(category_data)
            }
            
            category_analysis.append(category_summary)
        
        # Sort by total volume descending
        category_analysis.sort(key=lambda x: x['total_volume'], reverse=True)
        
        return {
            'category_breakdown': category_analysis,
            'summary_insights': self._generate_category_insights(category_analysis)
        }
    
    def _calculate_piece_picking_impact(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate the impact of piece picking on operations."""
        self.logger.info("Calculating piece picking impact")
        
        # Lines with any piece picking (eaches > 0)
        piece_lines = data[data['Qty in Eaches'] > 0]
        total_lines = len(data)
        total_volume = data['Case_Equivalent'].sum()
        
        piece_impact = {
            'total_piece_lines': len(piece_lines),
            'piece_lines_percentage': round(len(piece_lines) / total_lines * 100, 1) if total_lines > 0 else 0,
            'piece_volume_contribution': piece_lines['Case_Equivalent'].sum(),
            'piece_volume_percentage': round(piece_lines['Case_Equivalent'].sum() / total_volume * 100, 1) if total_volume > 0 else 0,
            'avg_eaches_per_piece_line': round(piece_lines['Qty in Eaches'].mean(), 1) if len(piece_lines) > 0 else 0,
            'max_eaches_per_line': piece_lines['Qty in Eaches'].max() if len(piece_lines) > 0 else 0
        }
        
        # Key insight from screenshots: "Piece picking - around 10-15% by lines but contributing to just <1% by volume"
        piece_impact['insight'] = {
            'lines_impact_range': '10-15%' if 10 <= piece_impact['piece_lines_percentage'] <= 15 else f"{piece_impact['piece_lines_percentage']}%",
            'volume_impact_assessment': 'minimal' if piece_impact['piece_volume_percentage'] < 1 else 'significant',
            'operational_efficiency_impact': 'high_touch_low_volume' if piece_impact['piece_lines_percentage'] > piece_impact['piece_volume_percentage'] else 'balanced'
        }
        
        return piece_impact
    
    def _analyze_eaches_only_lines(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze lines that are eaches/pieces only."""
        self.logger.info("Analyzing eaches-only lines")
        
        # Eaches-only lines (eaches > 0 AND cases = 0)
        eaches_only = data[(data['Qty in Eaches'] > 0) & (data['Qty in Cases'] == 0)]
        total_lines = len(data)
        
        eaches_analysis = {
            'eaches_only_lines_count': len(eaches_only),
            'eaches_only_percentage': round(len(eaches_only) / total_lines * 100, 1) if total_lines > 0 else 0,
            'avg_eaches_per_line': round(eaches_only['Qty in Eaches'].mean(), 1) if len(eaches_only) > 0 else 0,
            'total_eaches_volume': eaches_only['Qty in Eaches'].sum() if len(eaches_only) > 0 else 0
        }
        
        # Category breakdown of eaches-only lines
        if 'Category' in data.columns and len(eaches_only) > 0:
            eaches_by_category = eaches_only.groupby('Category').agg({
                'Qty in Eaches': ['count', 'sum', 'mean']
            })
            eaches_by_category.columns = ['Line_Count', 'Total_Eaches', 'Avg_Eaches']
            eaches_by_category = eaches_by_category.reset_index()
            
            # Round the Avg_Eaches column safely
            eaches_by_category['Avg_Eaches'] = eaches_by_category['Avg_Eaches'].round(1)
            
            eaches_analysis['category_breakdown'] = eaches_by_category
        
        return eaches_analysis
    
    def _calculate_picking_complexity(self, data: pd.DataFrame, category_analysis: Dict) -> Dict[str, Any]:
        """Calculate picking complexity score."""
        self.logger.info("Calculating picking complexity")
        
        complexity_factors = {}
        
        # Mixed picking complexity (lines with both cases and eaches)
        mixed_lines = data[(data['Qty in Cases'] > 0) & (data['Qty in Eaches'] > 0)]
        complexity_factors['mixed_picking_percentage'] = round(len(mixed_lines) / len(data) * 100, 1) if len(data) > 0 else 0
        
        # Category diversity in picking methods
        categories_with_pieces = sum(1 for cat in category_analysis['category_breakdown'] 
                                   if cat['pcs_lines_percentage'] > 5)  # Categories with >5% piece lines
        total_categories = len(category_analysis['category_breakdown'])
        complexity_factors['picking_diversity'] = round(categories_with_pieces / total_categories * 100, 1) if total_categories > 0 else 0
        
        # Small quantity complexity (high number of small picks)
        small_eaches = data[(data['Qty in Eaches'] > 0) & (data['Qty in Eaches'] <= 5)]
        complexity_factors['small_quantity_picks'] = round(len(small_eaches) / len(data) * 100, 1) if len(data) > 0 else 0
        
        # Overall complexity score
        weights = {'mixed_picking_percentage': 0.4, 'picking_diversity': 0.3, 'small_quantity_picks': 0.3}
        overall_score = sum(complexity_factors[factor] * weights[factor] for factor in complexity_factors)
        
        complexity_level = 'Low' if overall_score < 20 else 'Medium' if overall_score < 40 else 'High'
        
        return {
            'complexity_factors': complexity_factors,
            'overall_complexity_score': overall_score,
            'complexity_level': complexity_level
        }
    
    def _generate_picking_recommendations(self, overall: Dict, category: Dict, 
                                        piece_impact: Dict, eaches_only: Dict) -> Dict[str, Any]:
        """Generate picking optimization recommendations."""
        self.logger.info("Generating picking recommendations")
        
        recommendations = {
            'immediate_actions': [],
            'medium_term_initiatives': [],
            'strategic_improvements': []
        }
        
        # Immediate actions based on analysis
        piece_pct = piece_impact['piece_lines_percentage']
        if piece_pct > 10:
            recommendations['immediate_actions'].append({
                'action': 'Optimize piece picking zones',
                'rationale': f'{piece_pct:.1f}% of lines require piece picking but contribute minimal volume',
                'expected_benefit': 'Reduce picking time and improve efficiency'
            })
        
        eaches_only_pct = eaches_only['eaches_only_percentage']
        if eaches_only_pct > 5:
            recommendations['immediate_actions'].append({
                'action': 'Consolidate eaches-only picks',
                'rationale': f'{eaches_only_pct:.1f}% of lines are eaches-only picks',
                'expected_benefit': 'Batch small picks for improved productivity'
            })
        
        # Medium-term initiatives
        high_piece_categories = [cat for cat in category['category_breakdown'] 
                               if cat['pcs_lines_percentage'] > 20]
        if high_piece_categories:
            recommendations['medium_term_initiatives'].append({
                'initiative': 'Category-specific picking optimization',
                'details': f"Focus on {len(high_piece_categories)} categories with high piece picking rates",
                'timeline': '3-6 months'
            })
        
        # Strategic improvements
        recommendations['strategic_improvements'].append({
            'improvement': 'Implement pick-and-pack optimization',
            'description': 'Deploy advanced picking methodologies based on volume-complexity analysis',
            'roi_potential': 'High - significant efficiency gains in high-volume operations'
        })
        
        return recommendations
    
    def _classify_picking_type(self, row) -> str:
        """Classify picking type for a single row."""
        has_cases = row['Qty in Cases'] > 0
        has_eaches = row['Qty in Eaches'] > 0
        
        if has_cases and has_eaches:
            return 'Mixed'
        elif has_cases and not has_eaches:
            return 'Case_Only'
        elif not has_cases and has_eaches:
            return 'Piece_Only'
        else:
            return 'Unknown'
    
    def _calculate_category_complexity(self, category_data: pd.DataFrame) -> float:
        """Calculate operational complexity score for a category."""
        mixed_pct = len(category_data[(category_data['Qty in Cases'] > 0) & 
                                    (category_data['Qty in Eaches'] > 0)]) / len(category_data) * 100
        
        eaches_variety = category_data['Qty in Eaches'].nunique() if 'Qty in Eaches' in category_data.columns else 1
        cases_variety = category_data['Qty in Cases'].nunique() if 'Qty in Cases' in category_data.columns else 1
        
        complexity = (mixed_pct * 0.6) + (min(eaches_variety + cases_variety, 20) * 0.4)
        return round(complexity, 1)
    
    def _generate_category_insights(self, category_analysis: List[Dict]) -> Dict[str, Any]:
        """Generate insights from category analysis."""
        total_categories = len(category_analysis)
        
        high_pcs_categories = [cat for cat in category_analysis if cat['pcs_lines_percentage'] > 15]
        case_dominant_categories = [cat for cat in category_analysis if cat['case_only_lines_percentage'] > 80]
        
        insights = {
            'total_categories_analyzed': total_categories,
            'high_piece_picking_categories': len(high_pcs_categories),
            'case_dominant_categories': len(case_dominant_categories),
            'mixed_picking_prevalence': sum(1 for cat in category_analysis 
                                          if cat['operational_complexity'] > 30) / total_categories * 100 if total_categories > 0 else 0
        }
        
        return insights