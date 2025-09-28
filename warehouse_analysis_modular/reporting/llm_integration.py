#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM Integration Module

Handles Gemini API integration for generating report summaries with caching.
Extracted from the original Warehouse Analysis (2).py file.
"""

import os
import json
import logging
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

# Import from parent directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import (
    USE_GEMINI, GEMINI_API_KEY, GEMINI_ENDPOINT, LLM_TIMEOUT,
    CACHE_FILE, LLM_PROMPTS
)
from warehouse_analysis_modular.utils.helpers import setup_logging

logger = setup_logging()


class LLMIntegration:
    """
    Handles LLM (Gemini) API integration for generating analysis summaries.
    """
    
    def __init__(self, 
                 use_llm: bool = USE_GEMINI,
                 api_key: Optional[str] = None,
                 endpoint: Optional[str] = None,
                 cache_file: Optional[Path] = None,
                 timeout: int = LLM_TIMEOUT):
        """
        Initialize the LLM integration.
        
        Args:
            use_llm: Whether to use LLM for generating summaries
            api_key: Gemini API key (defaults to config)
            endpoint: Gemini API endpoint (defaults to config)
            cache_file: Path to cache file (defaults to config)
            timeout: Request timeout in seconds
        """
        self.use_llm = use_llm
        self.api_key = api_key or GEMINI_API_KEY
        self.endpoint = endpoint or GEMINI_ENDPOINT
        self.cache_file = cache_file or CACHE_FILE
        self.timeout = timeout
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load existing cache
        self.cache = self._load_cache()
        
        if not self.use_llm:
            self.logger.info("LLM integration disabled")
        elif not self.api_key:
            self.logger.warning("LLM API key not provided, summaries will be disabled")
            self.use_llm = False
        else:
            self.logger.info("LLM integration initialized")
    
    def _load_cache(self) -> Dict:
        """Load cached LLM responses from file."""
        if self.cache_file and self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                self.logger.info(f"Loaded {len(cache)} cached LLM responses")
                return cache
            except Exception as e:
                self.logger.warning(f"Failed to load LLM cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save cache to file."""
        try:
            # Ensure directory exists
            if self.cache_file:
                self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.cache, f, indent=2)
                self.logger.debug("Cache saved successfully")
        except Exception as e:
            self.logger.warning(f"Failed to save LLM cache: {e}")
    
    def _generate_cache_key(self, prompt: str) -> str:
        """Generate a cache key for the prompt."""
        return str(abs(hash(prompt)))
    
    def build_prompt(self, section_name: str, facts: Dict[str, Any], 
                    instruction: Optional[str] = None) -> str:
        """
        Build a structured prompt for the LLM.
        
        Args:
            section_name: Name of the analysis section
            facts: Dictionary of key facts to include
            instruction: Custom instruction (uses default from config if None)
            
        Returns:
            Formatted prompt string
        """
        # Use instruction from config if not provided
        if instruction is None:
            instruction = LLM_PROMPTS.get(section_name.lower(), {}).get('instruction', 
                'Provide a brief analysis and recommendations.')
        
        lines = [f"Section: {section_name}", ""]
        lines.append("Key facts:")
        for key, value in facts.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
        lines.append("Task:")
        lines.append(instruction)
        
        return "\n".join(lines)
    
    def call_gemini(self, prompt: str) -> str:
        """
        Call Gemini API to generate content.
        
        Args:
            prompt: The prompt to send to the API
            
        Returns:
            Generated text response
        """
        if not self.use_llm or not self.api_key:
            return "(LLM summaries disabled or API key missing.)"
        
        # Check cache first
        cache_key = self._generate_cache_key(prompt)
        if cache_key in self.cache:
            self.logger.debug("Using cached LLM response")
            return self.cache[cache_key]
        
        # Prepare API request
        url = f"{self.endpoint}?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }
        
        try:
            self.logger.debug("Making LLM API request")
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            data = response.json()
            
            if response.status_code != 200:
                error_msg = f"(LLM call HTTP {response.status_code}) {json.dumps(data)[:500]}"
                self.logger.error(f"LLM API error: {error_msg}")
                self.cache[cache_key] = error_msg
                self._save_cache()
                return error_msg
            
            # Extract text from response
            text_output = self._extract_text_from_response(data)
            
            if not text_output:
                text_output = f"(Empty response) {json.dumps(data)[:500]}"
            
        except requests.exceptions.Timeout:
            text_output = "(LLM call timed out)"
            self.logger.error("LLM API request timed out")
        except Exception as e:
            text_output = f"(LLM call failed: {str(e)})"
            self.logger.error(f"LLM API call failed: {str(e)}")
        
        # Cache the response
        self.cache[cache_key] = text_output
        self._save_cache()
        
        return text_output
    
    def _extract_text_from_response(self, data: Dict) -> str:
        """
        Extract text content from Gemini API response.
        
        Args:
            data: API response data
            
        Returns:
            Extracted text content
        """
        try:
            text_output = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
            return text_output
        except (IndexError, KeyError, TypeError) as e:
            self.logger.warning(f"Failed to extract text from response: {e}")
            return ""
    
    def generate_cover_summary(self, analysis_results: Dict, enhanced_facts: Dict = None) -> str:
        """
        Generate executive summary for the cover section with enhanced data.
        
        Args:
            analysis_results: Dictionary containing analysis results
            enhanced_facts: Optional dictionary with pre-calculated enhanced facts
            
        Returns:
            Generated summary text
        """
        if enhanced_facts:
            # Use enhanced facts provided by the caller
            facts = enhanced_facts
        else:
            # Extract key facts for cover summary (fallback)
            stats = analysis_results.get('order_statistics', {})
            facts = {
                "Generated on": datetime.now().isoformat(),
                "Total dates analyzed": stats.get('unique_dates', 'N/A'),
                "Unique SKUs": stats.get('unique_skus', 'N/A'),
                "Total order lines": stats.get('total_order_lines', 'N/A'),
                "Total case equivalent": f"{stats.get('total_case_equivalent', 0):.0f}"
            }
        
        prompt = self.build_prompt("Cover", facts)
        return self.call_gemini(prompt)
    
    def generate_date_profile_summary(self, analysis_results: Dict) -> str:
        """
        Generate summary for date profile analysis.
        
        Args:
            analysis_results: Dictionary containing analysis results
            
        Returns:
            Generated summary text
        """
        date_summary = analysis_results.get('date_order_summary')
        percentile_profile = analysis_results.get('percentile_profile')
        
        if date_summary is None or date_summary.empty:
            return "(No date summary data available)"
        
        # Get top dates by volume
        top_dates = date_summary.nlargest(3, 'Total_Case_Equiv')
        
        facts = {
            "Date range": f"{date_summary['Date'].min().date()} - {date_summary['Date'].max().date()}",
            "Peak dates (by volume)": "; ".join([
                f"{row['Date'].date()} -> {row['Total_Case_Equiv']:.0f}" 
                for _, row in top_dates.iterrows()
            ]),
            "Average daily volume": f"{date_summary['Total_Case_Equiv'].mean():.0f}",
            "95th percentile volume": f"{percentile_profile.set_index('Percentile').at['95%ile', 'Total_Case_Equiv']:.0f}" if percentile_profile is not None else "N/A"
        }
        
        prompt = self.build_prompt("Date Profile", facts)
        return self.call_gemini(prompt)
    
    def generate_percentile_summary(self, analysis_results: Dict) -> str:
        """
        Generate summary for percentile analysis with enhanced calculations.
        
        Args:
            analysis_results: Dictionary containing analysis results
            
        Returns:
            Generated summary text
        """
        from .prompts_config import get_prompt_by_type
        
        percentile_profile = analysis_results.get('percentile_profile')
        
        if percentile_profile is None or percentile_profile.empty:
            return "(No percentile data available)"
        
        # Enhanced percentile calculations
        pct_data = percentile_profile.set_index('Percentile')['Total_Case_Equiv']
        
        # Calculate key ratios and differences
        p95 = pct_data.get('95%ile', 0)
        p50 = pct_data.get('50%ile', 0) 
        p75 = pct_data.get('75%ile', 0)
        max_val = pct_data.get('Max', 0)
        
        # Capacity planning insights
        capacity_spread_ratio = p95 / p50 if p50 > 0 else 0
        peak_buffer = ((max_val - p95) / p95) * 100 if p95 > 0 else 0
        operational_range = p75 - p50
        
        facts = {
            "50th percentile": f"{p50:.0f}",
            "75th percentile": f"{p75:.0f}",
            "95th percentile": f"{p95:.0f}",
            "Maximum value": f"{max_val:.0f}",
            "Capacity spread ratio": f"{capacity_spread_ratio:.1f}x",
            "Peak buffer above 95th": f"{peak_buffer:.0f}%",
            "Operational range (75th-50th)": f"{operational_range:.0f}",
            "Planning threshold": f"95th percentile ({p95:.0f}) recommended for infrastructure sizing"
        }
        
        prompt_config = get_prompt_by_type('section', 'percentiles')
        prompt = self.build_prompt(
            prompt_config['context'], 
            facts, 
            prompt_config['instruction']
        )
        return self.call_gemini(prompt)
    
    def generate_sku_profile_summary(self, analysis_results: Dict) -> str:
        """
        Generate summary for SKU profile analysis with enhanced calculations.
        
        Args:
            analysis_results: Dictionary containing analysis results
            
        Returns:
            Generated summary text
        """
        from .prompts_config import get_prompt_by_type
        
        sku_summary = analysis_results.get('sku_order_summary')
        sku_profile = analysis_results.get('sku_profile_abc_fms')
        
        # Use whichever data is available
        data_source = sku_summary if sku_summary is not None else sku_profile
        
        if data_source is None or data_source.empty:
            return "(No SKU data available)"
        
        # Find volume column
        volume_col = None
        for col in ["Order_Volume_CE", "Total_Case_Equiv", "Case_Equivalent"]:
            if col in data_source.columns:
                volume_col = col
                break
        
        if volume_col is None:
            return "(No volume column found in SKU data)"
        
        # Enhanced calculations
        total_skus = len(data_source)
        total_volume = data_source[volume_col].sum()
        
        # Calculate Pareto analysis (80/20 rule)
        sorted_data = data_source.sort_values(volume_col, ascending=False)
        cumulative_volume = sorted_data[volume_col].cumsum()
        pareto_80_threshold = total_volume * 0.8
        
        skus_for_80_percent = len(cumulative_volume[cumulative_volume <= pareto_80_threshold])
        pareto_percentage = (skus_for_80_percent / total_skus) * 100
        
        # Top concentration analysis
        top_10_pct_count = max(1, int(total_skus * 0.1))
        top_10_pct_volume = sorted_data.head(top_10_pct_count)[volume_col].sum()
        top_concentration_pct = (top_10_pct_volume / total_volume) * 100
        
        # Volume distribution insights
        avg_sku_volume = total_volume / total_skus
        top_sku_volume = sorted_data.iloc[0][volume_col]
        volume_concentration_ratio = top_sku_volume / avg_sku_volume
        
        facts = {
            "Total SKUs analyzed": f"{total_skus:,}",
            "Total volume processed": f"{total_volume:,.0f}",
            "Pareto analysis": f"{pareto_percentage:.0f}% of SKUs contribute to 80% of volume",
            "Top 10% concentration": f"Top 10% of SKUs control {top_concentration_pct:.0f}% of volume",
            "Volume concentration ratio": f"{volume_concentration_ratio:.1f}x",
            "Average SKU volume": f"{avg_sku_volume:.0f}",
            "Top SKU volume": f"{top_sku_volume:.0f}"
        }
        
        prompt_config = get_prompt_by_type('section', 'sku_profile')
        prompt = self.build_prompt(
            prompt_config['context'], 
            facts, 
            prompt_config['instruction']
        )
        return self.call_gemini(prompt)
    
    def generate_date_profile_merged_summary(self, analysis_results: Dict) -> str:
        """
        Generate merged summary for date profile analysis (combines volume and customer patterns).
        
        Args:
            analysis_results: Dictionary containing analysis results
            
        Returns:
            Generated summary text using the new merged prompt
        """
        from .prompts_config import get_prompt_by_type
        
        date_summary = analysis_results.get('date_order_summary')
        
        if date_summary is None or date_summary.empty:
            return "(No date summary data available)"
        
        # Build enhanced facts for merged analysis
        facts = {}
        
        # Volume analysis
        if 'Total_Case_Equiv' in date_summary.columns:
            peak_volume = date_summary['Total_Case_Equiv'].max()
            avg_volume = date_summary['Total_Case_Equiv'].mean()
            facts.update({
                "Peak daily volume": f"{peak_volume:.0f}",
                "Average daily volume": f"{avg_volume:.0f}",
                "Peak vs average ratio": f"{peak_volume/avg_volume:.1f}x" if avg_volume > 0 else "N/A",
                "Volume variability": f"{(date_summary['Total_Case_Equiv'].std()/avg_volume)*100:.0f}%" if avg_volume > 0 else "N/A"
            })
        
        # Customer analysis
        if 'Distinct_Customers' in date_summary.columns:
            peak_customers = date_summary['Distinct_Customers'].max()
            avg_customers = date_summary['Distinct_Customers'].mean()
            facts.update({
                "Peak daily customers": f"{peak_customers:.0f}",
                "Average daily customers": f"{avg_customers:.0f}",
                "Customer peak ratio": f"{peak_customers/avg_customers:.1f}x" if avg_customers > 0 else "N/A"
            })
        
        # Date range
        facts["Analysis period"] = f"{date_summary['Date'].min().date()} to {date_summary['Date'].max().date()}"
        facts["Total analysis days"] = len(date_summary)
        
        # Use the new merged prompt
        prompt_config = get_prompt_by_type('section', 'date_profile_merged')
        prompt = self.build_prompt(
            prompt_config['context'], 
            facts, 
            prompt_config['instruction']
        )
        return self.call_gemini(prompt)
    
    def generate_abc_fms_summary(self, analysis_results: Dict) -> str:
        """
        Generate summary for ABC-FMS analysis with enhanced calculations.
        
        Args:
            analysis_results: Dictionary containing analysis results
            
        Returns:
            Generated summary text
        """
        from .prompts_config import get_prompt_by_type
        
        abc_fms_summary = analysis_results.get('abc_fms_summary')
        cross_tab_insights = analysis_results.get('cross_tabulation_insights', {})
        
        if abc_fms_summary is None or abc_fms_summary.empty:
            return "(No ABC-FMS summary data available)"
        
        # Enhanced ABC distribution calculations
        abc_data = abc_fms_summary[abc_fms_summary['ABC'] != 'Grand Total']
        
        if 'ABC' in abc_data.columns:
            abc_counts = abc_data['ABC'].value_counts()
            total_items = len(abc_data)
            
            abc_distribution = {
                'A_percentage': f"{(abc_counts.get('A', 0) / total_items) * 100:.0f}%" if total_items > 0 else "0%",
                'B_percentage': f"{(abc_counts.get('B', 0) / total_items) * 100:.0f}%" if total_items > 0 else "0%",
                'C_percentage': f"{(abc_counts.get('C', 0) / total_items) * 100:.0f}%" if total_items > 0 else "0%"
            }
        else:
            abc_distribution = {'A_percentage': 'N/A', 'B_percentage': 'N/A', 'C_percentage': 'N/A'}
        
        # FMS distribution calculations
        if 'FMS' in abc_data.columns:
            fms_counts = abc_data['FMS'].value_counts()
            fms_distribution = {
                'Fast_percentage': f"{(fms_counts.get('Fast', 0) / total_items) * 100:.0f}%" if total_items > 0 else "0%",
                'Medium_percentage': f"{(fms_counts.get('Medium', 0) / total_items) * 100:.0f}%" if total_items > 0 else "0%",
                'Slow_percentage': f"{(fms_counts.get('Slow', 0) / total_items) * 100:.0f}%" if total_items > 0 else "0%"
            }
        else:
            fms_distribution = {'Fast_percentage': 'N/A', 'Medium_percentage': 'N/A', 'Slow_percentage': 'N/A'}
        
        # Dominant classification insight
        dominant_abc = abc_counts.idxmax() if not abc_counts.empty else 'N/A'
        dominant_fms = fms_counts.idxmax() if 'FMS' in abc_data.columns and not fms_counts.empty else 'N/A'
        
        facts = {
            "ABC distribution - A class": abc_distribution['A_percentage'],
            "ABC distribution - B class": abc_distribution['B_percentage'], 
            "ABC distribution - C class": abc_distribution['C_percentage'],
            "FMS distribution - Fast": fms_distribution['Fast_percentage'],
            "FMS distribution - Medium": fms_distribution['Medium_percentage'],
            "FMS distribution - Slow": fms_distribution['Slow_percentage'],
            "Dominant ABC class": dominant_abc,
            "Dominant FMS class": dominant_fms,
            "Total classified items": f"{total_items:,}"
        }
        
        prompt_config = get_prompt_by_type('section', 'abc_fms')
        prompt = self.build_prompt(
            prompt_config['context'], 
            facts, 
            prompt_config['instruction']
        )
        return self.call_gemini(prompt)
    
    def generate_advanced_analysis_summaries(self, analysis_results: Dict) -> Dict[str, str]:
        """
        Generate LLM summaries for advanced analysis modules.
        
        Args:
            analysis_results: Dictionary containing all analysis results
            
        Returns:
            Dictionary with generated advanced summaries
        """
        self.logger.info("Generating advanced analysis summaries")
        
        advanced_summaries = {}
        
        try:
            # Multi-metric correlation summary
            if 'advanced_order_analysis' in analysis_results:
                advanced_order = analysis_results['advanced_order_analysis']
                if 'correlation_analysis' in advanced_order:
                    correlation_summary = self.generate_correlation_summary(advanced_order['correlation_analysis'])
                    advanced_summaries['multi_metric_correlation'] = correlation_summary
                
                # Capacity planning summary
                if 'enhanced_percentile_analysis' in advanced_order and 'peak_ratios' in advanced_order:
                    capacity_summary = self.generate_advanced_capacity_summary(
                        advanced_order['enhanced_percentile_analysis'], 
                        advanced_order['peak_ratios']
                    )
                    advanced_summaries['capacity_planning_advanced'] = capacity_summary
                
                # Operational complexity summary
                if 'operational_complexity' in advanced_order:
                    complexity_summary = self.generate_complexity_summary(advanced_order['operational_complexity'])
                    advanced_summaries['operational_complexity'] = complexity_summary
            
            # Picking methodology summary
            if 'picking_analysis' in analysis_results:
                picking_summary = self.generate_picking_summary(analysis_results['picking_analysis'])
                advanced_summaries['picking_methodology'] = picking_summary
            
            # Enhanced ABC-FMS matrix summary
            if 'enhanced_abc_fms_analysis' in analysis_results:
                enhanced_abc_summary = self.generate_enhanced_abc_fms_summary(
                    analysis_results['enhanced_abc_fms_analysis']
                )
                advanced_summaries['enhanced_abc_fms_matrix'] = enhanced_abc_summary
            
        except Exception as e:
            self.logger.error(f"Error generating advanced summaries: {e}")
        
        self.logger.info(f"Generated {len(advanced_summaries)} advanced analysis summaries")
        return advanced_summaries
    
    def generate_correlation_summary(self, correlation_analysis: Dict) -> str:
        """Generate summary for multi-metric correlation analysis."""
        from .prompts_config import get_prompt_by_type
        
        facts = {}
        
        if 'key_correlations' in correlation_analysis:
            correlations = correlation_analysis['key_correlations']
            facts.update({
                "Volume-Lines correlation": f"{correlations.get('volume_lines', 0):.2f}",
                "Volume-Customers correlation": f"{correlations.get('volume_customers', 0):.2f}",
                "Lines-Customers correlation": f"{correlations.get('lines_customers', 0):.2f}",
                "Shipments-Orders correlation": f"{correlations.get('shipments_orders', 0):.2f}"
            })
        
        if 'statistical_analysis' in correlation_analysis:
            stats = correlation_analysis['statistical_analysis']
            for key, data in stats.items():
                facts[f"{key}_significance"] = data.get('significance', 'unknown')
                facts[f"{key}_p_value"] = f"{data.get('p_value', 0):.3f}"
        
        prompt_config = get_prompt_by_type('advanced', 'multi_metric_correlation')
        prompt = self.build_prompt(
            prompt_config['context'], 
            facts, 
            prompt_config['instruction']
        )
        return self.call_gemini(prompt)
    
    def generate_picking_summary(self, picking_analysis: Dict) -> str:
        """Generate summary for picking methodology analysis."""
        from .prompts_config import get_prompt_by_type
        
        facts = {}
        
        if 'piece_picking_impact' in picking_analysis:
            impact = picking_analysis['piece_picking_impact']
            facts.update({
                "Piece lines percentage": f"{impact.get('piece_lines_percentage', 0):.1f}%",
                "Piece volume percentage": f"{impact.get('piece_volume_percentage', 0):.1f}%",
                "Average eaches per line": f"{impact.get('avg_eaches_per_piece_line', 0):.1f}"
            })
        
        if 'overall_picking_patterns' in picking_analysis:
            overall = picking_analysis['overall_picking_patterns']
            if 'key_metrics' in overall:
                metrics = overall['key_metrics']
                facts.update({
                    "Case only lines percentage": f"{metrics.get('case_only_lines_percentage', 0):.1f}%",
                    "Piece only lines percentage": f"{metrics.get('piece_only_lines_percentage', 0):.1f}%",
                    "Mixed lines percentage": f"{metrics.get('mixed_lines_percentage', 0):.1f}%"
                })
        
        # Category-level insights
        if 'category_picking_analysis' in picking_analysis:
            category_data = picking_analysis['category_picking_analysis']['category_breakdown']
            if category_data:
                # Find highest PCS category
                highest_pcs_cat = max(category_data, key=lambda x: x['pcs_lines_percentage'])
                facts["Highest PCS category"] = f"{highest_pcs_cat['category']} at {highest_pcs_cat['pcs_lines_percentage']:.1f}%"
        
        prompt_config = get_prompt_by_type('advanced', 'picking_methodology')
        prompt = self.build_prompt(
            prompt_config['context'], 
            facts, 
            prompt_config['instruction']
        )
        return self.call_gemini(prompt)
    
    def generate_enhanced_abc_fms_summary(self, enhanced_analysis: Dict) -> str:
        """Generate summary for enhanced ABC-FMS matrix analysis."""
        from .prompts_config import get_prompt_by_type
        
        facts = {}
        
        if 'advanced_segmentation' in enhanced_analysis:
            segmentation = enhanced_analysis['advanced_segmentation']
            
            if 'key_insights' in segmentation:
                insights = segmentation['key_insights']
                for key, insight in insights.items():
                    if 'high_impact' in key:
                        facts[f"{insight['segment']} high impact"] = insight['impact_description']
                    elif 'low_impact' in key:
                        facts[f"{insight['segment']} low impact"] = insight['impact_description']
        
        if 'percentage_breakdowns' in enhanced_analysis:
            breakdowns = enhanced_analysis['percentage_breakdowns']
            
            if 'abc_distribution' in breakdowns:
                abc_dist = breakdowns['abc_distribution']
                facts.update({
                    "A class percentage": abc_dist.get('A_percentage', '0%'),
                    "B class percentage": abc_dist.get('B_percentage', '0%'),
                    "C class percentage": abc_dist.get('C_percentage', '0%')
                })
            
            if 'fms_distribution' in breakdowns:
                fms_dist = breakdowns['fms_distribution']
                facts.update({
                    "Fast class percentage": fms_dist.get('Fast_percentage', '0%'),
                    "Medium class percentage": fms_dist.get('Medium_percentage', '0%'),
                    "Slow class percentage": fms_dist.get('Slow_percentage', '0%')
                })
        
        prompt_config = get_prompt_by_type('advanced', 'enhanced_abc_fms_matrix')
        prompt = self.build_prompt(
            prompt_config['context'], 
            facts, 
            prompt_config['instruction']
        )
        return self.call_gemini(prompt)
    
    def generate_advanced_capacity_summary(self, percentile_analysis: Dict, peak_ratios: Dict) -> str:
        """Generate summary for advanced capacity planning."""
        from .prompts_config import get_prompt_by_type
        
        facts = {}
        
        # Extract capacity planning metrics
        if 'Total_Case_Equiv' in percentile_analysis:
            volume_pct = percentile_analysis['Total_Case_Equiv']
            facts.update({
                "95th percentile volume": f"{volume_pct.get('95%ile', 0):.0f}",
                "Maximum volume": f"{volume_pct.get('Max', 0):.0f}",
                "Average volume": f"{volume_pct.get('Avg', 0):.0f}"
            })
        
        if 'Total_Case_Equiv' in peak_ratios:
            volume_ratios = peak_ratios['Total_Case_Equiv']
            facts.update({
                "Peak to average ratio": f"{volume_ratios.get('peak_to_avg', 0):.1f}x",
                "Peak to 95th ratio": f"{volume_ratios.get('peak_to_95th', 0):.1f}x",
                "95th to 50th ratio": f"{volume_ratios.get('p95_to_p50', 0):.1f}x"
            })
        
        # Calculate recommended buffer
        if 'Total_Case_Equiv' in percentile_analysis:
            p95 = percentile_analysis['Total_Case_Equiv'].get('95%ile', 0)
            max_val = percentile_analysis['Total_Case_Equiv'].get('Max', 0)
            if p95 > 0:
                buffer_pct = ((max_val - p95) / p95) * 100
                facts["Recommended buffer percentage"] = f"{max(15, buffer_pct):.0f}%"
        
        prompt_config = get_prompt_by_type('advanced', 'capacity_planning_advanced')
        prompt = self.build_prompt(
            prompt_config['context'], 
            facts, 
            prompt_config['instruction']
        )
        return self.call_gemini(prompt)
    
    def generate_complexity_summary(self, complexity_analysis: Dict) -> str:
        """Generate summary for operational complexity analysis."""
        from .prompts_config import get_prompt_by_type
        
        facts = {
            "Overall complexity score": f"{complexity_analysis.get('overall_complexity_score', 0):.1f}",
            "Complexity level": complexity_analysis.get('complexity_level', 'Unknown'),
            "Interpretation": complexity_analysis.get('interpretation', 'No interpretation available')
        }
        
        if 'complexity_factors' in complexity_analysis:
            factors = complexity_analysis['complexity_factors']
            facts.update({
                "Volume variability factor": f"{factors.get('volume_variability', 0):.1f}",
                "Multi-truck complexity factor": f"{factors.get('multi_truck_complexity', 0):.1f}",
                "SKU complexity factor": f"{factors.get('sku_complexity', 0):.1f}"
            })
        
        prompt_config = get_prompt_by_type('advanced', 'operational_complexity')
        prompt = self.build_prompt(
            prompt_config['context'], 
            facts, 
            prompt_config['instruction']
        )
        return self.call_gemini(prompt)
    
    def generate_all_summaries(self, analysis_results: Dict) -> Dict[str, str]:
        """
        Generate all LLM summaries for the analysis results including advanced analysis.
        
        Args:
            analysis_results: Dictionary containing all analysis results
            
        Returns:
            Dictionary with generated summaries
        """
        self.logger.info("Generating LLM summaries for all sections")
        
        # Basic summaries
        summaries = {
            'cover': self.generate_cover_summary(analysis_results),
            'date_profile': self.generate_date_profile_summary(analysis_results),
            'date_profile_merged': self.generate_date_profile_merged_summary(analysis_results),
            'percentiles': self.generate_percentile_summary(analysis_results),
            'sku_profile': self.generate_sku_profile_summary(analysis_results),
            'abc_fms': self.generate_abc_fms_summary(analysis_results)
        }
        
        # Advanced summaries
        advanced_summaries = self.generate_advanced_analysis_summaries(analysis_results)
        summaries.update(advanced_summaries)
        
        self.logger.info(f"Generated {len(summaries)} total LLM summaries")
        return summaries
    
    def test_api_connection(self) -> bool:
        """
        Test the API connection with a simple request.
        
        Returns:
            True if API is working, False otherwise
        """
        if not self.use_llm or not self.api_key:
            return False
        
        test_prompt = "Say one short sentence about warehouse efficiency."
        
        try:
            response = self.call_gemini(test_prompt)
            # Check if response indicates success (not an error message)
            return not response.startswith("(LLM call")
        except Exception as e:
            self.logger.error(f"API test failed: {e}")
            return False
    
    def generate_category_performance_summary(self, category_analysis: Dict) -> str:
        """Generate AI summary for category performance distribution analysis."""
        
        try:
            # Extract key data for the prompt
            performance_summary = category_analysis.get('performance_summary', pd.DataFrame())
            slotting_insights = category_analysis.get('slotting_insights', {})
            statistics = category_analysis.get('statistics', {})
            
            # Get top performing categories
            top_categories = []
            if not performance_summary.empty:
                top_5 = performance_summary.head(5)
                for _, row in top_5.iterrows():
                    top_categories.append({
                        'category': row.get('Category', 'Unknown'),
                        'priority_score': row.get('Priority_Score', 0),
                        'sku_percent': row.get('SKU_Percent', 0),
                        'cases_percent': row.get('Cases_Percent', 0),
                        'lines_percent': row.get('Lines_Percent', 0),
                        'slotting_priority': row.get('Slotting_Priority', 'Unknown')
                    })
            
            # Get slotting recommendations
            recommendations = slotting_insights.get('slotting_recommendations', {})
            key_findings = slotting_insights.get('key_findings', {})
            
            prompt = f"""
            As a warehouse operations expert, analyze this Category Performance Distribution data and provide strategic slotting insights:

            **Overall Statistics:**
            - Total Categories: {statistics.get('total_categories', 'N/A')}
            - Total SKUs: {statistics.get('total_skus', 'N/A')}
            - Total Volume: {statistics.get('total_volume', 'N/A'):,.0f} cases
            - Total Lines: {statistics.get('total_lines', 'N/A'):,.0f}

            **Top 5 Categories by Priority Score:**
            {chr(10).join([f"• {cat['category']}: Score {cat['priority_score']} ({cat['slotting_priority']} priority) - SKU: {cat['sku_percent']}%, Volume: {cat['cases_percent']}%, Lines: {cat['lines_percent']}%" for cat in top_categories])}

            **Key Findings:**
            - Top Volume Category: {key_findings.get('top_volume_category', 'N/A')}
            - Top Velocity Category: {key_findings.get('top_velocity_category', 'N/A')}
            - Critical ABC-FMS Classes: {', '.join(key_findings.get('critical_abc_fms_classes', []))}

            **Slotting Strategy Recommendations:**
            - Dock Proximity (High Priority): {', '.join(recommendations.get('dock_proximity', []))}
            - Structured Bins (Medium Priority): {', '.join(recommendations.get('structured_bins', [])[:5])}
            - Standard Storage: {len(recommendations.get('standard_storage', []))} categories

            **High Priority Categories Performance:**
            - Volume Share: {slotting_insights.get('high_priority_volume_share', 0)}%
            - Velocity Share: {slotting_insights.get('high_priority_velocity_share', 0)}%

            Provide a strategic analysis with:

            1. **Critical Insights** (3-4 bullet points about the most important findings)
            2. **Slotting Strategy Recommendations** (specific actionable advice for warehouse layout)
            3. **Operational Impact** (how this analysis drives efficiency improvements)
            4. **Implementation Priorities** (which categories to focus on first)

            Focus on practical warehouse management advice that directly impacts picker efficiency, throughput, and operational costs.
            """
            
            return self.call_gemini(prompt.strip())
            
        except Exception as e:
            self.logger.error(f"Failed to generate category performance summary: {str(e)}")
            return f"(Category performance summary generation failed: {str(e)})"
    
    def generate_manpower_summary(self, manpower_analysis: Dict) -> str:
        """
        Generate AI summary for manpower and FTE analysis.
        
        Args:
            manpower_analysis: Dictionary containing manpower analysis results
            
        Returns:
            AI-generated summary text
        """
        try:
            if not manpower_analysis:
                return "(No manpower analysis data available)"
            
            # Extract key metrics for facts
            summary_metrics = manpower_analysis.get('summary_metrics', {})
            cost_analysis = manpower_analysis.get('cost_analysis', {})
            peak_analysis = manpower_analysis.get('peak_analysis', {})
            daily_patterns = manpower_analysis.get('daily_staffing_analysis', {})
            
            facts = {
                "recommended_core_fte": summary_metrics.get('recommended_core_fte', 0),
                "peak_fte_requirement": summary_metrics.get('peak_fte_requirement', 0),
                "average_fte_requirement": summary_metrics.get('average_fte_requirement', 0),
                "flex_capacity_needed": summary_metrics.get('flex_capacity_needed', 0),
                "peak_days_percentage": summary_metrics.get('peak_days_percentage', 0),
                "monthly_labor_budget": cost_analysis.get('total_monthly_labor', 0),
                "annual_labor_budget": cost_analysis.get('annual_labor_budget', 0),
                "cost_per_case": cost_analysis.get('cost_per_case', 0),
                "baseline_fte": peak_analysis.get('baseline_fte', 0),
                "peak_multiplier": peak_analysis.get('peak_multiplier', 1),
                "peak_days_count": peak_analysis.get('peak_days_count', 0),
                "avg_fte_required": daily_patterns.get('avg_fte_required', 0),
                "max_fte_required": daily_patterns.get('max_fte_required', 0),
                "std_fte_required": daily_patterns.get('std_fte_required', 0)
            }
            
            prompt = f"""
            Based on the comprehensive manpower and FTE analysis data provided:
            
            **FTE Requirements:**
            - Recommended Core FTE: {facts['recommended_core_fte']:.0f}
            - Peak FTE Requirement: {facts['peak_fte_requirement']:.0f}
            - Average FTE Requirement: {facts['average_fte_requirement']:.1f}
            - Flex Capacity Needed: {facts['flex_capacity_needed']:.0f}
            - Peak Days: {facts['peak_days_percentage']:.1f}% of operating days
            
            **Labor Cost Analysis:**
            - Monthly Labor Budget: ${facts['monthly_labor_budget']:,.0f}
            - Annual Labor Budget: ${facts['annual_labor_budget']:,.0f}
            - Cost per Case: ${facts['cost_per_case']:.2f}
            
            **Staffing Patterns:**
            - Peak Multiplier: {facts['peak_multiplier']:.2f}x above baseline
            - FTE Variability (StdDev): {facts['std_fte_required']:.1f}
            - Max FTE Required: {facts['max_fte_required']:.0f}
            
            Using the 'manpower_workforce_planning' prompt configuration, provide exactly 3 bullets for workforce planning and FTE analysis:
            
            • **Core Staffing Requirements**: Use actual recommended core FTE and average FTE from data with operational justification
            • **Peak Capacity Planning**: Use actual peak FTE requirements and flex capacity needs with percentage of peak days from data  
            • **Cost Optimization**: Use actual labor cost metrics (monthly budget, cost per case) with efficiency recommendations
            
            Use only actual FTE numbers and cost data from provided facts. Bold all FTE values, percentages, and cost metrics.
            
            Structure the response as:
            
            1. **Strategic Workforce Planning Overview** (2-3 sentences about overall FTE strategy)
            2. **Core Insights** (3 bullet points as specified above)
            3. **Implementation Recommendations** (2-3 actionable items for workforce optimization)
            
            Focus on practical workforce management advice that directly impacts operational efficiency, cost management, and staffing optimization.
            """
            
            return self.call_gemini(prompt.strip())
            
        except Exception as e:
            self.logger.error(f"Failed to generate manpower summary: {str(e)}")
            return f"(Manpower summary generation failed: {str(e)})"


def generate_llm_summaries(analysis_results: Dict, 
                          use_llm: bool = USE_GEMINI,
                          api_key: Optional[str] = None) -> Dict[str, str]:
    """
    Convenience function to generate all LLM summaries.
    
    Args:
        analysis_results: Dictionary containing analysis results
        use_llm: Whether to use LLM
        api_key: API key to use
        
    Returns:
        Dictionary with generated summaries
    """
    integration = LLMIntegration(use_llm=use_llm, api_key=api_key)
    return integration.generate_all_summaries(analysis_results)