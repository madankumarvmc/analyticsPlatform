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
    
    def generate_cover_summary(self, analysis_results: Dict) -> str:
        """
        Generate executive summary for the cover section.
        
        Args:
            analysis_results: Dictionary containing analysis results
            
        Returns:
            Generated summary text
        """
        # Extract key facts for cover summary
        stats = analysis_results.get('order_statistics', {})
        sku_stats = analysis_results.get('sku_statistics', {})
        
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
        Generate summary for percentile analysis.
        
        Args:
            analysis_results: Dictionary containing analysis results
            
        Returns:
            Generated summary text
        """
        percentile_profile = analysis_results.get('percentile_profile')
        
        if percentile_profile is None or percentile_profile.empty:
            return "(No percentile data available)"
        
        # Extract percentile facts
        pct_data = percentile_profile.set_index('Percentile')['Total_Case_Equiv']
        facts = {row: f"{pct_data[row]:.0f}" for row in pct_data.index}
        
        prompt = self.build_prompt("Percentiles", facts)
        return self.call_gemini(prompt)
    
    def generate_sku_profile_summary(self, analysis_results: Dict) -> str:
        """
        Generate summary for SKU profile analysis.
        
        Args:
            analysis_results: Dictionary containing analysis results
            
        Returns:
            Generated summary text
        """
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
        
        # Get top SKUs
        top_skus = data_source.nlargest(5, volume_col)
        
        facts = {
            "Total SKUs analyzed": len(data_source),
            "Top 5 SKUs by volume": "; ".join([
                f"{row['Sku Code']}:{row[volume_col]:.0f}" 
                for _, row in top_skus.iterrows()
            ])
        }
        
        prompt = self.build_prompt("SKU Profile", facts)
        return self.call_gemini(prompt)
    
    def generate_abc_fms_summary(self, analysis_results: Dict) -> str:
        """
        Generate summary for ABC-FMS analysis.
        
        Args:
            analysis_results: Dictionary containing analysis results
            
        Returns:
            Generated summary text
        """
        abc_fms_summary = analysis_results.get('abc_fms_summary')
        cross_tab_insights = analysis_results.get('cross_tabulation_insights', {})
        
        if abc_fms_summary is None or abc_fms_summary.empty:
            return "(No ABC-FMS summary data available)"
        
        # Get sample rows (exclude grand total)
        sample_data = abc_fms_summary[abc_fms_summary['ABC'] != 'Grand Total'].head(3)
        
        facts = {
            "ABC distribution": json.dumps(cross_tab_insights.get('distribution_summary', {}).get('abc_volume_distribution', {})),
            "Dominant volume category": cross_tab_insights.get('dominant_categories', {}).get('highest_volume_category', 'N/A'),
            "Sample cross-tab data": sample_data.to_dict('records')[:2]  # Limit to first 2 rows
        }
        
        prompt = self.build_prompt("ABC-FMS", facts)
        return self.call_gemini(prompt)
    
    def generate_all_summaries(self, analysis_results: Dict) -> Dict[str, str]:
        """
        Generate all LLM summaries for the analysis results.
        
        Args:
            analysis_results: Dictionary containing all analysis results
            
        Returns:
            Dictionary with generated summaries
        """
        self.logger.info("Generating LLM summaries for all sections")
        
        summaries = {
            'cover': self.generate_cover_summary(analysis_results),
            'date_profile': self.generate_date_profile_summary(analysis_results),
            'percentiles': self.generate_percentile_summary(analysis_results),
            'sku_profile': self.generate_sku_profile_summary(analysis_results),
            'abc_fms': self.generate_abc_fms_summary(analysis_results)
        }
        
        self.logger.info(f"Generated {len(summaries)} LLM summaries")
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