#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prompts Configuration for AI-Generated Insights

This module contains all prompts used for generating AI-powered insights
for different sections and charts in warehouse analysis reports.
"""

# =============================================================================
# EXECUTIVE SUMMARY PROMPTS
# =============================================================================

EXECUTIVE_SUMMARY_PROMPTS = {
    'cover': {
        'instruction': '''Create exactly 4 bullet points using the actual data provided in facts:
        
        • **Data Scope**: Use actual numbers for dates, SKUs, order lines, case equivalents
        • **Operational Scale**: Use real date range and volume from data with calculated metrics
        • **Demand Pattern**: Use calculated peak vs average ratio and variability percentage from data
        • **Classification**: Use actual ABC distribution percentages from analysis
        
        Use only real numbers from the provided facts. Bold all actual percentages, ratios, and metrics. No placeholders.''',
        'context': 'Executive Summary for Management Report'
    }
}

# =============================================================================
# SECTION ANALYSIS PROMPTS
# =============================================================================

SECTION_ANALYSIS_PROMPTS = {
    'date_profile_merged': {
        'instruction': '''Provide exactly 3 bullets using actual data from facts:
        
        • **Volume Pattern**: Use calculated peak vs average ratio and variability percentage from data
        • **Customer Pattern**: Use actual customer count correlation data and peak ratios
        • **Operational Impact**: Key insight about demand concentration using real numbers
        
        Use only actual numbers from provided facts. Bold all real ratios and percentages. No placeholders.''',
        'context': 'Daily Operations Analysis'
    },
    
    'percentiles': {
        'instruction': '''Provide exactly 3 bullets using actual percentile data from facts:
        
        • **Capacity Spread**: Use calculated 95th to 50th percentile ratio from real data
        • **Peak Planning**: Use actual percentile values for infrastructure sizing recommendations
        • **Operational Buffer**: Insight about capacity buffer using real percentile differences
        
        Use only actual percentile numbers from provided facts. Bold all real percentages and values. No placeholders.''',
        'context': 'Capacity Planning Analysis'
    },
    
    'sku_profile': {
        'instruction': '''Provide exactly 3 bullets using actual SKU data from facts:
        
        • **Pareto Analysis**: Use calculated percentage of SKUs that contribute to 80% of volume from data
        • **Top Concentration**: Use actual top SKU percentages and their volume contribution from analysis
        • **Distribution Pattern**: Key insight about SKU velocity using real distribution data
        
        Use only actual numbers from provided facts. Bold all real percentages and SKU counts. No placeholders.''',
        'context': 'SKU Distribution Analysis'
    },
    
    'abc_fms': {
        'instruction': '''Provide exactly 3 bullets using actual ABC-FMS data from facts:
        
        • **ABC Distribution**: Use calculated A/B/C percentages by volume contribution from data
        • **Movement Patterns**: Use actual Fast/Medium/Slow percentages from analysis
        • **Classification Insight**: Key observation about effectiveness using real distribution data
        
        Use only actual percentages from provided facts. Bold all real classification percentages. No placeholders.''',
        'context': 'ABC-FMS Classification Analysis'
    },
    
    'fte_analysis': {
        'instruction': '''Analyze the Full-Time Equivalent workforce requirements:
        1. Interpret daily FTE variations and staffing patterns
        2. Identify peak staffing periods and resource constraints
        3. Recommend workforce planning strategies including:
           - Optimal staffing levels
           - Flexible capacity solutions
           - Cross-training opportunities
        
        Focus on labor cost optimization and service level maintenance.''',
        'context': 'Workforce Planning Analysis'
    }
}

# =============================================================================
# CHART INSIGHT PROMPTS
# =============================================================================

CHART_INSIGHT_PROMPTS = {
    'date_total_case_equiv': {
        'instruction': '''Provide exactly 2 bullets using actual chart data from facts:
        
        • **Volume Trend**: Use calculated peak-to-average ratio and trend patterns from real data
        • **Operational Impact**: Key insight about volume variability using actual numbers
        
        Use only actual data from provided facts. Bold all real ratios and metrics. No placeholders.''',
        'context': 'Daily Volume Trend Analysis'
    },
    
    'date_distinct_customers': {
        'instruction': '''Provide exactly 2 bullets using actual customer data from facts:
        
        • **Customer Pattern**: Use calculated peak customer ratios and patterns from real data
        • **Service Impact**: Operational complexity insight using actual customer metrics
        
        Use only actual data from provided facts. Bold all real ratios and customer metrics. No placeholders.''',
        'context': 'Customer Demand Pattern Analysis'
    },
    
    'percentile_total_case_equiv': {
        'instruction': '''Provide exactly 2 bullets using actual percentile data from facts:
        
        • **Capacity Spread**: Use real 95th/50th percentile values and calculated ratios from data
        • **Planning Insight**: Infrastructure recommendation using actual percentile differences
        
        Use only actual percentile values from provided facts. Bold all real percentiles and ratios. No placeholders.''',
        'context': 'Percentile Distribution Analysis'
    },
    
    'sku_pareto': {
        'instruction': '''Provide exactly 2 bullets using actual Pareto data from facts:
        
        • **80/20 Analysis**: Use calculated percentage of SKUs that drive 80% of volume from real data
        • **Concentration Impact**: Inventory management insight using actual concentration ratios
        
        Use only actual percentages from provided facts. Bold all real Pareto percentages. No placeholders.''',
        'context': 'SKU Pareto Analysis'
    },
    
    'abc_volume_stacked': {
        'instruction': '''Provide exactly 2 bullets using actual ABC data from facts:
        
        • **ABC Distribution**: Use calculated A/B/C class percentages by volume from real data
        • **Investment Impact**: Resource allocation insight using actual volume distribution
        
        Use only actual percentages from provided facts. Bold all real ABC percentages. No placeholders.''',
        'context': 'ABC Volume Distribution Analysis'
    },
    
    'abc_fms_heatmap': {
        'instruction': '''Provide exactly 2 bullets using actual ABC-FMS data from facts:
        
        • **Matrix Pattern**: Use actual dominant classification patterns and percentages from data
        • **Slotting Strategy**: Warehouse layout insight using real classification distribution
        
        Use only actual classification data from provided facts. Bold all real percentages and patterns. No placeholders.''',
        'context': 'ABC-FMS Cross-Classification Analysis'
    }
}

# =============================================================================
# WORD DOCUMENT SPECIFIC PROMPTS
# =============================================================================

WORD_DOCUMENT_PROMPTS = {
    'document_introduction': {
        'instruction': '''Create a professional introduction paragraph for a warehouse analysis report:
        1. Set the context for the analysis
        2. Explain the business value of the insights
        3. Guide the reader through the report structure
        
        Maintain executive-level language and focus on strategic value.''',
        'context': 'Document Introduction'
    },
    
    'key_findings_summary': {
        'instruction': '''Create exactly 4 bullet points using actual analysis data from facts:
        
        • **Volume Concentration**: Use calculated Pareto ratios and percentages from real data
        • **Demand Variability**: Use actual peak vs average ratios and operational impact from data
        • **Classification Patterns**: Use real ABC/FMS distribution percentages and insights
        • **Operational Scale**: Use actual volume and complexity metrics from analysis
        
        Use only real numbers from provided facts. Bold all actual percentages, ratios, and metrics. No placeholders.''',
        'context': 'Key Findings Summary'
    },
    
    'recommendations_summary': {
        'instruction': '''Create exactly 3 bullet points for action items using analysis insights:
        
        • **Immediate (30 days)**: Most critical action based on actual data findings with quantified impact
        • **Medium-term (6 months)**: Key initiative based on real analysis patterns with measurable benefit
        • **Strategic (12+ months)**: Long-term optimization using actual data insights with ROI potential
        
        Base recommendations on actual analysis findings. Bold timeframes and quantified benefits.''',
        'context': 'Consolidated Recommendations'
    }
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_prompt_by_type(prompt_type: str, section_key: str) -> dict:
    """
    Get prompt configuration by type and section key.
    
    Args:
        prompt_type: Type of prompt ('executive', 'section', 'chart', 'word')
        section_key: Specific section key
        
    Returns:
        Dictionary containing prompt instruction and context
    """
    prompt_maps = {
        'executive': EXECUTIVE_SUMMARY_PROMPTS,
        'section': SECTION_ANALYSIS_PROMPTS,
        'chart': CHART_INSIGHT_PROMPTS,
        'word': WORD_DOCUMENT_PROMPTS
    }
    
    return prompt_maps.get(prompt_type, {}).get(section_key, {
        'instruction': 'Provide a brief analysis and recommendations.',
        'context': 'General Analysis'
    })

def get_all_prompts() -> dict:
    """
    Get all prompt configurations.
    
    Returns:
        Dictionary containing all prompt configurations
    """
    return {
        'executive_summary': EXECUTIVE_SUMMARY_PROMPTS,
        'section_analysis': SECTION_ANALYSIS_PROMPTS,
        'chart_insights': CHART_INSIGHT_PROMPTS,
        'word_document': WORD_DOCUMENT_PROMPTS
    }

def validate_prompt_key(prompt_type: str, section_key: str) -> bool:
    """
    Validate if a prompt key exists.
    
    Args:
        prompt_type: Type of prompt
        section_key: Section key to validate
        
    Returns:
        True if prompt exists, False otherwise
    """
    prompt_config = get_prompt_by_type(prompt_type, section_key)
    return bool(prompt_config.get('instruction'))