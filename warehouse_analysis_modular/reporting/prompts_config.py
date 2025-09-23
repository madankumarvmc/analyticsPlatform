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
        'instruction': '''Produce a professional 3-sentence executive summary that highlights:
        1. Dataset scope and time period analyzed
        2. Key operational metrics and performance indicators
        3. Top-level strategic implications for warehouse management
        
        Focus on business value and actionable insights suitable for executive leadership.''',
        'context': 'Executive Summary for Management Report'
    }
}

# =============================================================================
# SECTION ANALYSIS PROMPTS
# =============================================================================

SECTION_ANALYSIS_PROMPTS = {
    'date_profile': {
        'instruction': '''Analyze the daily demand patterns and provide:
        1. A 3-sentence description of demand variability and trends
        2. Peak demand periods and their operational impact
        3. Three specific operational recommendations for:
           - Workforce staffing optimization
           - Pallet allocation strategies
           - Inventory buffer planning
        
        Frame recommendations in terms of cost reduction and efficiency gains.''',
        'context': 'Daily Operations Analysis'
    },
    
    'percentiles': {
        'instruction': '''Interpret these daily volume percentiles for capacity planning:
        1. Explain what each percentile means for operational planning
        2. Identify capacity utilization patterns and stress points
        3. Recommend infrastructure and resource provisioning strategies
        
        Focus on scalability and peak demand preparedness.''',
        'context': 'Capacity Planning Analysis'
    },
    
    'sku_profile': {
        'instruction': '''Analyze the SKU distribution and Pareto characteristics:
        1. Summarize the volume concentration among SKUs (3 sentences)
        2. Identify inventory management implications
        3. Provide one specific slotting optimization recommendation
        
        Emphasize inventory efficiency and picking optimization.''',
        'context': 'SKU Distribution Analysis'
    },
    
    'abc_fms': {
        'instruction': '''Analyze the ABC-FMS classification distribution:
        1. Interpret the volume and frequency distribution patterns
        2. Assess current inventory classification effectiveness
        3. Provide 3 prioritized recommendations for:
           - Optimal slotting strategies
           - Replenishment frequency optimization
           - Resource allocation improvements
        
        Focus on operational efficiency and cost optimization.''',
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
        'instruction': '''Analyze this daily volume trend chart and provide:
        1. Two key observations about demand patterns
        2. One operational insight about volume variability
        3. One recommendation for capacity planning
        
        Keep insights concise and actionable.''',
        'context': 'Daily Volume Trend Analysis'
    },
    
    'date_distinct_customers': {
        'instruction': '''Analyze this daily customer count chart and provide:
        1. Customer demand pattern insights
        2. Correlation between customer count and operational complexity
        3. One recommendation for customer service optimization
        
        Focus on service level and operational efficiency.''',
        'context': 'Customer Demand Pattern Analysis'
    },
    
    'percentile_total_case_equiv': {
        'instruction': '''Analyze this percentile distribution chart and provide:
        1. Capacity planning insights from the percentile spread
        2. Risk assessment for peak demand scenarios
        3. One infrastructure recommendation based on the distribution
        
        Emphasize planning for operational resilience.''',
        'context': 'Percentile Distribution Analysis'
    },
    
    'sku_pareto': {
        'instruction': '''Analyze this SKU Pareto chart and provide:
        1. Concentration ratio insights (80/20 principle application)
        2. Inventory management implications
        3. One specific recommendation for high-volume SKU handling
        
        Focus on inventory efficiency and picking optimization.''',
        'context': 'SKU Pareto Analysis'
    },
    
    'abc_volume_stacked': {
        'instruction': '''Analyze this ABC volume distribution chart and provide:
        1. Volume concentration insights across ABC classes
        2. Inventory investment allocation observations
        3. One recommendation for ABC classification optimization
        
        Emphasize inventory cost management.''',
        'context': 'ABC Volume Distribution Analysis'
    },
    
    'abc_fms_heatmap': {
        'instruction': '''Analyze this ABC-FMS heatmap and provide:
        1. Cross-classification pattern insights
        2. Slotting strategy implications from the distribution
        3. One specific recommendation for warehouse layout optimization
        
        Focus on picking efficiency and storage optimization.''',
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
        'instruction': '''Summarize the top 5 key findings from the warehouse analysis:
        1. Most critical operational insights
        2. Biggest opportunities for improvement
        3. Highest-impact recommendations
        
        Present as numbered list with business impact focus.''',
        'context': 'Key Findings Summary'
    },
    
    'recommendations_summary': {
        'instruction': '''Create a consolidated action plan with:
        1. Top 3 immediate actions (0-30 days)
        2. Top 3 medium-term initiatives (1-6 months)
        3. Long-term strategic improvements (6+ months)
        
        Include expected business impact and implementation complexity for each.''',
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