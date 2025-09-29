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
        'instruction': '''Create exactly 4 bullet points that summarize the warehouse analysis with comprehensive business impact:
        
        • **Operational Scale**: Highlight the data scope with specific numbers - total order lines processed, unique SKUs analyzed, date range analyzed, and total case equivalents handled
        • **Key Performance Patterns**: Identify the most critical operational patterns including peak-to-average ratios, demand concentration by top SKUs, and seasonal variations
        • **Strategic Insights**: Highlight the most impactful findings such as ABC class dominance, fast-moving items concentration, and operational efficiency opportunities
        • **Business Impact**: Summarize the key recommendations for capacity planning, inventory optimization, and workforce planning with quantified benefits
        
        Use only actual data from the analysis. Bold all specific numbers, percentages, and key metrics. Focus on actionable insights that drive business decisions.''',
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
        'instruction': '''Provide exactly 3 bullets using actual percentile data from the analysis (must show all configured percentiles: Max, 95%ile, 90%ile, 85%ile, Average):
        
        • **Capacity Distribution Analysis**: Report all percentile values with specific numbers - Maximum volume, 95th percentile, 90th percentile, 85th percentile, and average, highlighting the capacity spread and variability patterns
        • **Infrastructure Planning Recommendations**: Use actual 95th percentile value as design capacity base, calculate recommended buffer percentage (typically 15%), and provide total design capacity recommendation with justification
        • **Operational Buffer Strategy**: Analyze capacity gaps between percentiles, identify operational buffer zones, recommend flex capacity strategies, and highlight utilization optimization opportunities based on actual percentile distributions
        
        Use only actual calculated percentile values from the analysis. Bold all specific percentile numbers and capacity values. Never use placeholder values like 0 or N/A.''',
        'context': 'Capacity Planning Analysis'
    },
    
    'sku_profile': {
        'instruction': '''Provide exactly 3 bullets using comprehensive SKU data from facts (ensure top SKU table shows at least 10 SKUs):
        
        • **Pareto Analysis**: Calculate exact percentage of SKUs that contribute to 80% of total volume, identify the top 10-20 SKUs by volume with specific volume numbers, and highlight concentration impact on operations
        • **Top Performer Analysis**: Detail the highest performing SKUs with actual volume and line counts, analyze the volume contribution of the top 10% of SKUs, and assess the operational impact of SKU concentration
        • **Distribution Strategy**: Provide velocity-based insights using actual SKU movement data, recommend strategic slotting priorities for high-velocity SKUs, and identify opportunities for inventory optimization based on actual distribution patterns
        
        Use only actual SKU performance data from the analysis. Bold all specific SKU counts, volume numbers, and percentages. Ensure comprehensive coverage of top-performing SKUs with detailed metrics.''',
        'context': 'SKU Distribution Analysis'
    },
    
    'abc_fms': {
        'instruction': '''Provide exactly 3 bullets using actual ABC-FMS cross-classification data from facts:
        
        • **ABC Distribution Analysis**: Report exact percentage breakdown by volume (A-class: X%, B-class: Y%, C-class: Z%) and by SKU count, highlighting dominance patterns and strategic implications
        • **FMS Movement Classification**: Detail actual Fast/Medium/Slow movement percentages from analysis data, including frequency patterns and turnover insights for each class
        • **Strategic Classification Effectiveness**: Assess the distribution balance using real data - identify over-concentration in specific classes, recommend classification threshold adjustments, and highlight opportunities for improved slotting based on actual patterns
        
        Use only actual percentages and distributions from provided analysis results. Bold all real classification percentages and provide specific actionable recommendations based on data patterns.''',
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
    },
    
    'enhanced_order_trend_profile': {
        'instruction': '''Provide exactly 3 bullets using actual multi-metric trend data from facts:
        
        • **Multi-Metric Correlation**: Use calculated correlation between volume, lines, customers, shipments from real data
        • **Demand Synchronization**: Analyze peak alignment patterns across all metrics using actual peak ratios
        • **Operational Complexity**: Key insight about order complexity trends using real metric relationships
        
        Use only actual data from provided facts. Bold all real correlation coefficients and peak ratios. No placeholders.''',
        'context': 'Enhanced Order Profile Trend Analysis'
    },
    
    'sku_profile_2d_classification': {
        'instruction': '''Provide exactly 3 bullets using actual SKU 2D classification data from facts:
        
        • **Concentration Analysis**: Use actual percentages showing SKU count vs Volume vs Lines contribution from data
        • **AF Class Impact**: Analyze critical fast-moving A-class items percentage contribution using real numbers
        • **Strategic Slotting**: Key insight about warehouse slotting priorities using actual classification distribution
        
        Use only actual percentages from provided facts. Bold all real SKU%, Volume%, Lines% ratios. No placeholders.''',
        'context': 'SKU Profile 2D Classification Analysis'
    }
}

# =============================================================================
# ADVANCED ANALYSIS PROMPTS
# =============================================================================

ADVANCED_ANALYSIS_PROMPTS = {
    'multi_metric_correlation': {
        'instruction': '''Analyze multi-metric correlations and provide exactly 3 bullets:
        
        • **Volume-Lines Correlation**: Use actual correlation coefficient and significance from data
        • **Customer-Volume Relationship**: Use calculated correlation and operational impact
        • **Operational Synchronization**: Key insight about metric alignment using real correlation data
        
        Use only actual correlation values from provided facts. Bold all correlation coefficients and significance levels.''',
        'context': 'Multi-Metric Correlation Analysis'
    },
    
    'picking_methodology': {
        'instruction': '''Provide exactly 3 bullets for picking methodology analysis:
        
        • **Case vs Piece Impact**: Use actual percentages from data for lines vs volume contribution
        • **Category Complexity**: Identify categories with highest PCS lines percentage from real data
        • **Operational Efficiency**: Key insight about piece picking efficiency using calculated metrics
        
        Use only actual picking percentages from provided facts. Bold all percentage values and category names.''',
        'context': 'Picking Methodology Analysis'
    },
    
    'enhanced_abc_fms_matrix': {
        'instruction': '''Provide exactly 3 bullets for 2D classification matrix analysis:
        
        • **High-Impact Segments**: Use actual AF-class percentages (SKUs vs volume vs lines) from data
        • **Low-Impact Segments**: Use actual CS-class distribution patterns from real data
        • **Classification Effectiveness**: Overall matrix balance assessment using calculated metrics
        
        Use only actual classification percentages from provided facts. Bold all segment percentages and class names.''',
        'context': '2D Classification Matrix Analysis'
    },
    
    'capacity_planning_advanced': {
        'instruction': '''Provide exactly 3 bullets for advanced capacity planning:
        
        • **Design Capacity**: Use actual 95th percentile values and recommended buffer percentages
        • **Peak Management**: Use calculated peak-to-95th ratios and operational strategies
        • **Utilization Optimization**: Average utilization rates and efficiency opportunities from data
        
        Use only actual percentile values and ratios from provided facts. Bold all capacity numbers and percentages.''',
        'context': 'Advanced Capacity Planning Analysis'
    },
    
    'operational_complexity': {
        'instruction': '''Provide exactly 2 bullets for operational complexity assessment:
        
        • **Complexity Score**: Use calculated overall complexity score and level from data
        • **Management Strategy**: Specific recommendations based on actual complexity factors
        
        Use only actual complexity scores from provided facts. Bold complexity levels and key metrics.''',
        'context': 'Operational Complexity Analysis'
    },
    
    'manpower_workforce_planning': {
        'instruction': '''Provide exactly 3 bullets for workforce planning and FTE analysis:
        
        • **Core Staffing Requirements**: Use actual recommended core FTE and average FTE from data with operational justification
        • **Peak Capacity Planning**: Use actual peak FTE requirements and flex capacity needs with percentage of peak days from data
        • **Cost Optimization**: Use actual labor cost metrics (monthly budget, cost per case) with efficiency recommendations
        
        Use only actual FTE numbers and cost data from provided facts. Bold all FTE values, percentages, and cost metrics.''',
        'context': 'Workforce Planning & FTE Analysis'
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
        'instruction': '''Create exactly 8 bullet points with comprehensive operational insights using actual analysis data:
        
        • **Volume Concentration**: Identify exact percentage of SKUs that drive 80% of volume, top performing SKU volumes, and concentration ratios with business impact
        • **Demand Variability**: Calculate specific peak-to-average ratios across all metrics (volume, customers, lines), seasonal patterns, and capacity implications
        • **ABC Classification Performance**: Provide actual ABC class distribution percentages by volume, lines, and SKU count with strategic recommendations
        • **FMS Movement Analysis**: Detail Fast/Medium/Slow classification percentages, movement frequency patterns, and inventory turnover insights
        • **Customer Behavior Patterns**: Analyze customer concentration, order patterns, peak customer ratios, and service level implications
        • **Operational Efficiency**: Calculate picking methodology distributions, case vs piece percentages, and efficiency optimization opportunities
        • **Capacity Planning Insights**: Provide specific percentile values (95th, 90th, 85th), capacity recommendations, and infrastructure sizing guidance
        • **Strategic Opportunities**: Identify key improvement areas with quantified benefits including slotting optimization, workforce planning, and inventory management
        
        Use only actual data from the analysis. Bold all specific numbers, percentages, and metrics. Provide actionable insights with clear business value for each finding.''',
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
        prompt_type: Type of prompt ('executive', 'section', 'chart', 'word', 'advanced')
        section_key: Specific section key
        
    Returns:
        Dictionary containing prompt instruction and context
    """
    prompt_maps = {
        'executive': EXECUTIVE_SUMMARY_PROMPTS,
        'section': SECTION_ANALYSIS_PROMPTS,
        'chart': CHART_INSIGHT_PROMPTS,
        'word': WORD_DOCUMENT_PROMPTS,
        'advanced': ADVANCED_ANALYSIS_PROMPTS
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
        'word_document': WORD_DOCUMENT_PROMPTS,
        'advanced_analysis': ADVANCED_ANALYSIS_PROMPTS
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