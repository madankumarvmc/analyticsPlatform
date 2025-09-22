#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HTML Report Generation Module

Handles HTML report generation with templates and data formatting.
Extracted from the original Warehouse Analysis (2).py file.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

import pandas as pd
from jinja2 import Environment, select_autoescape
import logging

# Import from parent directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import (
    HTML_FILE, METADATA_FILE, TOP_N_TABLE_ROWS, OPEN_AFTER_BUILD
)
from warehouse_analysis_modular.utils.helpers import setup_logging

logger = setup_logging()


class HTMLReportGenerator:
    """
    Handles generation of HTML reports with embedded charts and data tables.
    """
    
    def __init__(self, 
                 output_file: Optional[Path] = None,
                 metadata_file: Optional[Path] = None,
                 max_table_rows: int = TOP_N_TABLE_ROWS,
                 open_after_build: bool = OPEN_AFTER_BUILD):
        """
        Initialize the HTML report generator.
        
        Args:
            output_file: Path for the output HTML file
            metadata_file: Path for the metadata JSON file
            max_table_rows: Maximum rows to show in HTML tables
            open_after_build: Whether to open the report in browser after generation
        """
        self.output_file = output_file or HTML_FILE
        self.metadata_file = metadata_file or METADATA_FILE
        self.max_table_rows = max_table_rows
        self.open_after_build = open_after_build
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Ensure output directories exist
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        if self.metadata_file:
            self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"HTML report generator initialized with output: {self.output_file}")
    
    def safe_df_to_html(self, df: pd.DataFrame, 
                       max_rows: Optional[int] = None,
                       float_format: str = "%.2f") -> str:
        """
        Convert DataFrame to HTML with safe handling and row limits.
        
        Args:
            df: DataFrame to convert
            max_rows: Maximum rows to display
            float_format: Format string for float values
            
        Returns:
            HTML string representation of the DataFrame
        """
        if df is None or df.empty:
            return "<p><em>No data available.</em></p>"
        
        max_rows = max_rows or self.max_table_rows
        show_df = df.copy()
        truncated = False
        
        if max_rows is not None and len(show_df) > max_rows:
            show_df = show_df.head(max_rows)
            truncated = True
        
        try:
            html_table = show_df.to_html(
                classes="table", 
                index=False, 
                float_format=float_format, 
                border=0,
                escape=False
            )
        except Exception as e:
            self.logger.warning(f"Failed to convert DataFrame to HTML: {e}")
            return f"<p><em>Error displaying data: {str(e)}</em></p>"
        
        note = ""
        if truncated:
            note = f'<p class="muted small">Showing top {max_rows} rows. Full table available in the Excel workbook.</p>'
        
        return html_table + note
    
    def prepare_chart_paths(self, chart_paths: Dict[str, str], 
                           base_path: Optional[Path] = None) -> Dict[str, str]:
        """
        Convert chart paths to be relative to the HTML report location.
        
        Args:
            chart_paths: Dictionary of chart paths
            base_path: Base path for relative calculation
            
        Returns:
            Dictionary with relative chart paths
        """
        if base_path is None:
            base_path = self.output_file.parent
        
        relative_paths = {}
        for name, path in chart_paths.items():
            if path:
                try:
                    # Convert to relative path from the HTML file location
                    rel_path = os.path.relpath(path, start=str(base_path))
                    relative_paths[name] = rel_path
                except Exception as e:
                    self.logger.warning(f"Failed to create relative path for {name}: {e}")
                    relative_paths[name] = path
            else:
                relative_paths[name] = None
        
        return relative_paths
    
    def prepare_table_html(self, analysis_results: Dict) -> Dict[str, str]:
        """
        Prepare HTML representations of all data tables.
        
        Args:
            analysis_results: Dictionary containing analysis results
            
        Returns:
            Dictionary with HTML table strings
        """
        self.logger.info("Preparing HTML tables")
        
        tables = {}
        
        # Date order summary
        if 'date_order_summary' in analysis_results:
            tables['date_order_summary'] = self.safe_df_to_html(analysis_results['date_order_summary'])
        
        # Percentile profile
        if 'percentile_profile' in analysis_results:
            tables['percentile_profile'] = self.safe_df_to_html(analysis_results['percentile_profile'])
        
        # SKU order summary - use whichever is available
        sku_data = analysis_results.get('sku_order_summary')
        if sku_data is None:
            sku_data = analysis_results.get('sku_profile_abc_fms')
        if sku_data is not None:
            tables['sku_order_summary'] = self.safe_df_to_html(sku_data)
        
        # SKU profile ABC-FMS
        if 'sku_profile_abc_fms' in analysis_results:
            tables['sku_profile_abc_fms'] = self.safe_df_to_html(analysis_results['sku_profile_abc_fms'])
        
        # ABC-FMS summary (show full table since it's usually small)
        if 'abc_fms_summary' in analysis_results:
            tables['abc_fms_summary'] = self.safe_df_to_html(
                analysis_results['abc_fms_summary'], 
                max_rows=None
            )
        
        self.logger.info(f"Prepared {len(tables)} HTML tables")
        return tables
    
    def prepare_metadata(self, analysis_results: Dict) -> Dict[str, Any]:
        """
        Prepare metadata for the report.
        
        Args:
            analysis_results: Dictionary containing analysis results
            
        Returns:
            Dictionary with report metadata
        """
        # Extract statistics from various analysis results
        order_stats = analysis_results.get('order_statistics', {})
        sku_stats = analysis_results.get('sku_statistics', {})
        cross_tab_insights = analysis_results.get('cross_tabulation_insights', {})
        
        # Count available DataFrames
        dataframe_counts = {}
        for key, value in analysis_results.items():
            if isinstance(value, pd.DataFrame):
                dataframe_counts[key] = len(value)
        
        # Handle date range with proper serialization
        date_range = order_stats.get('date_range', {})
        if isinstance(date_range, dict):
            # Convert any timestamp objects to ISO format strings
            serialized_date_range = {}
            for k, v in date_range.items():
                if hasattr(v, 'isoformat'):
                    serialized_date_range[k] = v.isoformat()
                else:
                    serialized_date_range[k] = str(v)
        else:
            serialized_date_range = {}
        
        metadata = {
            'generated_on': datetime.now().isoformat(),
            'total_dates': order_stats.get('unique_dates', 0),
            'total_skus': order_stats.get('unique_skus', 0),
            'total_order_lines': order_stats.get('total_order_lines', 0),
            'total_case_equivalent': order_stats.get('total_case_equivalent', 0),
            'date_range': serialized_date_range,
            'dataframe_counts': dataframe_counts,
            'abc_distribution': sku_stats.get('abc_distribution', {}),
            'fms_distribution': sku_stats.get('fms_distribution', {}),
            'report_sections': [
                'Date Profile',
                'Percentile Summary', 
                'SKU Profile',
                'ABC-FMS Analysis',
                'Cross-tabulation Summary'
            ]
        }
        
        return metadata
    
    def get_html_template(self) -> str:
        """
        Get the HTML template for the report.
        
        Returns:
            HTML template string
        """
        return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Order Profiles Analysis</title>
  <style>
    body{font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial; margin:30px; color:#111;}
    header {text-align:center; margin-bottom: 20px;}
    h1{font-size:28px; margin-bottom:5px;}
    h2{font-size:20px; margin-top:30px; border-bottom:1px solid #eee; padding-bottom:6px;}
    .kpi{display:inline-block; margin-right:14px; padding:8px 12px; background:#f7f7f7; border-radius:6px;}
    .section-summary{background:#fcfcfc; padding:12px; border-left:4px solid #ddd; margin:8px 0 14px 0;}
    .table {width:100%; border-collapse:collapse; margin-bottom:10px;}
    .table th, .table td{padding:6px 8px; border:1px solid #eee; font-size:12px;}
    .muted{color:#666;}
    .small{font-size:12px;}
    .chart{margin:14px 0; text-align:center;}
    .toc{margin-bottom:20px;}
    .toc a{display:block; margin:4px 0;}
    footer{margin-top:40px; font-size:12px; color:#666; border-top:1px solid #eee; padding-top:10px;}
    .grid {display:grid; grid-template-columns: 1fr 1fr; gap:18px;}
    .mono{font-family:monospace; font-size:13px; background:#fafafa; padding:6px; border-radius:4px;}
    .note{font-size:12px; color:#555}
    .error{color:#d32f2f; background:#ffebee; padding:8px; border-radius:4px; margin:8px 0;}
  </style>
  <script>
    function toggle(id){var e=document.getElementById(id); if(e.style.display==='none') e.style.display='block'; else e.style.display='none';}
  </script>
</head>
<body>
  <header>
    <h1>Order Profiles Analysis</h1>
    <div class="muted">Generated: {{ metadata.generated_on }}</div>
    <div style="margin-top:10px;">
      <span class="kpi">Dates analyzed: {{ metadata.total_dates }}</span>
      <span class="kpi">Unique SKUs: {{ metadata.total_skus }}</span>
      <span class="kpi">Order lines: {{ metadata.total_order_lines }}</span>
    </div>
  </header>

  <div class="toc">
    <strong>Contents</strong>
    <a href="#date">1. Date Profile</a>
    <a href="#percentile">2. Percentile Summary</a>
    <a href="#sku">3. SKU Profile</a>
    <a href="#abc_fms">4. ABC-FMS & 2D</a>
    <a href="#abc_fms_summary">5. ABC×FMS Summary</a>
    <a href="#appendix">Appendix</a>
  </div>

  {% if llm_summaries.cover %}
  <section>
    <div class="section-summary">
      {{ llm_summaries.cover | safe }}
    </div>
  </section>
  {% endif %}

  <section id="date">
    <h2>1. Date Profile</h2>
    {% if llm_summaries.date_profile %}
    <div class="section-summary">
      {{ llm_summaries.date_profile | safe }}
    </div>
    {% endif %}
    {% if charts.date_line %}
    <div class="chart">
      <img src="{{ charts.date_line }}" alt="Total Case Equivalent by Date" style="max-width:100%; height:auto;">
    </div>
    {% endif %}
    {% if charts.date_customers %}
    <div class="chart">
      <img src="{{ charts.date_customers }}" alt="Distinct Customers by Date" style="max-width:100%; height:auto;">
    </div>
    {% endif %}
    <h3>Data (top rows)</h3>
    <div>{{ tables.date_order_summary | safe }}</div>
  </section>

  <section id="percentile">
    <h2>2. Percentile Summary</h2>
    {% if llm_summaries.percentiles %}
    <div class="section-summary">{{ llm_summaries.percentiles | safe }}</div>
    {% endif %}
    {% if charts.percentile %}
    <div class="chart">
      <img src="{{ charts.percentile }}" alt="Percentiles" style="max-width:80%; height:auto;">
    </div>
    {% endif %}
    <div>{{ tables.percentile_profile | safe }}</div>
  </section>

  <section id="sku">
    <h2>3. SKU Profile</h2>
    {% if llm_summaries.sku_profile %}
    <div class="section-summary">{{ llm_summaries.sku_profile | safe }}</div>
    {% endif %}
    {% if charts.sku_pareto %}
      <div class="chart"><img src="{{ charts.sku_pareto }}" alt="SKU Pareto" style="max-width:100%;"></div>
    {% endif %}
    <div>{{ tables.sku_order_summary | safe }}</div>
  </section>

  <section id="abc_fms">
    <h2>4. SKU ABC & FMS</h2>
    {% if llm_summaries.abc_fms %}
    <div class="section-summary">{{ llm_summaries.abc_fms | safe }}</div>
    {% endif %}
    {% if charts.abc_volume %}
      <div class="chart"><img src="{{ charts.abc_volume }}" alt="ABC Volume stacked" style="max-width:90%;"></div>
    {% endif %}
    {% if charts.abc_heatmap %}
      <div class="chart"><img src="{{ charts.abc_heatmap }}" alt="ABCxFMS heatmap" style="max-width:60%;"></div>
    {% endif %}
    <div>{{ tables.sku_profile_abc_fms | safe }}</div>
  </section>

  <section id="abc_fms_summary">
    <h2>5. ABC × FMS Summary</h2>
    <div class="section-summary">Summary cross-tab of SKU counts, Volume and Lines across ABC vs FMS.</div>
    <div>{{ tables.abc_fms_summary | safe }}</div>
  </section>

  <section id="appendix">
    <h2>Appendix</h2>
    <h3>Methodology & Assumptions</h3>
    <p class="note">
      Case equivalent computed using Case Config; pallet equivalence used earlier not shown here.
      ABC cutoffs: &lt;70% → A; 70–90% → B; &gt;90% → C. FMS cutoffs applied similarly on order-line cumulative %.
    </p>

    <h3>Metadata</h3>
    <pre class="mono">{{ metadata | tojson }}</pre>

    <h3>Raw tables / downloads</h3>
    <p class="muted small">Full Excel workbook is expected to be alongside this report: <code>Order_Profiles.xlsx</code></p>
  </section>

  <footer>
    Report generated by Warehouse Analysis Modular Tool. <span class="muted">Generated: {{ metadata.generated_on }}</span>
  </footer>
</body>
</html>
"""
    
    def generate_report(self, 
                       analysis_results: Dict,
                       chart_paths: Dict[str, str],
                       llm_summaries: Dict[str, str]) -> str:
        """
        Generate the complete HTML report.
        
        Args:
            analysis_results: Dictionary containing all analysis results
            chart_paths: Dictionary with chart file paths
            llm_summaries: Dictionary with LLM-generated summaries
            
        Returns:
            Path to generated HTML file
        """
        self.logger.info("Generating HTML report")
        
        try:
            # Prepare all components
            metadata = self.prepare_metadata(analysis_results)
            tables = self.prepare_table_html(analysis_results)
            relative_chart_paths = self.prepare_chart_paths(chart_paths)
            
            # Set up Jinja2 environment
            env = Environment(autoescape=select_autoescape(["html", "xml"]))
            template = env.from_string(self.get_html_template())
            
            # Render the template
            html_content = template.render(
                metadata=metadata,
                tables=tables,
                charts=relative_chart_paths,
                llm_summaries=llm_summaries
            )
            
            # Write HTML file
            with open(self.output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Save metadata file
            if self.metadata_file:
                with open(self.metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)
            
            self.logger.info(f"HTML report generated successfully: {self.output_file}")
            
            # Optionally open in browser
            if self.open_after_build:
                self._open_report_in_browser()
            
            return str(self.output_file)
            
        except Exception as e:
            self.logger.error(f"Failed to generate HTML report: {str(e)}")
            raise
    
    def _open_report_in_browser(self):
        """Attempt to open the report in the default browser."""
        try:
            import webbrowser
            webbrowser.open_new_tab(self.output_file.resolve().as_uri())
            self.logger.info("Report opened in browser")
        except Exception as e:
            self.logger.warning(f"Failed to open report in browser: {e}")
    
    def generate_simple_report(self, 
                             analysis_results: Dict,
                             chart_paths: Optional[Dict[str, str]] = None,
                             llm_summaries: Optional[Dict[str, str]] = None) -> str:
        """
        Generate a simple report with minimal dependencies.
        
        Args:
            analysis_results: Dictionary containing analysis results
            chart_paths: Dictionary with chart file paths
            llm_summaries: Dictionary with LLM summaries
            
        Returns:
            Path to generated HTML file
        """
        # Use empty dictionaries if None provided
        chart_paths = chart_paths or {}
        llm_summaries = llm_summaries or {}
        
        return self.generate_report(analysis_results, chart_paths, llm_summaries)


def generate_html_report(analysis_results: Dict,
                        chart_paths: Dict[str, str],
                        llm_summaries: Dict[str, str],
                        output_file: Optional[Path] = None) -> str:
    """
    Convenience function to generate HTML report.
    
    Args:
        analysis_results: Dictionary containing analysis results
        chart_paths: Dictionary with chart file paths
        llm_summaries: Dictionary with LLM summaries
        output_file: Path for output file
        
    Returns:
        Path to generated HTML file
    """
    generator = HTMLReportGenerator(output_file=output_file)
    return generator.generate_report(analysis_results, chart_paths, llm_summaries)