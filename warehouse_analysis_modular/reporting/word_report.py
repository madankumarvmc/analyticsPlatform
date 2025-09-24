#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Microsoft Word Report Generator

Generates professional MS Word documents for warehouse analysis reports
with AI-powered insights and embedded charts.
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    from docx import Document
    from docx.shared import Inches, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml.shared import OxmlElement, qn
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

import pandas as pd

# Import from parent directories
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import REPORT_DIR, CHARTS_DIR
from warehouse_analysis_modular.reporting.llm_integration import LLMIntegration
from warehouse_analysis_modular.reporting.prompts_config import get_prompt_by_type, get_all_prompts
from warehouse_analysis_modular.utils.helpers import setup_logging

logger = setup_logging()


class WordReportGenerator:
    """
    Generates professional Microsoft Word reports with AI insights.
    """
    
    def __init__(self, 
                 output_dir: Path = REPORT_DIR,
                 charts_dir: Path = CHARTS_DIR,
                 llm_integration: Optional[LLMIntegration] = None):
        """
        Initialize the Word report generator.
        
        Args:
            output_dir: Directory for output files
            charts_dir: Directory containing chart images
            llm_integration: LLM integration instance for AI insights
        """
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx package is required for Word report generation. Install with: pip install python-docx")
        
        self.output_dir = Path(output_dir)
        self.charts_dir = Path(charts_dir)
        self.llm_integration = llm_integration or LLMIntegration()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Ensure output directory exists and test write permissions
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            # Test write permissions by creating a temporary file
            test_file = self.output_dir / "test_write_permissions.tmp"
            test_file.write_text("test")
            test_file.unlink()  # Remove test file
            self.logger.info(f"Output directory confirmed writable: {self.output_dir}")
        except Exception as e:
            self.logger.error(f"Cannot write to output directory {self.output_dir}: {e}")
            # Fallback to current working directory
            import tempfile
            self.output_dir = Path(tempfile.gettempdir()) / "warehouse_reports"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Using fallback directory: {self.output_dir}")
    
    def _add_heading_with_style(self, doc: Document, text: str, level: int = 1):
        """Add a styled heading to the document."""
        heading = doc.add_heading(text, level=level)
        if level == 1:
            heading.runs[0].font.size = Pt(16)
            heading.runs[0].font.bold = True
        elif level == 2:
            heading.runs[0].font.size = Pt(14)
            heading.runs[0].font.bold = True
        return heading
    
    def _add_paragraph_with_style(self, doc: Document, text: str, style: str = None):
        """Add a styled paragraph to the document."""
        paragraph = doc.add_paragraph(text)
        if style:
            paragraph.style = style
        return paragraph
    
    def _add_ai_insight_section(self, doc: Document, title: str, insight_text: str):
        """Add an AI insight section with special formatting."""
        # Add insight heading
        heading = doc.add_paragraph()
        run = heading.add_run(f"💡 AI Insights: {title}")
        run.font.size = Pt(12)
        run.font.bold = True
        
        # Add insight content in a styled box
        insight_para = doc.add_paragraph()
        insight_para.style = 'Quote'
        insight_para.add_run(insight_text)
        
        return heading, insight_para
    
    def _add_chart_with_insights(self, doc: Document, chart_name: str, chart_path: Path, analysis_results: Dict):
        """Add a chart with AI-generated insights."""
        # Add chart title regardless of whether image is available
        chart_title = chart_name.replace('_', ' ').title()
        self._add_heading_with_style(doc, chart_title, level=3)
        
        # Try to add chart image
        chart_added = False
        if chart_path.exists():
            try:
                self.logger.info(f"Adding chart image: {chart_path}")
                doc.add_picture(str(chart_path), width=Inches(6))
                last_paragraph = doc.paragraphs[-1]
                last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                chart_added = True
                self.logger.info(f"Chart {chart_name} added successfully")
            except Exception as e:
                self.logger.error(f"Failed to add chart image {chart_name}: {e}")
        else:
            self.logger.warning(f"Chart image not found: {chart_path}")
        
        # Add placeholder if chart couldn't be added
        if not chart_added:
            placeholder = doc.add_paragraph(f"📊 Chart: {chart_title}")
            placeholder.alignment = WD_ALIGN_PARAGRAPH.CENTER
            placeholder.style = 'Intense Quote'
            note = doc.add_paragraph(f"Note: Chart image ({chart_name}) not available in this report version.")
            note.style = 'Caption'
        
        # Generate AI insights for this chart (even without the image)
        try:
            chart_insights = self._generate_chart_insights(chart_name, analysis_results)
            if chart_insights and not chart_insights.startswith("("):
                self._add_ai_insight_section(doc, f"{chart_title} Analysis", chart_insights)
        except Exception as e:
            self.logger.error(f"Failed to generate insights for chart {chart_name}: {e}")
        
        # Add spacing
        doc.add_paragraph()
    
    def _generate_chart_insights(self, chart_name: str, analysis_results: Dict) -> str:
        """Generate AI insights for a specific chart."""
        # Get prompt for this chart
        prompt_config = get_prompt_by_type('chart', chart_name)
        
        if not prompt_config.get('instruction'):
            return ""
        
        # Build context facts based on chart type
        facts = self._build_chart_context_facts(chart_name, analysis_results)
        
        # Generate prompt
        prompt = self.llm_integration.build_prompt(
            prompt_config['context'], 
            facts, 
            prompt_config['instruction']
        )
        
        # Get AI response
        return self.llm_integration.call_gemini(prompt)
    
    def _build_chart_context_facts(self, chart_name: str, analysis_results: Dict) -> Dict[str, Any]:
        """Build context facts for chart analysis."""
        facts = {"Chart type": chart_name}
        
        # Add relevant data based on chart type
        if 'date' in chart_name:
            date_summary = analysis_results.get('date_order_summary')
            if date_summary is not None and not date_summary.empty:
                facts.update({
                    "Date range": f"{date_summary['Date'].min().date()} to {date_summary['Date'].max().date()}",
                    "Peak volume": f"{date_summary['Total_Case_Equiv'].max():.0f}",
                    "Average volume": f"{date_summary['Total_Case_Equiv'].mean():.0f}",
                    "Volume variation": f"{date_summary['Total_Case_Equiv'].std():.0f}"
                })
        
        elif 'sku' in chart_name:
            sku_summary = analysis_results.get('sku_order_summary')
            if sku_summary is not None and not sku_summary.empty:
                facts.update({
                    "Total SKUs": len(sku_summary),
                    "Top SKU volume": f"{sku_summary.iloc[0].get('Order_Volume_CE', 0):.0f}" if len(sku_summary) > 0 else "N/A"
                })
        
        elif 'abc' in chart_name:
            abc_summary = analysis_results.get('abc_fms_summary')
            if abc_summary is not None and not abc_summary.empty:
                facts.update({
                    "ABC classes": len(abc_summary['ABC'].unique()) if 'ABC' in abc_summary.columns else "N/A"
                })
        
        return facts
    
    def _add_data_table(self, doc: Document, title: str, data: pd.DataFrame, max_rows: int = 10):
        """Add a formatted data table to the document."""
        if data is None or data.empty:
            return
        
        # Add table title
        self._add_heading_with_style(doc, title, level=3)
        
        # Limit rows
        display_data = data.head(max_rows)
        
        # Create table
        table = doc.add_table(rows=1, cols=len(display_data.columns))
        table.style = 'Table Grid'
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        
        # Add header row
        header_cells = table.rows[0].cells
        for i, column in enumerate(display_data.columns):
            header_cells[i].text = str(column)
            header_cells[i].paragraphs[0].runs[0].font.bold = True
        
        # Add data rows
        for _, row in display_data.iterrows():
            row_cells = table.add_row().cells
            for i, value in enumerate(row):
                if pd.isna(value):
                    row_cells[i].text = "N/A"
                elif isinstance(value, (int, float)):
                    if isinstance(value, float) and value != int(value):
                        row_cells[i].text = f"{value:.2f}"
                    else:
                        row_cells[i].text = f"{int(value):,}"
                else:
                    row_cells[i].text = str(value)
        
        doc.add_paragraph()  # Add spacing
    
    def _add_executive_summary(self, doc: Document, analysis_results: Dict):
        """Add executive summary with AI insights."""
        self._add_heading_with_style(doc, "Executive Summary", level=1)
        
        # Generate AI-powered executive summary
        try:
            executive_summary = self.llm_integration.generate_cover_summary(analysis_results)
            if executive_summary and not executive_summary.startswith("("):
                self._add_paragraph_with_style(doc, executive_summary)
            else:
                # Fallback summary
                stats = analysis_results.get('order_statistics', {})
                fallback_summary = f"""This warehouse analysis covers {stats.get('unique_dates', 'N/A')} days of operational data, 
                analyzing {stats.get('unique_skus', 'N/A')} unique SKUs across {stats.get('total_order_lines', 'N/A')} order lines. 
                The analysis provides insights into demand patterns, inventory classification, and operational efficiency opportunities."""
                self._add_paragraph_with_style(doc, fallback_summary)
        except Exception as e:
            self.logger.error(f"Failed to generate executive summary: {e}")
        
        doc.add_page_break()
    
    def _add_key_findings(self, doc: Document, analysis_results: Dict):
        """Add key findings section."""
        self._add_heading_with_style(doc, "Key Findings", level=1)
        
        # Generate AI-powered key findings
        try:
            prompt_config = get_prompt_by_type('word', 'key_findings_summary')
            stats = analysis_results.get('order_statistics', {})
            sku_stats = analysis_results.get('sku_statistics', {})
            
            facts = {
                "Analysis period": f"{stats.get('unique_dates', 'N/A')} days",
                "SKU diversity": f"{stats.get('unique_skus', 'N/A')} unique SKUs",
                "Order complexity": f"{stats.get('total_order_lines', 'N/A')} order lines",
                "Volume processed": f"{stats.get('total_case_equivalent', 0):.0f} case equivalents"
            }
            
            prompt = self.llm_integration.build_prompt(
                prompt_config['context'], 
                facts, 
                prompt_config['instruction']
            )
            
            key_findings = self.llm_integration.call_gemini(prompt)
            if key_findings and not key_findings.startswith("("):
                self._add_paragraph_with_style(doc, key_findings)
            else:
                # Fallback findings
                self._add_paragraph_with_style(doc, "Key operational insights have been identified across demand patterns, inventory classification, and resource utilization areas.")
        except Exception as e:
            self.logger.error(f"Failed to generate key findings: {e}")
        
        doc.add_page_break()
    
    def generate_word_report(self, analysis_results: Dict, filename: str = None) -> Path:
        """
        Generate a comprehensive Word report.
        
        Args:
            analysis_results: Dictionary containing all analysis results
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to the generated Word document
        """
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx package is required")
        
        self.logger.info("Starting Word report generation")
        
        # Detect environment (helpful for debugging deployment issues)
        import os
        env_info = {
            "platform": os.name,
            "cwd": os.getcwd(),
            "streamlit_detected": "streamlit" in os.environ.get("PATH", "").lower() or "STREAMLIT_SERVER_PORT" in os.environ,
            "python_path": sys.executable if hasattr(sys, 'executable') else "Unknown"
        }
        self.logger.info(f"Environment info: {env_info}")
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Warehouse_Analysis_Report_{timestamp}.docx"
        
        # Create document
        doc = Document()
        
        # Set document properties
        doc.core_properties.title = "Warehouse Analysis Report"
        doc.core_properties.subject = "Operational Analysis and Recommendations"
        doc.core_properties.author = "Warehouse Analysis Tool"
        doc.core_properties.created = datetime.now()
        
        # Add title page
        title = doc.add_paragraph()
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title_run = title.add_run("WAREHOUSE ANALYSIS REPORT")
        title_run.font.size = Pt(24)
        title_run.font.bold = True
        
        subtitle = doc.add_paragraph()
        subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
        subtitle_run = subtitle.add_run("Operational Insights and Strategic Recommendations")
        subtitle_run.font.size = Pt(14)
        
        date_para = doc.add_paragraph()
        date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        date_run = date_para.add_run(f"Generated: {datetime.now().strftime('%B %d, %Y')}")
        date_run.font.size = Pt(12)
        
        doc.add_page_break()
        
        # Add executive summary
        self._add_executive_summary(doc, analysis_results)
        
        # Add key findings
        self._add_key_findings(doc, analysis_results)
        
        # Add daily operations analysis
        self._add_heading_with_style(doc, "Daily Operations Analysis", level=1)
        
        # Date summary insights
        try:
            date_insights = self.llm_integration.generate_date_profile_summary(analysis_results)
            if date_insights and not date_insights.startswith("("):
                self._add_ai_insight_section(doc, "Daily Demand Patterns", date_insights)
        except Exception as e:
            self.logger.error(f"Failed to generate date insights: {e}")
        
        # Add date summary table
        date_summary = analysis_results.get('date_order_summary')
        if date_summary is not None and not date_summary.empty:
            # Format date column for display
            display_date_summary = date_summary.copy()
            if 'Date' in display_date_summary.columns:
                display_date_summary['Date'] = display_date_summary['Date'].dt.strftime('%Y-%m-%d')
            self._add_data_table(doc, "Daily Operations Summary (Top 10 Days by Volume)", 
                               display_date_summary.nlargest(10, 'Total_Case_Equiv'))
        
        # Add daily volume chart
        date_chart_path = self.charts_dir / 'date_total_case_equiv.png'
        if date_chart_path.exists():
            self._add_chart_with_insights(doc, 'date_total_case_equiv', date_chart_path, analysis_results)
        
        # Add customer chart
        customer_chart_path = self.charts_dir / 'date_distinct_customers.png'
        if customer_chart_path.exists():
            self._add_chart_with_insights(doc, 'date_distinct_customers', customer_chart_path, analysis_results)
        
        doc.add_page_break()
        
        # Add capacity planning section
        self._add_heading_with_style(doc, "Capacity Planning Analysis", level=1)
        
        # Percentile insights
        try:
            percentile_insights = self.llm_integration.generate_percentile_summary(analysis_results)
            if percentile_insights and not percentile_insights.startswith("("):
                self._add_ai_insight_section(doc, "Capacity Requirements", percentile_insights)
        except Exception as e:
            self.logger.error(f"Failed to generate percentile insights: {e}")
        
        # Add percentile table
        percentile_data = analysis_results.get('percentile_profile')
        if percentile_data is not None and not percentile_data.empty:
            self._add_data_table(doc, "Daily Volume Percentiles", percentile_data)
        
        # Add percentile chart
        percentile_chart_path = self.charts_dir / 'percentile_total_case_equiv.png'
        if percentile_chart_path.exists():
            self._add_chart_with_insights(doc, 'percentile_total_case_equiv', percentile_chart_path, analysis_results)
        
        doc.add_page_break()
        
        # Add SKU analysis section
        self._add_heading_with_style(doc, "SKU Distribution Analysis", level=1)
        
        # SKU insights
        try:
            sku_insights = self.llm_integration.generate_sku_profile_summary(analysis_results)
            if sku_insights and not sku_insights.startswith("("):
                self._add_ai_insight_section(doc, "SKU Performance Patterns", sku_insights)
        except Exception as e:
            self.logger.error(f"Failed to generate SKU insights: {e}")
        
        # Add SKU summary table
        sku_summary = analysis_results.get('sku_order_summary')
        if sku_summary is not None and not sku_summary.empty:
            self._add_data_table(doc, "Top SKUs by Volume", sku_summary.head(15))
        
        # Add SKU Pareto chart
        sku_pareto_path = self.charts_dir / 'sku_pareto.png'
        if sku_pareto_path.exists():
            self._add_chart_with_insights(doc, 'sku_pareto', sku_pareto_path, analysis_results)
        
        doc.add_page_break()
        
        # Add ABC-FMS classification section
        self._add_heading_with_style(doc, "ABC-FMS Classification Analysis", level=1)
        
        # ABC-FMS insights
        try:
            abc_fms_insights = self.llm_integration.generate_abc_fms_summary(analysis_results)
            if abc_fms_insights and not abc_fms_insights.startswith("("):
                self._add_ai_insight_section(doc, "Classification Strategy", abc_fms_insights)
        except Exception as e:
            self.logger.error(f"Failed to generate ABC-FMS insights: {e}")
        
        # Add ABC-FMS summary table
        abc_fms_summary = analysis_results.get('abc_fms_summary')
        if abc_fms_summary is not None and not abc_fms_summary.empty:
            self._add_data_table(doc, "ABC-FMS Classification Summary", abc_fms_summary)
        
        # Add ABC volume chart
        abc_volume_path = self.charts_dir / 'abc_volume_stacked.png'
        if abc_volume_path.exists():
            self._add_chart_with_insights(doc, 'abc_volume_stacked', abc_volume_path, analysis_results)
        
        # Add ABC-FMS heatmap
        abc_heatmap_path = self.charts_dir / 'abc_fms_heatmap.png'
        if abc_heatmap_path.exists():
            self._add_chart_with_insights(doc, 'abc_fms_heatmap', abc_heatmap_path, analysis_results)
        
        doc.add_page_break()
        
        # Add recommendations section
        self._add_heading_with_style(doc, "Strategic Recommendations", level=1)
        
        try:
            prompt_config = get_prompt_by_type('word', 'recommendations_summary')
            facts = {
                "Analysis scope": f"{analysis_results.get('order_statistics', {}).get('unique_dates', 'N/A')} days analyzed",
                "Operational complexity": f"{analysis_results.get('order_statistics', {}).get('unique_skus', 'N/A')} SKUs managed",
                "Volume scale": f"{analysis_results.get('order_statistics', {}).get('total_case_equivalent', 0):.0f} case equivalents"
            }
            
            prompt = self.llm_integration.build_prompt(
                prompt_config['context'], 
                facts, 
                prompt_config['instruction']
            )
            
            recommendations = self.llm_integration.call_gemini(prompt)
            if recommendations and not recommendations.startswith("("):
                self._add_paragraph_with_style(doc, recommendations)
            else:
                # Fallback recommendations
                self._add_paragraph_with_style(doc, "Strategic recommendations have been developed based on the analysis findings to optimize warehouse operations and improve efficiency.")
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
        
        # Save document with enhanced error handling
        output_path = self.output_dir / filename
        
        try:
            self.logger.info(f"Saving Word document to: {output_path}")
            doc.save(str(output_path))
            
            # Verify file was created successfully
            if not output_path.exists():
                raise FileNotFoundError(f"Document was not created at expected path: {output_path}")
            
            file_size = output_path.stat().st_size
            if file_size == 0:
                raise ValueError("Generated document is empty (0 bytes)")
            
            self.logger.info(f"Word report generated successfully: {output_path} ({file_size:,} bytes)")
            return output_path
            
        except PermissionError as e:
            self.logger.error(f"Permission denied writing to {output_path}: {e}")
            # Try alternative location
            import tempfile
            alt_path = Path(tempfile.gettempdir()) / filename
            self.logger.info(f"Attempting to save to alternative location: {alt_path}")
            doc.save(str(alt_path))
            self.logger.info(f"Word report saved to alternative location: {alt_path}")
            return alt_path
            
        except Exception as e:
            self.logger.error(f"Failed to save Word document: {e}")
            raise


def generate_word_report(analysis_results: Dict, 
                        output_dir: Path = REPORT_DIR,
                        filename: str = None) -> Path:
    """
    Convenience function to generate a Word report.
    
    Args:
        analysis_results: Dictionary containing analysis results
        output_dir: Output directory
        filename: Output filename
        
    Returns:
        Path to generated Word document
    """
    generator = WordReportGenerator(output_dir=output_dir)
    return generator.generate_word_report(analysis_results, filename)