# Warehouse Analysis Expansion Framework

## Table of Contents
1. [Quick Start Guide](#quick-start-guide)
2. [System Architecture](#system-architecture)
3. [Implementation Templates](#implementation-templates)
4. [Excel Integration](#excel-integration)
5. [Word Report Integration](#word-report-integration)
6. [Chart Generation & Embedding](#chart-generation--embedding)
7. [Testing & Validation](#testing--validation)
8. [Advanced Features](#advanced-features)
9. [Maintenance & Scaling](#maintenance--scaling)

---

## Quick Start Guide

### The Expansion Process Flow
**Analysis Sheet → Excel Worksheet → Word Report with AI Insights**

```
1. Create Analyzer Class
   ↓
2. Add Excel Worksheet Integration  
   ↓
3. Add Word Report Section
   ↓
4. Generate AI Insights
   ↓ 
5. Test & Validate
```

### 5-Step Implementation Checklist

- [ ] **Step 1**: Create analyzer class using template
- [ ] **Step 2**: Add Excel worksheet integration in `excel_exporter.py`
- [ ] **Step 3**: Add Word report section in `word_report.py`
- [ ] **Step 4**: Add LLM prompt for AI insights in `llm_integration.py`
- [ ] **Step 5**: Integrate into main pipeline in `enhanced_main.py`

---

## System Architecture

### Core Components Overview

```
warehouse_analysis_modular/
├── analyzers/                    # Analysis logic
│   ├── your_new_analyzer.py     # ← Your new analysis
│   └── new_analyzer_template.py # Template to copy
├── reporting/
│   ├── excel_exporter.py        # ← Add Excel worksheet
│   ├── word_report.py           # ← Add Word section
│   ├── llm_integration.py       # ← Add AI insights
│   └── advanced_chart_generator.py # ← Add charts
└── enhanced_main.py             # ← Orchestration
```

### Data Flow Architecture

```
Input Data → Analyzer Class → Results Dictionary
    ↓
Excel Integration (Worksheets + Charts)
    ↓
Word Integration (Sections + Tables + AI Insights)
    ↓
Final Reports (Excel + Word with complete analysis)
```

### Integration Points

1. **Analyzer Registration** - `enhanced_main.py:run_advanced_analysis()`
2. **Excel Export** - `excel_exporter.py:_process_advanced_analysis()`
3. **Word Report** - `word_report.py:_add_advanced_analysis_sections()`
4. **AI Insights** - `llm_integration.py` (new method for each analysis)

---

## Implementation Templates

### Step 1: Create Analyzer Class

Copy the template:
```bash
cd warehouse_analysis_modular/analyzers/
cp new_analyzer_template.py your_analysis_name.py
```

**Basic Analyzer Structure:**
```python
class YourAnalysisAnalyzer:
    def __init__(self, input_data: pd.DataFrame):
        self.input_data = input_data
        self.logger = logging.getLogger(self.__class__.__name__)
        self._validate_input_data()
        
    def analyze_primary_metric(self) -> pd.DataFrame:
        """Main analysis logic - returns DataFrame with results"""
        # Your analysis implementation here
        return results_dataframe
        
    def calculate_insights(self, analysis_results: Dict) -> Dict[str, Any]:
        """Calculate key insights and statistics"""
        return insights_dict
        
    def run_full_analysis(self) -> Dict[str, Any]:
        """Main method - must return standardized structure"""
        return {
            'primary_analysis': self.analyze_primary_metric(),
            'insights': self.calculate_insights(analysis_results),
            'statistics': self.get_summary_statistics()
        }
```

**Key Requirements:**
- Must implement `run_full_analysis()` returning dictionary
- Results should include DataFrames for Excel export
- Include insights dictionary for Word report
- Follow naming convention: `{feature}_analysis` as dictionary key

---

## Excel Integration

### Step 2: Add to Excel Exporter

**Location**: `warehouse_analysis_modular/reporting/excel_exporter.py`

**Method**: `_process_advanced_analysis()`

**Template Code:**
```python
# Process Your New Analysis
if 'your_analysis_name' in analysis_results:
    your_analysis = analysis_results['your_analysis_name']
    
    # Main analysis worksheet
    if 'primary_analysis' in your_analysis:
        primary_data = your_analysis['primary_analysis']
        if not primary_data.empty:
            export_data['Your Analysis Results'] = primary_data
    
    # Secondary analysis worksheet (optional)
    if 'secondary_analysis' in your_analysis:
        secondary_data = your_analysis['secondary_analysis']
        if not secondary_data.empty:
            export_data['Your Analysis Details'] = secondary_data
            
    # Summary insights worksheet
    if 'insights' in your_analysis:
        insights = your_analysis['insights']
        if insights:
            insights_df = pd.DataFrame([
                {'Metric': key, 'Value': value} 
                for key, value in insights.items()
            ])
            export_data['Your Analysis Insights'] = insights_df
```

### Excel Worksheet Naming Convention
- **Main Analysis**: `"{Feature} Analysis Results"`
- **Detailed Breakdown**: `"{Feature} Details"`
- **Summary Insights**: `"{Feature} Insights"`
- **Charts**: `"{Feature} Charts"` (if applicable)

### Excel Chart Integration
```python
# If your analysis includes chart data
if 'chart_data' in your_analysis:
    chart_data = your_analysis['chart_data']
    export_data['Your Analysis Chart Data'] = chart_data
    # Chart will be generated automatically by chart_generator
```

---

## Word Report Integration

### Step 3: Add to Word Report Generator

**Location**: `warehouse_analysis_modular/reporting/word_report.py`

**Method**: `_add_advanced_analysis_sections()`

**Template Code:**
```python
# Your New Analysis Section
if 'your_analysis_name' in analysis_results:
    self._add_heading_with_style(doc, "Your Analysis Title", level=1)
    
    try:
        # Generate AI insights
        your_insights = self.llm_integration.generate_your_analysis_summary(
            analysis_results['your_analysis_name']
        )
        if your_insights and not your_insights.startswith("("):
            self._add_ai_insight_section(doc, "Key Findings", your_insights)
    except Exception as e:
        self.logger.error(f"Failed to generate your analysis insights: {e}")
    
    # Add data table
    your_analysis = analysis_results['your_analysis_name']
    if 'primary_analysis' in your_analysis:
        primary_data = your_analysis['primary_analysis']
        if not primary_data.empty:
            self._add_data_table(doc, "Analysis Results", primary_data, max_rows=10)
    
    doc.add_page_break()
```

### Step 4: Add LLM Integration

**Location**: `warehouse_analysis_modular/reporting/llm_integration.py`

**Add New Method:**
```python
def generate_your_analysis_summary(self, your_analysis_data: Dict) -> str:
    """Generate AI summary for your analysis."""
    
    prompt_key = "your_analysis_summary"
    
    # Add to prompts_config.py
    prompt_template = """
    Analyze the following warehouse {your_feature} data and provide actionable insights:
    
    Data Summary:
    {data_summary}
    
    Key Metrics:
    {key_metrics}
    
    Provide:
    1. 3 key findings about {your_feature}
    2. Business impact analysis
    3. Specific recommendations for improvement
    """
    
    return self._generate_with_template(
        prompt_key=prompt_key,
        template=prompt_template,
        data=your_analysis_data
    )
```

---

## Chart Generation & Embedding

### Chart Integration Options

**Option 1: Advanced Chart Generator**
```python
# In advanced_chart_generator.py
def create_your_analysis_chart(self, your_analysis_data: Dict) -> str:
    """Create visualization for your analysis."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Your chart creation logic here
    data = your_analysis_data['primary_analysis']
    data.plot(kind='bar', ax=ax)
    
    # Styling
    ax.set_title('Your Analysis Visualization', fontsize=16, fontweight='bold')
    ax.set_xlabel('Categories')
    ax.set_ylabel('Values')
    
    # Save chart
    chart_path = self.charts_dir / 'your_analysis_chart.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(chart_path)
```

**Option 2: Excel Chart Embedding**
```python
# In excel_exporter.py - for charts within Excel
if 'chart_data' in your_analysis:
    chart_data = your_analysis['chart_data']
    
    # Create chart data worksheet
    export_data['Your Analysis Chart Data'] = chart_data
    
    # Chart will be automatically created by Excel's chart engine
    # or by openpyxl chart creation methods
```

---

## Testing & Validation

### Testing Checklist

**Unit Testing:**
```python
def test_your_analyzer():
    # Create sample data
    sample_data = create_sample_warehouse_data()
    
    # Test analyzer
    analyzer = YourAnalysisAnalyzer(sample_data)
    results = analyzer.run_full_analysis()
    
    # Validate structure
    assert 'primary_analysis' in results
    assert isinstance(results['primary_analysis'], pd.DataFrame)
    assert 'insights' in results
    assert isinstance(results['insights'], dict)
```

**Integration Testing:**
```python
def test_full_pipeline():
    # Test complete pipeline with your analyzer
    pipeline = EnhancedWarehouseAnalysisPipeline(
        run_advanced_analysis=True
    )
    
    results = pipeline.run_full_analysis()
    
    # Validate your analysis is included
    assert 'your_analysis_name' in results
    
    # Check Excel export
    excel_path = results.get('excel_export_path')
    assert excel_path and os.path.exists(excel_path)
    
    # Check Word report
    word_path = results.get('word_report_path') 
    assert word_path and os.path.exists(word_path)
```

---

## Advanced Features

### Error Handling Best Practices

```python
def run_full_analysis(self) -> Dict[str, Any]:
    """Run analysis with proper error handling."""
    try:
        primary_results = self.analyze_primary_metric()
        insights = self.calculate_insights({'primary': primary_results})
        
        return {
            'primary_analysis': primary_results,
            'insights': insights,
            'statistics': self.get_summary_statistics()
        }
        
    except Exception as e:
        self.logger.error(f"Analysis failed: {str(e)}")
        # Return empty structure to prevent pipeline failure
        return {
            'primary_analysis': pd.DataFrame(),
            'insights': {},
            'statistics': {}
        }
```

### Performance Optimization

```python
class YourAnalysisAnalyzer:
    def __init__(self, input_data: pd.DataFrame):
        # Cache expensive operations
        self._cached_aggregations = {}
        self._data_prepared = False
        
    def _prepare_data(self):
        """Prepare data once and cache results."""
        if not self._data_prepared:
            # Expensive data preparation here
            self._data_prepared = True
```

### Multiple Chart Types

```python
def create_your_analysis_charts(self, analysis_data: Dict) -> Dict[str, str]:
    """Create multiple charts for comprehensive analysis."""
    
    chart_paths = {}
    
    # Main trend chart
    chart_paths['trend'] = self._create_trend_chart(analysis_data)
    
    # Distribution chart  
    chart_paths['distribution'] = self._create_distribution_chart(analysis_data)
    
    # Comparison chart
    chart_paths['comparison'] = self._create_comparison_chart(analysis_data)
    
    return chart_paths
```

---

## Maintenance & Scaling

### Integration Pipeline Registration

**Step 5: Register in Enhanced Main**

**Location**: `warehouse_analysis_modular/enhanced_main.py`

**Method**: `run_advanced_analysis()`

```python
def run_advanced_analysis(self, enriched_data, basic_results):
    """Add your analyzer to the pipeline."""
    
    advanced_results = {}
    
    try:
        # Existing analyzers...
        
        # Your New Analysis
        if self.your_analyzer:  # Add this check
            self.logger.info("Running your analysis")
            your_results = self.your_analyzer.run_full_analysis()
            advanced_results['your_analysis_name'] = your_results
            
    except Exception as e:
        self.logger.error(f"Advanced analysis error: {e}")
        
    return advanced_results
```

**Initialize in Constructor:**
```python
def __init__(self, run_advanced_analysis: bool = True):
    # Existing initialization...
    
    # Add your analyzer
    self.your_analyzer = YourAnalysisAnalyzer() if run_advanced_analysis else None
```

### Dependency Management

**Import Structure:**
```python
# In enhanced_main.py
from warehouse_analysis_modular.analyzers.your_analysis_analyzer import YourAnalysisAnalyzer

# In excel_exporter.py - no additional imports needed

# In word_report.py - no additional imports needed

# In llm_integration.py - add prompt to prompts_config.py
```

### Scaling Considerations

1. **Memory Management**: Large datasets should use chunked processing
2. **Performance**: Cache expensive calculations 
3. **Error Isolation**: Each analyzer should fail independently
4. **Logging**: Comprehensive logging for debugging
5. **Configuration**: Make analysis parameters configurable

### Version Control Best Practices

```bash
# Create feature branch
git checkout -b feature/your-analysis-name

# Make changes following the 5-step process
# Test thoroughly
# Commit with descriptive messages

git commit -m "Add [YourAnalysis]: Excel worksheet integration"
git commit -m "Add [YourAnalysis]: Word report section with AI insights"
git commit -m "Add [YourAnalysis]: Chart generation and embedding"
```

---

## Success Criteria

### Definition of Done
- [ ] Analyzer class follows template structure
- [ ] Excel worksheet exports successfully with data
- [ ] Word report section appears with AI insights
- [ ] Charts are generated and embedded (if applicable)
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Documentation updated
- [ ] Code review completed

### Quality Checklist
- [ ] Error handling implemented
- [ ] Logging added throughout
- [ ] Performance considerations addressed
- [ ] Memory usage optimized
- [ ] User-friendly output formatting
- [ ] AI insights are meaningful and actionable

---

## Examples & References

### Successful Implementation Examples

1. **Multi-Metric Correlation Analysis**
   - File: `analyzers/advanced_order_analyzer.py`
   - Excel: Creates "Multi-Metric Correlations" worksheet
   - Word: Adds correlation matrix tables with AI insights
   - Charts: Correlation heatmaps

2. **Case vs Piece Picking Analysis**
   - File: `analyzers/picking_analyzer.py`
   - Excel: Creates "Picking Method Breakdown" and "Picking by Category" worksheets
   - Word: Adds picking tables with operational complexity insights
   - Charts: Picking distribution charts

3. **2D Classification Matrix**
   - File: `analyzers/enhanced_abc_fms_analyzer.py`
   - Excel: Creates "ABC-FMS SKU Count Matrix" and "ABC-FMS Volume Matrix" worksheets
   - Word: Adds classification matrices with strategic insights
   - Charts: Heatmap visualizations

### Quick Reference Commands

```bash
# Copy analyzer template
cp analyzers/new_analyzer_template.py analyzers/your_analysis.py

# Test your analyzer
python -c "from analyzers.your_analysis import YourAnalyzer; print('Import successful')"

# Run full pipeline test
python enhanced_main.py

# Check Excel output
python -c "import pandas as pd; print(pd.ExcelFile('Order_Profiles.xlsx').sheet_names)"
```

---

## Support & Troubleshooting

### Common Issues

**Issue**: Excel worksheet not appearing
- **Solution**: Check `_process_advanced_analysis()` method in `excel_exporter.py`
- **Debug**: Add logging to confirm your analysis results are present

**Issue**: Word section not generating
- **Solution**: Verify key name matches in `_add_advanced_analysis_sections()`
- **Debug**: Check if AI insights generation is failing

**Issue**: Charts not embedding
- **Solution**: Confirm chart generation returns valid file paths
- **Debug**: Check chart file permissions and existence

### Getting Help

1. **Check Logs**: All components have comprehensive logging
2. **Test Incrementally**: Test each integration step independently  
3. **Use Templates**: Follow existing successful implementations
4. **Validate Data**: Ensure your analysis returns expected data structures

---

*This framework ensures consistent, scalable expansion of the warehouse analysis platform while maintaining high-quality report generation and AI-powered insights.*