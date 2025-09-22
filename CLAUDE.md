# Warehouse Analysis Project - Claude Context

## Project Overview

This project involves a comprehensive warehouse analysis tool that was successfully modularized from a monolithic 1,155-line Python script into a well-organized, maintainable codebase. The tool performs advanced analytics on warehouse order data, including ABC-FMS classification, SKU profiling, and automated report generation.

## Project Structure

### Original Files (Preserved)
- `Warehouse Analysis (2).py` - Original monolithic script (1,155 lines) - **KEPT INTACT**
- `config.py` - Configuration settings and constants
- `data_loader.py` - Data loading and enrichment functionality

### New Modular Structure
```
warehouse_analysis_modular/
├── __init__.py                    # Main package entry point
├── main.py                       # 🎯 Main orchestration script
├── analyzers/                   # 📊 Core analysis modules
│   ├── __init__.py
│   ├── order_analyzer.py        # Date-wise summaries & percentiles
│   ├── sku_analyzer.py          # SKU profiling & ABC-FMS classification
│   └── cross_tabulation.py      # ABC×FMS cross-tabulation analysis
├── reporting/                   # 📈 Report generation modules
│   ├── __init__.py
│   ├── chart_generator.py       # Matplotlib chart generation
│   ├── llm_integration.py       # Gemini API integration with caching
│   ├── html_report.py           # HTML report generation
│   └── excel_exporter.py        # Excel workbook export
└── utils/                       # 🛠️ Common utilities
    ├── __init__.py
    └── helpers.py               # Utility functions

tests/                           # 🧪 Test modules
├── __init__.py
└── test_gemini.py              # Gemini API testing
```

## Key Features & Capabilities

### Core Analysis Functions
1. **Order Analysis** (`order_analyzer.py`)
   - Date-wise order summaries
   - Percentile calculations (95th, 90th, 85th percentiles)
   - Demand pattern analysis
   - Customer and shipment metrics

2. **SKU Analysis** (`sku_analyzer.py`)
   - ABC classification (based on volume)
   - FMS classification (Fast/Medium/Slow based on order frequency)
   - Movement frequency calculations
   - 2D classification (ABC + FMS combinations)

3. **Cross-Tabulation Analysis** (`cross_tabulation.py`)
   - ABC×FMS summary matrices
   - Volume and line distribution analysis
   - Statistical insights and dominant category identification

### Reporting & Visualization
1. **Chart Generation** (`chart_generator.py`)
   - Date time series charts
   - SKU Pareto analysis
   - ABC volume stacked charts
   - ABC×FMS heatmaps
   - Percentile visualizations

2. **LLM Integration** (`llm_integration.py`)
   - Gemini API integration for AI-generated insights
   - Response caching for efficiency
   - Structured prompts for different analysis sections

3. **Report Generation**
   - **Excel Export**: Multi-sheet workbooks with formatted data
   - **HTML Reports**: Interactive web-based reports with embedded charts
   - **Metadata**: JSON metadata for report tracking

## Data Sources & Configuration

### Input Data
- **Primary Data File**: `/Users/MKSBX/Documents/Analytics Tool/TestData.xlsx`
- **Order Data Sheet**: Contains order lines with dates, shipments, SKUs, quantities
- **SKU Master Sheet**: SKU details including case configurations and pallet fits

### Key Configuration Settings
- **ABC Thresholds**: A (<70%), B (70-90%), C (>90%)
- **FMS Thresholds**: Fast (<70%), Medium (70-90%), Slow (>90%)
- **LLM API**: Gemini 2.0-flash model integration
- **Output Formats**: Excel, HTML, PNG charts

## Recent Analysis Results

### Dataset Statistics
- **Total Order Lines**: 123,307
- **Unique SKUs**: 890
- **Date Range**: 79 days (February 1 - April 30, 2025)
- **Total Case Equivalent**: 883,465

### Key Insights Generated
1. **Peak Demand**: Late March (25th-28th) shows significantly higher volumes
2. **SKU Concentration**: Top 5 SKUs dominate volume distribution
3. **ABC Distribution**: Category 'A' accounts for ~70% of total volume
4. **Capacity Planning**: 95th percentile volume is 17,536 (vs average 11,183)

## Generated Outputs

### Files Created
1. `Order_Profiles.xlsx` (87KB) - Comprehensive Excel workbook
2. `report/Order_Profiles_Analysis.html` (47KB) - Interactive HTML report
3. `report/charts/` - 6 PNG chart files
4. `report/llm_cache.json` (16KB) - Cached AI responses
5. `report/metadata.json` - Report metadata

### Charts Generated
- Date total case equivalent time series
- Date distinct customers bar chart
- Percentile analysis horizontal bar chart
- SKU Pareto chart (top 50 SKUs)
- ABC volume stacked bar chart
- ABC×FMS heatmap

## Usage Instructions

### Option 1: Use Original Script
```python
exec(open('Warehouse Analysis (2).py').read())
```

### Option 2: Use Modular Version (Full Analysis)
```python
from warehouse_analysis_modular.main import run_full_analysis
results = run_full_analysis()
```

### Option 3: Use Specific Components
```python
from warehouse_analysis_modular.analyzers import SkuAnalyzer
from warehouse_analysis_modular.reporting import ChartGenerator

# Use individual components as needed
```

### Option 4: Analysis Only (No Reports)
```python
from warehouse_analysis_modular.main import run_analysis_only
results = run_analysis_only()
```

## Technical Dependencies

### Required Python Packages
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `matplotlib` - Chart generation
- `jinja2` - HTML template rendering
- `requests` - API calls to LLM services
- `openpyxl` - Excel file operations

### External Services
- **Gemini API** - For AI-generated insights and recommendations
- **Environment Variables**: `GEMINI_API_KEY` for LLM integration

## Development Notes

### Modularization Benefits Achieved
1. **Separation of Concerns**: Each module has single responsibility
2. **Reusability**: Components can be imported independently
3. **Maintainability**: Easy to modify specific functionality
4. **Testing**: Each component can be tested in isolation
5. **Scalability**: Easy to add new analyzers or report types
6. **Error Handling**: Improved logging and error management

### Code Quality Improvements
- Comprehensive docstrings and type hints
- Consistent error handling and logging
- Configuration-driven design
- Proper input validation
- Caching for performance optimization

## Future Enhancement Opportunities

### Potential Improvements
1. **Additional Analysis Types**
   - Seasonal pattern analysis
   - Customer segmentation
   - Demand forecasting models
   - Cost analysis integration

2. **Reporting Enhancements**
   - Dashboard creation
   - Real-time updates
   - PDF report generation
   - Email automation

3. **Data Integration**
   - Database connectivity
   - Real-time data feeds
   - Multiple data source support
   - Data quality monitoring

### Testing & Validation
- Unit tests for each module
- Integration tests for full pipeline
- Performance benchmarking
- Data validation routines

## Commands & Shortcuts

### Running Analysis
```bash
# Full analysis with all outputs
python warehouse_analysis_modular/main.py

# Test Gemini API connectivity
python tests/test_gemini.py

# Module-specific testing
python -c "from warehouse_analysis_modular.analyzers import OrderAnalyzer; print('OK')"
```

### File Locations
- **Data**: `/Users/MKSBX/Documents/Analytics Tool/TestData.xlsx`
- **Config**: `config.py`
- **Main Script**: `warehouse_analysis_modular/main.py`
- **Output**: `Order_Profiles.xlsx` and `report/` directory

## Project Status

### Completed ✅
- [x] Complete modularization of monolithic script
- [x] All analysis functionality preserved
- [x] Enhanced error handling and logging
- [x] LLM integration with caching
- [x] Comprehensive reporting (Excel + HTML)
- [x] Chart generation and visualization
- [x] Documentation and code organization
- [x] Successful execution and validation

### Architecture Decision Records
- **Non-destructive approach**: Original file preserved untouched
- **Configuration-driven**: All settings centralized in config.py
- **Modular design**: Clear separation between analysis, reporting, and utilities
- **Caching strategy**: LLM responses cached for efficiency
- **Error resilience**: Pipeline continues even if individual components fail

## Contact & Maintenance

This modular warehouse analysis tool represents a complete transformation from a monolithic script to a production-ready, maintainable codebase. The architecture supports future enhancements while maintaining backward compatibility and operational reliability.

**Last Updated**: September 22, 2024
**Status**: Production Ready
**Performance**: Successfully processes 123K+ order lines across 890 SKUs