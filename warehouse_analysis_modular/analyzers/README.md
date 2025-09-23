# 📊 Analyzers Directory

This directory contains the core analysis modules for the warehouse analysis framework. Each analyzer handles a specific domain of warehouse data analysis.

## 🔧 **Available Analyzers**

| Analyzer | Purpose | Input | Key Outputs |
|----------|---------|-------|-------------|
| **OrderAnalyzer** | Date-wise order analysis, demand patterns | Enriched order data | Daily summaries, percentiles, demand insights |
| **SkuAnalyzer** | SKU profiling, ABC-FMS classification | Enriched order data | SKU classifications, movement metrics |
| **CrossTabulationAnalyzer** | ABC×FMS cross-analysis | SKU profile with classifications | Cross-tabulation matrices, strategic insights |

## 🚀 **Quick Start**

### **Using Existing Analyzers**
```python
from warehouse_analysis_modular.analyzers import OrderAnalyzer, SkuAnalyzer, CrossTabulationAnalyzer

# Order analysis
order_analyzer = OrderAnalyzer(enriched_data)
order_results = order_analyzer.run_full_analysis()

# SKU analysis
sku_analyzer = SkuAnalyzer(enriched_data)
sku_results = sku_analyzer.run_full_analysis()

# Cross-tabulation analysis
sku_profile = sku_results['sku_profile_abc_fms']
cross_analyzer = CrossTabulationAnalyzer(sku_profile)
cross_results = cross_analyzer.run_full_analysis()
```

### **Adding New Analyzers**
1. **Copy template**: `cp new_analyzer_template.py your_analyzer.py`
2. **Customize**: Replace placeholders with your analysis logic
3. **Test**: Verify with sample data
4. **Integrate**: Update web interface and package exports

## 📚 **Documentation**

- **[Complete Documentation](../ANALYZER_DOCUMENTATION.md)**: Comprehensive analyzer reference
- **[Adding New Analyzers Guide](../ADDING_NEW_ANALYZERS.md)**: Step-by-step implementation guide
- **[Template](new_analyzer_template.py)**: Standardized template for new analyzers

## 🏗️ **Architecture Pattern**

All analyzers follow this standardized structure:

```python
class YourAnalyzer:
    def __init__(self, input_data: pd.DataFrame):
        # Data validation and initialization
        
    def _validate_input_data(self):
        # Input validation logic
        
    def your_analysis_methods(self):
        # Core analysis logic
        
    def run_full_analysis(self) -> Dict:
        # Required main method - orchestrates all analysis
        return {
            'main_output': analysis_dataframe,
            'insights': insights_dict,
            'statistics': statistics_dict
        }
```

## 🔌 **Integration Points**

- **Web Interface**: Automatic integration through `web_utils/analysis_integration.py`
- **Reporting**: Output generation via `reporting/` modules
- **Configuration**: Configurable parameters in `config.py`
- **Utilities**: Common functions in `utils/helpers.py`

## 🧪 **Testing**

Each analyzer should include:
- Input validation tests
- Analysis logic tests
- Integration tests
- Performance tests with realistic data

## 📈 **Extension Examples**

Common analyzer extensions:
- **Customer Analysis**: RFM segmentation, loyalty metrics
- **Seasonal Analysis**: Time-based patterns, forecasting
- **Performance Metrics**: KPI calculations, benchmarking
- **Cost Analysis**: Profitability, cost breakdowns
- **Optimization**: Slotting, inventory optimization

## 🎯 **Best Practices**

- **Consistent Structure**: Follow the template pattern
- **Robust Validation**: Validate inputs thoroughly
- **Error Handling**: Graceful failure handling
- **Performance**: Use vectorized operations
- **Documentation**: Comprehensive docstrings
- **Testing**: Unit and integration tests

Ready to extend the framework? Start with the [Adding New Analyzers Guide](../ADDING_NEW_ANALYZERS.md)! 🚀