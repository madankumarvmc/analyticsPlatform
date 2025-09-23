# 🚀 Adding New Analyzers to the Warehouse Analysis Framework

This guide provides step-by-step instructions for adding new analysis capabilities to the warehouse analysis framework. Follow these steps to ensure proper integration with the existing system and web interface.

## 📋 **Prerequisites**

Before adding a new analyzer, ensure you have:
- ✅ Reviewed the [Analyzer Documentation](ANALYZER_DOCUMENTATION.md)
- ✅ Identified the specific analysis you want to implement
- ✅ Understood the input data requirements
- ✅ Planned the output structure and insights

## 🎯 **Step-by-Step Implementation Guide**

### **Step 1: Create Your Analyzer Class**

#### **1.1 Copy the Template**
```bash
cd warehouse_analysis_modular/analyzers/
cp new_analyzer_template.py your_analyzer_name.py
```

#### **1.2 Customize the Template**
Replace the following in your new file:
- `NewAnalyzerTemplate` → `YourAnalyzerName`
- `analyze_with_template` → `analyze_your_domain`
- Update class docstring with your specific purpose
- Implement your analysis logic

#### **1.3 Example: Customer Analyzer**
```python
class CustomerAnalyzer:
    """Analyzes customer ordering patterns and segmentation."""
    
    def __init__(self, enriched_data: pd.DataFrame):
        self.enriched_data = enriched_data
        self._validate_input_data()
    
    def _validate_input_data(self):
        required_columns = ['Date', 'Order No.', 'Customer ID', 'Sku Code']
        validate_dataframe(self.enriched_data, required_columns, min_rows=1)
    
    def segment_customers_rfm(self):
        """Perform RFM (Recency, Frequency, Monetary) analysis."""
        # Your RFM logic here
        
    def analyze_customer_patterns(self):
        """Analyze customer ordering patterns."""
        # Your pattern analysis here
        
    def run_full_analysis(self):
        return {
            'customer_segments': self.segment_customers_rfm(),
            'customer_patterns': self.analyze_customer_patterns(),
            'statistics': self.get_summary_statistics()
        }
```

### **Step 2: Update Package Exports**

#### **2.1 Add to `__init__.py`**
```python
# In warehouse_analysis_modular/analyzers/__init__.py
from .order_analyzer import OrderAnalyzer
from .sku_analyzer import SkuAnalyzer
from .cross_tabulation import CrossTabulationAnalyzer
from .your_analyzer_name import YourAnalyzerName  # Add this line

__all__ = [
    'OrderAnalyzer',
    'SkuAnalyzer', 
    'CrossTabulationAnalyzer',
    'YourAnalyzerName'  # Add this line
]
```

### **Step 3: Integration with Web Interface**

#### **3.1 Update Analysis Integration**
```python
# In warehouse_web_app/web_utils/analysis_integration.py

# Add import at the top
from warehouse_analysis_modular.analyzers import YourAnalyzerName

class WebAnalysisIntegrator:
    # Add new analysis method
    def _run_your_analysis(self, enriched_data: pd.DataFrame) -> Dict[str, Any]:
        """Run your custom analysis."""
        if not BACKEND_AVAILABLE:
            return {}
        
        analyzer = YourAnalyzerName(enriched_data)
        return analyzer.run_full_analysis()
    
    def _combine_analysis_results(self, order_results, sku_results, cross_tab_results):
        """Update to include your results."""
        combined = {}
        
        # Existing results...
        
        # Add your results
        your_results = self._run_your_analysis(enriched_data)
        if your_results:
            combined.update({
                'your_analysis_key': your_results.get('main_output'),
                'your_insights': your_results.get('insights')
            })
        
        return combined
```

#### **3.2 Update Results Display**
```python
# In warehouse_web_app/components/results_display.py

class ResultsDisplayManager:
    def _display_your_analysis_fullwidth(self, analysis_results: Dict[str, Any]) -> None:
        """Display your analysis results."""
        if 'your_analysis_key' in analysis_results:
            your_data = analysis_results['your_analysis_key']
            
            st.subheader("🎯 Your Analysis Title")
            
            # Display your results
            if isinstance(your_data, pd.DataFrame):
                st.dataframe(your_data, use_container_width=True)
            
            # Add charts if applicable
            if not your_data.empty:
                fig = px.bar(your_data, x='category', y='value', 
                            title="Your Analysis Chart")
                st.plotly_chart(fig, use_container_width=True)
    
    def display_analysis_results(self, analysis_results, outputs=None):
        """Update main display method."""
        # Existing tabs...
        tabs = st.tabs([
            "📊 Overview", "📅 Date Analysis", "🏷️ SKU Analysis", 
            "🔤 ABC-FMS Analysis", "📈 Charts", "📄 Reports",
            "🎯 Your Analysis"  # Add your tab
        ])
        
        # Existing tab content...
        
        with tabs[6]:  # Your new tab
            self._display_your_analysis_fullwidth(analysis_results)
```

### **Step 4: Add Configuration (If Needed)**

#### **4.1 Update Main Configuration**
```python
# In config.py (if your analyzer needs configuration)
YOUR_ANALYZER_CONFIG = {
    'threshold_value': 50.0,
    'categories': ['A', 'B', 'C'],
    'calculation_method': 'weighted'
}
```

#### **4.2 Update Web Configuration**
```python
# In warehouse_web_app/config_web.py (if web-specific config needed)
YOUR_ANALYZER_DEFAULTS = {
    'enable_advanced_features': True,
    'display_charts': True
}
```

### **Step 5: Add Parameter Controls (Optional)**

If your analyzer needs user-configurable parameters:

```python
# In warehouse_web_app/components/parameter_controls.py

class ParameterController:
    def create_your_analyzer_controls(self) -> Dict[str, Any]:
        """Create controls for your analyzer parameters."""
        st.subheader("🎯 Your Analyzer Settings")
        
        threshold = st.slider(
            "Your Threshold",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=1.0,
            help="Configure your analyzer threshold"
        )
        
        enable_feature = st.checkbox(
            "Enable Advanced Feature",
            value=True,
            help="Enable advanced processing"
        )
        
        return {
            'threshold': threshold,
            'enable_feature': enable_feature
        }
    
    # Update main parameter creation function
    def create_parameter_controls(self):
        # Existing controls...
        
        your_params = self.create_your_analyzer_controls()
        
        # Include in final parameters
        parameters.update({
            'your_analyzer_params': your_params
        })
```

### **Step 6: Testing and Validation**

#### **6.1 Create Unit Tests**
```python
# Create tests/test_your_analyzer.py
import pytest
import pandas as pd
from warehouse_analysis_modular.analyzers import YourAnalyzerName

class TestYourAnalyzer:
    def test_initialization(self):
        """Test analyzer initialization."""
        sample_data = self.create_sample_data()
        analyzer = YourAnalyzerName(sample_data)
        assert analyzer is not None
    
    def test_run_full_analysis(self):
        """Test complete analysis pipeline."""
        sample_data = self.create_sample_data()
        analyzer = YourAnalyzerName(sample_data)
        results = analyzer.run_full_analysis()
        
        # Verify expected keys exist
        assert 'main_output' in results
        assert isinstance(results['main_output'], pd.DataFrame)
    
    def create_sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'Date': pd.date_range('2025-01-01', periods=100),
            'Order No.': [f'ORD{i:04d}' for i in range(100)],
            'Customer ID': [f'CUST{i%10:03d}' for i in range(100)],
            'Sku Code': [f'SKU{i%20:03d}' for i in range(100)]
        })
```

#### **6.2 Test with Sample Data**
```python
# Test your analyzer independently
if __name__ == "__main__":
    # Create sample data
    sample_data = pd.DataFrame({...})
    
    # Test analyzer
    analyzer = YourAnalyzerName(sample_data)
    results = analyzer.run_full_analysis()
    
    print("Analysis completed successfully!")
    print(f"Results keys: {list(results.keys())}")
```

#### **6.3 Test Web Integration**
1. Start the web application
2. Upload test data
3. Verify your analyzer tab appears
4. Check that your results display correctly
5. Test parameter controls (if applicable)

### **Step 7: Documentation and Finalization**

#### **7.1 Update Documentation**
- Add your analyzer to the main documentation
- Include usage examples
- Document any new configuration options

#### **7.2 Add to Main Pipeline**
```python
# In warehouse_analysis_modular/main.py
class WarehouseAnalysisPipeline:
    def run_analysis_pipeline(self):
        # Existing analyses...
        
        # Add your analysis
        if self.enable_your_analysis:
            your_results = self._run_your_analysis(enriched_data)
            self.analysis_results['your_analysis'] = your_results
```

## 🎯 **Common Patterns and Best Practices**

### **Naming Conventions**
- Class names: `YourDomainAnalyzer` (e.g., `CustomerAnalyzer`, `SeasonalAnalyzer`)
- Method names: `analyze_specific_aspect()`, `calculate_metrics()`
- Output keys: `your_domain_analysis`, `your_domain_insights`

### **Error Handling**
```python
def your_analysis_method(self):
    try:
        # Your analysis logic
        results = perform_analysis()
        self.logger.info("Analysis completed successfully")
        return results
    except Exception as e:
        self.logger.error(f"Analysis failed: {str(e)}")
        return pd.DataFrame()  # Return empty structure on failure
```

### **Performance Optimization**
- Use vectorized pandas operations
- Cache expensive calculations
- Validate data early to fail fast
- Log progress for long-running analyses

### **Output Structure Consistency**
```python
def run_full_analysis(self):
    return {
        'main_analysis': your_main_dataframe,
        'secondary_analysis': your_secondary_dataframe,  # Optional
        'insights': your_insights_dict,
        'statistics': your_statistics_dict
    }
```

## 🚨 **Common Pitfalls to Avoid**

### **❌ Don't:**
- Modify existing analyzer interfaces
- Skip input validation
- Return inconsistent output structures
- Ignore error handling
- Forget to update package exports
- Skip documentation

### **✅ Do:**
- Follow the template structure
- Use consistent naming conventions
- Include comprehensive error handling
- Test with realistic data sizes
- Document your analyzer thoroughly
- Update all integration points

## 🔧 **Troubleshooting**

### **Import Errors**
- Verify your analyzer is added to `__init__.py`
- Check that all required packages are installed
- Ensure proper path handling for imports

### **Web Integration Issues**
- Check that analysis integration is updated
- Verify results display methods are added
- Ensure tab structure is updated correctly

### **Data Validation Problems**
- Use `validate_dataframe()` helper function
- Check column name consistency
- Handle missing or invalid data gracefully

### **Performance Issues**
- Profile your analysis with realistic data
- Use pandas vectorized operations
- Consider chunking for very large datasets

## 📞 **Getting Help**

If you encounter issues:
1. Review the [Analyzer Documentation](ANALYZER_DOCUMENTATION.md)
2. Check existing analyzer implementations for patterns
3. Test with the provided template first
4. Verify integration at each step

## 🎉 **Completion Checklist**

- [ ] Analyzer class created and tested
- [ ] Package exports updated (`__init__.py`)
- [ ] Web integration added (analysis_integration.py)
- [ ] Results display implemented (results_display.py)
- [ ] Configuration added (if needed)
- [ ] Parameter controls added (if needed)
- [ ] Unit tests created and passing
- [ ] Web interface tested end-to-end
- [ ] Documentation updated
- [ ] Performance validated

Once all items are checked, your analyzer is ready for production use! 🚀