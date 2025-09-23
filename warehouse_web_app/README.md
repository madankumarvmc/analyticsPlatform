# Warehouse Analysis Web Application

A comprehensive Streamlit-based web interface for the modular warehouse analysis tool, providing an intuitive way to perform ABC-FMS classification, order analysis, and generate detailed reports.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ (tested with Python 3.13)
- pip package manager

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd warehouse_web_app
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser:**
   - Local URL: `http://localhost:8501`
   - The application will automatically open in your default browser

## 📁 Project Structure

```
warehouse_web_app/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── config_web.py                   # Web application configuration
├── test_app.py                     # Comprehensive test suite
├── README.md                       # This file
│
├── components/                     # Reusable UI components
│   ├── __init__.py
│   ├── header.py                  # Professional header with logo
│   ├── file_upload.py             # Excel file upload & validation
│   ├── parameter_controls.py      # ABC/FMS threshold controls
│   └── results_display.py         # Results visualization
│
├── pages/                          # Multi-page application structure
│   ├── 1_📊_Order_Analysis.py     # Main analysis page (active)
│   ├── 2_👥_Manpower_Analysis.py  # Future: Staffing optimization
│   ├── 3_📦_Slotting_Analysis.py  # Future: Layout optimization
│   └── 4_⚙️_Settings.py           # Application settings
│
├── assets/                         # Static assets (logos, etc.)
│   └── README.md                  # Asset usage guide
│
└── web_utils/                      # Web-specific utilities
    ├── __init__.py
    ├── session_manager.py          # Session state management
    ├── error_handler.py            # Error handling & logging
    └── analysis_integration.py     # Backend integration
```

## 🎯 Features

### Current Features (v2.0)

#### 🎨 Professional Interface
- **Header with Logo**: Professional header component with logo on left side
- **Default Branding**: Built-in warehouse-themed SVG logo
- **Custom Logo Support**: Easy integration of company logos (PNG, JPG, SVG)
- **Responsive Design**: Adapts to desktop and mobile devices
- **Gradient Styling**: Modern gradient backgrounds and professional typography

#### 📊 Order Analysis
- **File Upload**: Drag-and-drop Excel file upload with validation
- **Parameter Configuration**: Interactive sliders for ABC/FMS thresholds
- **Data Validation**: Automatic validation of required sheets and columns
- **Progress Tracking**: Real-time analysis progress with status updates
- **Results Display**: Interactive tables and charts

#### 🔧 Configuration Options
- **ABC Classification**: Customize volume-based classification thresholds
- **FMS Classification**: Adjust frequency-based movement categories
- **Output Options**: Select which reports to generate
- **Advanced Settings**: Chart styles, table limits, performance options

#### 📈 Outputs
- **Interactive Charts**: Plotly-based visualizations
- **Excel Reports**: Detailed spreadsheet exports
- **HTML Reports**: Comprehensive web-based reports
- **AI Insights**: LLM-generated recommendations (when integrated)

### 🔮 Future Modules (Planned)

#### 👥 Manpower Analysis
- Staffing requirement calculations
- Peak hour analysis
- Labor cost optimization
- Productivity benchmarking

#### 📦 Slotting Analysis
- Warehouse layout optimization
- SKU placement strategies
- Travel time minimization
- Space utilization analysis

## 🛠️ Technical Details

### Dependencies
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
- **OpenPyXL**: Excel file processing
- **NumPy**: Numerical computations

### Architecture

#### Session Management
The application includes robust session state management:
- **Persistence**: User settings and analysis history
- **Error Tracking**: Comprehensive error logging
- **Performance Metrics**: Analysis timing and resource usage
- **Session Recovery**: Automatic session restoration

#### Error Handling
Centralized error handling system:
- **User-Friendly Messages**: Clear, actionable error descriptions
- **Technical Details**: Expandable debug information
- **Error Categories**: File upload, data validation, analysis, system errors
- **Logging**: File-based error logging with rotation

#### Component Architecture
Modular component design:
- **Reusable Components**: File upload, parameter controls, results display
- **Independent Modules**: Each component can be used standalone
- **State Management**: Components communicate through session state
- **Error Boundaries**: Isolated error handling per component

## 📋 Data Requirements

### Excel File Format
Your Excel file must contain two sheets:

#### OrderData Sheet
Required columns:
- `Date`: Order date (YYYY-MM-DD format)
- `Shipment No.`: Unique shipment identifier
- `Order No.`: Unique order identifier  
- `Sku Code`: SKU identifier matching SkuMaster
- `Qty in Cases`: Quantity in case units
- `Qty in Eaches`: Quantity in each units

#### SkuMaster Sheet
Required columns:
- `Sku Code`: Unique SKU identifier
- `Category`: Product category
- `Case Config`: Number of eaches per case
- `Pallet Fit`: Number of cases per pallet

### Example Data
```
OrderData:
Date,Shipment No.,Order No.,Sku Code,Qty in Cases,Qty in Eaches
2025-01-01,SH001,ORD001,SKU123,10,24
2025-01-01,SH002,ORD002,SKU456,5,0

SkuMaster:
Sku Code,Category,Case Config,Pallet Fit
SKU123,Electronics,12,48
SKU456,Furniture,1,20
```

## ⚙️ Configuration

### Default Settings
The application comes with sensible defaults:
- **ABC Thresholds**: A=70%, B=90%
- **FMS Thresholds**: Fast=70%, Medium=90%
- **Percentiles**: 95th, 90th, 85th
- **Outputs**: All reports enabled by default

### Customization
Settings can be customized through:
1. **Parameter Controls**: Interactive sliders on main page
2. **Settings Page**: Comprehensive configuration options
3. **Configuration Export/Import**: Save and share settings
4. **Session Persistence**: Settings remembered across sessions

### Header & Branding Customization

#### Using Your Company Logo
1. **Add logo file** to the `assets/` directory:
   ```
   assets/
   └── company_logo.png  # Your logo file
   ```

2. **Update header calls** in your application:
   ```python
   # Main app header
   create_header(
       title="Your Company Name",
       subtitle="Warehouse Analysis Platform",
       logo_path="assets/company_logo.png",
       show_navigation=True
   )
   
   # Page headers
   create_simple_header(
       title="📊 Order Analysis",
       logo_path="assets/company_logo.png"
   )
   ```

#### Default Professional Design
- **Built-in Logo**: Professional warehouse-themed SVG logo
- **Gradient Header**: Purple to blue gradient background
- **Responsive Layout**: Adapts to desktop and mobile
- **Professional Typography**: Modern fonts with shadows and effects

## 🧪 Testing

Run the comprehensive test suite:
```bash
python test_app.py
```

The test suite validates:
- ✅ All module imports
- ✅ Configuration loading
- ✅ Session management
- ✅ Error handling
- ✅ Component functionality
- ✅ Page availability

## 🚀 Integration with Backend

The web application integrates with the existing modular warehouse analysis backend:

```python
# Backend integration example
from web_utils.analysis_integration import run_web_analysis

# Run analysis with uploaded file and parameters
result = run_web_analysis(uploaded_file, parameters)

if result['success']:
    # Display results
    analysis_results = result['analysis_results']
    outputs = result['outputs']
else:
    # Handle errors
    error_message = result['message']
```

## 📊 Performance

### Optimization Features
- **Caching**: Analysis results cached for faster repeated access
- **Memory Management**: Configurable memory limits
- **Parallel Processing**: Multi-core utilization for large datasets
- **Progress Tracking**: Real-time progress updates

### File Size Limits
- **Default Limit**: 100MB per Excel file
- **Configurable**: Adjustable through settings
- **Validation**: Pre-upload size checking

## 🔧 Development

### Adding New Pages
1. Create new file in `pages/` directory with format: `N_📊_Page_Name.py`
2. Follow existing page structure
3. Import necessary components from `components/`
4. Use session state for data sharing

### Custom Components
1. Create component in `components/` directory
2. Follow the existing component pattern
3. Use error boundaries for robust error handling
4. Document component parameters and return values

### Testing New Features
1. Add tests to `test_app.py`
2. Run test suite: `python test_app.py`
3. Ensure 100% test pass rate
4. Test manually with Streamlit

## 🐛 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Check if dependencies are installed
pip list | grep streamlit
pip install -r requirements.txt
```

#### File Upload Issues
- Ensure Excel file has both OrderData and SkuMaster sheets
- Check column names match exactly (case-sensitive)
- Verify file size is under limit

#### Performance Issues
- Reduce file size or increase memory limits in settings
- Enable caching in advanced options
- Close other applications to free memory

#### Connection Issues
- Check if port 8501 is available
- Try different port: `streamlit run app.py --server.port 8502`
- Check firewall settings

### Debug Mode
Enable debug mode in `config_web.py`:
```python
APP_CONFIG = {
    'debug_mode': True,  # Enable debug mode
    # ... other settings
}
```

### Error Logs
Check error logs:
- Session errors: Available in Settings > Error Dashboard
- System errors: `warehouse_web_app.log` file
- Streamlit logs: Terminal output

## 📈 Future Enhancements

### Planned Features
- [ ] Real-time collaboration
- [ ] Advanced chart customization
- [ ] Export to PowerBI/Tableau
- [ ] API integration
- [ ] Mobile-responsive design
- [ ] Dark mode theme
- [ ] Multi-language support

### Integration Roadmap
- [ ] Complete backend integration with existing analysis modules
- [ ] LLM integration for advanced insights
- [ ] Database connectivity for large datasets
- [ ] Cloud deployment options
- [ ] Authentication and user management

## 📞 Support

### Getting Help
1. **Documentation**: Check this README and inline help text
2. **Error Dashboard**: Use Settings > Error Dashboard for debugging
3. **Test Suite**: Run `python test_app.py` to validate setup
4. **Logs**: Check `warehouse_web_app.log` for detailed error information

### Reporting Issues
When reporting issues, please include:
- Python version and OS
- Error messages from logs
- Steps to reproduce
- Sample data (if applicable)

## 📄 License

This project is part of the Warehouse Analysis Tool suite. Please refer to the main project license.

---

## 🎉 Congratulations!

You now have a fully functional web interface for warehouse analysis! 

**Next Steps:**
1. Upload your Excel data file
2. Configure ABC/FMS thresholds  
3. Run your first analysis
4. Explore the interactive results
5. Download comprehensive reports

**Happy Analyzing!** 📊🏭