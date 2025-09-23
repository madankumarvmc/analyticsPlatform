#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Script for Warehouse Analysis Web App
Tests basic functionality and imports.
"""

import sys
from pathlib import Path
import traceback

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test all critical imports."""
    print("🧪 Testing imports...")
    
    tests = [
        ("Streamlit", lambda: __import__('streamlit')),
        ("Pandas", lambda: __import__('pandas')), 
        ("Config Web", lambda: __import__('config_web')),
        ("Session Manager", lambda: __import__('web_utils.session_manager', fromlist=['SessionManager'])),
        ("Error Handler", lambda: __import__('web_utils.error_handler', fromlist=['ErrorHandler'])),
        ("Analysis Integration", lambda: __import__('web_utils.analysis_integration', fromlist=['WebAnalysisIntegrator'])),
        ("File Upload Component", lambda: __import__('components.file_upload', fromlist=['create_file_upload_section'])),
        ("Parameter Controls", lambda: __import__('components.parameter_controls', fromlist=['create_parameter_controls'])),
        ("Results Display", lambda: __import__('components.results_display', fromlist=['create_results_display_section']))
    ]
    
    passed = 0
    failed = 0
    
    for name, import_func in tests:
        try:
            import_func()
            print(f"  ✅ {name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {str(e)}")
            failed += 1
    
    print(f"\n📊 Import Results: {passed} passed, {failed} failed")
    return failed == 0

def test_session_manager():
    """Test session manager functionality."""
    print("\n🧪 Testing Session Manager...")
    
    try:
        from web_utils.session_manager import SessionManager
        
        # Test initialization
        sm = SessionManager()
        print("  ✅ SessionManager initialization")
        
        # Test session info
        info = sm.get_session_info()
        assert isinstance(info, dict)
        print("  ✅ Session info generation")
        
        # Test error logging
        test_error = ValueError("Test error")
        sm.log_error(test_error, "Test context")
        errors = sm.get_error_log()
        assert len(errors) > 0
        print("  ✅ Error logging")
        
        # Test performance metrics
        sm.update_performance_metrics("test_metric", 42)
        metrics = sm.get_performance_metrics()
        assert "test_metric" in metrics
        print("  ✅ Performance metrics")
        
        print("✅ Session Manager tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Session Manager test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_error_handler():
    """Test error handler functionality."""
    print("\n🧪 Testing Error Handler...")
    
    try:
        from web_utils.error_handler import ErrorHandler
        
        # Test initialization
        eh = ErrorHandler()
        print("  ✅ ErrorHandler initialization")
        
        # Test error handling
        test_error = ValueError("Test error message")
        result = eh.handle_error(test_error, category="test", show_user_message=False)
        assert isinstance(result, dict)
        assert result['error_type'] == 'ValueError'
        print("  ✅ Error handling")
        
        # Test user-friendly messages
        message = eh._get_user_friendly_message(test_error)
        assert isinstance(message, str)
        print("  ✅ User-friendly messages")
        
        print("✅ Error Handler tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Error Handler test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_config():
    """Test configuration loading."""
    print("\n🧪 Testing Configuration...")
    
    try:
        from config_web import (
            APP_CONFIG, DEFAULT_ABC_THRESHOLDS, DEFAULT_FMS_THRESHOLDS,
            CUSTOM_CSS, ERROR_MESSAGES, SUCCESS_MESSAGES
        )
        
        # Test APP_CONFIG
        assert isinstance(APP_CONFIG, dict)
        assert 'title' in APP_CONFIG
        print("  ✅ APP_CONFIG")
        
        # Test thresholds
        assert isinstance(DEFAULT_ABC_THRESHOLDS, dict)
        assert 'A_THRESHOLD' in DEFAULT_ABC_THRESHOLDS
        print("  ✅ ABC thresholds")
        
        assert isinstance(DEFAULT_FMS_THRESHOLDS, dict)
        assert 'F_THRESHOLD' in DEFAULT_FMS_THRESHOLDS
        print("  ✅ FMS thresholds")
        
        # Test CSS
        assert isinstance(CUSTOM_CSS, str)
        assert 'main-header' in CUSTOM_CSS
        print("  ✅ Custom CSS")
        
        # Test messages
        assert isinstance(ERROR_MESSAGES, dict)
        assert isinstance(SUCCESS_MESSAGES, dict)
        print("  ✅ Messages")
        
        print("✅ Configuration tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_components():
    """Test component imports."""
    print("\n🧪 Testing Components...")
    
    try:
        # Test file upload component
        from components.file_upload import FileUploadValidator, create_file_upload_section
        validator = FileUploadValidator()
        assert hasattr(validator, 'upload_file_component')
        print("  ✅ File upload component")
        
        # Test parameter controls
        from components.parameter_controls import ParameterController, create_parameter_controls
        controller = ParameterController()
        assert hasattr(controller, 'create_abc_controls')
        print("  ✅ Parameter controls component")
        
        # Test results display
        from components.results_display import create_results_display_section
        print("  ✅ Results display component")
        
        # Test header component
        from components.header import HeaderComponent, create_header, create_simple_header
        header = HeaderComponent()
        assert hasattr(header, 'render_header')
        print("  ✅ Header component")
        
        print("✅ Component tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_pages():
    """Test page imports."""
    print("\n🧪 Testing Pages...")
    
    page_files = [
        "pages/1_📊_Order_Analysis.py",
        "pages/2_👥_Manpower_Analysis.py", 
        "pages/3_📦_Slotting_Analysis.py",
        "pages/4_⚙️_Settings.py"
    ]
    
    passed = 0
    for page_file in page_files:
        page_path = Path(__file__).parent / page_file
        if page_path.exists():
            print(f"  ✅ {page_file}")
            passed += 1
        else:
            print(f"  ❌ {page_file} - File not found")
    
    print(f"✅ Page tests: {passed}/{len(page_files)} pages found")
    return passed == len(page_files)

def run_all_tests():
    """Run all tests."""
    print("🧪 Running Warehouse Analysis Web App Tests\n")
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Session Manager", test_session_manager),
        ("Error Handler", test_error_handler),
        ("Components", test_components),
        ("Pages", test_pages)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {str(e)}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"🧪 TEST SUMMARY")
    print(f"{'='*50}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print(f"\n🎉 All tests passed! The web application is ready to use.")
        print(f"🚀 To start the app, run: streamlit run app.py")
    else:
        print(f"\n⚠️  Some tests failed. Please review the errors above.")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)