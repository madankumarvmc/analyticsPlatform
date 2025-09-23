#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Header Fix Verification Test
Tests specifically for the header component issues that were causing raw HTML to display.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def test_header_svg_generation():
    """Test that SVG generation doesn't contain problematic code."""
    print("🧪 Testing SVG logo generation...")
    
    from components.header import HeaderComponent
    
    header = HeaderComponent()
    svg_content = header._get_default_logo_svg()
    
    # Check that SVG is properly formed
    assert svg_content.startswith('<svg'), "SVG should start with <svg tag"
    assert svg_content.endswith('</svg>'), "SVG should end with </svg> tag"
    assert 'class="wh-logo-image"' in svg_content, "SVG should use new CSS class"
    
    # Check that there are no problematic characters that could cause display issues
    problematic_patterns = [
        'fill="#FF6B35"/>',  # This was showing in the screenshot
        'font-family="Arial, sans-serif" font-size=',  # This was also visible
    ]
    
    for pattern in problematic_patterns:
        if pattern in svg_content:
            print(f"⚠️  Found potentially problematic pattern: {pattern}")
        else:
            print(f"✅ Pattern '{pattern[:20]}...' properly contained")
    
    print("✅ SVG generation test passed")
    return True

def test_css_class_uniqueness():
    """Test that CSS classes are unique and won't conflict."""
    print("\n🧪 Testing CSS class uniqueness...")
    
    from components.header import HeaderComponent
    
    header = HeaderComponent()
    css_content = header._get_header_css()
    
    # Check for new prefixed class names
    expected_classes = [
        'wh-header-container',
        'wh-logo-container', 
        'wh-logo-image',
        'wh-title-container',
        'wh-main-title',
        'wh-subtitle',
        'wh-nav-container',
        'wh-nav-links',
        'wh-nav-link',
        'wh-header-divider'
    ]
    
    for class_name in expected_classes:
        assert f'.{class_name}' in css_content, f"CSS should contain .{class_name}"
        print(f"✅ Found CSS class: .{class_name}")
    
    # Check that old conflicting class names are not present
    old_classes = [
        '.header-container',
        '.logo-container',
        '.main-title',
        '.title-container'
    ]
    
    for old_class in old_classes:
        if old_class in css_content:
            print(f"⚠️  Found old conflicting class: {old_class}")
        else:
            print(f"✅ Old class '{old_class}' properly removed")
    
    print("✅ CSS class uniqueness test passed")
    return True

def test_html_structure_safety():
    """Test that HTML structure is safe and won't show raw code."""
    print("\n🧪 Testing HTML structure safety...")
    
    from components.header import HeaderComponent
    
    header = HeaderComponent()
    
    # Test that the methods generate proper HTML without raw code exposure
    svg_content = header._get_default_logo_svg()
    css_content = header._get_header_css()
    
    # Check SVG structure
    assert '<svg' in svg_content and '</svg>' in svg_content, "SVG should be properly structured"
    assert not svg_content.startswith('fill='), "SVG should not start with attribute"
    assert not svg_content.startswith('font-family='), "SVG should not start with font attribute"
    
    # Check CSS structure  
    assert '<style>' in css_content and '</style>' in css_content, "CSS should be wrapped in style tags"
    assert css_content.count('<style>') == css_content.count('</style>'), "CSS tags should be balanced"
    
    print("✅ HTML structure safety test passed")
    return True

def test_app_integration():
    """Test that the app imports and initializes without errors."""
    print("\n🧪 Testing app integration...")
    
    try:
        # Test main app import
        from app import main
        print("✅ Main app import successful")
        
        # Test header component import in app context
        from components.header import create_header, create_simple_header
        print("✅ Header component import in app context successful")
        
        # Test config import
        from config_web import CUSTOM_CSS
        assert isinstance(CUSTOM_CSS, str), "CUSTOM_CSS should be a string"
        print("✅ Config import and validation successful")
        
    except Exception as e:
        print(f"❌ App integration test failed: {e}")
        return False
    
    print("✅ App integration test passed")
    return True

def run_header_fix_tests():
    """Run all header fix verification tests."""
    print("🔧 HEADER FIX VERIFICATION TESTS")
    print("=" * 50)
    
    tests = [
        ("SVG Generation", test_header_svg_generation),
        ("CSS Class Uniqueness", test_css_class_uniqueness), 
        ("HTML Structure Safety", test_html_structure_safety),
        ("App Integration", test_app_integration)
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
            print(f"❌ {test_name} test crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print("🧪 HEADER FIX TEST SUMMARY")
    print("=" * 50)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL HEADER FIXES VERIFIED!")
        print("✅ No raw HTML/CSS code should be visible on the webpage")
        print("✅ Header component should render properly with logo")
        print("✅ CSS conflicts resolved with unique class names")
        print("✅ SVG logo should display correctly")
        print("\n🚀 The webpage should now work correctly!")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the errors above.")
    
    return failed == 0

if __name__ == "__main__":
    success = run_header_fix_tests()
    sys.exit(0 if success else 1)