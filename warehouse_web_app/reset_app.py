#!/usr/bin/env python
"""
Reset App Script - Clear session state and cache for fresh start
"""

import streamlit as st
import os
import glob

def clear_streamlit_cache():
    """Clear Streamlit cache and session files."""
    print("🧹 Clearing Streamlit cache...")
    
    # Clear .streamlit cache directories
    cache_dirs = [
        os.path.expanduser("~/.streamlit"),
        ".streamlit",
        "__pycache__",
        "components/__pycache__",
        "web_utils/__pycache__",
        "pages/__pycache__"
    ]
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                import shutil
                if os.path.isdir(cache_dir):
                    shutil.rmtree(cache_dir)
                    print(f"   ✅ Cleared {cache_dir}")
            except Exception as e:
                print(f"   ⚠️  Could not clear {cache_dir}: {e}")
    
    # Clear Python cache files
    for pattern in ["**/*.pyc", "**/__pycache__"]:
        for file in glob.glob(pattern, recursive=True):
            try:
                if os.path.isfile(file):
                    os.remove(file)
                elif os.path.isdir(file):
                    import shutil
                    shutil.rmtree(file)
                print(f"   ✅ Cleared {file}")
            except Exception as e:
                print(f"   ⚠️  Could not clear {file}: {e}")

def main():
    print("🚀 Warehouse Analysis App Reset")
    print("=" * 40)
    
    clear_streamlit_cache()
    
    print("\n🎯 Reset Complete!")
    print("   ✅ Cache cleared")
    print("   ✅ Session state will be fresh")
    print("   ✅ Full-width layout will be active")
    
    print("\n🚀 To start the app with fresh state:")
    print("   streamlit run app.py --server.port 8501")
    print("\n💡 If you see the old layout:")
    print("   1. Hard refresh browser (Ctrl+F5 or Cmd+Shift+R)")
    print("   2. Clear browser cache")
    print("   3. Try incognito/private window")

if __name__ == "__main__":
    main()