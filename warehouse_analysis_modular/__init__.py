#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Warehouse Analysis Modular Package

A modularized version of the warehouse analysis tool with improved organization,
maintainability, and reusability.

Modules:
- analyzers: Core analysis logic (order analysis, SKU profiling, cross-tabulation)
- reporting: Report generation (charts, HTML, Excel, LLM integration)
- utils: Common utility functions
"""

__version__ = "1.0.0"
__author__ = "Analytics Tool"

# Import main functions for easy access
from .main import run_full_analysis

__all__ = ['run_full_analysis']