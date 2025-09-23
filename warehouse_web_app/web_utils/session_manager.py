#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Session Manager for Warehouse Analysis Web App
Handles session state initialization, persistence, and cleanup.
"""

import streamlit as st
import json
import logging
import traceback
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages session state for the warehouse analysis web application."""
    
    def __init__(self):
        self.session_keys = [
            'analysis_complete',
            'analysis_results', 
            'analysis_outputs',
            'uploaded_file',
            'file_validator',
            'parameters',
            'user_settings',
            'analysis_history',
            'current_analysis_id',
            'error_log',
            'performance_metrics'
        ]
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize all session state variables with default values."""
        default_values = {
            'analysis_complete': False,
            'analysis_results': None,
            'analysis_outputs': None,
            'uploaded_file': None,
            'file_validator': None,
            'parameters': None,
            'user_settings': self._get_default_user_settings(),
            'analysis_history': [],
            'current_analysis_id': None,
            'error_log': [],
            'performance_metrics': {},
            'session_initialized': True,
            'session_start_time': datetime.now(),
            'last_activity': datetime.now()
        }
        
        for key, default_value in default_values.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    def _get_default_user_settings(self) -> Dict[str, Any]:
        """Get default user settings."""
        return {
            'default_abc_thresholds': {'A_THRESHOLD': 70.0, 'B_THRESHOLD': 90.0},
            'default_fms_thresholds': {'F_THRESHOLD': 70.0, 'M_THRESHOLD': 90.0},
            'default_percentiles': [95, 90, 85],
            'default_output_options': {
                'generate_charts': True,
                'generate_llm_summaries': True,
                'generate_html_report': True,
                'generate_excel_export': True
            },
            'ui_preferences': {
                'theme': 'light',
                'auto_run_analysis': False,
                'show_advanced_options': False,
                'default_chart_style': 'default'
            },
            'data_preferences': {
                'auto_save_parameters': True,
                'cache_analysis_results': True,
                'max_file_size_mb': 50
            }
        }
    
    def update_last_activity(self):
        """Update the last activity timestamp."""
        st.session_state.last_activity = datetime.now()
    
    def get_session_duration(self) -> timedelta:
        """Get the current session duration."""
        if 'session_start_time' in st.session_state:
            return datetime.now() - st.session_state.session_start_time
        return timedelta(0)
    
    def is_session_expired(self, timeout_hours: int = 24) -> bool:
        """Check if the session has expired."""
        if 'last_activity' in st.session_state:
            return datetime.now() - st.session_state.last_activity > timedelta(hours=timeout_hours)
        return False
    
    def clear_analysis_state(self):
        """Clear analysis-related session state."""
        analysis_keys = [
            'analysis_complete',
            'analysis_results',
            'analysis_outputs',
            'uploaded_file',
            'file_validator',
            'parameters',
            'current_analysis_id'
        ]
        
        for key in analysis_keys:
            if key in st.session_state:
                st.session_state[key] = None if key != 'analysis_complete' else False
        
        logger.info("Analysis state cleared")
    
    def save_analysis_to_history(self, analysis_results: Dict[str, Any], parameters: Dict[str, Any]):
        """Save completed analysis to history."""
        analysis_entry = {
            'id': self._generate_analysis_id(),
            'timestamp': datetime.now().isoformat(),
            'parameters': parameters,
            'results_summary': self._create_results_summary(analysis_results),
            'file_name': getattr(st.session_state.uploaded_file, 'name', 'Unknown') if st.session_state.uploaded_file else 'Unknown'
        }
        
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
        
        st.session_state.analysis_history.append(analysis_entry)
        
        # Keep only last 10 analyses
        if len(st.session_state.analysis_history) > 10:
            st.session_state.analysis_history = st.session_state.analysis_history[-10:]
        
        logger.info(f"Analysis {analysis_entry['id']} saved to history")
    
    def _generate_analysis_id(self) -> str:
        """Generate a unique analysis ID."""
        return f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _create_results_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of analysis results for history."""
        summary = {}
        
        if 'order_statistics' in analysis_results:
            stats = analysis_results['order_statistics']
            summary.update({
                'total_order_lines': stats.get('total_order_lines', 0),
                'unique_skus': stats.get('unique_skus', 0),
                'date_range_days': stats.get('unique_dates', 0),
                'total_volume': stats.get('total_case_equivalent', 0)
            })
        
        if 'sku_profile_abc_fms' in analysis_results and analysis_results['sku_profile_abc_fms'] is not None:
            df = analysis_results['sku_profile_abc_fms']
            if not df.empty:
                summary.update({
                    'abc_distribution': df['ABC'].value_counts().to_dict() if 'ABC' in df.columns else {},
                    'fms_distribution': df['FMS'].value_counts().to_dict() if 'FMS' in df.columns else {}
                })
        
        return summary
    
    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """Get the analysis history."""
        return st.session_state.get('analysis_history', [])
    
    def clear_analysis_history(self):
        """Clear the analysis history."""
        st.session_state.analysis_history = []
        logger.info("Analysis history cleared")
    
    def log_error(self, error: Exception, context: str = ""):
        """Log an error to the session error log."""
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'traceback': traceback.format_exc()
        }
        
        if 'error_log' not in st.session_state:
            st.session_state.error_log = []
        
        st.session_state.error_log.append(error_entry)
        
        # Keep only last 20 errors
        if len(st.session_state.error_log) > 20:
            st.session_state.error_log = st.session_state.error_log[-20:]
        
        logger.error(f"Error logged: {error_entry['error_type']} - {error_entry['error_message']}")
    
    def get_error_log(self) -> List[Dict[str, Any]]:
        """Get the error log."""
        return st.session_state.get('error_log', [])
    
    def clear_error_log(self):
        """Clear the error log."""
        st.session_state.error_log = []
        logger.info("Error log cleared")
    
    def update_performance_metrics(self, metric_name: str, value: Any):
        """Update performance metrics."""
        if 'performance_metrics' not in st.session_state:
            st.session_state.performance_metrics = {}
        
        st.session_state.performance_metrics[metric_name] = {
            'value': value,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return st.session_state.get('performance_metrics', {})
    
    def export_session_state(self) -> str:
        """Export session state to JSON string."""
        exportable_state = {}
        
        for key in self.session_keys:
            if key in st.session_state:
                value = st.session_state[key]
                # Convert non-serializable objects
                if hasattr(value, 'to_dict'):
                    exportable_state[key] = value.to_dict()
                elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    exportable_state[key] = value
                else:
                    exportable_state[key] = str(value)
        
        return json.dumps(exportable_state, indent=2, default=str)
    
    def validate_session_state(self) -> List[str]:
        """Validate session state and return any issues found."""
        issues = []
        
        # Check for required keys
        for key in ['session_initialized', 'session_start_time']:
            if key not in st.session_state:
                issues.append(f"Missing required session key: {key}")
        
        # Check for expired session
        if self.is_session_expired():
            issues.append("Session has expired")
        
        # Check for corrupted data
        if 'analysis_history' in st.session_state:
            if not isinstance(st.session_state.analysis_history, list):
                issues.append("Analysis history is corrupted")
        
        if 'error_log' in st.session_state:
            if not isinstance(st.session_state.error_log, list):
                issues.append("Error log is corrupted")
        
        return issues
    
    def reset_session(self):
        """Reset the entire session state."""
        for key in list(st.session_state.keys()):
            if key.startswith('_') or key in ['session_initialized']:
                continue
            del st.session_state[key]
        
        self._initialize_session_state()
        logger.info("Session state reset")
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get information about the current session."""
        return {
            'session_duration': str(self.get_session_duration()),
            'total_analyses': len(self.get_analysis_history()),
            'total_errors': len(self.get_error_log()),
            'last_activity': st.session_state.get('last_activity', 'Unknown'),
            'session_start': st.session_state.get('session_start_time', 'Unknown'),
            'current_analysis_id': st.session_state.get('current_analysis_id'),
            'analysis_in_progress': st.session_state.get('analysis_complete', False)
        }