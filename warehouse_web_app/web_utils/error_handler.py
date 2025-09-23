#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Error Handler for Warehouse Analysis Web App
Centralized error handling, logging, and user-friendly error display.
"""

import streamlit as st
import logging
import traceback
import sys
from typing import Optional, Callable, Any, Dict
from functools import wraps
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('warehouse_web_app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class ErrorHandler:
    """Centralized error handling for the warehouse analysis web application."""
    
    def __init__(self):
        self.error_categories = {
            'file_upload': 'File Upload Error',
            'data_validation': 'Data Validation Error',
            'analysis': 'Analysis Error',
            'export': 'Export Error',
            'system': 'System Error',
            'configuration': 'Configuration Error'
        }
        
        self.user_friendly_messages = {
            'FileNotFoundError': 'The requested file could not be found.',
            'PermissionError': 'Permission denied. Please check file access rights.',
            'ValueError': 'Invalid data format or values detected.',
            'KeyError': 'Required data field is missing.',
            'ImportError': 'Required module could not be loaded.',
            'ConnectionError': 'Network connection issue.',
            'TimeoutError': 'Operation timed out. Please try again.',
            'MemoryError': 'Insufficient memory to complete the operation.',
            'AttributeError': 'Internal data structure error.',
            'TypeError': 'Data type mismatch detected.'
        }
    
    def handle_error(self, 
                    error: Exception, 
                    category: str = 'system',
                    context: str = "",
                    show_user_message: bool = True,
                    log_error: bool = True) -> Dict[str, Any]:
        """
        Handle an error with logging and user notification.
        
        Args:
            error: The exception that occurred
            category: Error category for classification
            context: Additional context about when/where the error occurred
            show_user_message: Whether to display error message to user
            log_error: Whether to log the error
            
        Returns:
            Dictionary with error details
        """
        error_details = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'category': category,
            'context': context,
            'traceback': traceback.format_exc(),
            'user_message': self._get_user_friendly_message(error)
        }
        
        # Log the error
        if log_error:
            self._log_error(error_details)
        
        # Show user-friendly message
        if show_user_message:
            self._display_user_error(error_details)
        
        # Store in session state for debugging
        self._store_error_in_session(error_details)
        
        return error_details
    
    def _get_user_friendly_message(self, error: Exception) -> str:
        """Generate a user-friendly error message."""
        error_type = type(error).__name__
        
        if error_type in self.user_friendly_messages:
            base_message = self.user_friendly_messages[error_type]
        else:
            base_message = "An unexpected error occurred."
        
        # Add specific context for common errors
        if isinstance(error, FileNotFoundError):
            return f"{base_message} Please check the file path and try again."
        elif isinstance(error, ValueError) and "sheet" in str(error).lower():
            return "The uploaded Excel file is missing required sheets (OrderData, SkuMaster)."
        elif isinstance(error, ValueError) and "column" in str(error).lower():
            return "The uploaded file is missing required columns. Please check the data format."
        elif isinstance(error, MemoryError):
            return f"{base_message} Try uploading a smaller file or contact support."
        elif isinstance(error, ImportError):
            return f"{base_message} Please ensure all required dependencies are installed."
        
        return base_message
    
    def _log_error(self, error_details: Dict[str, Any]):
        """Log error details to the application log."""
        log_message = (
            f"[{error_details['category'].upper()}] "
            f"{error_details['error_type']}: {error_details['error_message']}"
        )
        
        if error_details['context']:
            log_message += f" | Context: {error_details['context']}"
        
        logger.error(log_message)
        logger.debug(f"Full traceback: {error_details['traceback']}")
    
    def _display_user_error(self, error_details: Dict[str, Any]):
        """Display user-friendly error message in the Streamlit interface."""
        category_title = self.error_categories.get(error_details['category'], 'Error')
        
        st.error(f"**{category_title}**: {error_details['user_message']}")
        
        # Show expandable details only in debug mode
        from config_web import is_debug_mode
        if is_debug_mode():
            with st.expander("🔍 Technical Details (for debugging)"):
                st.write(f"**Error Type:** {error_details['error_type']}")
                st.write(f"**Timestamp:** {error_details['timestamp']}")
                if error_details['context']:
                    st.write(f"**Context:** {error_details['context']}")
                st.write(f"**Original Message:** {error_details['error_message']}")
                
                if st.checkbox("Show full traceback", key=f"traceback_{error_details['timestamp']}"):
                    st.code(error_details['traceback'])
    
    def _store_error_in_session(self, error_details: Dict[str, Any]):
        """Store error details in session state for later review."""
        if 'error_log' not in st.session_state:
            st.session_state.error_log = []
        
        st.session_state.error_log.append(error_details)
        
        # Keep only last 20 errors
        if len(st.session_state.error_log) > 20:
            st.session_state.error_log = st.session_state.error_log[-20:]
    
    def with_error_handling(self, 
                          category: str = 'system',
                          context: str = "",
                          show_user_message: bool = True,
                          return_on_error: Any = None):
        """
        Decorator for automatic error handling.
        
        Args:
            category: Error category for classification
            context: Additional context about the function
            show_user_message: Whether to display error message to user
            return_on_error: Value to return if an error occurs
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self.handle_error(
                        error=e,
                        category=category,
                        context=context or f"Function: {func.__name__}",
                        show_user_message=show_user_message
                    )
                    return return_on_error
            return wrapper
        return decorator
    
    def create_error_boundary(self, 
                            title: str = "Error Boundary",
                            fallback_content: Optional[Callable] = None):
        """
        Create an error boundary that catches and handles exceptions.
        
        Args:
            title: Title for the error boundary
            fallback_content: Function to call to render fallback content
        """
        class ErrorBoundary:
            def __init__(self, error_handler: 'ErrorHandler', title: str, fallback: Optional[Callable]):
                self.error_handler = error_handler
                self.title = title
                self.fallback = fallback
            
            def __enter__(self):
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is not None:
                    self.error_handler.handle_error(
                        error=exc_val,
                        context=f"Error boundary: {self.title}"
                    )
                    
                    if self.fallback:
                        try:
                            self.fallback()
                        except Exception as fallback_error:
                            st.error(f"Fallback content also failed: {str(fallback_error)}")
                    
                    return True  # Suppress the exception
                return False
        
        return ErrorBoundary(self, title, fallback_content)
    
    def validate_and_handle(self, 
                          validation_func: Callable,
                          error_message: str = "Validation failed",
                          category: str = 'data_validation') -> bool:
        """
        Validate data and handle validation errors.
        
        Args:
            validation_func: Function that returns True if validation passes
            error_message: Error message to display if validation fails
            category: Error category
            
        Returns:
            True if validation passed, False otherwise
        """
        try:
            if validation_func():
                return True
            else:
                # Create a custom validation error
                validation_error = ValueError(error_message)
                self.handle_error(
                    error=validation_error,
                    category=category,
                    context="Data validation"
                )
                return False
        except Exception as e:
            self.handle_error(
                error=e,
                category=category,
                context="Validation function execution"
            )
            return False
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of recent errors."""
        if 'error_log' not in st.session_state:
            return {'total_errors': 0, 'recent_errors': [], 'error_categories': {}}
        
        errors = st.session_state.error_log
        
        # Count errors by category
        category_counts = {}
        for error in errors:
            category = error.get('category', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Get recent errors (last 5)
        recent_errors = errors[-5:] if len(errors) > 5 else errors
        
        return {
            'total_errors': len(errors),
            'recent_errors': recent_errors,
            'error_categories': category_counts,
            'last_error_time': errors[-1]['timestamp'] if errors else None
        }
    
    def display_error_dashboard(self):
        """Display an error dashboard for debugging."""
        st.subheader("🚨 Error Dashboard")
        
        error_summary = self.get_error_summary()
        
        if error_summary['total_errors'] == 0:
            st.success("✅ No errors recorded in this session!")
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Errors", error_summary['total_errors'])
        
        with col2:
            most_common_category = max(error_summary['error_categories'], 
                                     key=error_summary['error_categories'].get)
            st.metric("Most Common Type", most_common_category)
        
        with col3:
            if error_summary['last_error_time']:
                last_error = datetime.fromisoformat(error_summary['last_error_time'])
                time_since = datetime.now() - last_error
                st.metric("Last Error", f"{time_since.seconds // 60} min ago")
        
        # Error category breakdown
        if error_summary['error_categories']:
            st.subheader("📊 Error Categories")
            category_df = pd.DataFrame([
                {'Category': cat, 'Count': count} 
                for cat, count in error_summary['error_categories'].items()
            ])
            st.bar_chart(category_df.set_index('Category'))
        
        # Recent errors table
        if error_summary['recent_errors']:
            st.subheader("🕒 Recent Errors")
            recent_df = pd.DataFrame([
                {
                    'Time': error['timestamp'][:19],  # Remove microseconds
                    'Type': error['error_type'],
                    'Category': error['category'],
                    'Message': error['error_message'][:100] + '...' if len(error['error_message']) > 100 else error['error_message']
                }
                for error in error_summary['recent_errors']
            ])
            st.dataframe(recent_df, use_container_width=True)
        
        # Clear errors button
        if st.button("🗑️ Clear Error Log"):
            st.session_state.error_log = []
            st.success("Error log cleared!")
            st.experimental_rerun()


# Global error handler instance
error_handler = ErrorHandler()


# Convenience functions
def handle_error(error: Exception, category: str = 'system', context: str = ""):
    """Convenience function for error handling."""
    return error_handler.handle_error(error, category, context)


def with_error_handling(category: str = 'system', context: str = ""):
    """Convenience decorator for error handling."""
    return error_handler.with_error_handling(category, context)