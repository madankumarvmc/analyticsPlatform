#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Header Component for Warehouse Analysis Web App
Displays title and navigation elements.
"""

import streamlit as st
from typing import Optional


class HeaderComponent:
    """Header component with title and navigation."""
    
    def __init__(self):
        pass
    
    def render_header(self, 
                     title: str = "Warehouse Analysis Tool",
                     subtitle: str = "Advanced Analytics for Warehouse Operations",
                     show_navigation: bool = True) -> None:
        """
        Render the application header with title and navigation.
        
        Args:
            title: Main application title
            subtitle: Subtitle/description
            show_navigation: Whether to show navigation links
        """
        # Custom CSS for header
        st.markdown(self._get_header_css(), unsafe_allow_html=True)
        
        # Header content wrapped in container
        st.markdown('<div class="wh-header-container">', unsafe_allow_html=True)
        
        # Header content - single column layout centered
        if show_navigation:
            # Layout with navigation
            col1, col2 = st.columns([4, 1])
            
            with col1:
                # Title section
                self._render_title(title, subtitle)
            
            with col2:
                # Navigation/actions section
                self._render_navigation()
        else:
            # Simple single-column layout without navigation
            # Title section - full width
            self._render_title(title, subtitle)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_title(self, title: str, subtitle: str) -> None:
        """Render the title section."""
        st.markdown(f"""
        <div class="wh-title-container">
            <h1 class="wh-main-title">{title}</h1>
            <p class="wh-subtitle">{subtitle}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_navigation(self) -> None:
        """Render the navigation section."""
        st.markdown("""
        <div class="wh-nav-container">
            <div class="wh-nav-links">
                <a href="#analysis" class="wh-nav-link">📊 Analysis</a>
                <a href="#reports" class="wh-nav-link">📋 Reports</a>
                <a href="#settings" class="wh-nav-link">⚙️ Settings</a>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _get_header_css(self) -> str:
        """Get CSS styles for the header."""
        return """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            .wh-header-container {
                background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
                padding: 2rem 2rem 1.5rem 2rem;
                margin: -1rem -1rem 2.5rem -1rem;
                border-bottom: 1px solid #e2e8f0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                overflow: hidden;
                position: relative;
            }
            
            .wh-header-container::before {
                content: '';
                position: absolute;
                bottom: 0;
                left: 0;
                right: 0;
                height: 2px;
                background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 50%, #06b6d4 100%);
            }
            
            .wh-title-container {
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                height: 60px;
                text-align: center;
            }
            
            .wh-main-title {
                color: #1e293b;
                font-size: 2.2rem;
                font-weight: 600;
                margin: 0;
                font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                letter-spacing: -0.5px;
                line-height: 1.2;
            }
            
            .wh-subtitle {
                color: #64748b;
                font-size: 0.95rem;
                margin: 0.3rem 0 0 0;
                font-weight: 400;
                font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.4;
            }
            
            .wh-nav-container {
                display: flex;
                align-items: center;
                justify-content: center;
                height: 80px;
            }
            
            .wh-nav-links {
                display: flex;
                flex-direction: column;
                gap: 0.5rem;
            }
            
            .wh-nav-link {
                color: rgba(255, 255, 255, 0.9);
                text-decoration: none;
                font-size: 0.9rem;
                padding: 0.25rem 0.5rem;
                border-radius: 5px;
                transition: all 0.3s ease;
                text-align: center;
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
            }
            
            .wh-nav-link:hover {
                color: white;
                background: rgba(255, 255, 255, 0.2);
                text-decoration: none;
                transform: translateY(-1px);
            }
            
            .wh-header-divider {
                height: 3px;
                background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
                margin: 1rem 0 2rem 0;
                border-radius: 2px;
            }
            
            /* Responsive design */
            @media (max-width: 768px) {
                .wh-header-container {
                    padding: 1.5rem 1rem;
                }
                
                .wh-main-title {
                    font-size: 1.8rem;
                }
                
                .wh-subtitle {
                    font-size: 0.85rem;
                }
                
                .wh-nav-links {
                    flex-direction: row;
                    gap: 0.25rem;
                }
                .wh-nav-link {
                    font-size: 0.8rem;
                    padding: 0.2rem 0.3rem;
                }
            }
            
            @media (max-width: 480px) {
                .wh-main-title {
                    font-size: 1.5rem;
                }
                
                .wh-subtitle {
                    font-size: 0.8rem;
                }
            }
        </style>
        """


def create_header(title: str = "Warehouse Analysis Tool",
                 subtitle: str = "Advanced Analytics for Warehouse Operations", 
                 show_navigation: bool = True) -> None:
    """
    Create and render the application header.
    
    Args:
        title: Main application title
        subtitle: Subtitle/description
        show_navigation: Whether to show navigation links
    """
    header = HeaderComponent()
    header.render_header(title, subtitle, show_navigation)


def create_simple_header(title: str) -> None:
    """
    Create a simple header with just title.
    
    Args:
        title: Page title
    """
    header = HeaderComponent()
    header.render_header(title, "", show_navigation=False)