#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Header Component for Warehouse Analysis Web App
Displays logo, title, and navigation elements.
"""

import streamlit as st
import base64
from pathlib import Path
from typing import Optional


class HeaderComponent:
    """Header component with logo and title."""
    
    def __init__(self):
        self.logo_path = None
        self.default_logo_svg = self._get_default_logo_svg()
        # Base64 encoded SBX logos for reliable deployment
        self.SBX_LOGO1_BASE64 = "data:image/png;base64,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

... [1 lines truncated] ...
    
    def render_header(self, 
                     title: str = "Warehouse Analysis Tool",
                     subtitle: str = "Advanced Analytics for Warehouse Operations",
                     show_navigation: bool = True) -> None:
        """
        Render the application header with logo and title.
        
        Args:
            title: Main application title
            subtitle: Subtitle/description
            show_navigation: Whether to show navigation links
        """
        # Custom CSS for header
        st.markdown(self._get_header_css(), unsafe_allow_html=True)
        
        # Header content wrapped in container
        st.markdown('<div class="wh-header-container">', unsafe_allow_html=True)
        
        # Header content - minimalistic layout
        if show_navigation:
            # Original 3-column layout when navigation is needed
            col1, col2, col3 = st.columns([1, 4, 1])
            
            with col1:
                # Logo section
                self._render_logo()
            
            with col2:
                # Title section
                self._render_title(title, subtitle)
            
            with col3:
                # Navigation/actions section
                self._render_navigation()
        else:
            # Minimalistic 2-column layout without navigation
            col1, col2 = st.columns([1, 4])
            
            with col1:
                # Logo section
                self._render_logo()
            
            with col2:
                # Title section
                self._render_title(title, subtitle)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_logo(self) -> None:
        """Render the logo section using base64 encoded logos for reliable deployment."""
        # Create a container with custom CSS styling
        st.markdown('<div class="wh-logo-container">', unsafe_allow_html=True)
        
        # Use base64 encoded SBX logo for reliable deployment
        try:
            st.image(
                self.SBX_LOGO1_BASE64,
                width=None,  # Let CSS control the size
                use_container_width=False,
                output_format='auto'
            )
        except Exception as e:
            # Fallback to default logo on any error
            self._render_default_logo_content()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_default_logo_content(self) -> None:
        """Render just the default SVG logo content."""
        st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: flex-start; height: 60px;">
            {self.default_logo_svg}
        </div>
        """, unsafe_allow_html=True)
    
    
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
    
    def _get_default_logo_svg(self) -> str:
        """Generate a default warehouse-themed SVG logo."""
        return """<svg class="wh-logo-image" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
            <rect x="20" y="60" width="80" height="40" fill="#4A90E2" stroke="#2C5985" stroke-width="2"/>
            <polygon points="10,60 60,30 110,60" fill="#2C5985" stroke="#1A3A5C" stroke-width="2"/>
            <rect x="50" y="75" width="20" height="25" fill="#E8F4FD" stroke="#4A90E2" stroke-width="1"/>
            <rect x="30" y="70" width="12" height="8" fill="#E8F4FD" stroke="#4A90E2" stroke-width="1"/>
            <rect x="78" y="70" width="12" height="8" fill="#E8F4FD" stroke="#4A90E2" stroke-width="1"/>
            <rect x="15" y="85" width="15" height="15" fill="#94C9F0" stroke="#4A90E2" stroke-width="1"/>
            <rect x="90" y="85" width="15" height="15" fill="#94C9F0" stroke="#4A90E2" stroke-width="1"/>
            <g transform="translate(25, 105)">
                <rect x="0" y="0" width="4" height="8" fill="#FF6B35"/>
                <rect x="0" y="8" width="12" height="4" fill="#FF6B35"/>
                <circle cx="2" r="2" fill="#333"/>
                <circle cx="10" r="2" fill="#333"/>
                <rect x="4" y="2" width="1" height="6" fill="#FF6B35"/>
            </g>
            <g transform="translate(75, 105)">
                <rect x="0" y="6" width="2" height="6" fill="#28A745"/>
                <rect x="3" y="4" width="2" height="8" fill="#28A745"/>
                <rect x="6" y="2" width="2" height="10" fill="#28A745"/>
                <rect x="9" y="0" width="2" height="12" fill="#28A745"/>
            </g>
            <text x="60" y="115" text-anchor="middle" font-family="Arial, sans-serif" font-size="8" font-weight="bold" fill="#2C5985">WH ANALYTICS</text>
        </svg>"""
    
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
            
            .wh-logo-container {
                display: flex;
                align-items: center;
                justify-content: flex-start;
                height: 60px;
                overflow: hidden;
            }
            
            /* Style Streamlit's generated img elements within logo container */
            .wh-logo-container img {
                max-height: 50px !important;
                max-width: 100px !important;
                width: auto !important;
                height: auto !important;
                filter: drop-shadow(0px 1px 3px rgba(0,0,0,0.1));
                border-radius: 4px;
                object-fit: contain;
            }
            
            /* Streamlit generates a div wrapper for images - style it too */
            .wh-logo-container > div {
                display: flex !important;
                align-items: center !important;
                justify-content: flex-start !important;
            }
            
            /* Legacy support for old image class */
            .wh-logo-image {
                max-height: 50px;
                max-width: 100px;
                width: auto;
                height: auto;
                filter: drop-shadow(0px 1px 3px rgba(0,0,0,0.1));
                border-radius: 4px;
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
                
                .wh-logo-image {
                    max-height: 40px;
                    max-width: 80px;
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
                .wh-main-title {
                    font-size: 1.8rem;
                }
                .wh-subtitle {
                    font-size: 0.9rem;
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
    Create a simple header with just title and logo.
    
    Args:
        title: Page title
    """
    header = HeaderComponent()
    header.render_header(title, "", show_navigation=False)