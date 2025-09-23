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
    
    def render_header(self, 
                     title: str = "Warehouse Analysis Tool",
                     subtitle: str = "Advanced Analytics for Warehouse Operations",
                     logo_path: Optional[str] = None,
                     show_navigation: bool = True) -> None:
        """
        Render the application header with logo and title.
        
        Args:
            title: Main application title
            subtitle: Subtitle/description
            logo_path: Path to logo image file (optional)
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
                self._render_logo(logo_path)
            
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
                self._render_logo(logo_path)
            
            with col2:
                # Title section
                self._render_title(title, subtitle)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_logo(self, logo_path: Optional[str] = None) -> None:
        """Render the logo section using Streamlit's native image handling."""
        # Create a container with custom CSS styling
        st.markdown('<div class="wh-logo-container">', unsafe_allow_html=True)
        
        if logo_path:
            try:
                # Use st.image() which works reliably on both local and Streamlit Cloud
                st.image(
                    logo_path,
                    width=None,  # Let CSS control the size
                    use_container_width=False,
                    output_format='auto'
                )
            except Exception as e:
                # Fallback to default logo on any error
                self._render_default_logo_content()
        else:
            # Use default logo when no path provided
            self._render_default_logo_content()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_default_logo_content(self) -> None:
        """Render just the default SVG logo content."""
        st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: flex-start; height: 60px;">
            {self.default_logo_svg}
        </div>
        """, unsafe_allow_html=True)
    
    def _render_default_logo(self) -> None:
        """Render the default SVG logo."""
        st.markdown('<div class="wh-logo-container">', unsafe_allow_html=True)
        self._render_default_logo_content()
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
                 logo_path: Optional[str] = None,
                 show_navigation: bool = True) -> None:
    """
    Create and render the application header.
    
    Args:
        title: Main application title
        subtitle: Subtitle/description
        logo_path: Path to custom logo image file
        show_navigation: Whether to show navigation links
    """
    header = HeaderComponent()
    header.render_header(title, subtitle, logo_path, show_navigation)


def create_simple_header(title: str, logo_path: Optional[str] = None) -> None:
    """
    Create a simple header with just title and logo.
    
    Args:
        title: Page title
        logo_path: Path to logo image file
    """
    header = HeaderComponent()
    header.render_header(title, "", logo_path, show_navigation=False)