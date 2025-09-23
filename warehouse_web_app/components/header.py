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
        
        # Header content
        col1, col2, col3 = st.columns([1, 4, 1])
        
        with col1:
            # Logo section
            self._render_logo(logo_path)
        
        with col2:
            # Title section
            self._render_title(title, subtitle)
        
        with col3:
            # Navigation/actions section
            if show_navigation:
                self._render_navigation()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Header divider
        st.markdown('<div class="wh-header-divider"></div>', unsafe_allow_html=True)
    
    def _render_logo(self, logo_path: Optional[str] = None) -> None:
        """Render the logo section."""
        if logo_path and Path(logo_path).exists():
            # Use custom logo
            try:
                with open(logo_path, "rb") as f:
                    logo_data = f.read()
                    logo_base64 = base64.b64encode(logo_data).decode()
                    
                # Determine image type
                ext = Path(logo_path).suffix.lower()
                mime_type = {
                    '.png': 'image/png',
                    '.jpg': 'image/jpeg', 
                    '.jpeg': 'image/jpeg',
                    '.svg': 'image/svg+xml',
                    '.gif': 'image/gif'
                }.get(ext, 'image/png')
                
                st.markdown(f"""
                <div class="wh-logo-container">
                    <img src="data:{mime_type};base64,{logo_base64}" class="wh-logo-image" alt="Logo">
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error loading logo: {e}")
                self._render_default_logo()
        else:
            # Use default logo
            self._render_default_logo()
    
    def _render_default_logo(self) -> None:
        """Render the default SVG logo."""
        st.markdown(f"""
        <div class="wh-logo-container">
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
            .wh-header-container {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1rem 2rem;
                margin: -1rem -1rem 2rem -1rem;
                border-radius: 0 0 15px 15px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                overflow: hidden;
            }
            
            .wh-logo-container {
                display: flex;
                align-items: center;
                justify-content: center;
                height: 80px;
                overflow: hidden;
            }
            
            .wh-logo-image {
                max-height: 70px;
                max-width: 120px;
                width: auto;
                height: auto;
                filter: drop-shadow(2px 2px 4px rgba(0,0,0,0.2));
            }
            
            .wh-title-container {
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                height: 80px;
                text-align: center;
            }
            
            .wh-main-title {
                color: white;
                font-size: 2.5rem;
                font-weight: 700;
                margin: 0;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            
            .wh-subtitle {
                color: rgba(255, 255, 255, 0.9);
                font-size: 1.1rem;
                margin: 0.25rem 0 0 0;
                font-weight: 300;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
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