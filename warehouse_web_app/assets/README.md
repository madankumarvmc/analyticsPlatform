# Assets Directory

This directory contains static assets for the warehouse analysis web application.

## Logo Usage

### Adding Your Custom Logo

1. **Place your logo file** in this directory:
   - Supported formats: PNG, JPG, JPEG, SVG, GIF
   - Recommended size: 120x80 pixels (or proportional)
   - Recommended file name: `logo.png` or `company_logo.svg`

2. **Update the header component** to use your logo:

```python
# In app.py or any page
create_header(
    title="Your Company Name",
    subtitle="Warehouse Analysis Tool",
    logo_path="assets/logo.png",  # Path to your logo
    show_navigation=True
)
```

### Logo Requirements

- **Size**: Maximum 120px wide, 80px tall for best fit
- **Format**: PNG with transparency recommended for best quality
- **Background**: Transparent background works best with the gradient header
- **Style**: Clean, professional design that scales well

### Example Logo Integration

```python
# Using custom logo in main app
from components.header import create_header

create_header(
    title="ACME Warehousing",
    subtitle="Advanced Analytics Platform", 
    logo_path="assets/acme_logo.png",
    show_navigation=True
)

# Using custom logo in individual pages
from components.header import create_simple_header

create_simple_header(
    title="📊 Order Analysis",
    logo_path="assets/acme_logo.png"
)
```

### Default Logo

If no custom logo is provided, the application uses a built-in SVG warehouse logo that includes:
- Warehouse building icon
- Forklift and analytics symbols
- "WH ANALYTICS" text
- Professional blue color scheme

### Logo Positioning

The logo appears in the left column of the header with:
- Automatic scaling to fit the header height
- Center alignment within its container
- Drop shadow effect for visual depth
- Responsive design for mobile devices

## Adding Other Assets

You can also place other assets in this directory:
- Icons for specific features
- Background images
- Custom graphics
- Documentation images

Remember to update the `.gitignore` file if you don't want to commit certain assets to version control.