# ═══════════════════════════════════════════════════════════════════════════════
#                           QUANTTS-CORE ICONS LIBRARY
#                    Professional Material Design Icons for Quant Finance
# ═══════════════════════════════════════════════════════════════════════════════

"""
Professional SVG icon library for QuantTS-Core Dashboard.
Uses Material Design and custom finance-specific icons.
All icons are inline SVG for maximum compatibility with Streamlit.
"""

# Icon sizes
ICON_SM = 16
ICON_MD = 20
ICON_LG = 24
ICON_XL = 32

# Theme colors
COLORS = {
    'primary': '#00D4AA',
    'secondary': '#667eea',
    'success': '#00D4AA',
    'danger': '#ff6b6b',
    'warning': '#feca57',
    'info': '#54a0ff',
    'muted': '#a0aec0',
    'white': '#ffffff',
}


def _svg_wrapper(svg_content: str, size: int = ICON_MD, color: str = None, 
                 style: str = "", css_class: str = "") -> str:
    """Wrap SVG content with proper attributes."""
    fill_color = color or COLORS['white']
    return f'''<svg xmlns="http://www.w3.org/2000/svg" 
                    width="{size}" height="{size}" 
                    viewBox="0 0 24 24" 
                    fill="none" 
                    stroke="{fill_color}" 
                    stroke-width="2" 
                    stroke-linecap="round" 
                    stroke-linejoin="round"
                    style="display: inline-block; vertical-align: middle; {style}"
                    class="{css_class}">
                {svg_content}
            </svg>'''


def _svg_filled(svg_content: str, size: int = ICON_MD, color: str = None,
                style: str = "", css_class: str = "") -> str:
    """Wrap filled SVG content."""
    fill_color = color or COLORS['white']
    return f'''<svg xmlns="http://www.w3.org/2000/svg" 
                    width="{size}" height="{size}" 
                    viewBox="0 0 24 24" 
                    fill="{fill_color}"
                    style="display: inline-block; vertical-align: middle; {style}"
                    class="{css_class}">
                {svg_content}
            </svg>'''


# ═══════════════════════════════════════════════════════════════════════════════
#                              CHART & DATA ICONS
# ═══════════════════════════════════════════════════════════════════════════════

def icon_trending_up(size: int = ICON_MD, color: str = None) -> str:
    """Trending up arrow - for positive returns."""
    return _svg_wrapper(
        '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline>'
        '<polyline points="17 6 23 6 23 12"></polyline>',
        size, color or COLORS['success']
    )


def icon_trending_down(size: int = ICON_MD, color: str = None) -> str:
    """Trending down arrow - for negative returns."""
    return _svg_wrapper(
        '<polyline points="23 18 13.5 8.5 8.5 13.5 1 6"></polyline>'
        '<polyline points="17 18 23 18 23 12"></polyline>',
        size, color or COLORS['danger']
    )


def icon_bar_chart(size: int = ICON_MD, color: str = None) -> str:
    """Bar chart icon."""
    return _svg_wrapper(
        '<line x1="12" y1="20" x2="12" y2="10"></line>'
        '<line x1="18" y1="20" x2="18" y2="4"></line>'
        '<line x1="6" y1="20" x2="6" y2="16"></line>',
        size, color
    )


def icon_line_chart(size: int = ICON_MD, color: str = None) -> str:
    """Line chart icon - for time series."""
    return _svg_wrapper(
        '<path d="M3 3v18h18"></path>'
        '<path d="m19 9-5 5-4-4-3 3"></path>',
        size, color or COLORS['primary']
    )


def icon_candlestick(size: int = ICON_MD, color: str = None) -> str:
    """Candlestick chart icon - for OHLC data."""
    return _svg_wrapper(
        '<path d="M9 5v4"></path>'
        '<rect x="7" y="9" width="4" height="6" rx="1"></rect>'
        '<path d="M9 15v4"></path>'
        '<path d="M17 3v4"></path>'
        '<rect x="15" y="7" width="4" height="8" rx="1"></rect>'
        '<path d="M17 15v6"></path>',
        size, color
    )


def icon_pie_chart(size: int = ICON_MD, color: str = None) -> str:
    """Pie chart icon - for allocation/distribution."""
    return _svg_wrapper(
        '<path d="M21.21 15.89A10 10 0 1 1 8 2.83"></path>'
        '<path d="M22 12A10 10 0 0 0 12 2v10z"></path>',
        size, color
    )


def icon_scatter(size: int = ICON_MD, color: str = None) -> str:
    """Scatter plot icon."""
    return _svg_wrapper(
        '<circle cx="7" cy="14" r="2"></circle>'
        '<circle cx="11" cy="6" r="2"></circle>'
        '<circle cx="16" cy="16" r="2"></circle>'
        '<circle cx="19" cy="9" r="2"></circle>'
        '<circle cx="13" cy="12" r="2"></circle>',
        size, color
    )


# ═══════════════════════════════════════════════════════════════════════════════
#                              FINANCE ICONS
# ═══════════════════════════════════════════════════════════════════════════════

def icon_dollar(size: int = ICON_MD, color: str = None) -> str:
    """Dollar sign - for monetary values."""
    return _svg_wrapper(
        '<line x1="12" y1="1" x2="12" y2="23"></line>'
        '<path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path>',
        size, color or COLORS['success']
    )


def icon_wallet(size: int = ICON_MD, color: str = None) -> str:
    """Wallet icon - for portfolio."""
    return _svg_wrapper(
        '<path d="M21 12V7H5a2 2 0 0 1 0-4h14v4"></path>'
        '<path d="M3 5v14a2 2 0 0 0 2 2h16v-5"></path>'
        '<path d="M18 12a2 2 0 0 0 0 4h4v-4Z"></path>',
        size, color
    )


def icon_briefcase(size: int = ICON_MD, color: str = None) -> str:
    """Briefcase - for portfolio/assets."""
    return _svg_wrapper(
        '<rect x="2" y="7" width="20" height="14" rx="2" ry="2"></rect>'
        '<path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"></path>',
        size, color
    )


def icon_bank(size: int = ICON_MD, color: str = None) -> str:
    """Bank icon - for financial institutions."""
    return _svg_wrapper(
        '<line x1="3" y1="22" x2="21" y2="22"></line>'
        '<line x1="6" y1="18" x2="6" y2="11"></line>'
        '<line x1="10" y1="18" x2="10" y2="11"></line>'
        '<line x1="14" y1="18" x2="14" y2="11"></line>'
        '<line x1="18" y1="18" x2="18" y2="11"></line>'
        '<polygon points="12 2 20 7 4 7"></polygon>',
        size, color
    )


def icon_percent(size: int = ICON_MD, color: str = None) -> str:
    """Percent icon - for returns/rates."""
    return _svg_wrapper(
        '<line x1="19" y1="5" x2="5" y2="19"></line>'
        '<circle cx="6.5" cy="6.5" r="2.5"></circle>'
        '<circle cx="17.5" cy="17.5" r="2.5"></circle>',
        size, color
    )


# ═══════════════════════════════════════════════════════════════════════════════
#                              RISK & ANALYSIS ICONS
# ═══════════════════════════════════════════════════════════════════════════════

def icon_alert_triangle(size: int = ICON_MD, color: str = None) -> str:
    """Alert triangle - for warnings/risk."""
    return _svg_wrapper(
        '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path>'
        '<line x1="12" y1="9" x2="12" y2="13"></line>'
        '<line x1="12" y1="17" x2="12.01" y2="17"></line>',
        size, color or COLORS['warning']
    )


def icon_shield(size: int = ICON_MD, color: str = None) -> str:
    """Shield - for risk protection/VaR."""
    return _svg_wrapper(
        '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>',
        size, color
    )


def icon_shield_alert(size: int = ICON_MD, color: str = None) -> str:
    """Shield with alert - for risk warning."""
    return _svg_wrapper(
        '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>'
        '<line x1="12" y1="8" x2="12" y2="12"></line>'
        '<line x1="12" y1="16" x2="12.01" y2="16"></line>',
        size, color or COLORS['danger']
    )


def icon_activity(size: int = ICON_MD, color: str = None) -> str:
    """Activity/heartbeat - for volatility."""
    return _svg_wrapper(
        '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>',
        size, color or COLORS['white']
    )


def icon_zap(size: int = ICON_MD, color: str = None) -> str:
    """Lightning bolt - for momentum/signals."""
    return _svg_wrapper(
        '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>',
        size, color or COLORS['warning']
    )


def icon_target(size: int = ICON_MD, color: str = None) -> str:
    """Target - for objectives/benchmarks."""
    return _svg_wrapper(
        '<circle cx="12" cy="12" r="10"></circle>'
        '<circle cx="12" cy="12" r="6"></circle>'
        '<circle cx="12" cy="12" r="2"></circle>',
        size, color
    )


def icon_crosshair(size: int = ICON_MD, color: str = None) -> str:
    """Crosshair - for precision analysis."""
    return _svg_wrapper(
        '<circle cx="12" cy="12" r="10"></circle>'
        '<line x1="22" y1="12" x2="18" y2="12"></line>'
        '<line x1="6" y1="12" x2="2" y2="12"></line>'
        '<line x1="12" y1="6" x2="12" y2="2"></line>'
        '<line x1="12" y1="22" x2="12" y2="18"></line>',
        size, color
    )


# ═══════════════════════════════════════════════════════════════════════════════
#                              NAVIGATION & UI ICONS
# ═══════════════════════════════════════════════════════════════════════════════

def icon_home(size: int = ICON_MD, color: str = None) -> str:
    """Home icon."""
    return _svg_wrapper(
        '<path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>'
        '<polyline points="9 22 9 12 15 12 15 22"></polyline>',
        size, color
    )


def icon_compass(size: int = ICON_MD, color: str = None) -> str:
    """Compass - for navigation."""
    return _svg_wrapper(
        '<circle cx="12" cy="12" r="10"></circle>'
        '<polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"></polygon>',
        size, color
    )


def icon_layers(size: int = ICON_MD, color: str = None) -> str:
    """Layers - for stacked views."""
    return _svg_wrapper(
        '<polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>'
        '<polyline points="2 17 12 22 22 17"></polyline>'
        '<polyline points="2 12 12 17 22 12"></polyline>',
        size, color
    )


def icon_grid(size: int = ICON_MD, color: str = None) -> str:
    """Grid - for overview/dashboard."""
    return _svg_wrapper(
        '<rect x="3" y="3" width="7" height="7"></rect>'
        '<rect x="14" y="3" width="7" height="7"></rect>'
        '<rect x="14" y="14" width="7" height="7"></rect>'
        '<rect x="3" y="14" width="7" height="7"></rect>',
        size, color
    )


def icon_sliders(size: int = ICON_MD, color: str = None) -> str:
    """Sliders - for settings/parameters."""
    return _svg_wrapper(
        '<line x1="4" y1="21" x2="4" y2="14"></line>'
        '<line x1="4" y1="10" x2="4" y2="3"></line>'
        '<line x1="12" y1="21" x2="12" y2="12"></line>'
        '<line x1="12" y1="8" x2="12" y2="3"></line>'
        '<line x1="20" y1="21" x2="20" y2="16"></line>'
        '<line x1="20" y1="12" x2="20" y2="3"></line>'
        '<line x1="1" y1="14" x2="7" y2="14"></line>'
        '<line x1="9" y1="8" x2="15" y2="8"></line>'
        '<line x1="17" y1="16" x2="23" y2="16"></line>',
        size, color
    )


def icon_settings(size: int = ICON_MD, color: str = None) -> str:
    """Settings gear icon."""
    return _svg_wrapper(
        '<circle cx="12" cy="12" r="3"></circle>'
        '<path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>',
        size, color
    )


# ═══════════════════════════════════════════════════════════════════════════════
#                              STATUS & INDICATOR ICONS
# ═══════════════════════════════════════════════════════════════════════════════

def icon_check_circle(size: int = ICON_MD, color: str = None) -> str:
    """Check in circle - for success."""
    return _svg_wrapper(
        '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>'
        '<polyline points="22 4 12 14.01 9 11.01"></polyline>',
        size, color or COLORS['success']
    )


def icon_x_circle(size: int = ICON_MD, color: str = None) -> str:
    """X in circle - for error/failure."""
    return _svg_wrapper(
        '<circle cx="12" cy="12" r="10"></circle>'
        '<line x1="15" y1="9" x2="9" y2="15"></line>'
        '<line x1="9" y1="9" x2="15" y2="15"></line>',
        size, color or COLORS['danger']
    )


def icon_info(size: int = ICON_MD, color: str = None) -> str:
    """Info icon."""
    return _svg_wrapper(
        '<circle cx="12" cy="12" r="10"></circle>'
        '<line x1="12" y1="16" x2="12" y2="12"></line>'
        '<line x1="12" y1="8" x2="12.01" y2="8"></line>',
        size, color or COLORS['info']
    )


def icon_arrow_up(size: int = ICON_MD, color: str = None) -> str:
    """Arrow up."""
    return _svg_wrapper(
        '<line x1="12" y1="19" x2="12" y2="5"></line>'
        '<polyline points="5 12 12 5 19 12"></polyline>',
        size, color or COLORS['success']
    )


def icon_arrow_down(size: int = ICON_MD, color: str = None) -> str:
    """Arrow down."""
    return _svg_wrapper(
        '<line x1="12" y1="5" x2="12" y2="19"></line>'
        '<polyline points="19 12 12 19 5 12"></polyline>',
        size, color or COLORS['danger']
    )


def icon_minus(size: int = ICON_MD, color: str = None) -> str:
    """Minus/neutral."""
    return _svg_wrapper(
        '<line x1="5" y1="12" x2="19" y2="12"></line>',
        size, color or COLORS['muted']
    )


# ═══════════════════════════════════════════════════════════════════════════════
#                              DATA & FILE ICONS
# ═══════════════════════════════════════════════════════════════════════════════

def icon_database(size: int = ICON_MD, color: str = None) -> str:
    """Database icon."""
    return _svg_wrapper(
        '<ellipse cx="12" cy="5" rx="9" ry="3"></ellipse>'
        '<path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path>'
        '<path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path>',
        size, color
    )


def icon_folder(size: int = ICON_MD, color: str = None) -> str:
    """Folder icon."""
    return _svg_wrapper(
        '<path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path>',
        size, color
    )


def icon_file_text(size: int = ICON_MD, color: str = None) -> str:
    """File with text."""
    return _svg_wrapper(
        '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>'
        '<polyline points="14 2 14 8 20 8"></polyline>'
        '<line x1="16" y1="13" x2="8" y2="13"></line>'
        '<line x1="16" y1="17" x2="8" y2="17"></line>'
        '<polyline points="10 9 9 9 8 9"></polyline>',
        size, color
    )


def icon_download(size: int = ICON_MD, color: str = None) -> str:
    """Download icon."""
    return _svg_wrapper(
        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>'
        '<polyline points="7 10 12 15 17 10"></polyline>'
        '<line x1="12" y1="15" x2="12" y2="3"></line>',
        size, color
    )


def icon_upload(size: int = ICON_MD, color: str = None) -> str:
    """Upload icon."""
    return _svg_wrapper(
        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>'
        '<polyline points="17 8 12 3 7 8"></polyline>'
        '<line x1="12" y1="3" x2="12" y2="15"></line>',
        size, color
    )


# ═══════════════════════════════════════════════════════════════════════════════
#                              TIME & CALENDAR ICONS
# ═══════════════════════════════════════════════════════════════════════════════

def icon_calendar(size: int = ICON_MD, color: str = None) -> str:
    """Calendar icon."""
    return _svg_wrapper(
        '<rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>'
        '<line x1="16" y1="2" x2="16" y2="6"></line>'
        '<line x1="8" y1="2" x2="8" y2="6"></line>'
        '<line x1="3" y1="10" x2="21" y2="10"></line>',
        size, color
    )


def icon_clock(size: int = ICON_MD, color: str = None) -> str:
    """Clock icon."""
    return _svg_wrapper(
        '<circle cx="12" cy="12" r="10"></circle>'
        '<polyline points="12 6 12 12 16 14"></polyline>',
        size, color
    )


def icon_refresh(size: int = ICON_MD, color: str = None) -> str:
    """Refresh icon."""
    return _svg_wrapper(
        '<polyline points="23 4 23 10 17 10"></polyline>'
        '<polyline points="1 20 1 14 7 14"></polyline>'
        '<path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path>',
        size, color
    )


# ═══════════════════════════════════════════════════════════════════════════════
#                              MATH & STATISTICS ICONS
# ═══════════════════════════════════════════════════════════════════════════════

def icon_sigma(size: int = ICON_MD, color: str = None) -> str:
    """Sigma - for statistics/sum."""
    return _svg_wrapper(
        '<path d="M18 4H6l6 6-6 6h12"></path>',
        size, color
    )


def icon_function(size: int = ICON_MD, color: str = None) -> str:
    """Function icon - for formulas."""
    return _svg_wrapper(
        '<path d="M9 17c2 0 2.8-1 2.8-2.8V10c0-2 1-3.3 3.2-3"></path>'
        '<path d="M9 11.2h5"></path>',
        size, color
    )


def icon_hash(size: int = ICON_MD, color: str = None) -> str:
    """Hash/number sign."""
    return _svg_wrapper(
        '<line x1="4" y1="9" x2="20" y2="9"></line>'
        '<line x1="4" y1="15" x2="20" y2="15"></line>'
        '<line x1="10" y1="3" x2="8" y2="21"></line>'
        '<line x1="16" y1="3" x2="14" y2="21"></line>',
        size, color
    )


def icon_divide(size: int = ICON_MD, color: str = None) -> str:
    """Divide sign."""
    return _svg_wrapper(
        '<circle cx="12" cy="6" r="2"></circle>'
        '<line x1="5" y1="12" x2="19" y2="12"></line>'
        '<circle cx="12" cy="18" r="2"></circle>',
        size, color
    )


# ═══════════════════════════════════════════════════════════════════════════════
#                              LINK & CORRELATION ICONS
# ═══════════════════════════════════════════════════════════════════════════════

def icon_link(size: int = ICON_MD, color: str = None) -> str:
    """Link icon - for correlation/dependency."""
    return _svg_wrapper(
        '<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path>'
        ' <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path>',
        size, color or COLORS['white']
    )


def icon_git_branch(size: int = ICON_MD, color: str = None) -> str:
    """Branch icon - for dependency tree."""
    return _svg_wrapper(
        '<line x1="6" y1="3" x2="6" y2="15"></line>'
        '<circle cx="18" cy="6" r="3"></circle>'
        '<circle cx="6" cy="18" r="3"></circle>'
        '<path d="M18 9a9 9 0 0 1-9 9"></path>',
        size, color
    )


def icon_git_merge(size: int = ICON_MD, color: str = None) -> str:
    """Merge icon."""
    return _svg_wrapper(
        '<circle cx="18" cy="18" r="3"></circle>'
        '<circle cx="6" cy="6" r="3"></circle>'
        '<path d="M6 21V9a9 9 0 0 0 9 9"></path>',
        size, color
    )


# ═══════════════════════════════════════════════════════════════════════════════
#                              BALANCE & STABILITY ICONS
# ═══════════════════════════════════════════════════════════════════════════════

def icon_scale(size: int = ICON_MD, color: str = None) -> str:
    """Scale/balance icon - for stability."""
    return _svg_wrapper(
        '<path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"></path>'
        '<path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"></path>'
        '<path d="M7 21h10"></path>'
        '<path d="M12 3v18"></path>'
        '<path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"></path>',
        size, color
    )


def icon_anchor(size: int = ICON_MD, color: str = None) -> str:
    """Anchor - for stability."""
    return _svg_wrapper(
        '<circle cx="12" cy="5" r="3"></circle>'
        '<line x1="12" y1="22" x2="12" y2="8"></line>'
        '<path d="M5 12H2a10 10 0 0 0 20 0h-3"></path>',
        size, color
    )


def icon_lock(size: int = ICON_MD, color: str = None) -> str:
    """Lock - for security/stability."""
    return _svg_wrapper(
        '<rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>'
        '<path d="M7 11V7a5 5 0 0 1 10 0v4"></path>',
        size, color
    )


# ═══════════════════════════════════════════════════════════════════════════════
#                              HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def icon_with_label(icon_html: str, label: str, gap: str = "8px", 
                    label_style: str = "") -> str:
    """Combine icon with a text label."""
    return f'''<span style="display: inline-flex; align-items: center; gap: {gap};">
        {icon_html}
        <span style="{label_style}">{label}</span>
    </span>'''


def icon_badge(icon_html: str, background: str = None, padding: str = "8px",
               border_radius: str = "8px") -> str:
    """Wrap icon in a badge container."""
    bg = background or "rgba(0, 212, 170, 0.15)"
    return f'''<span style="display: inline-flex; align-items: center; 
                          justify-content: center; background: {bg}; 
                          padding: {padding}; border-radius: {border_radius};">
        {icon_html}
    </span>'''


def delta_icon(value: float, size: int = ICON_SM) -> str:
    """Return appropriate icon based on value sign."""
    if value > 0:
        return icon_arrow_up(size, COLORS['success'])
    elif value < 0:
        return icon_arrow_down(size, COLORS['danger'])
    else:
        return icon_minus(size, COLORS['muted'])


def status_icon(status: str, size: int = ICON_MD) -> str:
    """Return appropriate icon based on status."""
    status_map = {
        'success': icon_check_circle,
        'error': icon_x_circle,
        'warning': icon_alert_triangle,
        'info': icon_info,
    }
    return status_map.get(status, icon_info)(size)


# ═══════════════════════════════════════════════════════════════════════════════
#                              ICON MAPPING FOR PAGES
# ═══════════════════════════════════════════════════════════════════════════════

# Pre-configured icons for common dashboard elements
ICONS = {
    # Navigation
    'overview': icon_grid,
    'returns': icon_trending_up,
    'volatility': icon_activity,
    'dependency': icon_link,
    'stability': icon_scale,
    
    # Metrics
    'price': icon_dollar,
    'return': icon_percent,
    'risk': icon_shield_alert,
    'sharpe': icon_target,
    'drawdown': icon_trending_down,
    
    # Charts
    'chart': icon_line_chart,
    'candlestick': icon_candlestick,
    'distribution': icon_bar_chart,
    'scatter': icon_scatter,
    'pie': icon_pie_chart,
    
    # Status
    'success': icon_check_circle,
    'error': icon_x_circle,
    'warning': icon_alert_triangle,
    'info': icon_info,
    
    # Data
    'data': icon_database,
    'calendar': icon_calendar,
    'clock': icon_clock,
    'folder': icon_folder,
    
    # Actions
    'download': icon_download,
    'upload': icon_upload,
    'refresh': icon_refresh,
    'settings': icon_settings,
}


def get_icon(name: str, size: int = ICON_MD, color: str = None) -> str:
    """Get an icon by name."""
    icon_func = ICONS.get(name.lower())
    if icon_func:
        return icon_func(size, color)
    return ""
