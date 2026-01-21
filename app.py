import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add components to path
sys.path.insert(0, str(Path(__file__).parent))
from components.icons import (
    icon_line_chart, icon_grid, icon_trending_up, icon_activity,
    icon_link, icon_scale, icon_folder, icon_calendar, icon_clock,
    icon_check_circle, icon_x_circle, icon_database, COLORS, ICON_LG, ICON_MD
)

# ═══════════════════════════════════════════════════════════════════════════════
#                           QUANTTS-CORE DASHBOARD
#                    Professional Quantitative Finance Analytics
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="QuantTS-Core | Quant Analytics",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS for Professional Quant Dashboard Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main theme colors - Dark mode quant finance style */
    :root {
        --primary-color: #00D4AA;
        --secondary-color: #667eea;
        --background-dark: #0e1117;
        --card-bg: #1a1f2e;
        --text-primary: #ffffff;
        --text-secondary: #a0aec0;
        --positive: #00D4AA;
        --negative: #ff6b6b;
        --warning: #feca57;
    }
    
    /* Hide default Streamlit branding and auto-generated page navigation */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Hide the default multipage navigation in sidebar */
    [data-testid="stSidebarNav"] {display: none !important;}
    section[data-testid="stSidebar"] > div > div:first-child > div:first-child {display: none !important;}
    
    /* Hide any auto-generated page links */
    [data-testid="stSidebar"] ul {display: none !important;}
    [data-testid="stSidebar"] nav {display: none !important;}
    
    /* Sidebar styling - Enhanced visibility */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e2538 0%, #141824 100%);
        border-right: 1px solid #3d4760;
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #00D4AA !important;
        font-weight: 700;
        letter-spacing: 1px;
    }
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: #e2e8f0 !important;
    }
    
    [data-testid="stSidebar"] .stRadio label {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stSidebar"] .stRadio label span {
        color: #ffffff !important;
    }
    
    /* Radio buttons in sidebar - High contrast */
    [data-testid="stSidebar"] .stRadio > div {
        background: transparent;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label {
        background: rgba(45, 55, 72, 0.6) !important;
        padding: 12px 16px !important;
        border-radius: 8px !important;
        border: 1px solid #4a5568 !important;
        margin: 4px 0 !important;
        color: #ffffff !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background: rgba(0, 212, 170, 0.15) !important;
        border-color: #00D4AA !important;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label[data-checked="true"],
    [data-testid="stSidebar"] .stRadio > div > label[aria-checked="true"] {
        background: rgba(0, 212, 170, 0.2) !important;
        border-color: #00D4AA !important;
        border-left: 3px solid #00D4AA !important;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #2d3748 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2d3748;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 212, 170, 0.15);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    
    .metric-delta-positive {
        color: #00D4AA;
        font-size: 0.9rem;
    }
    
    .metric-delta-negative {
        color: #ff6b6b;
        font-size: 0.9rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #ffffff;
        border-bottom: 2px solid #00D4AA;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    /* KPI Grid */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }
    
    /* Stat box */
    .stat-box {
        background: #1a1f2e;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        border: 1px solid #2d3748;
    }
    
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00D4AA;
    }
    
    .stat-label {
        font-size: 0.75rem;
        color: #a0aec0;
        text-transform: uppercase;
    }
    
    /* Risk indicator */
    .risk-low { color: #00D4AA; }
    .risk-medium { color: #feca57; }
    .risk-high { color: #ff6b6b; }
    
    /* Table styling */
    .dataframe {
        border-collapse: collapse;
        width: 100%;
    }
    
    .dataframe th {
        background: #2d3748 !important;
        color: #ffffff !important;
        padding: 12px !important;
        text-align: left !important;
        font-weight: 600 !important;
    }
    
    .dataframe td {
        padding: 10px !important;
        border-bottom: 1px solid #2d3748 !important;
    }
    
    /* Plotly chart container */
    .stPlotlyChart {
        background: #1a1f2e;
        border-radius: 12px;
        padding: 10px;
        border: 1px solid #2d3748;
    }
    
    /* Radio buttons styling - Main area */
    .stRadio > div {
        gap: 8px;
    }
    
    .stRadio > div > label {
        background: #1a1f2e;
        padding: 8px 16px;
        border-radius: 6px;
        border: 1px solid #2d3748;
        transition: all 0.2s ease;
        color: #ffffff !important;
    }
    
    .stRadio > div > label:hover {
        border-color: #00D4AA;
    }
    
    /* Logo and branding */
    .brand-container {
        text-align: center;
        padding: 25px 0;
        margin-bottom: 20px;
        background: rgba(0, 212, 170, 0.05);
        border-radius: 12px;
        border: 1px solid rgba(0, 212, 170, 0.2);
    }
    
    .brand-logo {
        font-size: 1.8rem;
        font-weight: 800;
        color: #00D4AA !important;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }
    
    .brand-tagline {
        font-size: 0.8rem;
        color: #a0aec0 !important;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 5px;
    }
    
    /* Divider with better visibility */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #4a5568, transparent);
        margin: 20px 0;
    }
    
    /* Navigation styling */
    .nav-item {
        padding: 12px 16px;
        border-radius: 8px;
        margin: 4px 0;
        cursor: pointer;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        gap: 12px;
        color: #ffffff !important;
    }
    
    .nav-item:hover {
        background: rgba(0, 212, 170, 0.1);
    }
    
    .nav-item.active {
        background: linear-gradient(135deg, rgba(0, 212, 170, 0.2) 0%, rgba(102, 126, 234, 0.2) 100%);
        border-left: 3px solid #00D4AA;
    }
    
    /* Alert boxes */
    .alert-success {
        background: rgba(0, 212, 170, 0.1);
        border-left: 4px solid #00D4AA;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 10px 0;
    }
    
    .alert-warning {
        background: rgba(254, 202, 87, 0.1);
        border-left: 4px solid #feca57;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 10px 0;
    }
    
    .alert-danger {
        background: rgba(255, 107, 107, 0.1);
        border-left: 4px solid #ff6b6b;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 10px 0;
    }
    
    /* Progress bars */
    .progress-container {
        background: #2d3748;
        border-radius: 10px;
        height: 8px;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #00D4AA, #667eea);
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #1a1f2e !important;
        border-radius: 8px !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #1a1f2e;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        border: 1px solid #2d3748;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 212, 170, 0.2) 0%, rgba(102, 126, 234, 0.2) 100%);
        border-bottom: 2px solid #00D4AA;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand Logo with Professional Icon
    st.markdown(f"""
    <div class="brand-container">
        <div class="brand-logo">{icon_line_chart(32, COLORS['primary'])} QuantTS</div>
        <div class="brand-tagline">Time Series Analytics</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Navigation with Material Icons
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px;">
        {icon_grid(20, '#e2e8f0')}
        <span style="color: #ffffff !important; font-size: 1.1rem; font-weight: 600;">Navigation</span>
    </div>
    """, unsafe_allow_html=True)
    
    pages = {
        "Overview": ("Overview", icon_grid),
        "Returns Analysis": ("Returns", icon_trending_up),
        "Volatility": ("Volatility", icon_activity),
        "Dependency": ("Dependency", icon_link),
        "Stability": ("Stability", icon_scale)
    }
    
    page = st.radio(
        "Select Module",
        list(pages.keys()),
        label_visibility="collapsed"
    )
    
    selected_page = pages[page][0]
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Data Status with Icons
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px;">
        {icon_database(20, '#e2e8f0')}
        <span style="color: #ffffff !important; font-size: 1.1rem; font-weight: 600;">Data Status</span>
    </div>
    """, unsafe_allow_html=True)
    
    data_path = Path("data/clean/yahoo/returns.parquet")
    if data_path.exists():
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; padding: 12px; 
                    background: rgba(0, 212, 170, 0.15); border-radius: 8px; margin-bottom: 10px;
                    border: 1px solid rgba(0, 212, 170, 0.3);">
            {icon_check_circle(18, COLORS['success'])}
            <span style="color: #00D4AA !important; font-weight: 500;">Data loaded</span>
        </div>
        """, unsafe_allow_html=True)
        df_check = pd.read_parquet(data_path)
        st.markdown(f"""
        <div style="padding: 5px 10px; background: rgba(255,255,255,0.05); border-radius: 6px;">
            <div style="display: flex; align-items: center; gap: 6px; color: #e2e8f0 !important; font-size: 0.85rem; margin-bottom: 8px;">
                {icon_folder(14, '#e2e8f0')} <span style="color: #ffffff !important;">{len(df_check):,} observations</span>
            </div>
            <div style="display: flex; align-items: center; gap: 6px; color: #e2e8f0 !important; font-size: 0.85rem;">
                {icon_calendar(14, '#e2e8f0')} <span style="color: #ffffff !important;">{df_check['date'].min().strftime('%Y-%m-%d')} → {df_check['date'].max().strftime('%Y-%m-%d')}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; padding: 10px; 
                    background: rgba(255, 107, 107, 0.1); border-radius: 8px;">
            {icon_x_circle(18, COLORS['danger'])}
            <span style="color: #ff6b6b;">No data found</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown(f"""
    <div style="text-align: center; padding: 20px 0; border-top: 1px solid #3d4760; margin-top: 20px;">
        <p style="display: flex; align-items: center; justify-content: center; gap: 6px; 
                  color: #e2e8f0 !important; font-size: 0.8rem; margin-bottom: 5px;">
            {icon_clock(14, '#e2e8f0')} <span style="color: #ffffff !important;">QuantTS-Core v1.0</span>
        </p>
        <p style="color: #a0aec0 !important; font-size: 0.75rem;">© 2026 Adama COULIBALY</p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Page Router
# ─────────────────────────────────────────────────────────────────────────────
if selected_page == "Overview":
    from pages.overview import run
elif selected_page == "Returns":
    from pages.returns import run
elif selected_page == "Volatility":
    from pages.volatility import run
elif selected_page == "Dependency":
    from pages.dependance import run
else:
    from pages.stability import run

run()