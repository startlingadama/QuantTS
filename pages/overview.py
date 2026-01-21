import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add components to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from components.icons import (
    icon_line_chart, icon_trending_up, icon_trending_down, icon_bar_chart,
    icon_activity, icon_shield_alert, icon_target, icon_percent, icon_dollar,
    icon_calendar, icon_clock, icon_database, icon_alert_triangle,
    icon_pie_chart, icon_hash, COLORS, ICON_LG, ICON_MD, ICON_SM
)

# ═══════════════════════════════════════════════════════════════════════════════
#                              OVERVIEW DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    """Load and prepare data for the overview dashboard."""
    data_path = Path("data/clean/yahoo/returns.parquet")
    if data_path.exists():
        df = pd.read_parquet(data_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    return None

def calculate_metrics(df):
    """Calculate key performance metrics."""
    if df is None or df.empty:
        return {}
    
    returns = df['log_return']
    prices = df['adj_close']
    
    # Annualization factor (252 trading days)
    ann_factor = 252
    
    # Core metrics
    total_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
    ann_return = returns.mean() * ann_factor * 100
    ann_volatility = returns.std() * np.sqrt(ann_factor) * 100
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(ann_factor) if returns.std() != 0 else 0
    
    # Risk metrics
    var_95 = np.percentile(returns, 5) * 100
    var_99 = np.percentile(returns, 1) * 100
    cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
    
    # Drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100
    
    # Distribution stats
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    
    # Recent performance
    last_price = prices.iloc[-1]
    daily_change = returns.iloc[-1] * 100
    weekly_return = returns.tail(5).sum() * 100
    monthly_return = returns.tail(21).sum() * 100
    ytd_return = returns.tail(min(len(returns), 252)).sum() * 100
    
    return {
        'total_return': total_return,
        'ann_return': ann_return,
        'ann_volatility': ann_volatility,
        'sharpe_ratio': sharpe_ratio,
        'var_95': var_95,
        'var_99': var_99,
        'cvar_95': cvar_95,
        'max_drawdown': max_drawdown,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'last_price': last_price,
        'daily_change': daily_change,
        'weekly_return': weekly_return,
        'monthly_return': monthly_return,
        'ytd_return': ytd_return,
        'n_observations': len(df)
    }

def create_price_chart(df):
    """Create an interactive price chart with volume."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC',
            increasing_line_color='#00D4AA',
            decreasing_line_color='#ff6b6b'
        ),
        row=1, col=1
    )
    
    # Add moving averages
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_200'] = df['close'].rolling(window=200).mean()
    
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['SMA_20'], name='SMA 20', 
                   line=dict(color='#feca57', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['SMA_50'], name='SMA 50',
                   line=dict(color='#667eea', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['SMA_200'], name='SMA 200',
                   line=dict(color='#ff9ff3', width=1)),
        row=1, col=1
    )
    
    # Volume bars
    colors = ['#00D4AA' if row['close'] >= row['open'] else '#ff6b6b' 
              for _, row in df.iterrows()]
    
    fig.add_trace(
        go.Bar(x=df['date'], y=df['volume'], name='Volume',
               marker_color=colors, opacity=0.7),
        row=2, col=1
    )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,31,46,1)',
        height=500,
        margin=dict(l=50, r=50, t=30, b=30),
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0)'
        ),
        font=dict(color='#a0aec0')
    )
    
    fig.update_xaxes(
        gridcolor='rgba(45,55,72,0.5)',
        showgrid=True
    )
    fig.update_yaxes(
        gridcolor='rgba(45,55,72,0.5)',
        showgrid=True
    )
    
    return fig

def create_returns_distribution(df):
    """Create returns distribution histogram."""
    fig = go.Figure()
    
    returns = df['log_return'] * 100  # Convert to percentage
    
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=100,
        name='Returns Distribution',
        marker_color='#667eea',
        opacity=0.75
    ))
    
    # Add normal distribution overlay
    x_range = np.linspace(returns.min(), returns.max(), 100)
    normal_dist = (1 / (returns.std() * np.sqrt(2 * np.pi))) * \
                  np.exp(-0.5 * ((x_range - returns.mean()) / returns.std()) ** 2)
    normal_dist = normal_dist * len(returns) * (returns.max() - returns.min()) / 100
    
    fig.add_trace(go.Scatter(
        x=x_range,
        y=normal_dist,
        name='Normal Distribution',
        line=dict(color='#00D4AA', width=2, dash='dash')
    ))
    
    # Add VaR lines
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)
    
    fig.add_vline(x=var_95, line_dash="dot", line_color="#feca57",
                  annotation_text=f"VaR 95%: {var_95:.2f}%")
    fig.add_vline(x=var_99, line_dash="dot", line_color="#ff6b6b",
                  annotation_text=f"VaR 99%: {var_99:.2f}%")
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,31,46,1)',
        height=350,
        margin=dict(l=50, r=50, t=30, b=30),
        xaxis_title='Daily Returns (%)',
        yaxis_title='Frequency',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0)'
        ),
        font=dict(color='#a0aec0')
    )
    
    fig.update_xaxes(gridcolor='rgba(45,55,72,0.5)')
    fig.update_yaxes(gridcolor='rgba(45,55,72,0.5)')
    
    return fig

def create_cumulative_returns(df):
    """Create cumulative returns chart."""
    cumulative = (1 + df['log_return']).cumprod() - 1
    
    fig = go.Figure()
    
    # Cumulative returns
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=cumulative * 100,
        fill='tozeroy',
        name='Cumulative Return',
        line=dict(color='#00D4AA', width=2),
        fillcolor='rgba(0,212,170,0.2)'
    ))
    
    # Rolling Sharpe (252d window)
    rolling_mean = df['log_return'].rolling(252).mean()
    rolling_std = df['log_return'].rolling(252).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,31,46,1)',
        height=300,
        margin=dict(l=50, r=50, t=30, b=30),
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(color='#a0aec0')
    )
    
    fig.update_xaxes(gridcolor='rgba(45,55,72,0.5)')
    fig.update_yaxes(gridcolor='rgba(45,55,72,0.5)')
    
    return fig

def create_drawdown_chart(df):
    """Create drawdown chart."""
    returns = df['log_return']
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=drawdown,
        fill='tozeroy',
        name='Drawdown',
        line=dict(color='#ff6b6b', width=1),
        fillcolor='rgba(255,107,107,0.3)'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,31,46,1)',
        height=250,
        margin=dict(l=50, r=50, t=30, b=30),
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        showlegend=False,
        font=dict(color='#a0aec0')
    )
    
    fig.update_xaxes(gridcolor='rgba(45,55,72,0.5)')
    fig.update_yaxes(gridcolor='rgba(45,55,72,0.5)')
    
    return fig

def render_metric_card(label, value, delta=None, prefix="", suffix="", is_positive_good=True):
    """Render a styled metric card."""
    if delta is not None:
        delta_class = "metric-delta-positive" if (delta >= 0) == is_positive_good else "metric-delta-negative"
        delta_symbol = "▲" if delta >= 0 else "▼"
        delta_html = f'<span class="{delta_class}">{delta_symbol} {abs(delta):.2f}%</span>'
    else:
        delta_html = ""
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{prefix}{value}{suffix}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def run():
    """Main function to run the Overview dashboard."""
    
    # Header with professional icon
    st.markdown(f"""
    <div style="margin-bottom: 30px;">
        <h1 style="color: #ffffff; font-weight: 700; margin-bottom: 5px; display: flex; align-items: center; gap: 12px;">
            {icon_line_chart(32, COLORS['primary'])} Market Overview
        </h1>
        <p style="color: #a0aec0; font-size: 1.1rem;">
            Comprehensive quantitative analysis and performance metrics
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    if df is None:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 10px; padding: 15px; 
                    background: rgba(255, 107, 107, 0.1); border-left: 4px solid #ff6b6b; 
                    border-radius: 4px; margin: 20px 0;">
            {icon_alert_triangle(24, COLORS['danger'])}
            <span style="color: #ff6b6b;">No data available. Please run the data ingestion pipeline first.</span>
        </div>
        """, unsafe_allow_html=True)
        st.code("""
# Run these commands to load data:
python src/ingestion/load_yahoo.py
python src/preprocessing/make_returns.py
        """)
        return
    
    # Calculate metrics
    metrics = calculate_metrics(df)
    
    # ─────────────────────────────────────────────────────────────────────────
    # TOP KPI CARDS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px;">
        {icon_target(20, COLORS['primary'])}
        <span style="color: #ffffff; font-size: 1.2rem; font-weight: 600;">Key Performance Indicators</span>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Last Price",
            value=f"${metrics['last_price']:.2f}",
            delta=f"{metrics['daily_change']:.2f}%"
        )
    
    with col2:
        st.metric(
            label="Annual Return",
            value=f"{metrics['ann_return']:.2f}%",
            delta=f"{metrics['monthly_return']:.2f}% MTD"
        )
    
    with col3:
        st.metric(
            label="Volatility (Ann.)",
            value=f"{metrics['ann_volatility']:.2f}%",
            delta=None
        )
    
    with col4:
        st.metric(
            label="Sharpe Ratio",
            value=f"{metrics['sharpe_ratio']:.3f}",
            delta=None
        )
    
    with col5:
        st.metric(
            label="Max Drawdown",
            value=f"{metrics['max_drawdown']:.2f}%",
            delta=None
        )
    
    st.markdown("---")
    
    # ─────────────────────────────────────────────────────────────────────────
    # PRICE CHART
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px;">
        {icon_line_chart(20, COLORS['primary'])}
        <span style="color: #ffffff; font-size: 1.2rem; font-weight: 600;">Price Chart & Technical Analysis</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Time range selector
    col1, col2 = st.columns([3, 1])
    with col2:
        time_range = st.selectbox(
            "Time Range",
            ["1M", "3M", "6M", "1Y", "3Y", "5Y", "All"],
            index=3
        )
    
    # Filter data based on time range
    end_date = df['date'].max()
    range_map = {
        "1M": 21, "3M": 63, "6M": 126, "1Y": 252, 
        "3Y": 756, "5Y": 1260, "All": len(df)
    }
    n_days = range_map.get(time_range, 252)
    df_filtered = df.tail(n_days).copy()
    
    st.plotly_chart(create_price_chart(df_filtered), width='stretch')
    
    # ─────────────────────────────────────────────────────────────────────────
    # RETURNS & DRAWDOWN
    # ─────────────────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
            {icon_trending_up(18, COLORS['success'])}
            <span style="color: #ffffff; font-size: 1.1rem; font-weight: 600;">Cumulative Returns</span>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(create_cumulative_returns(df), width='stretch')
    
    with col2:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
            {icon_trending_down(18, COLORS['danger'])}
            <span style="color: #ffffff; font-size: 1.1rem; font-weight: 600;">Drawdown Analysis</span>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(create_drawdown_chart(df), width='stretch')
    
    # ─────────────────────────────────────────────────────────────────────────
    # DISTRIBUTION & RISK METRICS
    # ─────────────────────────────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
            {icon_bar_chart(18, COLORS['secondary'])}
            <span style="color: #ffffff; font-size: 1.1rem; font-weight: 600;">Returns Distribution</span>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(create_returns_distribution(df), width='stretch')
    
    with col2:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
            {icon_shield_alert(18, COLORS['danger'])}
            <span style="color: #ffffff; font-size: 1.1rem; font-weight: 600;">Risk Metrics</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom: 15px;">
            <div class="metric-label">Value at Risk (95%)</div>
            <div class="metric-value risk-high">{metrics['var_95']:.3f}%</div>
            <span style="color: #a0aec0; font-size: 0.8rem;">Daily potential loss</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom: 15px;">
            <div class="metric-label">Value at Risk (99%)</div>
            <div class="metric-value risk-high">{metrics['var_99']:.3f}%</div>
            <span style="color: #a0aec0; font-size: 0.8rem;">Daily potential loss</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom: 15px;">
            <div class="metric-label">Expected Shortfall (CVaR 95%)</div>
            <div class="metric-value risk-high">{metrics['cvar_95']:.3f}%</div>
            <span style="color: #a0aec0; font-size: 0.8rem;">Expected loss beyond VaR</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px; margin-top: 15px;">
            {icon_hash(16, COLORS['secondary'])}
            <span style="color: #ffffff; font-size: 0.95rem; font-weight: 600;">Distribution Shape</span>
        </div>
        """, unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Skewness", f"{metrics['skewness']:.3f}")
        with col_b:
            st.metric("Kurtosis", f"{metrics['kurtosis']:.3f}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # PERFORMANCE TABLE
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px;">
        {icon_calendar(20, COLORS['primary'])}
        <span style="color: #ffffff; font-size: 1.2rem; font-weight: 600;">Performance Summary</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Create performance table
    perf_data = {
        'Period': ['Daily', 'Weekly', 'Monthly', 'YTD', 'Total'],
        'Return (%)': [
            f"{metrics['daily_change']:.2f}",
            f"{metrics['weekly_return']:.2f}",
            f"{metrics['monthly_return']:.2f}",
            f"{metrics['ytd_return']:.2f}",
            f"{metrics['total_return']:.2f}"
        ]
    }
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.dataframe(
            pd.DataFrame(perf_data),
            width='stretch',
            hide_index=True
        )
    
    # Footer stats
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 6px; color: #a0aec0; font-size: 0.85rem;">
            {icon_database(14, COLORS['muted'])} Total Observations: {metrics['n_observations']:,}
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 6px; color: #a0aec0; font-size: 0.85rem;">
            {icon_calendar(14, COLORS['muted'])} Start: {df['date'].min().strftime('%Y-%m-%d')}
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 6px; color: #a0aec0; font-size: 0.85rem;">
            {icon_calendar(14, COLORS['muted'])} End: {df['date'].max().strftime('%Y-%m-%d')}
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 6px; color: #a0aec0; font-size: 0.85rem;">
            {icon_clock(14, COLORS['muted'])} Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        </div>
        """, unsafe_allow_html=True)