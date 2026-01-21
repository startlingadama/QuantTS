import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from scipy import stats
import sys

# Add components to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from components.icons import (
    icon_activity, icon_line_chart, icon_bar_chart, icon_trending_up,
    icon_trending_down, icon_target, icon_percent, icon_alert_triangle,
    icon_scatter, icon_hash, icon_sliders, COLORS, ICON_MD
)

# ═══════════════════════════════════════════════════════════════════════════════
#                              VOLATILITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    """Load returns data with volatility if available."""
    # Try to load data with conditional volatility first
    vol_path = Path("data/clean/yahoo/returns_with_volatility.parquet")
    base_path = Path("data/clean/yahoo/returns.parquet")
    features_path = Path("data/features/returns.parquet")
    
    if features_path.exists():
        df = pd.read_parquet(features_path)
    elif vol_path.exists():
        df = pd.read_parquet(vol_path)
    elif base_path.exists():
        df = pd.read_parquet(base_path)
    else:
        return None
    
    df['date'] = pd.to_datetime(df['date'])
    return df

def calculate_realized_volatility(df, window=21):
    """Calculate realized volatility."""
    returns = df['log_return']
    realized_vol = returns.rolling(window).std() * np.sqrt(252) * 100
    return realized_vol

def create_volatility_chart(df):
    """Create volatility time series chart."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.5],
        subplot_titles=('Price', 'Volatility Measures')
    )
    
    # Price
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['adj_close'],
        name='Price',
        line=dict(color='#667eea', width=1.5)
    ), row=1, col=1)
    
    # Realized volatility (21-day)
    realized_vol_21 = calculate_realized_volatility(df, 21)
    fig.add_trace(go.Scatter(
        x=df['date'], y=realized_vol_21,
        name='21D Realized Vol',
        line=dict(color='#00D4AA', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(0,212,170,0.1)'
    ), row=2, col=1)
    
    # Realized volatility (63-day)
    realized_vol_63 = calculate_realized_volatility(df, 63)
    fig.add_trace(go.Scatter(
        x=df['date'], y=realized_vol_63,
        name='63D Realized Vol',
        line=dict(color='#feca57', width=1.5)
    ), row=2, col=1)
    
    # Conditional volatility if available
    if 'conditional_volatility' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['conditional_volatility'] * 100 * np.sqrt(252),
            name='GARCH Vol',
            line=dict(color='#ff6b6b', width=1.5)
        ), row=2, col=1)
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,31,46,1)',
        height=500,
        margin=dict(l=50, r=50, t=50, b=30),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(color='#a0aec0')
    )
    
    fig.update_xaxes(gridcolor='rgba(45,55,72,0.5)')
    fig.update_yaxes(gridcolor='rgba(45,55,72,0.5)')
    
    return fig

def create_volatility_cone(df):
    """Create volatility term structure / cone."""
    windows = [5, 10, 21, 42, 63, 126, 252]
    percentiles = [10, 25, 50, 75, 90]
    
    vol_data = {}
    for w in windows:
        vol = df['log_return'].rolling(w).std() * np.sqrt(252) * 100
        vol_data[w] = {
            'current': vol.iloc[-1],
            'min': vol.min(),
            'max': vol.max(),
            **{f'p{p}': np.percentile(vol.dropna(), p) for p in percentiles}
        }
    
    vol_df = pd.DataFrame(vol_data).T
    
    fig = go.Figure()
    
    # Cone bands
    fig.add_trace(go.Scatter(
        x=windows + windows[::-1],
        y=list(vol_df['p10']) + list(vol_df['p90'])[::-1],
        fill='toself',
        fillcolor='rgba(102,126,234,0.1)',
        line=dict(color='rgba(0,0,0,0)'),
        name='10-90 Percentile',
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter(
        x=windows + windows[::-1],
        y=list(vol_df['p25']) + list(vol_df['p75'])[::-1],
        fill='toself',
        fillcolor='rgba(102,126,234,0.2)',
        line=dict(color='rgba(0,0,0,0)'),
        name='25-75 Percentile',
        showlegend=True
    ))
    
    # Median
    fig.add_trace(go.Scatter(
        x=windows, y=vol_df['p50'],
        name='Median',
        line=dict(color='#667eea', width=2)
    ))
    
    # Current
    fig.add_trace(go.Scatter(
        x=windows, y=vol_df['current'],
        name='Current',
        line=dict(color='#00D4AA', width=3),
        mode='lines+markers',
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,31,46,1)',
        height=400,
        margin=dict(l=50, r=50, t=30, b=30),
        xaxis_title='Window (Days)',
        yaxis_title='Annualized Volatility (%)',
        xaxis=dict(tickvals=windows),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        font=dict(color='#a0aec0')
    )
    
    fig.update_xaxes(gridcolor='rgba(45,55,72,0.5)')
    fig.update_yaxes(gridcolor='rgba(45,55,72,0.5)')
    
    return fig

def create_volatility_distribution(df):
    """Create volatility distribution histogram."""
    realized_vol = calculate_realized_volatility(df, 21).dropna()
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=realized_vol,
        nbinsx=50,
        name='21D Realized Vol',
        marker_color='#667eea',
        opacity=0.75
    ))
    
    # Current volatility line
    current_vol = realized_vol.iloc[-1]
    fig.add_vline(
        x=current_vol,
        line_dash="dash",
        line_color="#00D4AA",
        annotation_text=f"Current: {current_vol:.1f}%"
    )
    
    # Percentile bands
    p25 = np.percentile(realized_vol, 25)
    p75 = np.percentile(realized_vol, 75)
    
    fig.add_vrect(
        x0=p25, x1=p75,
        fillcolor="rgba(0,212,170,0.1)",
        line_width=0,
        annotation_text="IQR"
    )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,31,46,1)',
        height=350,
        margin=dict(l=50, r=50, t=30, b=30),
        xaxis_title='Volatility (%)',
        yaxis_title='Frequency',
        showlegend=False,
        font=dict(color='#a0aec0')
    )
    
    fig.update_xaxes(gridcolor='rgba(45,55,72,0.5)')
    fig.update_yaxes(gridcolor='rgba(45,55,72,0.5)')
    
    return fig

def create_volatility_regime(df):
    """Create volatility regime chart."""
    realized_vol = calculate_realized_volatility(df, 21)
    
    # Define regimes based on percentiles
    p33 = np.percentile(realized_vol.dropna(), 33)
    p67 = np.percentile(realized_vol.dropna(), 67)
    
    regime = pd.Series(index=df.index, dtype=str)
    regime[realized_vol <= p33] = 'Low'
    regime[(realized_vol > p33) & (realized_vol <= p67)] = 'Medium'
    regime[realized_vol > p67] = 'High'
    
    df_regime = df.copy()
    df_regime['regime'] = regime
    df_regime['vol'] = realized_vol
    
    fig = go.Figure()
    
    colors = {'Low': '#00D4AA', 'Medium': '#feca57', 'High': '#ff6b6b'}
    
    for r in ['Low', 'Medium', 'High']:
        mask = df_regime['regime'] == r
        fig.add_trace(go.Scatter(
            x=df_regime.loc[mask, 'date'],
            y=df_regime.loc[mask, 'vol'],
            mode='markers',
            name=f'{r} Vol',
            marker=dict(color=colors[r], size=3, opacity=0.6)
        ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,31,46,1)',
        height=300,
        margin=dict(l=50, r=50, t=30, b=30),
        xaxis_title='Date',
        yaxis_title='Volatility (%)',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        font=dict(color='#a0aec0')
    )
    
    fig.update_xaxes(gridcolor='rgba(45,55,72,0.5)')
    fig.update_yaxes(gridcolor='rgba(45,55,72,0.5)')
    
    return fig, df_regime

def create_vol_vs_returns(df):
    """Create volatility vs returns scatter plot."""
    df_copy = df.copy()
    df_copy['vol_21d'] = calculate_realized_volatility(df, 21)
    df_copy['future_return'] = df_copy['log_return'].shift(-21).rolling(21).sum() * 100
    df_copy = df_copy.dropna()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_copy['vol_21d'],
        y=df_copy['future_return'],
        mode='markers',
        marker=dict(
            color=df_copy['future_return'],
            colorscale=[[0, '#ff6b6b'], [0.5, '#1a1f2e'], [1, '#00D4AA']],
            size=5,
            opacity=0.5,
            colorbar=dict(title='Return %')
        ),
        name='Vol vs Future Return'
    ))
    
    # Add trend line
    z = np.polyfit(df_copy['vol_21d'], df_copy['future_return'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_copy['vol_21d'].min(), df_copy['vol_21d'].max(), 100)
    
    fig.add_trace(go.Scatter(
        x=x_line,
        y=p(x_line),
        mode='lines',
        name='Trend',
        line=dict(color='#feca57', width=2, dash='dash')
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,31,46,1)',
        height=350,
        margin=dict(l=50, r=50, t=30, b=30),
        xaxis_title='Current Volatility (%)',
        yaxis_title='21D Future Return (%)',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        font=dict(color='#a0aec0')
    )
    
    fig.update_xaxes(gridcolor='rgba(45,55,72,0.5)')
    fig.update_yaxes(gridcolor='rgba(45,55,72,0.5)')
    
    return fig

def calculate_vol_metrics(df):
    """Calculate volatility metrics."""
    returns = df['log_return']
    
    # Different realized volatility windows
    vol_5d = returns.rolling(5).std() * np.sqrt(252) * 100
    vol_21d = returns.rolling(21).std() * np.sqrt(252) * 100
    vol_63d = returns.rolling(63).std() * np.sqrt(252) * 100
    vol_252d = returns.rolling(252).std() * np.sqrt(252) * 100
    
    # Current values
    current_5d = vol_5d.iloc[-1]
    current_21d = vol_21d.iloc[-1]
    current_63d = vol_63d.iloc[-1]
    current_252d = vol_252d.iloc[-1]
    
    # Historical percentiles
    percentile_21d = stats.percentileofscore(vol_21d.dropna(), current_21d)
    
    # Volatility of volatility
    vol_of_vol = vol_21d.rolling(21).std().iloc[-1]
    
    # Max vol
    max_vol = vol_21d.max()
    max_vol_date = df.loc[vol_21d.idxmax(), 'date']
    
    # Min vol
    min_vol = vol_21d.min()
    min_vol_date = df.loc[vol_21d.idxmin(), 'date']
    
    return {
        'current_5d': current_5d,
        'current_21d': current_21d,
        'current_63d': current_63d,
        'current_252d': current_252d,
        'percentile_21d': percentile_21d,
        'vol_of_vol': vol_of_vol,
        'max_vol': max_vol,
        'max_vol_date': max_vol_date,
        'min_vol': min_vol,
        'min_vol_date': min_vol_date,
        'mean_vol': vol_21d.mean(),
        'median_vol': vol_21d.median()
    }

def run():
    """Main function to run the Volatility Analysis page."""
    
    # Header with icon
    st.markdown(f"""
    <div style="margin-bottom: 30px;">
        <h1 style="color: #ffffff; font-weight: 700; margin-bottom: 5px; display: flex; align-items: center; gap: 12px;">
            {icon_activity(32, COLORS['primary'])} Volatility Analysis
        </h1>
        <p style="color: #a0aec0; font-size: 1.1rem;">
            Comprehensive volatility modeling and regime analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    if df is None:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 10px; padding: 15px; 
                    background: rgba(255, 107, 107, 0.1); border-left: 4px solid #ff6b6b; 
                    border-radius: 4px;">
            {icon_alert_triangle(24, COLORS['danger'])}
            <span style="color: #ff6b6b;">No data available. Please run the data pipeline first.</span>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Calculate metrics
    vol_metrics = calculate_vol_metrics(df)
    
    # ─────────────────────────────────────────────────────────────────────────
    # TOP METRICS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px;">
        {icon_percent(20, COLORS['primary'])}
        <span style="color: #ffffff; font-size: 1.2rem; font-weight: 600;">Current Volatility Metrics</span>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("5D Vol (Ann.)", f"{vol_metrics['current_5d']:.1f}%")
    with col2:
        st.metric("21D Vol (Ann.)", f"{vol_metrics['current_21d']:.1f}%")
    with col3:
        st.metric("63D Vol (Ann.)", f"{vol_metrics['current_63d']:.1f}%")
    with col4:
        st.metric("252D Vol (Ann.)", f"{vol_metrics['current_252d']:.1f}%")
    with col5:
        st.metric("Percentile (21D)", f"{vol_metrics['percentile_21d']:.0f}%")
    with col6:
        st.metric("Vol of Vol", f"{vol_metrics['vol_of_vol']:.2f}%")
    
    st.markdown("---")
    
    # ─────────────────────────────────────────────────────────────────────────
    # VOLATILITY TIME SERIES
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px;">
        {icon_line_chart(20, COLORS['primary'])}
        <span style="color: #ffffff; font-size: 1.2rem; font-weight: 600;">Volatility Time Series</span>
    </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(create_volatility_chart(df), width='stretch')
    
    # ─────────────────────────────────────────────────────────────────────────
    # VOLATILITY CONE & DISTRIBUTION
    # ─────────────────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
            {icon_hash(18, COLORS['secondary'])}
            <span style="color: #ffffff; font-size: 1.1rem; font-weight: 600;">Volatility Term Structure</span>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(create_volatility_cone(df), width='stretch')
        
        st.caption("""
        **Interpretation:** The cone shows historical volatility percentiles across different 
        lookback windows. The current volatility line shows where we stand relative to history.
        """)
    
    with col2:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
            {icon_bar_chart(18, COLORS['secondary'])}
            <span style="color: #ffffff; font-size: 1.1rem; font-weight: 600;">Volatility Distribution</span>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(create_volatility_distribution(df), width='stretch')
        
        # Regime indicator
        realized_vol = calculate_realized_volatility(df, 21).iloc[-1]
        p33 = np.percentile(calculate_realized_volatility(df, 21).dropna(), 33)
        p67 = np.percentile(calculate_realized_volatility(df, 21).dropna(), 67)
        
        if realized_vol <= p33:
            regime = "Low Volatility Regime"
            regime_color = "#00D4AA"
            regime_icon = icon_trending_down(20, COLORS['success'])
        elif realized_vol <= p67:
            regime = "Normal Volatility Regime"
            regime_color = "#feca57"
            regime_icon = icon_activity(20, COLORS['warning'])
        else:
            regime = "High Volatility Regime"
            regime_color = "#ff6b6b"
            regime_icon = icon_trending_up(20, COLORS['danger'])
        
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <div class="metric-label">Current Regime</div>
            <div style="font-size: 1.3rem; font-weight: 600; color: {regime_color}; display: flex; align-items: center; justify-content: center; gap: 8px;">
                {regime_icon} {regime}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ─────────────────────────────────────────────────────────────────────────
    # VOLATILITY REGIMES
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px;">
        {icon_target(20, COLORS['warning'])}
        <span style="color: #ffffff; font-size: 1.2rem; font-weight: 600;">Volatility Regime Analysis</span>
    </div>
    """, unsafe_allow_html=True)
    
    regime_fig, df_regime = create_volatility_regime(df)
    st.plotly_chart(regime_fig, width='stretch')
    
    # Regime statistics
    regime_stats = df_regime.groupby('regime').agg({
        'log_return': ['count', 'mean', 'std']
    }).round(4)
    regime_stats.columns = ['Count', 'Mean Return', 'Std Return']
    regime_stats['Mean Return'] = regime_stats['Mean Return'] * 100
    regime_stats['Std Return'] = regime_stats['Std Return'] * 100
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("#### Regime Statistics")
        st.dataframe(regime_stats, width='stretch'
                     )
    
    # ─────────────────────────────────────────────────────────────────────────
    # VOL VS RETURNS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px;">
        {icon_scatter(20, COLORS['secondary'])}
        <span style="color: #ffffff; font-size: 1.2rem; font-weight: 600;">Volatility-Return Relationship</span>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.plotly_chart(create_vol_vs_returns(df), width='stretch')
    
    with col2:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
            {icon_hash(16, COLORS['info'])}
            <span style="color: #ffffff; font-weight: 500;">Interpretation</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        This scatter plot shows the relationship between current volatility 
        and subsequent 21-day returns.
        
        **Key insights:**
        - Higher volatility often precedes higher expected returns (risk premium)
        - Extreme volatility can signal market stress
        - Low volatility environments may persist (volatility clustering)
        """)
        
        # Historical extremes
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px; margin-top: 15px;">
            {icon_bar_chart(16, COLORS['info'])}
            <span style="color: #ffffff; font-weight: 500;">Historical Extremes</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-box" style="margin-bottom: 10px;">
            <div class="stat-label">Maximum Volatility</div>
            <div class="stat-value">{vol_metrics['max_vol']:.1f}%</div>
            <span style="font-size: 0.75rem; color: #a0aec0;">
                {vol_metrics['max_vol_date'].strftime('%Y-%m-%d')}
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">Minimum Volatility</div>
            <div class="stat-value" style="color: #00D4AA;">{vol_metrics['min_vol']:.1f}%</div>
            <span style="font-size: 0.75rem; color: #a0aec0;">
                {vol_metrics['min_vol_date'].strftime('%Y-%m-%d')}
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY TABLE
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px;">
        {icon_sliders(20, COLORS['primary'])}
        <span style="color: #ffffff; font-size: 1.2rem; font-weight: 600;">Volatility Summary</span>
    </div>
    """, unsafe_allow_html=True)
    
    summary_df = pd.DataFrame({
        'Metric': ['5D Realized Vol', '21D Realized Vol', '63D Realized Vol', 
                   '252D Realized Vol', 'Historical Mean', 'Historical Median',
                   'Current Percentile', 'Vol of Vol'],
        'Value': [f"{vol_metrics['current_5d']:.2f}%", f"{vol_metrics['current_21d']:.2f}%",
                  f"{vol_metrics['current_63d']:.2f}%", f"{vol_metrics['current_252d']:.2f}%",
                  f"{vol_metrics['mean_vol']:.2f}%", f"{vol_metrics['median_vol']:.2f}%",
                  f"{vol_metrics['percentile_21d']:.0f}%", f"{vol_metrics['vol_of_vol']:.2f}%"]
    })
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.dataframe(summary_df, width='stretch', hide_index=True)