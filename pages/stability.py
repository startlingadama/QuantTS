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
    icon_scale, icon_line_chart, icon_bar_chart, icon_activity,
    icon_target, icon_alert_triangle, icon_check_circle, icon_x_circle,
    icon_hash, icon_info, icon_anchor, icon_crosshair, COLORS, ICON_MD
)

# ═══════════════════════════════════════════════════════════════════════════════
#                              STABILITY ANALYSIS
#                    (Stationarity Tests & Structural Breaks)
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    """Load returns data."""
    data_path = Path("data/clean/yahoo/returns.parquet")
    if data_path.exists():
        df = pd.read_parquet(data_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    return None

def adf_test(series):
    """Perform Augmented Dickey-Fuller test."""
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(series.dropna(), autolag='AIC')
        return {
            'statistic': result[0],
            'p_value': result[1],
            'lags_used': result[2],
            'n_obs': result[3],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
    except:
        return None

def kpss_test(series):
    """Perform KPSS test."""
    try:
        from statsmodels.tsa.stattools import kpss
        result = kpss(series.dropna(), regression='c', nlags='auto')
        return {
            'statistic': result[0],
            'p_value': result[1],
            'lags_used': result[2],
            'critical_values': result[3],
            'is_stationary': result[1] > 0.05  # Note: opposite interpretation
        }
    except:
        return None

def create_price_returns_comparison(df):
    """Create price vs returns comparison chart."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.5],
        subplot_titles=('Price (Non-Stationary)', 'Log Returns (Stationary)')
    )
    
    # Price series
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['adj_close'],
        name='Price',
        line=dict(color='#667eea', width=1.5)
    ), row=1, col=1)
    
    # Returns series
    colors = ['#00D4AA' if r >= 0 else '#ff6b6b' for r in df['log_return']]
    
    fig.add_trace(go.Bar(
        x=df['date'], y=df['log_return'] * 100,
        name='Log Returns',
        marker_color=colors,
        opacity=0.7
    ), row=2, col=1)
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,31,46,1)',
        height=500,
        margin=dict(l=50, r=50, t=50, b=30),
        showlegend=False,
        font=dict(color='#a0aec0')
    )
    
    fig.update_xaxes(gridcolor='rgba(45,55,72,0.5)')
    fig.update_yaxes(gridcolor='rgba(45,55,72,0.5)')
    
    return fig

def create_rolling_mean_std(df, window=252):
    """Create rolling mean and std chart to check stationarity visually."""
    returns = df['log_return'] * 100
    
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{window}D Rolling Mean', f'{window}D Rolling Std')
    )
    
    # Rolling mean
    fig.add_trace(go.Scatter(
        x=df['date'], y=rolling_mean,
        name='Rolling Mean',
        line=dict(color='#00D4AA', width=2),
        fill='tozeroy',
        fillcolor='rgba(0,212,170,0.1)'
    ), row=1, col=1)
    
    # Overall mean reference line
    overall_mean = returns.mean()
    fig.add_hline(y=overall_mean, line_dash="dash", line_color="#feca57", 
                  row=1, col=1, annotation_text=f"Overall: {overall_mean:.4f}%")
    
    # Rolling std
    fig.add_trace(go.Scatter(
        x=df['date'], y=rolling_std,
        name='Rolling Std',
        line=dict(color='#ff6b6b', width=2),
        fill='tozeroy',
        fillcolor='rgba(255,107,107,0.1)'
    ), row=2, col=1)
    
    # Overall std reference line
    overall_std = returns.std()
    fig.add_hline(y=overall_std, line_dash="dash", line_color="#feca57",
                  row=2, col=1, annotation_text=f"Overall: {overall_std:.4f}%")
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,31,46,1)',
        height=400,
        margin=dict(l=50, r=50, t=50, b=30),
        showlegend=False,
        font=dict(color='#a0aec0')
    )
    
    fig.update_xaxes(gridcolor='rgba(45,55,72,0.5)')
    fig.update_yaxes(gridcolor='rgba(45,55,72,0.5)')
    
    return fig

def detect_structural_breaks(df, window=63):
    """Detect potential structural breaks using CUSUM-like approach."""
    returns = df['log_return'].dropna()
    n = len(returns)
    
    # Calculate cumulative sum of standardized residuals
    mean = returns.mean()
    std = returns.std()
    standardized = (returns - mean) / std
    cusum = np.cumsum(standardized) / np.sqrt(n)
    
    # Calculate boundaries (simplified)
    k = np.arange(1, n + 1)
    upper_bound = 0.948 * np.sqrt(k / n + 2 * k * (n - k) / (n ** 2))
    lower_bound = -upper_bound
    
    return cusum, upper_bound, lower_bound

def create_cusum_chart(df):
    """Create CUSUM chart for structural break detection."""
    cusum, upper, lower = detect_structural_breaks(df)
    
    fig = go.Figure()
    
    dates = df['date'].iloc[1:]  # Align with returns
    
    # CUSUM line
    fig.add_trace(go.Scatter(
        x=dates, y=cusum,
        name='CUSUM',
        line=dict(color='#00D4AA', width=2)
    ))
    
    # Boundaries
    fig.add_trace(go.Scatter(
        x=dates, y=upper,
        name='Upper Bound',
        line=dict(color='#ff6b6b', width=1, dash='dash'),
        fill=None
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=lower,
        name='Lower Bound',
        line=dict(color='#ff6b6b', width=1, dash='dash'),
        fill='tonexty',
        fillcolor='rgba(255,107,107,0.1)'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,31,46,1)',
        height=350,
        margin=dict(l=50, r=50, t=30, b=30),
        xaxis_title='Date',
        yaxis_title='CUSUM',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        font=dict(color='#a0aec0')
    )
    
    fig.update_xaxes(gridcolor='rgba(45,55,72,0.5)')
    fig.update_yaxes(gridcolor='rgba(45,55,72,0.5)')
    
    return fig

def create_subsample_comparison(df, n_subsamples=4):
    """Create box plot comparing subsamples."""
    returns = df['log_return'] * 100
    n = len(returns)
    sample_size = n // n_subsamples
    
    data = []
    for i in range(n_subsamples):
        start = i * sample_size
        end = start + sample_size
        subsample = returns.iloc[start:end]
        
        start_date = df['date'].iloc[start].strftime('%Y')
        end_date = df['date'].iloc[min(end-1, n-1)].strftime('%Y')
        
        for val in subsample:
            data.append({
                'Period': f'{start_date}-{end_date}',
                'Return': val
            })
    
    data_df = pd.DataFrame(data)
    
    fig = go.Figure()
    
    colors = ['#667eea', '#00D4AA', '#feca57', '#ff6b6b']
    
    for i, period in enumerate(data_df['Period'].unique()):
        subset = data_df[data_df['Period'] == period]['Return']
        fig.add_trace(go.Box(
            y=subset,
            name=period,
            marker_color=colors[i % len(colors)],
            boxmean=True
        ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,31,46,1)',
        height=350,
        margin=dict(l=50, r=50, t=30, b=30),
        yaxis_title='Return (%)',
        showlegend=False,
        font=dict(color='#a0aec0')
    )
    
    fig.update_xaxes(gridcolor='rgba(45,55,72,0.5)')
    fig.update_yaxes(gridcolor='rgba(45,55,72,0.5)')
    
    return fig

def create_subsample_stats_table(df, n_subsamples=4):
    """Create table comparing subsample statistics."""
    returns = df['log_return']
    n = len(returns)
    sample_size = n // n_subsamples
    
    stats_list = []
    for i in range(n_subsamples):
        start = i * sample_size
        end = start + sample_size
        subsample = returns.iloc[start:end]
        
        start_date = df['date'].iloc[start].strftime('%Y-%m')
        end_date = df['date'].iloc[min(end-1, n-1)].strftime('%Y-%m')
        
        stats_list.append({
            'Period': f'{start_date} to {end_date}',
            'N': len(subsample),
            'Mean (%)': subsample.mean() * 100,
            'Std (%)': subsample.std() * 100,
            'Skewness': subsample.skew(),
            'Kurtosis': subsample.kurtosis(),
            'Sharpe': (subsample.mean() / subsample.std()) * np.sqrt(252)
        })
    
    return pd.DataFrame(stats_list)

def variance_ratio_test(returns, k=5):
    """Perform variance ratio test."""
    returns = returns.dropna()
    n = len(returns)
    
    # Variance of k-period returns
    k_period_returns = returns.rolling(k).sum().dropna()
    var_k = k_period_returns.var()
    
    # Variance of 1-period returns
    var_1 = returns.var()
    
    # Variance ratio
    vr = var_k / (k * var_1)
    
    # Under random walk, VR should be close to 1
    # Standard error (simplified)
    se = np.sqrt(2 * (2 * k - 1) * (k - 1) / (3 * k * n))
    
    z_stat = (vr - 1) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return {
        'k': k,
        'variance_ratio': vr,
        'z_statistic': z_stat,
        'p_value': p_value,
        'is_random_walk': p_value > 0.05
    }

def run():
    """Main function to run the Stability Analysis page."""
    
    # Header with icon
    st.markdown(f"""
    <div style="margin-bottom: 30px;">
        <h1 style="color: #ffffff; font-weight: 700; margin-bottom: 5px; display: flex; align-items: center; gap: 12px;">
            {icon_scale(32, COLORS['primary'])} Stability Analysis
        </h1>
        <p style="color: #a0aec0; font-size: 1.1rem;">
            Stationarity tests, structural breaks, and parameter stability
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
    
    # ─────────────────────────────────────────────────────────────────────────
    # INFO SECTION
    # ─────────────────────────────────────────────────────────────────────────
    with st.expander("ℹ️ About Stationarity in Time Series", expanded=False):
        st.markdown("""
        **Stationarity** is a crucial concept in time series analysis. A stationary time series 
        has statistical properties (mean, variance, autocorrelation) that don't change over time.
        
        #### Why Does Stationarity Matter?
        - Most statistical models assume stationarity
        - Non-stationary series can lead to spurious regression results
        - Forecasting is more reliable with stationary data
        
        #### Key Tests:
        
        | Test | Null Hypothesis | Stationary if... |
        |------|-----------------|------------------|
        | **ADF** | Series has unit root (non-stationary) | p-value < 0.05 (reject H₀) |
        | **KPSS** | Series is stationary | p-value > 0.05 (fail to reject H₀) |
        
        #### Typical Findings:
        - **Prices**: Non-stationary (unit root, I(1) process)
        - **Log Returns**: Stationary (I(0) process)
        """)
    
    st.markdown("---")
    
    # ─────────────────────────────────────────────────────────────────────────
    # VISUAL STATIONARITY CHECK
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px;">
        {icon_line_chart(20, COLORS['primary'])}
        <span style="color: #ffffff; font-size: 1.2rem; font-weight: 600;">Price vs Returns: Visual Stationarity Check</span>
    </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(create_price_returns_comparison(df.tail(500)), width='stretch')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="alert-danger">
            <div style="display: flex; align-items: flex-start; gap: 10px;">
                {icon_x_circle(18, COLORS['danger'])}
                <div>
                    <strong>Prices:</strong> Non-stationary<br>
                    - Clear upward/downward trends<br>
                    - Mean and variance change over time<br>
                    - Contains a unit root
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="alert-success">
            <div style="display: flex; align-items: flex-start; gap: 10px;">
                {icon_check_circle(18, COLORS['success'])}
                <div>
                    <strong>Log Returns:</strong> Stationary<br>
                    - Fluctuates around zero mean<br>
                    - Relatively constant variance<br>
                    - No persistent trends
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ─────────────────────────────────────────────────────────────────────────
    # STATIONARITY TESTS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px;">
        {icon_target(20, COLORS['secondary'])}
        <span style="color: #ffffff; font-size: 1.2rem; font-weight: 600;">Formal Stationarity Tests</span>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Prices", "Log Returns"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        # ADF test on prices
        adf_price = adf_test(df['adj_close'])
        kpss_price = kpss_test(df['adj_close'])
        
        with col1:
            st.markdown("#### ADF Test (Prices)")
            if adf_price:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">ADF Statistic</div>
                    <div class="metric-value">{adf_price['statistic']:.4f}</div>
                    <p style="color: #a0aec0; margin-top: 10px;">p-value: {adf_price['p_value']:.4f}</p>
                    <p style="color: #a0aec0;">Lags used: {adf_price['lags_used']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("**Critical Values:**")
                for key, val in adf_price['critical_values'].items():
                    st.caption(f"{key}: {val:.4f}")
                
                if adf_price['is_stationary']:
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; gap: 8px; padding: 10px; 
                                background: rgba(0, 212, 170, 0.1); border-radius: 8px;">
                        {icon_check_circle(18, COLORS['success'])}
                        <span style="color: #00D4AA;">Reject H₀: Series appears stationary</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; gap: 8px; padding: 10px; 
                                background: rgba(255, 107, 107, 0.1); border-radius: 8px;">
                        {icon_x_circle(18, COLORS['danger'])}
                        <span style="color: #ff6b6b;">Cannot reject H₀: Series has unit root (non-stationary)</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Could not perform ADF test. Install statsmodels.")
        
        with col2:
            st.markdown("#### KPSS Test (Prices)")
            if kpss_price:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">KPSS Statistic</div>
                    <div class="metric-value">{kpss_price['statistic']:.4f}</div>
                    <p style="color: #a0aec0; margin-top: 10px;">p-value: {kpss_price['p_value']:.4f}</p>
                    <p style="color: #a0aec0;">Lags used: {kpss_price['lags_used']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("**Critical Values:**")
                for key, val in kpss_price['critical_values'].items():
                    st.caption(f"{key}: {val:.4f}")
                
                if kpss_price['is_stationary']:
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; gap: 8px; padding: 10px; 
                                background: rgba(0, 212, 170, 0.1); border-radius: 8px;">
                        {icon_check_circle(18, COLORS['success'])}
                        <span style="color: #00D4AA;">Cannot reject H₀: Series appears stationary</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; gap: 8px; padding: 10px; 
                                background: rgba(255, 107, 107, 0.1); border-radius: 8px;">
                        {icon_x_circle(18, COLORS['danger'])}
                        <span style="color: #ff6b6b;">Reject H₀: Series is non-stationary</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Could not perform KPSS test. Install statsmodels.")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        # ADF test on returns
        adf_returns = adf_test(df['log_return'])
        kpss_returns = kpss_test(df['log_return'])
        
        with col1:
            st.markdown("#### ADF Test (Log Returns)")
            if adf_returns:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">ADF Statistic</div>
                    <div class="metric-value">{adf_returns['statistic']:.4f}</div>
                    <p style="color: #a0aec0; margin-top: 10px;">p-value: {adf_returns['p_value']:.6f}</p>
                    <p style="color: #a0aec0;">Lags used: {adf_returns['lags_used']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("**Critical Values:**")
                for key, val in adf_returns['critical_values'].items():
                    st.caption(f"{key}: {val:.4f}")
                
                if adf_returns['is_stationary']:
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; gap: 8px; padding: 10px; 
                                background: rgba(0, 212, 170, 0.1); border-radius: 8px;">
                        {icon_check_circle(18, COLORS['success'])}
                        <span style="color: #00D4AA;">Reject H₀: Series is stationary</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; gap: 8px; padding: 10px; 
                                background: rgba(255, 107, 107, 0.1); border-radius: 8px;">
                        {icon_x_circle(18, COLORS['danger'])}
                        <span style="color: #ff6b6b;">Cannot reject H₀: Series has unit root</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Could not perform ADF test. Install statsmodels.")
        
        with col2:
            st.markdown("#### KPSS Test (Log Returns)")
            if kpss_returns:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">KPSS Statistic</div>
                    <div class="metric-value">{kpss_returns['statistic']:.4f}</div>
                    <p style="color: #a0aec0; margin-top: 10px;">p-value: {kpss_returns['p_value']:.4f}</p>
                    <p style="color: #a0aec0;">Lags used: {kpss_returns['lags_used']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("**Critical Values:**")
                for key, val in kpss_returns['critical_values'].items():
                    st.caption(f"{key}: {val:.4f}")
                
                if kpss_returns['is_stationary']:
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; gap: 8px; padding: 10px; 
                                background: rgba(0, 212, 170, 0.1); border-radius: 8px;">
                        {icon_check_circle(18, COLORS['success'])}
                        <span style="color: #00D4AA;">Cannot reject H₀: Series is stationary</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; gap: 8px; padding: 10px; 
                                background: rgba(255, 107, 107, 0.1); border-radius: 8px;">
                        {icon_x_circle(18, COLORS['danger'])}
                        <span style="color: #ff6b6b;">Reject H₀: Series is non-stationary</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Could not perform KPSS test. Install statsmodels.")
    
    # ─────────────────────────────────────────────────────────────────────────
    # ROLLING STATISTICS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px;">
        {icon_activity(20, COLORS['primary'])}
        <span style="color: #ffffff; font-size: 1.2rem; font-weight: 600;">Rolling Statistics (Stationarity Visualization)</span>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col2:
        window = st.selectbox("Rolling Window", [63, 126, 252, 504], index=2)
    
    st.plotly_chart(create_rolling_mean_std(df, window), width='stretch')
    
    st.caption("""
    **Interpretation:** For a stationary series, the rolling mean and standard deviation 
    should remain relatively constant and close to their overall values. Large deviations 
    suggest regime changes or structural breaks.
    """)
    
    # ─────────────────────────────────────────────────────────────────────────
    # STRUCTURAL BREAKS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px;">
        {icon_crosshair(20, COLORS['warning'])}
        <span style="color: #ffffff; font-size: 1.2rem; font-weight: 600;">Structural Break Detection (CUSUM)</span>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.plotly_chart(create_cusum_chart(df), width='stretch')
    
    with col2:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
            {icon_hash(16, COLORS['info'])}
            <span style="color: #ffffff; font-weight: 500;">CUSUM Test</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        The **Cumulative Sum (CUSUM)** test detects structural breaks by 
        tracking deviations from the expected cumulative sum.
        
        **Interpretation:**
        - CUSUM within bounds → No structural break
        - CUSUM crosses bounds → Potential break point
        
        **Common causes of breaks:**
        - Market crashes / crises
        - Policy changes
        - Regime shifts
        """)
    
    # ─────────────────────────────────────────────────────────────────────────
    # SUBSAMPLE ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px;">
        {icon_bar_chart(20, COLORS['secondary'])}
        <span style="color: #ffffff; font-size: 1.2rem; font-weight: 600;">Subsample Stability Analysis</span>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
            {icon_hash(16, COLORS['secondary'])}
            <span style="color: #ffffff; font-weight: 500;">Distribution by Period</span>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(create_subsample_comparison(df), width='stretch')
    
    with col2:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
            {icon_hash(16, COLORS['secondary'])}
            <span style="color: #ffffff; font-weight: 500;">Period Statistics</span>
        </div>
        """, unsafe_allow_html=True)
        stats_table = create_subsample_stats_table(df)
        st.dataframe(
            stats_table.style.format({
                'Mean (%)': '{:.4f}',
                'Std (%)': '{:.4f}',
                'Skewness': '{:.3f}',
                'Kurtosis': '{:.3f}',
                'Sharpe': '{:.3f}'
            }),
            width='stretch',
            hide_index=True
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # VARIANCE RATIO TEST
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px;">
        {icon_target(20, COLORS['primary'])}
        <span style="color: #ffffff; font-size: 1.2rem; font-weight: 600;">Variance Ratio Test (Random Walk)</span>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        vr_results = []
        for k in [2, 5, 10, 20, 40]:
            vr = variance_ratio_test(df['log_return'], k)
            vr_results.append(vr)
        
        vr_df = pd.DataFrame(vr_results)
        vr_df.columns = ['k (Periods)', 'Variance Ratio', 'Z-Statistic', 'p-value', 'Random Walk?']
        vr_df['Random Walk?'] = vr_df['Random Walk?'].map({True: 'Yes', False: 'No'})
        
        st.dataframe(
            vr_df.style.format({
                'Variance Ratio': '{:.4f}',
                'Z-Statistic': '{:.4f}',
                'p-value': '{:.4f}'
            }),
            width='stretch',
            hide_index=True
        )
    
    with col2:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
            {icon_hash(16, COLORS['info'])}
            <span style="color: #ffffff; font-weight: 500;">Interpretation</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        The **Variance Ratio Test** checks if returns follow a random walk.
        
        - **VR = 1**: Perfect random walk
        - **VR > 1**: Positive autocorrelation (momentum)
        - **VR < 1**: Negative autocorrelation (mean reversion)
        
        **H₀**: Returns follow random walk
        **p < 0.05**: Reject random walk hypothesis
        """)
    
    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px;">
        {icon_anchor(20, COLORS['primary'])}
        <span style="color: #ffffff; font-size: 1.2rem; font-weight: 600;">Stability Analysis Summary</span>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        price_stat = "Non-Stationary" if adf_price and not adf_price['is_stationary'] else "Stationary"
        price_icon = icon_x_circle(14, COLORS['danger']) if adf_price and not adf_price['is_stationary'] else icon_check_circle(14, COLORS['success'])
        st.markdown(f"""
        <div style="background: #1a1f2e; padding: 12px; border-radius: 8px; text-align: center;">
            <div style="color: #a0aec0; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 5px;">
                Prices (ADF)
            </div>
            <div style="color: #ffffff; font-size: 1rem; font-weight: 600; display: flex; align-items: center; justify-content: center; gap: 6px;">
                {price_icon} {price_stat}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        returns_stat = "Stationary" if adf_returns and adf_returns['is_stationary'] else "Non-Stationary"
        returns_icon = icon_check_circle(14, COLORS['success']) if adf_returns and adf_returns['is_stationary'] else icon_x_circle(14, COLORS['danger'])
        st.markdown(f"""
        <div style="background: #1a1f2e; padding: 12px; border-radius: 8px; text-align: center;">
            <div style="color: #a0aec0; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 5px;">
                Returns (ADF)
            </div>
            <div style="color: #ffffff; font-size: 1rem; font-weight: 600; display: flex; align-items: center; justify-content: center; gap: 6px;">
                {returns_icon} {returns_stat}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Check for structural breaks
        cusum, upper, _ = detect_structural_breaks(df)
        breaks_detected = np.any(np.abs(cusum) > upper)
        break_stat = "Detected" if breaks_detected else "None"
        break_icon = icon_alert_triangle(14, COLORS['warning']) if breaks_detected else icon_check_circle(14, COLORS['success'])
        st.markdown(f"""
        <div style="background: #1a1f2e; padding: 12px; border-radius: 8px; text-align: center;">
            <div style="color: #a0aec0; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 5px;">
                Structural Breaks
            </div>
            <div style="color: #ffffff; font-size: 1rem; font-weight: 600; display: flex; align-items: center; justify-content: center; gap: 6px;">
                {break_icon} {break_stat}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Random walk summary
        rw_count = sum(1 for vr in vr_results if vr['is_random_walk'])
        rw_icon = icon_check_circle(14, COLORS['success']) if rw_count >= 3 else icon_alert_triangle(14, COLORS['warning'])
        st.markdown(f"""
        <div style="background: #1a1f2e; padding: 12px; border-radius: 8px; text-align: center;">
            <div style="color: #a0aec0; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 5px;">
                Random Walk Tests
            </div>
            <div style="color: #ffffff; font-size: 1rem; font-weight: 600; display: flex; align-items: center; justify-content: center; gap: 6px;">
                {rw_icon} {rw_count}/5 pass
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Final interpretation
    st.markdown(f"""
    <div class="alert-success" style="margin-top: 20px;">
        <div style="display: flex; align-items: flex-start; gap: 10px;">
            {icon_check_circle(20, COLORS['success'])}
            <div>
                <strong>Key Takeaways:</strong><br>
                • <strong>Prices</strong> are typically non-stationary (unit root) - need to difference for modeling<br>
                • <strong>Log returns</strong> are typically stationary - suitable for most time series models<br>
                • Variance ratio close to 1 supports weak-form market efficiency<br>
                • No structural breaks suggests stable distribution over time
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)