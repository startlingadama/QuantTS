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
    icon_link, icon_line_chart, icon_bar_chart, icon_activity,
    icon_scatter, icon_scale, icon_alert_triangle, icon_check_circle,
    icon_hash, icon_target, icon_info, icon_git_branch, COLORS, ICON_MD
)

# ═══════════════════════════════════════════════════════════════════════════════
#                              DEPENDENCY ANALYSIS
#                       (Autocorrelation & Time Dependencies)
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    """Load returns data."""
    data_path = Path("data/clean/yahoo/returns.parquet")
    if data_path.exists():
        df = pd.read_parquet(data_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    return None

def calculate_acf(series, nlags=40):
    """Calculate autocorrelation function."""
    n = len(series)
    mean = series.mean()
    var = np.sum((series - mean) ** 2) / n
    
    acf = []
    for k in range(nlags + 1):
        if k == 0:
            acf.append(1.0)
        else:
            cov = np.sum((series[k:] - mean) * (series[:-k] - mean)) / n
            acf.append(cov / var if var != 0 else 0)
    
    return np.array(acf)

def calculate_pacf(series, nlags=40):
    """Calculate partial autocorrelation function using Yule-Walker."""
    acf = calculate_acf(series, nlags)
    pacf = [1.0, acf[1]]
    
    for k in range(2, nlags + 1):
        # Yule-Walker equations
        phi = np.zeros(k)
        for j in range(k):
            if j == 0:
                phi[j] = acf[k]
            else:
                phi[j] = acf[k - j]
        
        # Solve for PACF
        toeplitz = np.zeros((k-1, k-1))
        for i in range(k-1):
            for j in range(k-1):
                toeplitz[i, j] = acf[abs(i - j)]
        
        try:
            rhs = acf[1:k]
            coeffs = np.linalg.solve(toeplitz, rhs)
            pacf_val = (acf[k] - np.dot(coeffs, acf[k-1:0:-1])) / (1 - np.dot(coeffs, acf[1:k]))
            pacf.append(pacf_val if not np.isnan(pacf_val) else 0)
        except:
            pacf.append(0)
    
    return np.array(pacf)

def create_acf_plot(df, series_type='returns'):
    """Create ACF plot."""
    if series_type == 'returns':
        series = df['log_return'].dropna()
        title = 'Returns'
    elif series_type == 'abs_returns':
        series = df['log_return'].abs().dropna()
        title = 'Absolute Returns'
    else:
        series = (df['log_return'] ** 2).dropna()
        title = 'Squared Returns'
    
    nlags = 40
    acf = calculate_acf(series, nlags)
    
    # Confidence interval (95%)
    n = len(series)
    conf_int = 1.96 / np.sqrt(n)
    
    fig = go.Figure()
    
    # ACF bars
    colors = ['#00D4AA' if abs(a) <= conf_int else '#ff6b6b' for a in acf[1:]]
    
    fig.add_trace(go.Bar(
        x=list(range(1, nlags + 1)),
        y=acf[1:],
        marker_color=colors,
        name='ACF'
    ))
    
    # Confidence bands
    fig.add_hline(y=conf_int, line_dash="dash", line_color="#feca57", 
                  annotation_text="95% CI")
    fig.add_hline(y=-conf_int, line_dash="dash", line_color="#feca57")
    fig.add_hline(y=0, line_color="white", line_width=0.5)
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,31,46,1)',
        height=300,
        margin=dict(l=50, r=50, t=40, b=30),
        title=dict(text=f'ACF - {title}', font=dict(size=14)),
        xaxis_title='Lag',
        yaxis_title='Autocorrelation',
        showlegend=False,
        font=dict(color='#a0aec0')
    )
    
    fig.update_xaxes(gridcolor='rgba(45,55,72,0.5)')
    fig.update_yaxes(gridcolor='rgba(45,55,72,0.5)', range=[-0.3, 0.3])
    
    return fig

def create_pacf_plot(df, series_type='returns'):
    """Create PACF plot."""
    if series_type == 'returns':
        series = df['log_return'].dropna()
        title = 'Returns'
    elif series_type == 'abs_returns':
        series = df['log_return'].abs().dropna()
        title = 'Absolute Returns'
    else:
        series = (df['log_return'] ** 2).dropna()
        title = 'Squared Returns'
    
    nlags = 40
    
    # Use statsmodels if available, otherwise use our implementation
    try:
        from statsmodels.tsa.stattools import pacf
        pacf_vals = pacf(series, nlags=nlags)
    except:
        pacf_vals = calculate_pacf(series, nlags)
    
    # Confidence interval
    n = len(series)
    conf_int = 1.96 / np.sqrt(n)
    
    fig = go.Figure()
    
    # PACF bars
    colors = ['#667eea' if abs(p) <= conf_int else '#ff6b6b' for p in pacf_vals[1:]]
    
    fig.add_trace(go.Bar(
        x=list(range(1, nlags + 1)),
        y=pacf_vals[1:],
        marker_color=colors,
        name='PACF'
    ))
    
    # Confidence bands
    fig.add_hline(y=conf_int, line_dash="dash", line_color="#feca57")
    fig.add_hline(y=-conf_int, line_dash="dash", line_color="#feca57")
    fig.add_hline(y=0, line_color="white", line_width=0.5)
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,31,46,1)',
        height=300,
        margin=dict(l=50, r=50, t=40, b=30),
        title=dict(text=f'PACF - {title}', font=dict(size=14)),
        xaxis_title='Lag',
        yaxis_title='Partial Autocorrelation',
        showlegend=False,
        font=dict(color='#a0aec0')
    )
    
    fig.update_xaxes(gridcolor='rgba(45,55,72,0.5)')
    fig.update_yaxes(gridcolor='rgba(45,55,72,0.5)', range=[-0.3, 0.3])
    
    return fig

def create_lag_scatter(df, lag=1):
    """Create lag scatter plot."""
    returns = df['log_return'].dropna() * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=returns[:-lag],
        y=returns[lag:],
        mode='markers',
        marker=dict(
            color='#667eea',
            size=4,
            opacity=0.4
        ),
        name=f'Lag-{lag}'
    ))
    
    # Correlation
    corr = returns[:-lag].corr(returns[lag:].reset_index(drop=True))
    
    # Add regression line
    z = np.polyfit(returns[:-lag], returns[lag:].values, 1)
    p = np.poly1d(z)
    x_line = np.linspace(returns.min(), returns.max(), 100)
    
    fig.add_trace(go.Scatter(
        x=x_line,
        y=p(x_line),
        mode='lines',
        line=dict(color='#00D4AA', width=2),
        name=f'Trend (ρ={corr:.4f})'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,31,46,1)',
        height=350,
        margin=dict(l=50, r=50, t=30, b=30),
        xaxis_title=f'Return(t)',
        yaxis_title=f'Return(t+{lag})',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        font=dict(color='#a0aec0')
    )
    
    fig.update_xaxes(gridcolor='rgba(45,55,72,0.5)')
    fig.update_yaxes(gridcolor='rgba(45,55,72,0.5)')
    
    return fig, corr

def create_volatility_clustering_chart(df):
    """Create volatility clustering visualization."""
    returns = df['log_return'].dropna()
    abs_returns = returns.abs()
    
    # ACF of absolute returns (volatility proxy)
    nlags = 100
    acf_abs = calculate_acf(abs_returns, nlags)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(1, nlags + 1)),
        y=acf_abs[1:],
        mode='lines+markers',
        line=dict(color='#00D4AA', width=2),
        marker=dict(size=4),
        name='ACF of |Returns|',
        fill='tozeroy',
        fillcolor='rgba(0,212,170,0.1)'
    ))
    
    # Confidence band
    n = len(returns)
    conf_int = 1.96 / np.sqrt(n)
    
    fig.add_hline(y=conf_int, line_dash="dash", line_color="#feca57", 
                  annotation_text="95% CI")
    fig.add_hline(y=0, line_color="white", line_width=0.5)
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,31,46,1)',
        height=350,
        margin=dict(l=50, r=50, t=30, b=30),
        xaxis_title='Lag (Days)',
        yaxis_title='Autocorrelation',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        font=dict(color='#a0aec0')
    )
    
    fig.update_xaxes(gridcolor='rgba(45,55,72,0.5)')
    fig.update_yaxes(gridcolor='rgba(45,55,72,0.5)')
    
    return fig

def create_leverage_effect_chart(df):
    """Create leverage effect visualization."""
    returns = df['log_return'].dropna()
    
    # Calculate cross-correlations between returns and future squared returns
    lags = range(-20, 21)
    cross_corr = []
    
    for lag in lags:
        if lag < 0:
            # Past returns vs current vol
            corr = returns[:lag].corr((returns.shift(lag) ** 2).dropna()[:lag])
        elif lag == 0:
            corr = returns.corr(returns ** 2)
        else:
            # Current returns vs future vol
            corr = returns[:-lag].corr((returns ** 2).shift(-lag).dropna()[:-lag])
        cross_corr.append(corr if not np.isnan(corr) else 0)
    
    fig = go.Figure()
    
    colors = ['#ff6b6b' if l < 0 else '#00D4AA' for l in lags]
    
    fig.add_trace(go.Bar(
        x=list(lags),
        y=cross_corr,
        marker_color=colors,
        name='Cross-correlation'
    ))
    
    fig.add_hline(y=0, line_color="white", line_width=0.5)
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,31,46,1)',
        height=300,
        margin=dict(l=50, r=50, t=30, b=30),
        xaxis_title='Lag (Days)',
        yaxis_title='Cross-correlation',
        showlegend=False,
        font=dict(color='#a0aec0'),
        annotations=[
            dict(x=-10, y=max(cross_corr) * 0.8, text="Past Returns", 
                 showarrow=False, font=dict(color='#ff6b6b')),
            dict(x=10, y=max(cross_corr) * 0.8, text="Future Vol", 
                 showarrow=False, font=dict(color='#00D4AA'))
        ]
    )
    
    fig.update_xaxes(gridcolor='rgba(45,55,72,0.5)')
    fig.update_yaxes(gridcolor='rgba(45,55,72,0.5)')
    
    return fig

def ljung_box_test(df, lags=[10, 20, 40]):
    """Perform Ljung-Box test."""
    returns = df['log_return'].dropna()
    
    results = []
    for lag in lags:
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_result = acorr_ljungbox(returns, lags=[lag], return_df=True)
            lb_stat = lb_result['lb_stat'].values[0]
            lb_p = lb_result['lb_pvalue'].values[0]
        except:
            # Manual calculation
            n = len(returns)
            acf = calculate_acf(returns, lag)
            lb_stat = n * (n + 2) * sum([acf[k]**2 / (n - k) for k in range(1, lag + 1)])
            lb_p = 1 - stats.chi2.cdf(lb_stat, lag)
        
        results.append({
            'Lag': lag,
            'LB Statistic': lb_stat,
            'p-value': lb_p,
            'Significant': '✓' if lb_p < 0.05 else '✗'
        })
    
    return pd.DataFrame(results)

def run():
    """Main function to run the Dependency Analysis page."""
    
    # Header with icon
    st.markdown(f"""
    <div style="margin-bottom: 30px;">
        <h1 style="color: #ffffff; font-weight: 700; margin-bottom: 5px; display: flex; align-items: center; gap: 12px;">
            {icon_link(32, COLORS['secondary'])} Dependency Analysis
        </h1>
        <p style="color: #a0aec0; font-size: 1.1rem;">
            Autocorrelation, volatility clustering, and time dependencies
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
    # STYLIZED FACTS INFO
    # ─────────────────────────────────────────────────────────────────────────
    with st.expander("ℹ️ About Stylized Facts of Financial Returns", expanded=False):
        st.markdown("""
        Financial returns exhibit several well-documented statistical regularities (stylized facts):
        
        1. **Absence of autocorrelation in returns**: Linear autocorrelations of returns are negligible, 
           except for very short intervals (market microstructure effects).
        
        2. **Volatility clustering**: Large changes tend to be followed by large changes, 
           small changes by small changes. ACF of |returns| is significantly positive for many lags.
        
        3. **Leverage effect**: Negative returns tend to increase future volatility more than 
           positive returns of the same magnitude.
        
        4. **Heavy tails**: Return distributions have heavier tails than normal distribution 
           (excess kurtosis).
        
        5. **Gain/Loss asymmetry**: Large drawdowns are not matched by equally large upward movements.
        """)
    
    st.markdown("---")
    
    # ─────────────────────────────────────────────────────────────────────────
    # ACF / PACF ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px;">
        {icon_bar_chart(20, COLORS['primary'])}
        <span style="color: #ffffff; font-size: 1.2rem; font-weight: 600;">Autocorrelation Analysis</span>
    </div>
    """, unsafe_allow_html=True)
    
    tabs = st.tabs(["Returns", "Absolute Returns", "Squared Returns"])
    
    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_acf_plot(df, 'returns'), width='stretch')
        with col2:
            st.plotly_chart(create_pacf_plot(df, 'returns'), width='stretch')
        
        st.markdown(f"""
        <div class="alert-success">
            <div style="display: flex; align-items: flex-start; gap: 10px;">
                {icon_check_circle(18, COLORS['success'])}
                <div>
                    <strong>Interpretation:</strong> Returns typically show little to no autocorrelation, 
                    consistent with weak-form market efficiency. Bars outside confidence bands suggest 
                    potential predictability.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_acf_plot(df, 'abs_returns'), width='stretch')
        with col2:
            st.plotly_chart(create_pacf_plot(df, 'abs_returns'), width='stretch')
        
        st.markdown(f"""
        <div class="alert-warning">
            <div style="display: flex; align-items: flex-start; gap: 10px;">
                {icon_alert_triangle(18, COLORS['warning'])}
                <div>
                    <strong>Interpretation:</strong> Absolute returns (volatility proxy) typically show 
                    significant positive autocorrelation for many lags. This is the famous 
                    <strong>volatility clustering</strong> effect.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_acf_plot(df, 'squared_returns'), width='stretch')
        with col2:
            st.plotly_chart(create_pacf_plot(df, 'squared_returns'), width='stretch')
        
        st.markdown(f"""
        <div class="alert-warning">
            <div style="display: flex; align-items: flex-start; gap: 10px;">
                {icon_alert_triangle(18, COLORS['warning'])}
                <div>
                    <strong>Interpretation:</strong> Squared returns also exhibit significant 
                    autocorrelation, supporting ARCH/GARCH modeling approaches for volatility.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ─────────────────────────────────────────────────────────────────────────
    # LAG SCATTER PLOTS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px;">
        {icon_scatter(20, COLORS['secondary'])}
        <span style="color: #ffffff; font-size: 1.2rem; font-weight: 600;">Lag Scatter Plots</span>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig1, corr1 = create_lag_scatter(df, 1)
        st.plotly_chart(fig1, width ='stretch')
        st.caption(f"Lag-1 Correlation: ρ = {corr1:.4f}")
    
    with col2:
        fig5, corr5 = create_lag_scatter(df, 5)
        st.plotly_chart(fig5, width ='stretch')
        st.caption(f"Lag-5 Correlation: ρ = {corr5:.4f}")
    
    with col3:
        fig20, corr20 = create_lag_scatter(df, 20)
        st.plotly_chart(fig20, width='stretch')
        st.caption(f"Lag-20 Correlation: ρ = {corr20:.4f}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # VOLATILITY CLUSTERING
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px;">
        {icon_activity(20, COLORS['primary'])}
        <span style="color: #ffffff; font-size: 1.2rem; font-weight: 600;">Volatility Clustering</span>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.plotly_chart(create_volatility_clustering_chart(df), width='stretch')
    
    with col2:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
            {icon_hash(16, COLORS['info'])}
            <span style="color: #ffffff; font-weight: 500;">Key Findings</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **Volatility Clustering** is one of the most robust stylized facts:
        
        - Large price movements tend to cluster together
        - ACF of absolute returns decays slowly (long memory)
        - Justifies GARCH-type models for volatility
        - Important for risk management and option pricing
        
        The slow decay of autocorrelation in |returns| suggests **long memory** 
        in the volatility process.
        """)
    
    # ─────────────────────────────────────────────────────────────────────────
    # LEVERAGE EFFECT
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px;">
        {icon_scale(20, COLORS['warning'])}
        <span style="color: #ffffff; font-size: 1.2rem; font-weight: 600;">Leverage Effect</span>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.plotly_chart(create_leverage_effect_chart(df), width='stretch')
    
    with col2:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
            {icon_hash(16, COLORS['info'])}
            <span style="color: #ffffff; font-weight: 500;">Understanding Leverage Effect</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        The **leverage effect** refers to the negative correlation between 
        returns and future volatility:
        
        - **Negative returns** → Higher future volatility
        - **Positive returns** → Lower future volatility
        
        This asymmetry is captured by models like:
        - EGARCH (Exponential GARCH)
        - GJR-GARCH
        - TGARCH
        
        The chart shows cross-correlation between returns at time t 
        and squared returns at time t+k.
        """)
    
    # ─────────────────────────────────────────────────────────────────────────
    # LJUNG-BOX TEST
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px;">
        {icon_target(20, COLORS['primary'])}
        <span style="color: #ffffff; font-size: 1.2rem; font-weight: 600;">Ljung-Box Test for Serial Correlation</span>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
            {icon_hash(16, COLORS['secondary'])}
            <span style="color: #ffffff; font-weight: 500;">Returns</span>
        </div>
        """, unsafe_allow_html=True)
        lb_results = ljung_box_test(df)
        st.dataframe(lb_results, width='stretch', hide_index=True)
    
    with col2:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
            {icon_hash(16, COLORS['info'])}
            <span style="color: #ffffff; font-weight: 500;">Interpretation</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        The **Ljung-Box test** checks whether autocorrelations up to lag k 
        are jointly zero.
        
        - **H₀**: No serial correlation up to lag k
        - **H₁**: At least one autocorrelation is non-zero
        
        **p-value < 0.05**: Reject H₀ → Significant autocorrelation exists
        
        For returns, we expect **no significant autocorrelation** (market efficiency).
        For |returns| or returns², we expect **significant autocorrelation** 
        (volatility clustering).
        """)
    
    # Summary
    st.markdown("---")
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px;">
        {icon_bar_chart(20, COLORS['primary'])}
        <span style="color: #ffffff; font-size: 1.2rem; font-weight: 600;">Dependency Analysis Summary</span>
    </div>
    """, unsafe_allow_html=True)
    
    returns = df['log_return'].dropna()
    acf_returns = calculate_acf(returns, 1)[1]
    acf_abs = calculate_acf(returns.abs(), 1)[1]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Lag-1 ACF (Returns)", f"{acf_returns:.4f}")
    with col2:
        st.metric("Lag-1 ACF (|Returns|)", f"{acf_abs:.4f}")
    with col3:
        vol_clustering = "Present" if acf_abs > 0.1 else "Weak"
        vol_icon = icon_check_circle(14, COLORS['success']) if acf_abs > 0.1 else icon_alert_triangle(14, COLORS['warning'])
        st.markdown(f"""
        <div style="background: #1a1f2e; padding: 12px; border-radius: 8px; text-align: center;">
            <div style="color: #a0aec0; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 5px;">
                Volatility Clustering
            </div>
            <div style="color: #ffffff; font-size: 1.5rem; font-weight: 600; display: flex; align-items: center; justify-content: center; gap: 6px;">
                {vol_icon} {vol_clustering}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        market_eff = "Consistent" if abs(acf_returns) < 0.05 else "Anomaly"
        eff_icon = icon_check_circle(14, COLORS['success']) if abs(acf_returns) < 0.05 else icon_alert_triangle(14, COLORS['warning'])
        st.markdown(f"""
        <div style="background: #1a1f2e; padding: 12px; border-radius: 8px; text-align: center;">
            <div style="color: #a0aec0; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 5px;">
                Market Efficiency
            </div>
            <div style="color: #ffffff; font-size: 1.5rem; font-weight: 600; display: flex; align-items: center; justify-content: center; gap: 6px;">
                {eff_icon} {market_eff}
            </div>
        </div>
        """, unsafe_allow_html=True)