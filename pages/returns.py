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
    icon_trending_up, icon_trending_down, icon_bar_chart, icon_line_chart,
    icon_activity, icon_target, icon_percent, icon_calendar, icon_hash,
    icon_alert_triangle, icon_check_circle, icon_scatter, COLORS, ICON_MD
)

# ═══════════════════════════════════════════════════════════════════════════════
#                              RETURNS ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    """Load returns data."""
    data_path = Path("data/clean/yahoo/returns.parquet")
    if data_path.exists():
        df = pd.read_parquet(data_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    return None

def create_returns_timeseries(df):
    """Create returns time series chart."""
    fig = go.Figure()
    
    returns = df['log_return'] * 100
    
    # Color based on positive/negative
    colors = ['#00D4AA' if r >= 0 else '#ff6b6b' for r in returns]
    
    fig.add_trace(go.Bar(
        x=df['date'],
        y=returns,
        marker_color=colors,
        name='Daily Returns',
        opacity=0.8
    ))
    
    # Add rolling mean
    rolling_mean = returns.rolling(21).mean()
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=rolling_mean,
        name='21-Day MA',
        line=dict(color='#feca57', width=2)
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,31,46,1)',
        height=400,
        margin=dict(l=50, r=50, t=30, b=30),
        xaxis_title='Date',
        yaxis_title='Return (%)',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(color='#a0aec0')
    )
    
    fig.update_xaxes(gridcolor='rgba(45,55,72,0.5)')
    fig.update_yaxes(gridcolor='rgba(45,55,72,0.5)')
    
    return fig

def create_qq_plot(df):
    """Create Q-Q plot for returns."""
    returns = df['log_return'].dropna()
    
    # Calculate theoretical quantiles
    (osm, osr), (slope, intercept, r) = stats.probplot(returns, dist="norm")
    
    fig = go.Figure()
    
    # Scatter points
    fig.add_trace(go.Scatter(
        x=osm,
        y=osr,
        mode='markers',
        name='Sample Quantiles',
        marker=dict(color='#667eea', size=5, opacity=0.6)
    ))
    
    # Reference line
    line_x = np.array([osm.min(), osm.max()])
    line_y = slope * line_x + intercept
    
    fig.add_trace(go.Scatter(
        x=line_x,
        y=line_y,
        mode='lines',
        name='Normal Reference',
        line=dict(color='#00D4AA', width=2, dash='dash')
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,31,46,1)',
        height=350,
        margin=dict(l=50, r=50, t=30, b=30),
        xaxis_title='Theoretical Quantiles',
        yaxis_title='Sample Quantiles',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        font=dict(color='#a0aec0')
    )
    
    fig.update_xaxes(gridcolor='rgba(45,55,72,0.5)')
    fig.update_yaxes(gridcolor='rgba(45,55,72,0.5)')
    
    return fig

def create_rolling_stats(df, window=21):
    """Create rolling statistics chart."""
    returns = df['log_return']
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Rolling Mean', 'Rolling Std', 'Rolling Sharpe')
    )
    
    # Rolling mean
    rolling_mean = returns.rolling(window).mean() * 100 * 252
    fig.add_trace(go.Scatter(
        x=df['date'], y=rolling_mean,
        name=f'{window}D Mean',
        line=dict(color='#00D4AA', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(0,212,170,0.1)'
    ), row=1, col=1)
    
    # Rolling std
    rolling_std = returns.rolling(window).std() * 100 * np.sqrt(252)
    fig.add_trace(go.Scatter(
        x=df['date'], y=rolling_std,
        name=f'{window}D Volatility',
        line=dict(color='#ff6b6b', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(255,107,107,0.1)'
    ), row=2, col=1)
    
    # Rolling Sharpe
    rolling_sharpe = (returns.rolling(window).mean() / returns.rolling(window).std()) * np.sqrt(252)
    fig.add_trace(go.Scatter(
        x=df['date'], y=rolling_sharpe,
        name=f'{window}D Sharpe',
        line=dict(color='#667eea', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(102,126,234,0.1)'
    ), row=3, col=1)
    
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

def create_returns_heatmap(df):
    """Create monthly returns heatmap."""
    df_copy = df.copy()
    df_copy['year'] = df_copy['date'].dt.year
    df_copy['month'] = df_copy['date'].dt.month
    
    # Calculate monthly returns
    monthly = df_copy.groupby(['year', 'month'])['log_return'].sum().unstack()
    monthly = monthly * 100  # Convert to percentage
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig = go.Figure(data=go.Heatmap(
        z=monthly.values,
        x=month_names[:monthly.shape[1]],
        y=monthly.index.astype(str),
        colorscale=[
            [0, '#ff6b6b'],
            [0.5, '#1a1f2e'],
            [1, '#00D4AA']
        ],
        text=np.round(monthly.values, 1),
        texttemplate='%{text}%',
        textfont=dict(size=10, color='white'),
        hovertemplate='Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>',
        colorbar=dict(
            title=dict(text='Return %', side='right'),
            tickformat='.1f'
        )
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,31,46,1)',
        height=400,
        margin=dict(l=50, r=50, t=30, b=30),
        xaxis_title='Month',
        yaxis_title='Year',
        font=dict(color='#a0aec0')
    )
    
    return fig

def create_histogram_comparison(df):
    """Create histogram comparing periods."""
    returns = df['log_return'] * 100
    
    # Split into periods
    mid_point = len(df) // 2
    first_half = returns[:mid_point]
    second_half = returns[mid_point:]
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=first_half,
        name=f'First Half ({df["date"].iloc[0].year}-{df["date"].iloc[mid_point].year})',
        opacity=0.6,
        marker_color='#667eea',
        nbinsx=50
    ))
    
    fig.add_trace(go.Histogram(
        x=second_half,
        name=f'Second Half ({df["date"].iloc[mid_point].year}-{df["date"].iloc[-1].year})',
        opacity=0.6,
        marker_color='#00D4AA',
        nbinsx=50
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,31,46,1)',
        height=350,
        margin=dict(l=50, r=50, t=30, b=30),
        barmode='overlay',
        xaxis_title='Return (%)',
        yaxis_title='Frequency',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        font=dict(color='#a0aec0')
    )
    
    fig.update_xaxes(gridcolor='rgba(45,55,72,0.5)')
    fig.update_yaxes(gridcolor='rgba(45,55,72,0.5)')
    
    return fig

def calculate_return_stats(df):
    """Calculate detailed return statistics."""
    returns = df['log_return']
    
    # Basic stats
    mean_daily = returns.mean() * 100
    std_daily = returns.std() * 100
    mean_annual = mean_daily * 252
    std_annual = std_daily * np.sqrt(252)
    
    # Distribution stats
    skew = stats.skew(returns.dropna())
    kurt = stats.kurtosis(returns.dropna())
    
    # Normality tests
    jb_stat, jb_p = stats.jarque_bera(returns.dropna())
    shapiro_stat, shapiro_p = stats.shapiro(returns.dropna()[:5000]) if len(returns) > 5000 else stats.shapiro(returns.dropna())
    
    # Positive/Negative days
    pos_days = (returns > 0).sum()
    neg_days = (returns < 0).sum()
    pos_ratio = pos_days / len(returns) * 100
    
    # Best/Worst
    best_day = returns.max() * 100
    worst_day = returns.min() * 100
    best_date = df.loc[returns.idxmax(), 'date']
    worst_date = df.loc[returns.idxmin(), 'date']
    
    return {
        'mean_daily': mean_daily,
        'std_daily': std_daily,
        'mean_annual': mean_annual,
        'std_annual': std_annual,
        'skewness': skew,
        'kurtosis': kurt,
        'jb_stat': jb_stat,
        'jb_p': jb_p,
        'shapiro_p': shapiro_p,
        'pos_days': pos_days,
        'neg_days': neg_days,
        'pos_ratio': pos_ratio,
        'best_day': best_day,
        'worst_day': worst_day,
        'best_date': best_date,
        'worst_date': worst_date
    }

def run():
    """Main function to run the Returns Analysis page."""
    
    # Header with icon
    st.markdown(f"""
    <div style="margin-bottom: 30px;">
        <h1 style="color: #ffffff; font-weight: 700; margin-bottom: 5px; display: flex; align-items: center; gap: 12px;">
            {icon_trending_up(32, COLORS['success'])} Returns Analysis
        </h1>
        <p style="color: #a0aec0; font-size: 1.1rem;">
            Deep dive into return characteristics and statistical properties
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
    
    # Calculate statistics
    stats_dict = calculate_return_stats(df)
    
    # ─────────────────────────────────────────────────────────────────────────
    # TOP METRICS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px;">
        {icon_percent(20, COLORS['primary'])}
        <span style="color: #ffffff; font-size: 1.2rem; font-weight: 600;">Return Statistics</span>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Mean (Daily)", f"{stats_dict['mean_daily']:.4f}%")
    with col2:
        st.metric("Std (Daily)", f"{stats_dict['std_daily']:.4f}%")
    with col3:
        st.metric("Mean (Annual)", f"{stats_dict['mean_annual']:.2f}%")
    with col4:
        st.metric("Vol (Annual)", f"{stats_dict['std_annual']:.2f}%")
    with col5:
        st.metric("Win Rate", f"{stats_dict['pos_ratio']:.1f}%")
    with col6:
        st.metric("Skewness", f"{stats_dict['skewness']:.3f}")
    
    st.markdown("---")
    
    # ─────────────────────────────────────────────────────────────────────────
    # RETURNS TIME SERIES
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px;">
        {icon_line_chart(20, COLORS['primary'])}
        <span style="color: #ffffff; font-size: 1.2rem; font-weight: 600;">Daily Returns Time Series</span>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([4, 1])
    with col2:
        show_last = st.selectbox(
            "Show last",
            ["3 months", "6 months", "1 year", "3 years", "All"],
            index=2
        )
    
    range_map = {"3 months": 63, "6 months": 126, "1 year": 252, "3 years": 756, "All": len(df)}
    n_days = range_map.get(show_last, 252)
    df_filtered = df.tail(n_days).copy()
    
    st.plotly_chart(create_returns_timeseries(df_filtered), width='stretch')
    
    # ─────────────────────────────────────────────────────────────────────────
    # ROLLING STATISTICS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px;">
        {icon_activity(20, COLORS['secondary'])}
        <span style="color: #ffffff; font-size: 1.2rem; font-weight: 600;">Rolling Statistics</span>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([4, 1])
    with col2:
        window = st.slider("Window (days)", 5, 252, 21)
    
    st.plotly_chart(create_rolling_stats(df, window), width='stretch')
    
    # ─────────────────────────────────────────────────────────────────────────
    # DISTRIBUTION ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px;">
        {icon_hash(20, COLORS['primary'])}
        <span style="color: #ffffff; font-size: 1.2rem; font-weight: 600;">Distribution Analysis</span>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
            {icon_scatter(16, COLORS['secondary'])}
            <span style="color: #ffffff; font-weight: 500;">Q-Q Plot (Normality Check)</span>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(create_qq_plot(df), width='stretch')
        
        # Normality test results
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
            {icon_target(16, COLORS['info'])}
            <span style="color: #ffffff; font-weight: 500;">Normality Tests</span>
        </div>
        """, unsafe_allow_html=True)
        if stats_dict['jb_p'] < 0.05:
            st.markdown(f"""
            <div class="alert-warning">
                <div style="display: flex; align-items: flex-start; gap: 10px;">
                    {icon_alert_triangle(18, COLORS['warning'])}
                    <div>
                        <strong>Jarque-Bera Test:</strong> p-value = {stats_dict['jb_p']:.4e}<br>
                        Returns are NOT normally distributed (reject H₀ at 5%)
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-success">
                <div style="display: flex; align-items: flex-start; gap: 10px;">
                    {icon_check_circle(18, COLORS['success'])}
                    <div>
                        <strong>Jarque-Bera Test:</strong> p-value = {stats_dict['jb_p']:.4f}<br>
                        Cannot reject normality at 5% level
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
            {icon_bar_chart(16, COLORS['secondary'])}
            <span style="color: #ffffff; font-weight: 500;">Period Comparison</span>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(create_histogram_comparison(df), width='stretch')
        
        # Distribution shape
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
            {icon_hash(16, COLORS['info'])}
            <span style="color: #ffffff; font-weight: 500;">Distribution Shape</span>
        </div>
        """, unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            skew_interpretation = "Left-tailed" if stats_dict['skewness'] < 0 else "Right-tailed"
            st.info(f"**Skewness:** {stats_dict['skewness']:.3f}\n\n{skew_interpretation} distribution")
        with col_b:
            kurt_interpretation = "Heavy tails (leptokurtic)" if stats_dict['kurtosis'] > 0 else "Light tails (platykurtic)"
            st.info(f"**Excess Kurtosis:** {stats_dict['kurtosis']:.3f}\n\n{kurt_interpretation}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # MONTHLY RETURNS HEATMAP
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px;">
        {icon_calendar(20, COLORS['primary'])}
        <span style="color: #ffffff; font-size: 1.2rem; font-weight: 600;">Monthly Returns Heatmap</span>
    </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(create_returns_heatmap(df), width='stretch')
    
    # ─────────────────────────────────────────────────────────────────────────
    # EXTREMES TABLE
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px;">
        {icon_target(20, COLORS['warning'])}
        <span style="color: #ffffff; font-size: 1.2rem; font-weight: 600;">Extreme Returns</span>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
            {icon_trending_up(16, COLORS['success'])}
            <span style="color: #ffffff; font-weight: 500;">Best Days</span>
        </div>
        """, unsafe_allow_html=True)
        best_days = df.nlargest(10, 'log_return')[['date', 'log_return', 'close']].copy()
        best_days['log_return'] = best_days['log_return'] * 100
        best_days.columns = ['Date', 'Return (%)', 'Close Price']
        best_days['Date'] = best_days['Date'].dt.strftime('%Y-%m-%d')
        st.dataframe(best_days, width='stretch', hide_index=True)
    
    with col2:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
            {icon_trending_down(16, COLORS['danger'])}
            <span style="color: #ffffff; font-weight: 500;">Worst Days</span>
        </div>
        """, unsafe_allow_html=True)
        worst_days = df.nsmallest(10, 'log_return')[['date', 'log_return', 'close']].copy()
        worst_days['log_return'] = worst_days['log_return'] * 100
        worst_days.columns = ['Date', 'Return (%)', 'Close Price']
        worst_days['Date'] = worst_days['Date'].dt.strftime('%Y-%m-%d')
        st.dataframe(worst_days, width='stretch', hide_index=True)
    
    # Summary statistics table
    st.markdown("---")
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px;">
        {icon_bar_chart(20, COLORS['primary'])}
        <span style="color: #ffffff; font-size: 1.2rem; font-weight: 600;">Summary Statistics</span>
    </div>
    """, unsafe_allow_html=True)
    
    summary_df = pd.DataFrame({
        'Statistic': ['Mean (Daily)', 'Std Dev (Daily)', 'Mean (Annual)', 'Volatility (Annual)',
                      'Skewness', 'Kurtosis', 'Best Day', 'Worst Day', 
                      'Positive Days', 'Negative Days', 'Win Rate'],
        'Value': [f"{stats_dict['mean_daily']:.4f}%", f"{stats_dict['std_daily']:.4f}%",
                  f"{stats_dict['mean_annual']:.2f}%", f"{stats_dict['std_annual']:.2f}%",
                  f"{stats_dict['skewness']:.4f}", f"{stats_dict['kurtosis']:.4f}",
                  f"{stats_dict['best_day']:.2f}% ({stats_dict['best_date'].strftime('%Y-%m-%d')})",
                  f"{stats_dict['worst_day']:.2f}% ({stats_dict['worst_date'].strftime('%Y-%m-%d')})",
                  f"{stats_dict['pos_days']:,}", f"{stats_dict['neg_days']:,}",
                  f"{stats_dict['pos_ratio']:.1f}%"]
    })
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.dataframe(summary_df, width='stretch', hide_index=True)