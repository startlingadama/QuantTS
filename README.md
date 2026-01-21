# QuantTS-Core

**Quantitative Time Series Analysis Dashboard**

A professional-grade analytics platform for quantitative finance, featuring statistical analysis, volatility modeling, and market diagnostics.

---

## Overview

QuantTS-Core is a Streamlit-based dashboard designed for quantitative analysts and traders. It provides comprehensive tools for analyzing financial time series data with a focus on:

- **Returns Analysis** — Distribution testing, rolling statistics, and performance metrics
- **Volatility Modeling** — GARCH estimation, regime detection, and term structure
- **Dependency Analysis** — ACF/PACF, volatility clustering, and leverage effects
- **Stationarity Testing** — ADF, KPSS, structural breaks, and variance ratio tests

---

## Features

### Market Overview
- Real-time price visualization with candlestick charts
- Key performance indicators (returns, volatility, Sharpe ratio)
- Returns distribution with normality testing

### Returns Analysis
- Log returns time series with regime highlighting
- Rolling mean and volatility windows
- Monthly returns heatmap
- Statistical distribution tests (Jarque-Bera, Shapiro-Wilk)

### Volatility Analysis
- GARCH(1,1) model estimation
- Volatility regime classification (Low/Medium/High/Extreme)
- Realized vs implied volatility comparison
- Term structure analysis

### Dependency Structure
- Autocorrelation and partial autocorrelation functions
- Ljung-Box test for serial correlation
- Volatility clustering detection
- Leverage effect analysis (asymmetric volatility)

### Stability Tests
- Augmented Dickey-Fuller test
- KPSS stationarity test
- Structural break detection (Chow test, CUSUM)
- Variance ratio tests

---

## Installation

```bash
# Clone the repository
git clone https://github.com/startlingadama/QuantTS.git
cd QuantTS

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
# Run the dashboard
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`

---

## Project Structure

```
QuantTS-core/
├── app.py                 # Main application entry point
├── components/
│   ├── __init__.py
│   └── icons.py           # SVG icon library
├── pages/
│   ├── overview.py        # Market overview dashboard
│   ├── returns.py         # Returns analysis
│   ├── volatility.py      # Volatility modeling
│   ├── dependance.py      # Dependency analysis
│   └── stability.py       # Stationarity tests
├── src/
│   ├── analysis/          # Statistical analysis modules
│   ├── backtest/          # Backtesting engine
│   ├── ingestion/         # Data loading utilities
│   ├── models/            # Time series models
│   ├── preprocessing/     # Data transformation
│   └── signals/           # Signal generation
├── data/
│   ├── raw/               # Raw market data
│   ├── clean/             # Processed data
│   └── features/          # Computed features
└── notebooks/             # Jupyter notebooks
```

---

## Data Pipeline

1. **Ingestion** — Load data from Yahoo Finance or local sources
2. **Cleaning** — Handle missing values, outliers, and adjustments
3. **Feature Engineering** — Compute returns, volatility, and indicators
4. **Analysis** — Apply statistical tests and models
5. **Visualization** — Render interactive charts and metrics

---

## Requirements

- Python 3.10+
- Streamlit
- Plotly
- Pandas
- NumPy
- SciPy
- Statsmodels
- Arch (GARCH modeling)

---

## Configuration

Data sources and parameters can be configured in `src/utils/config.py`:

```python
TICKERS = ['AAPL', 'MSFT', 'GOOGL']
START_DATE = '2020-01-01'
DATA_PATH = 'data/'
```

---

## License

MIT License

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-analysis`)
3. Commit changes (`git commit -m 'Add new analysis module'`)
4. Push to branch (`git push origin feature/new-analysis`)
5. Open a Pull Request

---

**Adama COULIBALY** — AI/ML Engineer & Quantitative Analyst
- GitHub: [adama-coulibaly](https://github.com/startlingadama)