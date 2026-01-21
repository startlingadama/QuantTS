import pandas as pd

# stationarity tests
from statsmodels.tsa.stattools import adfuller, kpss

df = pd.read_parquet('data/clean/yahoo/returns.parquet')

r = df['log_return']

# ADF test
adf_stat, adf_p, *_ = adfuller(r)
print(f"ADF Statistic: {adf_stat}")

# KPSS test
kpss_stat, kpss_p, *_ = kpss(r, regression='c')
print(f"KPSS Statistic: {kpss_stat}")

# Interpretation

# ADF p < 0.05 → no unit root ✔️

# KPSS p > 0.05 → stationary ✔️
# Returns usually pass both.