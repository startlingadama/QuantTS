import  pickle
from arch import arch_model
import pandas as pd
from scipy.stats import norm

# Load data
with open('models/garch_model.pkl', 'rb') as f:
    garch_res = pickle.load(f)

returns = pd.read_parquet('./data/features/returns.parquet')

# forcast horizon (days)
h = 10
forecast = garch_res.forecast(horizon=h)

sigma_f = forecast.variance.iloc[-1] ** 0.5 / 100  # scale back

#  parametric Value-at-Risk (VaR) calculation
alpha = 0.05
VaR_parametric = norm.ppf(alpha) * sigma_f  # parametric VaR


# condtio  VaR
CVaR = - sigma_f * norm.pdf(norm.ppf(alpha)) / alpha

# Rolling backtest (Risk Validation)
returns["VaR_95"] = returns["conditional_volatility"] * norm.ppf(alpha)

returns["Violation"] = returns["log_return"] < returns["VaR_95"]

returns.to_parquet(
    'data/backtest/returns_with_var.parquet',
    engine ='pyarrow',
    compression='snappy'
)

print("Returns with VaR saved to data/backtest/returns_with_var.parquet")
