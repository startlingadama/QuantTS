import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model

# Load data
arima_data = pd.read_parquet("./models/arima_model_selection.parquet")
returns = pd.read_parquet('./data/clean/yahoo/returns.parquet')

# Prepare data
r = returns["log_return"]
results = arima_data.values.tolist()
p, q = results[0][0], results[0][1]

arma = ARIMA(r, order=(p, 0, q)).fit()



# residual and scale 

resid = arma.resid

resid = resid * 100 # scale to percentage

# fit GARCH(1,1) model

garch = arch_model(resid, vol='Garch', p=1, q=1, dist='normal')

garch_res = garch.fit(update_freq=10)

# Save the GARCH model
import pickle
with open('models/garch_model.pkl', 'wb') as f:
    pickle.dump(garch_res, f)

# volatilily extration

returns["conditional_volatility"] = garch_res.conditional_volatility / 100  # scale back

# Save the returns with conditional volatility
returns.to_parquet(
    'data/features/returns.parquet', 
    engine ='pyarrow', 
    compression='snappy'
)



# Volatility Modeling
# - Mean models
#     Constant mean
#     Heterogeneous Autoregression (HAR)
#     Autoregression (AR)
#     Zero mean
#     Models with and without exogenous regressors
# - Volatility models
#     ARCH
#     GARCH
#     TARCH
#     EGARCH
#     EWMA/RiskMetrics
# - Distributions
#     Normal
#     Student's T
#     Generalized Error Distribution