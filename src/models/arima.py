import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_parquet('data/clean/yahoo/returns.parquet')
r = df['log_return']

results = []

for p in range(0,3):
    for q in range(0,3):
        try:
            model = ARIMA(r, order=(p,0,q))
            res = model.fit()
            results.append((p, q, res.aic, res.bic))
        except:
            pass

results = sorted(results, key=lambda x: x[2])
results_df = pd.DataFrame(results, columns=['p', 'q', 'AIC', 'BIC'])

            
results_df.to_parquet('models/arima_model_selection.parquet', engine='pyarrow', compression='snappy')
results_df.to_parquet("./data/features/arima.parquet", engine='pyarrow', compression='snappy')

# Save the best model
res.save('models/arma_model.pkl')
print("ARIMA model selection results saved to models/arima_model_selection.parquet")