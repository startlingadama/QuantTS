import pandas as pd
import numpy as np
from pathlib import Path


RAW_DATA_PATH = Path('data/raw/yahoo')
CLEANED_PATH = Path('data/clean/yahoo')
CLEANED_PATH.mkdir(parents=True, exist_ok=True)

def make_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values('date')

    # enforce datetime
    df["date"] = pd.to_datetime(df["date"])

    # log price
    df['log_price'] = np.log(df['adj_close'])

    # log returns
    df['log_return'] = df['log_price'].diff()

    # drop first row (NaN return)
    df = df.dropna().reset_index(drop=True)

    return df

def main():
    df_raw = pd.read_parquet(RAW_DATA_PATH / 'AAPL_prices.parquet')
    df_ret = make_returns(df_raw)

    df_ret.to_parquet(
        CLEANED_PATH / 'returns.parquet',
        engine='pyarrow',
        compression='snappy'
    )

    print("Returns data saved.")


if __name__ == "__main__":
    main()