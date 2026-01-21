import yfinance as yf
import pandas as pd
from pathlib import Path

TICKER = 'AAPL'
START="2010-01-01"
END="2025-01-01"

OUTPUT_DIR = Path('data/raw/yahoo')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = yf.download(
    TICKER, 
    start=START, 
    end=END, 
    progress=False, 
    auto_adjust=False
    )
df.reset_index(inplace=True)

df = df.rename(columns={
    'Date': 'date',
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Adj Close': 'adj_close',
    'Volume': 'volume'
})

df['symbol'] = TICKER
df["source"] = "yahoo"

df.columns = df.columns.get_level_values(0)


df.to_parquet(
    OUTPUT_DIR / f'{TICKER}_prices.parquet',
    engine='pyarrow',
    compression='snappy'
)

print("Daily TS data saved.")
