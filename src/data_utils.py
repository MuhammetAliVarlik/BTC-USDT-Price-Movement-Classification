import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta

# Initialize Binance exchange client
exchange = ccxt.binance()

def load_bitcoin_ohlcv(frame=30, days=365):
    """
    Fetches BTC/USDT OHLCV data from Binance at a given timeframe (default 30 minutes)
    for the past specified number of days (default 365).
    """

    # Current time in milliseconds
    now = exchange.milliseconds()

    # Calculate timestamp for 'days' ago from now
    one_year_ago = datetime.utcnow() - timedelta(days=days)

    # Convert datetime to exchange timestamp format (milliseconds)
    since = exchange.parse8601(one_year_ago.strftime('%Y-%m-%dT%H:%M:%SZ'))
    
    all_ohlcv = []
    limit = 500  # max number of bars per request

    # Fetch data in chunks until reaching current time
    while since < now:
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe=f'{frame}m', since=since, limit=limit)
        if not ohlcv:
            break
        all_ohlcv += ohlcv
        since = ohlcv[-1][0] + 1  # move to just after last timestamp
        time.sleep(0.5)  # pause to respect API rate limits

    # Create DataFrame and convert timestamps to datetime objects
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Create target column: 1 if close price after forecast horizon (6 hours) is higher than current close, else 0
    hour_window = 6
    forecast_horizon = int(hour_window * 60 / frame)  # convert hours to number of bars
    df['Target'] = (df['close'].shift(-forecast_horizon) > df['close']).astype(int)

    # Remove last rows where target cannot be computed (due to shifting)
    df = df[:-forecast_horizon]

    return df, frame
