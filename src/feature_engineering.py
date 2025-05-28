import numpy as np

def add_technical_indicators(df, frame=30,
                             sma_hours=[5, 10, 14, 30],
                             ema_hours=[5, 10, 14, 30],
                             momentum_hours=[5, 10, 14],
                             stochastic_hours=14,
                             stochastic_smooth=3,
                             macd_fast_hours=12,
                             macd_slow_hours=26,
                             macd_signal_hours=9,
                             rsi_hours=[14, 28],
                             williams_hours=[14, 28],
                             cci_hours=[14]):
    """
    Calculates various technical indicators based on 30-minute interval data.
    Converts time-based periods (in hours) to equivalent number of bars (candles) and applies rolling computations.

    Args:
        df (pd.DataFrame): DataFrame containing OHLCV data with at least 'close', 'high', 'low', 'volume' columns.
        frame (int): Timeframe in minutes (default 30).
        sma_hours (list): Periods (hours) for Simple Moving Averages.
        ema_hours (list): Periods (hours) for Exponential Moving Averages.
        momentum_hours (list): Periods (hours) for Momentum indicator.
        stochastic_hours (int): Period (hours) for Stochastic Oscillator.
        stochastic_smooth (int): Smoothing window for Stochastic %D.
        macd_fast_hours (int): Fast EMA period (hours) for MACD.
        macd_slow_hours (int): Slow EMA period (hours) for MACD.
        macd_signal_hours (int): Signal line EMA period (hours) for MACD.
        rsi_hours (list): Periods (hours) for Relative Strength Index.
        williams_hours (list): Periods (hours) for Williams %R indicator.
        cci_hours (list): Periods (hours) for Commodity Channel Index.

    Returns:
        pd.DataFrame: DataFrame with new columns for each calculated technical indicator.
    """

    # Convert hours into equivalent number of bars (based on frame length)
    def to_bars(hours):
        return int(hours * 60 / frame)

    # Helper function to calculate Exponential Moving Average (EMA)
    def EMA(series, span):
        return series.ewm(span=span, adjust=False).mean()

    # Simple Moving Average (SMA) for specified hour periods
    for h in sma_hours:
        p = to_bars(h)
        df[f'SMA_{h}h'] = df['close'].rolling(window=p).mean()

    # Exponential Moving Average (EMA) for specified hour periods
    for h in ema_hours:
        p = to_bars(h)
        df[f'EMA_{h}h'] = EMA(df['close'], p)

    # Momentum: difference between current close and close 'p' bars ago
    for h in momentum_hours:
        p = to_bars(h)
        df[f'Momentum_{h}h'] = df['close'] - df['close'].shift(p)

    # Stochastic Oscillator: %K and smoothed %D
    p = to_bars(stochastic_hours)
    low_min = df['low'].rolling(window=p).min()
    high_max = df['high'].rolling(window=p).max()
    df['Stoch_%K'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
    df['Stoch_%D'] = df['Stoch_%K'].rolling(window=stochastic_smooth).mean()

    # MACD: difference of fast and slow EMAs and its signal line
    fast_p = to_bars(macd_fast_hours)
    slow_p = to_bars(macd_slow_hours)
    signal_p = to_bars(macd_signal_hours)
    ema_fast = EMA(df['close'], fast_p)
    ema_slow = EMA(df['close'], slow_p)
    df['MACD_line'] = ema_fast - ema_slow
    df['MACD_signal'] = EMA(df['MACD_line'], signal_p)

    # RSI: measures strength of recent gains vs losses
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    for h in rsi_hours:
        p = to_bars(h)
        ma_up = up.rolling(window=p).mean()
        ma_down = down.rolling(window=p).mean()
        rs = ma_up / ma_down
        df[f'RSI_{h}h'] = 100 - (100 / (1 + rs))

    # Williams %R: momentum indicator showing overbought/oversold levels
    for h in williams_hours:
        p = to_bars(h)
        high_p = df['high'].rolling(window=p).max()
        low_p = df['low'].rolling(window=p).min()
        df[f'Williams_%R_{h}h'] = -100 * ((high_p - df['close']) / (high_p - low_p))

    # Accumulation/Distribution Index (ADI): volume-based indicator of flow into/out of security
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, np.nan)
    mfm = mfm.fillna(0)  # replace NaNs (from division by zero) with 0
    mfv = mfm * df['volume']
    df['ADI'] = mfv.cumsum()

    # Commodity Channel Index (CCI): measures deviation from typical price moving average
    for h in cci_hours:
        p = to_bars(h)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(window=p).mean()
        mean_dev = typical_price.rolling(window=p).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        df[f'CCI_{h}h'] = (typical_price - sma_tp) / (0.015 * mean_dev)

    return df
