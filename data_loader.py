"""
data_loader.py
Chargement des données OHLCV depuis:
  - Binance (via ccxt, gratuit, sans clé API)
  - CSV local
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_from_binance(symbol="BTC/USDT", timeframe="15m", limit=1000) -> pd.DataFrame:
    """
    Télécharge les données OHLCV depuis Binance (gratuit, sans clé API).

    Args:
        symbol:    Paire (ex: "BTC/USDT", "ETH/USDT")
        timeframe: Intervalle ("1m","5m","15m","1h","4h","1d")
        limit:     Nombre de barres (max 1000 par appel Binance)

    Returns:
        DataFrame OHLCV avec DatetimeIndex UTC.
    """
    try:
        import ccxt
    except ImportError:
        raise ImportError("Installer ccxt: pip install ccxt")

    exchange = ccxt.binance({"enableRateLimit": True})
    print(f"Téléchargement {symbol} {timeframe} ({limit} barres) depuis Binance...")

    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()

    print(f"  Données chargées: {len(df)} barres ({df.index[0]} → {df.index[-1]})")
    return df


def load_from_csv(filepath: str) -> pd.DataFrame:
    """
    Charge un CSV local.

    Format attendu (avec header):
        timestamp,open,high,low,close,volume
        2024-01-01 00:00:00,42000,42500,41800,42300,1234.5
        ...

    Accepte aussi le format Binance export (colonnes en anglais ou français).
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {filepath}")

    df = pd.read_csv(filepath)

    # Normaliser les noms de colonnes
    df.columns = df.columns.str.lower().str.strip()
    rename_map = {
        "time": "timestamp", "date": "timestamp",
        "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
        "open time": "timestamp",
    }
    df = df.rename(columns=rename_map)

    # Parser le timestamp
    ts_col = "timestamp"
    if ts_col in df.columns:
        try:
            df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
        except Exception:
            df[ts_col] = pd.to_datetime(df[ts_col], unit="ms", utc=True)
        df = df.set_index(ts_col)

    # Garder seulement les colonnes OHLCV
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].astype(float).sort_index().dropna()

    print(f"CSV chargé: {len(df)} barres ({df.index[0]} → {df.index[-1]})")
    return df


def generate_sample_data(n_bars=2000, freq="15min", start_price=45000.0, seed=42) -> pd.DataFrame:
    """
    Génère des données OHLCV synthétiques BTC-like pour les tests.
    Mouvement brownien géométrique avec drift et volatilité calibrés sur BTC.
    """
    np.random.seed(seed)
    annual_drift = 0.50
    annual_vol   = 0.65
    bars_per_year = 365 * 24 * 4  # barres 15min par an
    dt = 1.0 / bars_per_year

    prices = [start_price]
    for _ in range(n_bars - 1):
        ret = (annual_drift - 0.5 * annual_vol**2) * dt + annual_vol * np.sqrt(dt) * np.random.randn()
        prices.append(prices[-1] * np.exp(ret))

    prices = np.array(prices)

    # Générer OHLCV réaliste
    vol_intrabar = annual_vol * np.sqrt(dt) * prices

    opens  = prices
    closes = prices * np.exp(annual_vol * np.sqrt(dt) * np.random.randn(n_bars) * 0.3)
    highs  = np.maximum(opens, closes) + np.abs(np.random.randn(n_bars)) * vol_intrabar * 0.5
    lows   = np.minimum(opens, closes) - np.abs(np.random.randn(n_bars)) * vol_intrabar * 0.5
    volumes = np.abs(np.random.randn(n_bars)) * 500 + 1000

    idx = pd.date_range("2024-01-01", periods=n_bars, freq=freq, tz="UTC")
    df = pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": volumes
    }, index=idx)

    print(f"Données synthétiques générées: {n_bars} barres")
    print(f"  Prix: ${df['close'].min():,.0f} → ${df['close'].max():,.0f}")
    return df
