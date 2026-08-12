"""
download_equity_markets.py
Télécharge les données 1 minute pour NASDAQ, S&P500, Or, et autres marchés
via yfinance (Yahoo Finance) — gratuit, sans clé API.

Marchés disponibles:
  Indices ETF : QQQ (NASDAQ-100), SPY (S&P500), DIA (Dow Jones)
  Futures     : NQ=F (NASDAQ futures), ES=F (S&P500 futures)
  Or/Métaux   : GLD (Gold ETF), SLV (Silver ETF), GC=F (Gold futures)
  Pétrole     : USO (Oil ETF), CL=F (Oil futures)
  Forex       : EURUSD=X, GBPUSD=X, USDJPY=X
  Crypto      : BTC-USD, ETH-USD

LIMITE Yahoo Finance 1min: maximum 7 jours d'historique
Pour plus d'historique: utiliser interval='5m' (60 jours) ou '1h' (730 jours)

Usage:
    pip install yfinance
    python download_equity_markets.py
    python download_equity_markets.py --symbols QQQ SPY GLD --days 7
    python download_equity_markets.py --interval 5m --days 60
"""

import argparse
import os
import time
from datetime import datetime, timedelta

import pandas as pd

SAVE_DIR = "data_equity"

# ─────────────────────────────────────────────
# Marchés disponibles
# ─────────────────────────────────────────────

MARKETS = {
    # Indices / ETFs (marchés US — sessions 9h30-16h EST)
    "QQQ":     {"name": "NASDAQ-100 ETF",    "type": "index",  "currency": "USD"},
    "SPY":     {"name": "S&P 500 ETF",       "type": "index",  "currency": "USD"},
    "DIA":     {"name": "Dow Jones ETF",     "type": "index",  "currency": "USD"},
    "IWM":     {"name": "Russell 2000 ETF",  "type": "index",  "currency": "USD"},
    # Futures (24h/5j, plus de volume)
    "NQ=F":    {"name": "NASDAQ Futures",    "type": "future", "currency": "USD"},
    "ES=F":    {"name": "S&P500 Futures",    "type": "future", "currency": "USD"},
    # Or et Métaux
    "GLD":     {"name": "Gold ETF (GLD)",    "type": "metal",  "currency": "USD"},
    "GC=F":    {"name": "Gold Futures",      "type": "metal",  "currency": "USD"},
    "SLV":     {"name": "Silver ETF",        "type": "metal",  "currency": "USD"},
    # Pétrole
    "USO":     {"name": "Oil ETF",           "type": "oil",    "currency": "USD"},
    "CL=F":    {"name": "Oil Futures",       "type": "oil",    "currency": "USD"},
    # Forex (via Yahoo Finance)
    "EURUSD=X":{"name": "EUR/USD",           "type": "forex",  "currency": "USD"},
    "GBPUSD=X":{"name": "GBP/USD",           "type": "forex",  "currency": "USD"},
    "USDJPY=X":{"name": "USD/JPY",           "type": "forex",  "currency": "JPY"},
    "USDCHF=X":{"name": "USD/CHF",           "type": "forex",  "currency": "CHF"},
    # Crypto via Yahoo
    "BTC-USD": {"name": "Bitcoin",           "type": "crypto", "currency": "USD"},
    "ETH-USD": {"name": "Ethereum",          "type": "crypto", "currency": "USD"},
}

# Sélection par défaut pour comparer avec le backtest crypto
DEFAULT_SYMBOLS = ["QQQ", "SPY", "GLD", "NQ=F", "EURUSD=X"]

# Intervalles disponibles Yahoo Finance
INTERVAL_MAX_DAYS = {
    "1m":  7,    # max 7 jours
    "2m":  60,
    "5m":  60,   # max 60 jours
    "15m": 60,
    "30m": 60,
    "60m": 730,  # max 2 ans
    "1h":  730,
    "1d":  None, # illimité
}


def download_yfinance(
    symbol:   str,
    interval: str  = "1m",
    days:     int  = 7,
    save_dir: str  = SAVE_DIR,
) -> pd.DataFrame:
    """
    Télécharge les données depuis Yahoo Finance via yfinance.

    Args:
        symbol:   Ticker Yahoo ex: "QQQ", "SPY", "GC=F", "EURUSD=X"
        interval: "1m", "5m", "15m", "1h", "1d"
        days:     Nombre de jours (limité selon interval)
        save_dir: Dossier de sauvegarde

    Returns:
        DataFrame OHLCV standardisé
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance non installé. Lancer: pip install yfinance")

    os.makedirs(save_dir, exist_ok=True)

    # Vérifier la limite de jours
    max_days = INTERVAL_MAX_DAYS.get(interval, 7)
    if max_days and days > max_days:
        print(f"  ⚠️  {interval} limité à {max_days} jours sur Yahoo Finance → ajusté")
        days = max_days

    period = f"{days}d"
    name   = MARKETS.get(symbol, {}).get("name", symbol)

    print(f"  Téléchargement {symbol} ({name}) | {interval} | {days} jours...", end="", flush=True)

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=True)

        if df.empty:
            print(f" VIDE — symbole invalide ou pas de données")
            return pd.DataFrame()

        # Standardiser les colonnes
        df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume"
        })
        df = df[["open", "high", "low", "close", "volume"]].dropna()

        # Sauvegarder
        safe_sym = symbol.replace("=", "_").replace("-", "_")
        fname    = f"{safe_sym}_{interval}_{days}d.csv"
        fpath    = os.path.join(save_dir, fname)
        df.to_csv(fpath)

        print(f" {len(df):,} barres | "
              f"{df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")

        return df

    except Exception as e:
        print(f" ERREUR: {e}")
        return pd.DataFrame()


def download_all(
    symbols:  list[str],
    interval: str = "1m",
    days:     int = 7,
    save_dir: str = SAVE_DIR,
) -> dict[str, pd.DataFrame]:
    """Télécharge plusieurs symboles."""
    results = {}
    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}]", end=" ")
        df = download_yfinance(symbol, interval, days, save_dir)
        if not df.empty:
            results[symbol] = df
        time.sleep(0.5)  # éviter le rate limiting
    return results


def list_available():
    """Affiche tous les marchés disponibles."""
    print("\nMarchés disponibles:")
    print("─" * 60)
    types = {}
    for sym, info in MARKETS.items():
        t = info["type"]
        if t not in types:
            types[t] = []
        types[t].append((sym, info["name"]))

    type_labels = {
        "index":  "📈 Indices / ETFs",
        "future": "📊 Futures",
        "metal":  "🥇 Or / Métaux",
        "oil":    "🛢️  Pétrole",
        "forex":  "💱 Forex",
        "crypto": "₿  Crypto",
    }
    for t, label in type_labels.items():
        if t in types:
            print(f"\n{label}:")
            for sym, name in types[t]:
                print(f"  {sym:<12} {name}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Téléchargeur données équités/indices/forex — Yahoo Finance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # NASDAQ + S&P500 en 1min (7 jours max)
  python download_equity_markets.py --symbols QQQ SPY --interval 1m --days 7

  # Plus d'historique en 5min (60 jours)
  python download_equity_markets.py --symbols QQQ SPY GLD --interval 5m --days 60

  # Tous les marchés par défaut
  python download_equity_markets.py

  # Comparer avec le backtest crypto (même intervalle 1min)
  python download_equity_markets.py --symbols QQQ SPY GC=F EURUSD=X --interval 1m --days 7

  # Voir tous les marchés disponibles
  python download_equity_markets.py --list
        """
    )
    parser.add_argument("--symbols",  nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--interval", default="1m",
                        choices=["1m","2m","5m","15m","30m","60m","1h","1d"])
    parser.add_argument("--days",     type=int, default=7)
    parser.add_argument("--dir",      default=SAVE_DIR)
    parser.add_argument("--list",     action="store_true", help="Lister les marchés")
    args = parser.parse_args()

    if args.list:
        list_available()
        return

    print("=" * 65)
    print("  TÉLÉCHARGEMENT DONNÉES MARCHÉS — Yahoo Finance")
    print("=" * 65)
    print(f"  Symboles  : {', '.join(args.symbols)}")
    print(f"  Intervalle: {args.interval}")
    print(f"  Jours     : {args.days}")
    print(f"  Dossier   : {args.dir}")
    print()

    results = download_all(args.symbols, args.interval, args.days, args.dir)

    print("\n" + "=" * 65)
    print(f"✅ {len(results)}/{len(args.symbols)} symboles téléchargés avec succès")
    print(f"   Fichiers dans: {args.dir}/")
    print()
    print("Prochaine étape — Lancer le backtest:")
    syms = " ".join(args.symbols)
    print(f"  python multi_equity_backtest.py --symbols {syms} --interval {args.interval}")


if __name__ == "__main__":
    main()
