"""
download_history.py
Télécharge des données historiques 1min depuis Binance.
Peut aller jusqu'à plusieurs années selon le symbole.

Usage:
    python download_history.py --symbol BTCUSDT --days 30
    python download_history.py --symbol BTCUSDT --from 2024-01-01 --to 2024-06-01
    python download_history.py --all --days 60
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

BINANCE_URL = "https://api.binance.com/api/v3/klines"
SAVE_DIR    = "data_1min"

# ms par bougie + étiquette de fichier par intervalle Binance supporté
INTERVAL_MS = {"1m": 60_000, "1h": 3_600_000, "4h": 14_400_000}
INTERVAL_LABEL = {"1m": "1min", "1h": "1h", "4h": "4h"}


def download_chunk(symbol: str, start_ms: int, end_ms: int, interval: str = "1m", limit=1000) -> list:
    """Télécharge un chunk de barres depuis Binance."""
    params = {
        "symbol":    symbol,
        "interval":  interval,
        "startTime": start_ms,
        "endTime":   end_ms,
        "limit":     limit,
    }
    resp = requests.get(BINANCE_URL, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def download_full_history(
    symbol:   str,
    start_dt: datetime,
    end_dt:   datetime,
    save_dir: str = SAVE_DIR,
    interval: str = "1m",
) -> pd.DataFrame:
    """
    Télécharge tout l'historique entre start_dt et end_dt pour l'intervalle
    donné (1m, 1h, 4h). Pagine automatiquement par chunks de 1000 bougies.

    Args:
        symbol:   ex "BTCUSDT"
        start_dt: date de début
        end_dt:   date de fin
        save_dir: dossier de sauvegarde
        interval: "1m", "1h" ou "4h"

    Returns:
        DataFrame OHLCV complet
    """
    os.makedirs(save_dir, exist_ok=True)
    bar_ms = INTERVAL_MS[interval]

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms   = int(end_dt.timestamp() * 1000)

    total_bars_est = int((end_ms - start_ms) / bar_ms)
    total_chunks   = (total_bars_est // 1000) + 1

    print(f"\n{'='*60}")
    print(f"Téléchargement {symbol} {interval}")
    print(f"  De  : {start_dt.strftime('%Y-%m-%d %H:%M')}")
    print(f"  À   : {end_dt.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Barres estimées : ~{total_bars_est:,}")
    print(f"  Chunks          : ~{total_chunks}")
    print(f"{'='*60}")

    all_data  = []
    current   = start_ms
    chunk_num = 0

    while current < end_ms:
        chunk_end = min(current + 1000 * bar_ms, end_ms)
        chunk_num += 1

        try:
            data = download_chunk(symbol, current, chunk_end, interval=interval)
            if not data:
                # Aucune bougie sur cette fenêtre — normal si le symbole n'était pas
                # encore listé à cette date. On avance la fenêtre au lieu de s'arrêter,
                # sinon un --from antérieur au listing renvoie un historique vide.
                current = chunk_end
                pct = min((current - start_ms) / (end_ms - start_ms) * 100, 100)
                print(f"\r  [{pct:5.1f}%] {chunk_num} chunks | pas encore de données à cette date...",
                      end="", flush=True)
                time.sleep(0.1)
                continue

            all_data.extend(data)
            last_ts = data[-1][0]
            current = last_ts + bar_ms  # prochaine bougie

            # Affichage progression
            pct = min((current - start_ms) / (end_ms - start_ms) * 100, 100)
            dt  = datetime.fromtimestamp(last_ts/1000, tz=timezone.utc)
            print(f"\r  [{pct:5.1f}%] {chunk_num} chunks | "
                  f"Dernière barre: {dt.strftime('%Y-%m-%d %H:%M')} | "
                  f"{len(all_data):,} barres", end="", flush=True)

            time.sleep(0.1)  # rate limit

        except requests.exceptions.HTTPError as e:
            print(f"\n  Erreur HTTP: {e} — pause 5s")
            time.sleep(5)
        except Exception as e:
            print(f"\n  Erreur: {e} — pause 3s")
            time.sleep(3)

    print()  # nouvelle ligne

    if not all_data:
        print("  Aucune donnée téléchargée.")
        return pd.DataFrame()

    # Convertir en DataFrame
    df = pd.DataFrame(all_data, columns=[
        "timestamp","open","high","low","close","volume",
        "close_time","quote_vol","n_trades","taker_buy_base",
        "taker_buy_quote","ignore"
    ])
    df = df[["timestamp","open","high","low","close","volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index().drop_duplicates()
    df = df.astype(float)

    # Sauvegarder
    label = INTERVAL_LABEL.get(interval, interval)
    fname = f"{symbol}_{label}_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.csv"
    fpath = os.path.join(save_dir, fname)
    df.to_csv(fpath)

    print(f"\n✅ {symbol}: {len(df):,} barres sauvegardées → {fpath}")
    print(f"   Période: {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"   Taille fichier: {os.path.getsize(fpath)/1024/1024:.1f} MB")

    return df


def merge_and_update(symbol: str, save_dir: str = SAVE_DIR) -> pd.DataFrame:
    """
    Fusionne tous les CSV d'un symbole et met à jour avec les données récentes.
    """
    import glob
    pattern = os.path.join(save_dir, f"{symbol}_1min_*.csv")
    files   = sorted(glob.glob(pattern))

    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        dfs.append(df)

    merged = pd.concat(dfs).sort_index().drop_duplicates()

    # Mettre à jour avec les données manquantes depuis la dernière barre
    last_ts = merged.index[-1]
    now     = datetime.now(tz=timezone.utc)

    if (now - last_ts).total_seconds() > 120:
        print(f"\nMise à jour {symbol} depuis {last_ts.strftime('%Y-%m-%d %H:%M')}...")
        update_df = download_full_history(
            symbol,
            start_dt = last_ts.to_pydatetime() + timedelta(minutes=1),
            end_dt   = now,
            save_dir = save_dir,
        )
        if not update_df.empty:
            merged = pd.concat([merged, update_df]).sort_index().drop_duplicates()

    # Sauvegarder version fusionnée
    fname = os.path.join(save_dir, f"{symbol}_1min_MERGED.csv")
    merged.to_csv(fname)
    print(f"Fichier fusionné: {fname} ({len(merged):,} barres)")

    return merged


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

SYMBOLS_DEFAULT = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

def main():
    parser = argparse.ArgumentParser(
        description="Téléchargeur historique 1min Binance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # 30 derniers jours BTC
  python download_history.py --symbol BTCUSDT --days 30

  # Période précise
  python download_history.py --symbol ETHUSDT --from 2024-01-01 --to 2024-06-01

  # Tous les symboles par défaut, 60 jours
  python download_history.py --all --days 60

  # Fusionner et mettre à jour les données existantes
  python download_history.py --symbol BTCUSDT --update
        """
    )

    parser.add_argument("--symbol",  default="BTCUSDT",   help="Symbole Binance")
    parser.add_argument("--days",    type=int, default=30, help="Nombre de jours")
    parser.add_argument("--from",    dest="date_from",    help="Date début YYYY-MM-DD")
    parser.add_argument("--to",      dest="date_to",      help="Date fin YYYY-MM-DD")
    parser.add_argument("--all",     action="store_true", help="Tous les symboles par défaut")
    parser.add_argument("--update",  action="store_true", help="Mettre à jour les données existantes")
    parser.add_argument("--dir",     default=SAVE_DIR,    help="Dossier de sauvegarde")
    parser.add_argument("--interval", default="1m", choices=list(INTERVAL_MS.keys()),
                        help="Intervalle des bougies (1m, 1h, 4h)")
    args = parser.parse_args()

    # Déterminer la période
    now = datetime.now(tz=timezone.utc).replace(second=0, microsecond=0)

    if args.date_from and args.date_to:
        start = datetime.strptime(args.date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end   = datetime.strptime(args.date_to,   "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        end   = now
        start = now - timedelta(days=args.days)

    # Symboles à télécharger
    symbols = SYMBOLS_DEFAULT if args.all else [args.symbol.upper()]

    print(f"\nTéléchargement de {len(symbols)} symbole(s)")
    print(f"Période: {start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')}")
    print(f"Dossier: {args.dir}\n")

    for symbol in symbols:
        if args.update:
            merge_and_update(symbol, args.dir)
        else:
            download_full_history(symbol, start, end, args.dir, interval=args.interval)

    print("\n✅ Téléchargement terminé!")
    print(f"\nPour lancer le backtest sur ces données:")
    print(f"  python multi_market_backtest.py --markets {' '.join(symbols)} --from-csv")


if __name__ == "__main__":
    main()
