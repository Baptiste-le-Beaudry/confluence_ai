"""
run_walkforward.py
Point d'entrée simple pour le walk-forward backtest.

Usage:
    python run_walkforward.py                        # données synthétiques
    python run_walkforward.py --source binance       # Binance BTC/USDT 15min
    python run_walkforward.py --source csv --file mon_btc.csv
    python run_walkforward.py --tf 1h --bars 3000   # 1h timeframe

Le walk-forward:
  - Entraîne l'AI sur 15 jours
  - Teste sur les 2 jours suivants
  - Glisse de 2 jours et recommence
  - Concatène tous les résultats

Critères resserrés vs backtest simple:
  - AI ET Confluence doivent être d'accord (AND)
  - SL serré (1.5× ATR) mais TP large (4.5× ATR) → ratio R/R ~3
  - Cooldown de 2.5h entre trades
  - Filtre chop actif (pas de trades en range)
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_from_binance, load_from_csv, generate_sample_data
from walk_forward import WalkForwardEngine, WFParams


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Confluence AI v4.0")
    parser.add_argument("--source",     default="synthetic",
                        choices=["synthetic", "binance", "csv"])
    parser.add_argument("--symbol",     default="BTC/USDT")
    parser.add_argument("--tf",         default="15m",
                        choices=["5m", "15m", "30m", "1h", "4h"])
    parser.add_argument("--bars",       type=int,   default=5000)
    parser.add_argument("--file",       default="data.csv")
    parser.add_argument("--capital",    type=float, default=10000.0)
    parser.add_argument("--train-days", type=int,   default=15)
    parser.add_argument("--test-days",  type=int,   default=2)
    parser.add_argument("--target-tpd", type=float, default=1.0,
                        help="Trades par jour cible")
    parser.add_argument("--sl",         type=float, default=1.5, help="SL en ATR")
    parser.add_argument("--tp",         type=float, default=4.5, help="TP en ATR")
    parser.add_argument("--cooldown",   type=int,   default=10)
    parser.add_argument("--conf",       type=float, default=0.30,
                        help="Seuil de confluence (fraction du max, ex: 0.30)")
    parser.add_argument("--output",     default="wf_report.png")
    args = parser.parse_args()

    # ── Chargement des données ──
    freq_map = {"5m": "5min", "15m": "15min", "30m": "30min",
                "1h": "h", "4h": "4h"}
    freq = freq_map.get(args.tf, "15min")

    if args.source == "synthetic":
        df = generate_sample_data(n_bars=args.bars, freq=freq)
    elif args.source == "binance":
        df = load_from_binance(args.symbol, args.tf, args.bars)
    else:
        df = load_from_csv(args.file)

    # ── Paramètres ──
    params = WFParams(
        train_days             = args.train_days,
        test_days              = args.test_days,
        step_days              = args.test_days,
        target_trades_per_day  = args.target_tpd,
        pretrain_epochs        = 12,
        conf_long_pct          = args.conf,
        conf_short_pct         = args.conf,
        cooldown_bars          = args.cooldown,
        sl_atr_mult            = args.sl,
        tp_atr_mult            = args.tp,
        capital                = args.capital,
        risk_pct               = 0.008,
        use_chop_filter        = True,
        use_candle_confirm     = False,
        use_htf                = False,
    )

    print(f"\nParamètres Walk-Forward:")
    print(f"  Source          : {args.source}")
    print(f"  Timeframe       : {args.tf}")
    print(f"  Barres          : {len(df)}")
    print(f"  Train / Test    : {args.train_days}j / {args.test_days}j")
    print(f"  Cible           : {args.target_tpd} trade/jour")
    print(f"  SL / TP         : {args.sl}× / {args.tp}× ATR  (R/R ~{args.tp/args.sl:.1f})")
    print(f"  Confluence min  : {args.conf*100:.0f}%")
    print(f"  Cooldown        : {args.cooldown} barres")

    # ── Exécution ──
    wf = WalkForwardEngine(params)
    m  = wf.run(df)

    if "error" in m:
        print(f"\nErreur: {m['error']}")
        print("Conseil: réduire --conf ou --cooldown pour avoir plus de trades")
        return

    wf.print_report(m)
    wf.plot_results(m, args.output)

    # Export CSV des trades
    import pandas as pd
    trades_df = pd.DataFrame([{
        "direction":   t.direction,
        "entry_time":  t.entry_time,
        "entry_price": t.entry_price,
        "exit_time":   t.exit_time,
        "exit_price":  t.exit_price,
        "exit_reason": t.exit_reason,
        "pnl_usd":     t.pnl_usd,
        "pnl_pct":     t.pnl_pct * 100,
        "conf_score":  t.conf_score,
        "ai_proba":    t.ai_proba,
    } for t in wf.all_trades])

    csv_out = args.output.replace(".png", "_trades.csv")
    trades_df.to_csv(csv_out, index=False)
    print(f"Trades exportés: {csv_out}")

    # Résumé des fenêtres
    windows_df = pd.DataFrame(wf.window_results)
    win_csv = args.output.replace(".png", "_windows.csv")
    windows_df.to_csv(win_csv, index=False)
    print(f"Fenêtres exportées: {win_csv}")


if __name__ == "__main__":
    main()
