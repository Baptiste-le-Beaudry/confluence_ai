"""
run_bot.py
Lance le bot Confluence AI avec vos paramètres MT5 et Telegram.
Utilise automatiquement les credentials de config.py

Usage:
    python run_bot.py              # live trading avec vos infos MT5
    python run_bot.py --paper      # simulation sans positions réelles
    python run_bot.py --paper --symbols BTCUSD XAUUSD   # simulation avec données MT5
    python run_bot.py --symbols BTCUSD XAUUSD   # seulement 2 symboles
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    MT5_LOGIN, MT5_PASSWORD, MT5_SERVER,
    SYMBOLS, TIMEFRAME, N_BARS, LOOP_SECONDS,
    RISK_PCT, SL_ATR_MULT, TP_ATR_MULT,
    CONF_PCT, COOLDOWN_BARS, TARGET_TPD,
    CAPITAL_REF, MAGIC_NUMBER, LOG_LEVEL
)
from mt5_live_bot import MT5LiveBot
import argparse
import logging

# Configuration du logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.DEBUG),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/mt5_bot.log", encoding="utf-8"),
    ]
)


def main():
    parser = argparse.ArgumentParser(description="Confluence AI Bot — MT5")
    parser.add_argument("--paper",   action="store_true", help="Mode simulation")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Symboles à trader (défaut: tous ceux de config.py)")
    parser.add_argument("--tf",      default=TIMEFRAME,
                        help=f"Timeframe (défaut: {TIMEFRAME})")
    args = parser.parse_args()

    symbols = args.symbols if args.symbols else SYMBOLS

    print("\n" + "="*60)
    print("  CONFLUENCE AI BOT v4.0")
    print("="*60)
    print(f"  Compte MT5  : {MT5_LOGIN} @ {MT5_SERVER}")
    print(f"  Mode        : {'PAPER (simulation)' if args.paper else '🔴 LIVE'}")
    print(f"  Symboles    : {', '.join(symbols)}")
    print(f"  Timeframe   : {args.tf}")
    print(f"  Risque/trade: {RISK_PCT*100:.1f}%")
    print(f"  SL / TP     : {SL_ATR_MULT}× / {TP_ATR_MULT}× ATR")
    print("="*60 + "\n")

    if not args.paper:
        confirm = input("⚠️  Confirmer le LIVE trading (oui/non): ").strip().lower()
        if confirm != "oui":
            print("Annulé.")
            return

    bot = MT5LiveBot(
        symbols       = symbols,
        timeframe     = args.tf,
        n_bars        = N_BARS,
        loop_seconds  = LOOP_SECONDS,
        capital       = CAPITAL_REF,
        risk_pct      = RISK_PCT,
        sl_atr_mult   = SL_ATR_MULT,
        tp_atr_mult   = TP_ATR_MULT,
        target_tpd    = TARGET_TPD,
        conf_pct      = CONF_PCT,
        cooldown_bars = COOLDOWN_BARS,
        use_chop      = True,
        paper_mode    = args.paper,
        force_mt5_data = True,
        magic         = MAGIC_NUMBER,
    )

    bot.run(
        account  = MT5_LOGIN,
        password = MT5_PASSWORD,
        server   = MT5_SERVER,
    )


if __name__ == "__main__":
    main()
