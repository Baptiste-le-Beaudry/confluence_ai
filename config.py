"""
config.py - Configuration centrale du bot Confluence AI v4.0
Charge les variables depuis .env et expose les constantes utilisées partout.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# TELEGRAM — Notifications de trades
# ─────────────────────────────────────────────

# Token du bot (créer avec @BotFather si vous n'en avez pas)
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")

# Votre ID personnel Telegram (récupéré via @userinfobot)
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "5290735931")

# API Telegram (pour lire les signaux des canaux — fonctionnalité future)
TELEGRAM_API_ID   = int(os.getenv("TELEGRAM_API_ID",   "23642327"))
TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH", "ba4d5c1073af95498d981d0b358d4c80")

# ─────────────────────────────────────────────
# MT5 — Compte Demo TradeMaxGlobal
# ─────────────────────────────────────────────

MT5_LOGIN    = int(os.getenv("MT5_LOGIN",    "70013614"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "H-7yOxUu")
MT5_SERVER   = os.getenv("MT5_SERVER",   "TradeMaxGlobal-Demo")
MT5_PATH     = os.getenv("MT5_PATH",     r"C:\Program Files\MetaTrader 5\terminal64.exe")

# ─────────────────────────────────────────────
# PARAMÈTRES DE TRADING
# ─────────────────────────────────────────────

VOLUME        = float(os.getenv("TRADE_VOLUME",    "0.05"))
DEVIATION     = int(os.getenv("PRICE_DEVIATION",   "20"))
MAGIC_NUMBER  = int(os.getenv("MAGIC_NUMBER",      "12345"))

# ─────────────────────────────────────────────
# SYMBOLES À TRADER
# ─────────────────────────────────────────────

# Ajuster selon ce qui est disponible sur TradeMaxGlobal
SYMBOLS = [
    "BTCUSD",   # Bitcoin
    "XAUUSD",   # Or
    "EURUSD",   # Euro/Dollar
    "GBPUSD",   # Livre/Dollar
    "USDJPY",   # Dollar/Yen
    "USDCHF",   # Dollar/Franc suisse
    "AUDUSD",   # Dollar australien
    "NAS100",   # Nasdaq 100
]

# ─────────────────────────────────────────────
# PARAMÈTRES DU BOT
# ─────────────────────────────────────────────

TIMEFRAME        = "M15"     # M5, M15, M30, H1, H4
N_BARS           = 500       # Barres historiques à charger
LOOP_SECONDS     = 60        # Intervalle de la boucle principale
RISK_PCT         = 0.01      # 1% du capital par trade
SL_ATR_MULT      = 2.0       # Stop loss = 2× ATR
TP_ATR_MULT      = 4.0       # Take profit = 4× ATR
CONF_PCT         = 0.30      # Seuil de confluence (30% du score max)
COOLDOWN_BARS    = 10        # Barres minimum entre deux trades
TARGET_TPD       = 5.0       # Trades par jour cible (mesuré empiriquement: 1.0 -> ~0.3 trade/j réel, 5.0 -> ~1.4 trade/j réel)
CAPITAL_REF      = 10_000.0  # Capital de référence pour le sizing

# ─────────────────────────────────────────────
# LOGS
# ─────────────────────────────────────────────

LOG_LEVEL    = os.getenv("LOG_LEVEL",    "DEBUG")
LOG_TIMEZONE = os.getenv("LOG_TIMEZONE", "Europe/Paris")
