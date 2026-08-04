"""
telegram_notify.py
Notifications Telegram — utilise les credentials de config.py
"""

import requests
import logging
from datetime import datetime
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger("confluence_bot")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"


def send_message(text: str, parse_mode: str = "HTML") -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram non configuré — ajouter TELEGRAM_TOKEN dans .env")
        return False
    try:
        resp = requests.post(
            f"{TELEGRAM_API}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": parse_mode},
            timeout=10
        )
        return resp.status_code == 200
    except Exception as e:
        logger.error(f"Telegram erreur: {e}")
        return False


def notify_trade_open(symbol, direction, price, sl, tp, lot,
                      risk_usd, ai_proba, conf_score, timeframe="M15"):
    emoji  = "🟢" if direction == "LONG" else "🔴"
    arrow  = "📈" if direction == "LONG" else "📉"
    sl_dist = abs(price - sl)
    tp_dist = abs(price - tp)
    rr = tp_dist / sl_dist if sl_dist > 0 else 0
    text = (
        f"{emoji} <b>TRADE OUVERT — {direction}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{arrow} <b>Symbole  :</b> {symbol} ({timeframe})\n"
        f"💰 <b>Prix     :</b> {price:.5f}\n"
        f"🛑 <b>Stop Loss:</b> {sl:.5f}  (-{sl_dist:.5f})\n"
        f"🎯 <b>Take Prof:</b> {tp:.5f}  (+{tp_dist:.5f})\n"
        f"📊 <b>Lot      :</b> {lot:.2f}\n"
        f"💵 <b>Risque   :</b> ${risk_usd:.2f}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🤖 <b>AI Proba :</b> {ai_proba:.1%}\n"
        f"📐 <b>Confluence:</b> {conf_score:.2f}\n"
        f"⚖️ <b>Ratio R/R :</b> 1:{rr:.2f}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    return send_message(text)


def notify_trade_close(symbol, direction, entry, exit_price,
                       pnl_usd, pnl_pct, reason, duration=""):
    emoji  = "✅" if pnl_usd >= 0 else "❌"
    result = "PROFIT" if pnl_usd >= 0 else "PERTE"
    arrow  = "📈" if direction == "LONG" else "📉"
    text = (
        f"{emoji} <b>TRADE FERMÉ — {result}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{arrow} <b>Symbole  :</b> {symbol} ({direction})\n"
        f"📥 <b>Entrée   :</b> {entry:.5f}\n"
        f"📤 <b>Sortie   :</b> {exit_price:.5f}\n"
        f"💵 <b>P&L      :</b> <b>{pnl_usd:+.2f}$</b> ({pnl_pct:+.2f}%)\n"
        f"📋 <b>Raison   :</b> {reason}\n"
    )
    if duration:
        text += f"⏱️ <b>Durée    :</b> {duration}\n"
    text += f"━━━━━━━━━━━━━━━━━━━━\n🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    return send_message(text)


def notify_signal_detected(symbol, direction, ai_proba, conf, zone, chop):
    emoji = "🎯" if not chop else "⚠️"
    text = (
        f"{emoji} <b>Signal {direction}</b> — {symbol}\n"
        f"AI: {ai_proba:.1%} | Conf: {conf:.2f} | Zone: {zone}\n"
        f"{'⛔ Bloqué (range)' if chop else '✅ Valide'}\n"
        f"🕐 {datetime.now().strftime('%H:%M:%S')}"
    )
    return send_message(text)


def notify_bot_start(symbols, timeframe, paper):
    mode = "📋 PAPER (simulation)" if paper else "🔴 LIVE TRADING"
    text = (
        f"🤖 <b>Confluence AI Bot démarré</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"⚙️ <b>Mode      :</b> {mode}\n"
        f"📊 <b>Symboles  :</b> {', '.join(symbols)}\n"
        f"⏱️ <b>Timeframe :</b> {timeframe}\n"
        f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    return send_message(text)


def notify_bot_stop(realized_pnl, n_trades):
    emoji = "✅" if realized_pnl >= 0 else "❌"
    text = (
        f"🛑 <b>Bot arrêté</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Trades   :</b> {n_trades}\n"
        f"{emoji} <b>P&L total:</b> {realized_pnl:+.2f}$\n"
        f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    return send_message(text)


def notify_error(symbol, error):
    text = (
        f"⚠️ <b>Erreur bot</b>\n"
        f"Symbole: {symbol}\n"
        f"Erreur: {error[:200]}\n"
        f"🕐 {datetime.now().strftime('%H:%M:%S')}"
    )
    return send_message(text)


def notify_daily_summary(stats):
    pnl   = stats.get("pnl_usd", 0)
    emoji = "✅" if pnl >= 0 else "❌"
    text = (
        f"📅 <b>Résumé quotidien</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Trades    :</b> {stats.get('n_trades', 0)}\n"
        f"🏆 <b>Gagnants  :</b> {stats.get('winners', 0)}\n"
        f"💔 <b>Perdants  :</b> {stats.get('losers', 0)}\n"
        f"🎯 <b>Win Rate  :</b> {stats.get('win_rate', 0):.1f}%\n"
        f"{emoji} <b>P&L jour  :</b> {pnl:+.2f}$\n"
        f"💰 <b>Balance   :</b> {stats.get('balance', 0):.2f}$\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    return send_message(text)


def test_connection():
    ok = send_message(
        "✅ <b>Test Confluence AI Bot</b>\n"
        "Connexion Telegram fonctionnelle!\n"
        f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print("✅ Telegram OK!" if ok else "❌ Telegram ERREUR — vérifier TELEGRAM_TOKEN dans .env")
    return ok


if __name__ == "__main__":
    test_connection()
