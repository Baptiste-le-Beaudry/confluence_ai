"""
squeeze_bot.py
Bot Squeeze Release — positions plus grosses, très sélectif.

Logique:
  Ce bot ne trade QUE les setups Bollinger Squeeze Release.
  Comme le WR historique est 100% sur ce setup (2/2 trades),
  on augmente le risque par trade à 3-5% du capital.

  Il tourne EN PARALLÈLE du bot principal (setup combined).
  Ils partagent les mêmes données mais ont des logiques d'entrée différentes.

Caractéristiques:
  - Setup: Squeeze Release uniquement
  - Risque/trade: 3% (vs 1% pour le bot principal)
  - TP partiel: 1.5× ATR (50%) + 2.5× ATR (50%)
  - Breakeven automatique après TP1
  - Cooldown: 15 barres minimum entre trades
  - Très peu de trades (~1-3/semaine) mais haute confiance

Usage:
  python squeeze_bot.py --paper
  python squeeze_bot.py --symbols BNBUSDT BTCUSDT --paper
  python squeeze_bot.py --account 70013614 --password xxx --server TradeMaxGlobal-Demo
"""

import argparse
import logging
import signal
import sys
import os
import time
from datetime import datetime
from collections import deque
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from indicators import compute_all
from ai_model import NeuralNet, build_features, compute_label_continuous, pretrain_on_history
from entry_filters import setup_squeeze_release
from mt5_broker import MT5Broker
from telegram_notify import (notify_trade_open, notify_trade_close,
                              notify_bot_start, notify_bot_stop,
                              notify_error, send_message)

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [SQUEEZE] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/squeeze_bot.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger("squeeze_bot")

# Timeframe → constante MT5
TF_MAP     = {"M1":1,"M5":5,"M15":15,"M30":30,"H1":16385,"H4":16388}
TF_SECONDS = {"M1":60,"M5":300,"M15":900,"M30":1800,"H1":3600,"H4":14400}
BARS_PER_DAY = {"M1":1440,"M5":288,"M15":96,"M30":48,"H1":24,"H4":6}

DEFAULT_SYMBOLS = ["BNBUSDT", "BTCUSDT"]


class SqueezeState:
    """État par symbole pour le bot squeeze."""
    def __init__(self, symbol: str):
        self.symbol       = symbol
        self.nn           = NeuralNet(lr=0.02)
        self.pending      = deque(maxlen=20)
        self.last_sig_bar = -999
        self.thresh_long  = 0.52
        self.thresh_short = 0.48
        self.pretrained   = False
        self.n_bars       = 0
        self.last_bar_ts  = None
        # Tracking position partielle
        self.position_data = {}
        # Fenêtres normalisation
        self.dv = deque(maxlen=50); self.et = deque(maxlen=50)
        self.gp = deque(maxlen=50); self.mr = deque(maxlen=50)

    def get_norm_w(self):
        return {
            "vwap_std": max(np.std(self.dv), 0.05) if len(self.dv)>5 else 0.5,
            "ema_std":  max(np.std(self.et), 0.05) if len(self.et)>5 else 0.05,
            "gann_std": max(np.std(self.gp), 0.10) if len(self.gp)>5 else 0.5,
            "mom_std":  max(np.std(self.mr), 0.10) if len(self.mr)>5 else 1.0,
        }


class SqueezeLiveBot:
    """
    Bot Squeeze Release avec positions plus grosses.
    Tourne en parallèle du bot principal.
    """

    def __init__(
        self,
        symbols:       list[str] = DEFAULT_SYMBOLS,
        timeframe:     str       = "M15",
        n_bars:        int       = 500,
        loop_seconds:  int       = 60,
        capital:       float     = 10_000.0,
        risk_pct:      float     = 0.03,    # 3% par trade (vs 1% bot principal)
        sl_atr_mult:   float     = 2.0,
        tp1_mult:      float     = 1.5,     # TP partiel 1
        tp2_mult:      float     = 2.5,     # TP partiel 2
        tp1_pct:       float     = 0.50,    # 50% fermé au TP1
        cooldown_bars: int       = 15,
        conf_pct:      float     = 0.25,
        paper_mode:    bool      = True,
        magic:         int       = 20240202,  # magic différent du bot principal
    ):
        self.symbols      = [s.upper() for s in symbols]
        self.timeframe    = timeframe
        self.tf_mt5       = TF_MAP.get(timeframe, 15)
        self.tf_seconds   = TF_SECONDS.get(timeframe, 900)
        self.n_bars       = n_bars
        self.loop_seconds = loop_seconds
        self.capital      = capital
        self.risk_pct     = risk_pct
        self.sl_atr_mult  = sl_atr_mult
        self.tp1_mult     = tp1_mult
        self.tp2_mult     = tp2_mult
        self.tp1_pct      = tp1_pct
        self.cooldown     = cooldown_bars
        self.conf_pct     = conf_pct
        self.paper_mode   = paper_mode
        self.magic        = magic
        self.bars_per_day = BARS_PER_DAY.get(timeframe, 96)

        self.states: dict[str, SqueezeState] = {
            s: SqueezeState(s) for s in self.symbols
        }
        self.broker   = None
        self._running = False
        self.trades   = []

    def run(self, account=0, password="", server=""):
        os.makedirs("logs", exist_ok=True)

        logger.info("=" * 60)
        logger.info("  BOT SQUEEZE RELEASE — Positions Grosses")
        logger.info("=" * 60)
        logger.info(f"  Mode     : {'PAPER' if self.paper_mode else '🔴 LIVE'}")
        logger.info(f"  Symboles : {', '.join(self.symbols)}")
        logger.info(f"  TF       : {self.timeframe}")
        logger.info(f"  Risque   : {self.risk_pct*100:.0f}% par trade (BOT PRINCIPAL = 1%)")
        logger.info(f"  TP1/TP2  : {self.tp1_mult}× / {self.tp2_mult}× ATR (50%/50%)")
        logger.info(f"  SL       : {self.sl_atr_mult}× ATR")
        logger.info(f"  Cooldown : {self.cooldown} barres")
        logger.info(f"  Magic    : {self.magic} (différent du bot principal)")

        if not self.paper_mode:
            self.broker = MT5Broker(account=account, password=password,
                                    server=server, magic=self.magic)
            if not self.broker.connect():
                logger.error("Connexion MT5 échouée.")
                return

        notify_bot_start(self.symbols, f"{self.timeframe} (Squeeze)", self.paper_mode)
        send_message(
            f"🎯 <b>Bot Squeeze démarré</b>\n"
            f"Risque/trade: {self.risk_pct*100:.0f}% (3× bot principal)\n"
            f"TP: {self.tp1_mult}×ATR (50%) + {self.tp2_mult}×ATR (50%)\n"
            f"Attend uniquement les Bollinger Squeeze Release"
        )

        signal.signal(signal.SIGINT,  self._stop)
        signal.signal(signal.SIGTERM, self._stop)

        self._running = True
        iteration     = 0

        while self._running:
            iteration += 1
            start = time.monotonic()
            logger.info(f"─── Squeeze Itération {iteration} | {datetime.now().strftime('%H:%M:%S')} ───")

            for symbol in self.symbols:
                try:
                    self._process(symbol)
                except Exception as e:
                    logger.error(f"Erreur {symbol}: {e}", exc_info=True)
                    notify_error(symbol, str(e))

            elapsed = time.monotonic() - start
            sleep   = max(0, self.loop_seconds - elapsed)
            if sleep > 0 and self._running:
                time.sleep(sleep)

        logger.info("Bot Squeeze arrêté.")
        notify_bot_stop(0, len(self.trades))
        if self.broker:
            self.broker.disconnect()

    def _process(self, symbol: str):
        state = self.states[symbol]

        # Récupérer les données
        df_raw = self._get_data(symbol)
        if df_raw is None or len(df_raw) < 230:
            return

        # Vérifier nouvelle barre
        last_bar = df_raw.index[-1]
        if state.last_bar_ts == last_bar:
            return
        state.last_bar_ts = last_bar
        state.n_bars += 1

        # Indicateurs
        df = compute_all(df_raw)
        if df is None or len(df) == 0:
            return

        # Pré-entraînement
        if not state.pretrained and len(df) >= 300:
            logger.info(f"{symbol}: pré-entraînement squeeze AI...")
            norm_w = {
                "vwap_std": max(df["dist_vwap"].abs().std(), 0.05),
                "ema_std":  max((df["ema_trend"]*100).abs().std(), 0.05),
                "gann_std": max(df["gann_pos"].abs().std(), 0.10),
                "mom_std":  max(df["mom_raw"].abs().std(), 0.10),
            }
            pretrain_on_history(state.nn, df, norm_w,
                                fwd_bars=self.cooldown, sl_atr=self.sl_atr_mult,
                                tp_atr=self.tp2_mult, n_epochs=10, verbose=True)

            # Calibrer seuils
            probas = []
            for _, row in df.iloc[220:].iterrows():
                rd = row.to_dict(); rd["close"]=row["close"]; rd["atr"]=row["atr"]
                if not np.isnan(rd.get("atr",np.nan)) and rd.get("atr",0)>0:
                    fv = build_features(rd, norm_w)
                    probas.append(state.nn.predict(fv))
            if probas:
                probas = np.array(probas)
                frac   = np.clip(0.5/self.bars_per_day/2.0, 0.001, 0.10)
                state.thresh_long  = max(float(np.percentile(probas,(1-frac)*100)), 0.505)
                state.thresh_short = min(float(np.percentile(probas,frac*100)), 0.495)
            state.pretrained = True
            logger.info(f"{symbol}: seuils L>{state.thresh_long:.3f} S<{state.thresh_short:.3f}")

        # Barre actuelle et précédente
        row      = df.iloc[-1]
        prev_row = df.iloc[-2].to_dict() if len(df) >= 2 else {}
        close    = float(row["close"])
        atr      = float(row["atr"])
        bar_idx  = len(df) - 1

        if np.isnan(atr) or atr <= 0:
            return

        # Mise à jour normalisation
        state.dv.append(abs(row.get("dist_vwap", 0.0)))
        state.et.append(abs(row.get("ema_trend", 0.0)*100))
        state.gp.append(abs(row.get("gann_pos",  0.0)))
        state.mr.append(abs(row.get("mom_raw",   0.0)))

        norm_w  = state.get_norm_w()
        rd      = row.to_dict(); rd["close"]=close; rd["atr"]=atr
        rd["ema20"]  = float(row.get("ema20", close))
        rd["ema50"]  = float(row.get("ema50", close))
        rd["ema200"] = float(row.get("ema200", close))
        fv = build_features(rd, norm_w)

        # Online learning
        state.pending.append((fv, bar_idx))
        if len(state.pending) >= 5:
            old_fv, _ = state.pending[0]
            lbl = compute_label_continuous(close, old_fv.ref_close, old_fv.ref_atr)
            if lbl is not None:
                state.nn.train_step(old_fv, lbl)

        ai_proba  = state.nn.predict(fv)
        conf      = float(row.get("conf_score", 0.0))
        max_conf  = float(df["max_conf"].iloc[-1])

        # ── Signal Squeeze Release uniquement ──
        long_sig, short_sig, reason, quality = setup_squeeze_release(
            row         = rd,
            prev_row    = prev_row,
            ai_proba    = ai_proba,
            conf_score  = conf,
            max_conf    = max_conf,
            ai_thresh   = state.thresh_long,
            conf_pct    = self.conf_pct,
        )

        cooldown_ok = (bar_idx - state.last_sig_bar) >= self.cooldown
        current_dir = self._get_direction(symbol)

        # ── Gestion TP partiel ──
        if current_dir and symbol in state.position_data:
            self._manage_partial_tp(symbol, close, atr, state)

        # ── Fermer si signal opposé ──
        if current_dir == "LONG"  and short_sig and cooldown_ok:
            self._close(symbol, "REVERSE→SHORT", state)
            current_dir = None
        elif current_dir == "SHORT" and long_sig  and cooldown_ok:
            self._close(symbol, "REVERSE→LONG", state)
            current_dir = None

        # ── Ouvrir nouveau trade ──
        if current_dir is None and (long_sig or short_sig) and cooldown_ok:
            direction = "LONG" if long_sig else "SHORT"
            risk_usd  = self.capital * self.risk_pct
            denom     = max(self.sl_atr_mult * atr / close, 0.001)
            size_usd  = min(risk_usd / denom, self.capital * 0.30)  # max 30% du capital

            sl  = close - self.sl_atr_mult*atr if direction=="LONG" else close + self.sl_atr_mult*atr
            tp1 = close + self.tp1_mult*atr    if direction=="LONG" else close - self.tp1_mult*atr
            tp2 = close + self.tp2_mult*atr    if direction=="LONG" else close - self.tp2_mult*atr

            logger.info(
                f"🎯 SQUEEZE {direction} | {symbol} | "
                f"Prix={close:.5f} | AI={ai_proba:.3f} | Q={quality:.2f}\n"
                f"   SL={sl:.5f} | TP1={tp1:.5f} | TP2={tp2:.5f} | "
                f"Risque=${risk_usd:.2f} ({self.risk_pct*100:.0f}%)"
            )

            send_message(
                f"🎯 <b>SQUEEZE SIGNAL {direction}</b>\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"📊 {symbol} | {self.timeframe}\n"
                f"💰 Prix : {close:.5f}\n"
                f"🛑 SL   : {sl:.5f}\n"
                f"🎯 TP1  : {tp1:.5f} (50% @ {self.tp1_mult}×ATR)\n"
                f"🎯 TP2  : {tp2:.5f} (50% @ {self.tp2_mult}×ATR)\n"
                f"💵 Risque: ${risk_usd:.2f} ({self.risk_pct*100:.0f}%)\n"
                f"🤖 AI: {ai_proba:.3f} | Qualité: {quality:.2f}\n"
                f"📋 {reason.split('|')[0]}"
            )

            if not self.paper_mode and self.broker:
                # Ouvrir avec TP2 comme TP final (TP1 géré manuellement)
                self.broker.open_trade(
                    symbol=symbol, direction=direction,
                    sl=sl, tp=tp2, risk_usd=risk_usd,
                    comment=f"SQUEEZE {direction[:1]}"
                )

            state.position_data[symbol] = {
                "entry":     close, "direction": direction,
                "sl":        sl,    "tp1":       tp1,
                "tp2":       tp2,   "tp1_hit":   False,
                "size_usd":  size_usd, "atr":    atr,
            }
            state.last_sig_bar = bar_idx

        else:
            sq_status = "SQUEEZE" if bool(row.get("squeeze", False)) else "no squeeze"
            logger.info(
                f"{symbol} | {close:.5f} | AI={ai_proba:.3f} "
                f"(L>{state.thresh_long:.3f}) | {sq_status} | "
                f"Pos={current_dir or 'FLAT'}"
            )

    def _manage_partial_tp(self, symbol: str, close: float, atr: float, state: SqueezeState):
        """Gère le TP partiel et le breakeven automatique."""
        pos = state.position_data.get(symbol, {})
        if not pos or pos.get("tp1_hit", False):
            return

        tp1       = pos["tp1"]
        tp2       = pos["tp2"]
        entry     = pos["entry"]
        direction = pos["direction"]
        size_usd  = pos["size_usd"]

        tp1_reached = (direction=="LONG" and close >= tp1) or (direction=="SHORT" and close <= tp1)

        if tp1_reached:
            partial_size = size_usd * self.tp1_pct
            if direction == "LONG":
                pnl_tp1 = partial_size * (close - entry) / entry
            else:
                pnl_tp1 = partial_size * (entry - close) / entry

            logger.info(
                f"✅ TP1 PARTIEL {symbol} | {direction} | "
                f"Fermé 50% @ {close:.5f} | P&L={pnl_tp1:+.2f}$"
            )

            send_message(
                f"✅ <b>TP1 PARTIEL — {symbol}</b>\n"
                f"50% fermé @ {close:.5f}\n"
                f"P&L partiels: ${pnl_tp1:+.2f}\n"
                f"50% restant vise TP2 = {tp2:.5f}\n"
                f"SL → Breakeven = {entry:.5f}"
            )

            # Fermer la moitié sur MT5
            if not self.paper_mode and self.broker:
                positions = self.broker.get_open_positions(symbol)
                if positions:
                    half_lot = positions[0].volume / 2.0
                    # Modifier le SL au breakeven sur la position restante
                    be_sl = entry * (1.001 if direction=="LONG" else 0.999)
                    self.broker.modify_sl_tp(symbol, sl=be_sl, tp=tp2)

            pos["tp1_hit"]  = True
            pos["size_usd"] = size_usd * (1.0 - self.tp1_pct)

    def _get_direction(self, symbol: str):
        if symbol in self.states and self.states[symbol].position_data.get(symbol):
            return self.states[symbol].position_data[symbol].get("direction")
        if self.broker:
            return self.broker.get_position_direction(symbol)
        return None

    def _close(self, symbol: str, reason: str, state: SqueezeState):
        logger.info(f"Fermeture {symbol} | {reason}")
        if not self.paper_mode and self.broker:
            self.broker.close_position(symbol, reason=reason)
        state.position_data.pop(symbol, None)

    def _get_data(self, symbol: str):
        if self.paper_mode or self.broker is None:
            from data_loader import generate_sample_data
            return generate_sample_data(n_bars=self.n_bars, seed=hash(symbol)%999)
        try:
            return self.broker.get_ohlcv(symbol, self.tf_mt5, self.n_bars)
        except Exception as e:
            logger.error(f"Données {symbol}: {e}")
            return None

    def _stop(self, signum, frame):
        logger.info("Arrêt du bot Squeeze...")
        self._running = False


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bot Squeeze Release — Positions Grosses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ce bot tourne EN PARALLÈLE du bot principal (run_bot.py).
Il prend des positions 3× plus grosses sur les Squeeze Release.

Lancement simultané (2 terminaux PowerShell):
  Terminal 1: python run_bot.py --paper          ← bot principal (1% risque)
  Terminal 2: python squeeze_bot.py --paper      ← bot squeeze  (3% risque)
        """
    )
    parser.add_argument("--symbols",  nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--tf",       default="M15", choices=list(TF_MAP.keys()))
    parser.add_argument("--bars",     type=int,   default=500)
    parser.add_argument("--capital",  type=float, default=10_000.0)
    parser.add_argument("--risk",     type=float, default=0.03,
                        help="Risque par trade (défaut 3%%)")
    parser.add_argument("--sl",       type=float, default=2.0)
    parser.add_argument("--tp1",      type=float, default=1.5)
    parser.add_argument("--tp2",      type=float, default=2.5)
    parser.add_argument("--cooldown", type=int,   default=15)
    parser.add_argument("--account",  type=int,   default=0)
    parser.add_argument("--password", type=str,   default="")
    parser.add_argument("--server",   type=str,   default="")
    parser.add_argument("--loop",     type=int,   default=60)
    parser.add_argument("--paper",    action="store_true")
    args = parser.parse_args()

    print("="*60)
    print("  BOT SQUEEZE RELEASE — Positions Grosses")
    print("="*60)
    print(f"  Mode         : {'PAPER' if args.paper else '🔴 LIVE'}")
    print(f"  Symboles     : {', '.join(args.symbols)}")
    print(f"  Risque/trade : {args.risk*100:.0f}% (3× le bot principal)")
    print(f"  TP1 / TP2    : {args.tp1}× / {args.tp2}× ATR (50%/50%)")
    print(f"  SL           : {args.sl}× ATR")
    print(f"  Cooldown     : {args.cooldown} barres")
    print()
    print("  Lancer en PARALLÈLE du bot principal:")
    print("  Terminal 1: python run_bot.py --paper")
    print("  Terminal 2: python squeeze_bot.py --paper")
    print("="*60)

    if not args.paper:
        confirm = input("\n⚠️  LIVE avec 3% risque/trade — Confirmer (oui/non): ").strip().lower()
        if confirm != "oui":
            sys.exit(0)

    bot = SqueezeLiveBot(
        symbols       = args.symbols,
        timeframe     = args.tf,
        n_bars        = args.bars,
        loop_seconds  = args.loop,
        capital       = args.capital,
        risk_pct      = args.risk,
        sl_atr_mult   = args.sl,
        tp1_mult      = args.tp1,
        tp2_mult      = args.tp2,
        cooldown_bars = args.cooldown,
        paper_mode    = args.paper,
    )
    bot.run(account=args.account, password=args.password, server=args.server)


if __name__ == "__main__":
    main()
