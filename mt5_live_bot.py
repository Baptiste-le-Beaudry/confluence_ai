"""
mt5_live_bot.py
Bot de trading live multi-symboles sur MetaTrader 5.

Symboles supportés: BTCUSD, XAUUSD, EURUSD, GBPUSD, USDJPY, etc.
Timeframes: M5, M15, M30, H1, H4

Flux par symbole (toutes les N secondes):
  1. Récupérer les barres OHLCV depuis MT5
  2. Calculer tous les indicateurs
  3. Prédire avec l'AI (réseau de neurones)
  4. Vérifier les filtres (chop, cooldown, confluence)
  5. Ouvrir / fermer les positions selon le signal
  6. Logger le statut

Utilisation:
  python mt5_live_bot.py
  python mt5_live_bot.py --symbols BTCUSD XAUUSD EURUSD
  python mt5_live_bot.py --symbols BTCUSD --tf M15 --paper
  python mt5_live_bot.py --account 12345 --password xxx --server Broker-MT5
"""

import argparse
import logging
import signal
import sys
import time
import os
from datetime import datetime
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from indicators import compute_all
from ai_model import NeuralNet, build_features, compute_label_continuous, pretrain_on_history
from mt5_broker import MT5Broker, ASSET_CONFIG
from telegram_notify import notify_trade_open, notify_trade_close

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/mt5_bot.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger("confluence_bot")

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

# Mapping timeframe string → constante MT5
TF_MAP = {
    "M1":  1,    # mt5.TIMEFRAME_M1
    "M5":  5,    # mt5.TIMEFRAME_M5
    "M15": 15,   # mt5.TIMEFRAME_M15
    "M30": 30,   # mt5.TIMEFRAME_M30
    "H1":  16385, # mt5.TIMEFRAME_H1
    "H4":  16388, # mt5.TIMEFRAME_H4
    "D1":  16408, # mt5.TIMEFRAME_D1
}

TF_SECONDS = {
    "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400, "D1": 86400,
}

BARS_PER_DAY = {
    "M1": 1440, "M5": 288, "M15": 96, "M30": 48,
    "H1": 24, "H4": 6, "D1": 1,
}


@dataclass
class SymbolState:
    """État courant d'un symbole géré par le bot."""
    symbol:           str
    nn:               NeuralNet = field(default_factory=lambda: NeuralNet(lr=0.02))
    pending:          deque = field(default_factory=lambda: deque(maxlen=20))
    last_signal_bar:  int = -999
    thresh_long:      float = 0.62
    thresh_short:     float = 0.38
    last_bar_time:    datetime = None
    pretrained:       bool = False
    n_bars_seen:      int = 0
    dv_win:  deque = field(default_factory=lambda: deque(maxlen=50))
    et_win:  deque = field(default_factory=lambda: deque(maxlen=50))
    gp_win:  deque = field(default_factory=lambda: deque(maxlen=50))
    mr_win:  deque = field(default_factory=lambda: deque(maxlen=50))
    paper_trade: dict | None = None

    def get_norm_windows(self) -> dict:
        return {
            "vwap_std": max(np.std(self.dv_win), 0.05) if len(self.dv_win) > 5 else 0.5,
            "ema_std":  max(np.std(self.et_win), 0.05) if len(self.et_win) > 5 else 0.05,
            "gann_std": max(np.std(self.gp_win), 0.10) if len(self.gp_win) > 5 else 0.5,
            "mom_std":  max(np.std(self.mr_win), 0.10) if len(self.mr_win) > 5 else 1.0,
        }


# ─────────────────────────────────────────────
# Bot principal
# ─────────────────────────────────────────────

class MT5LiveBot:
    """
    Bot de trading live multi-symboles connecté à MetaTrader 5.
    """

    def __init__(
        self,
        symbols:         list[str],
        timeframe:       str   = "M15",
        n_bars:          int   = 500,
        loop_seconds:    int   = 60,
        capital:         float = 10_000.0,
        risk_pct:        float = 0.01,
        sl_atr_mult:     float = 2.0,
        tp_atr_mult:     float = 4.0,
        target_tpd:      float = 1.0,
        conf_pct:        float = 0.30,
        cooldown_bars:   int   = 10,
        use_chop:        bool  = True,
        paper_mode:      bool  = False,
        force_mt5_data:  bool  = False,
        pretrain_bars:   int   = 300,
        magic:           int   = 20240101,
    ):
        self.symbols       = [s.upper() for s in symbols]
        self.timeframe     = timeframe
        self.tf_mt5        = TF_MAP.get(timeframe, 15)
        self.tf_seconds    = TF_SECONDS.get(timeframe, 900)
        self.n_bars        = n_bars
        self.loop_seconds  = loop_seconds
        self.capital       = capital
        self.risk_pct      = risk_pct
        self.sl_atr_mult   = sl_atr_mult
        self.tp_atr_mult   = tp_atr_mult
        self.target_tpd    = target_tpd
        self.conf_pct      = conf_pct
        self.cooldown_bars = cooldown_bars
        self.use_chop      = use_chop
        self.paper_mode    = paper_mode
        self.force_mt5_data = force_mt5_data
        self.pretrain_bars = pretrain_bars
        self.bars_per_day  = BARS_PER_DAY.get(timeframe, 96)

        # État par symbole
        self.states: dict[str, SymbolState] = {
            s: SymbolState(symbol=s) for s in self.symbols
        }

        # Broker (sera initialisé dans run())
        self.broker: Optional[MT5Broker] = None
        self.magic  = magic

        # Pour arrêt propre
        self._running = False

    # ─────────────────────────────────────────
    # Démarrage
    # ─────────────────────────────────────────

    def run(self, account=0, password="", server=""):
        """
        Démarre le bot live.

        Args:
            account:  Numéro de compte MT5
            password: Mot de passe MT5
            server:   Serveur du broker
        """
        os.makedirs("logs", exist_ok=True)

        logger.info("=" * 60)
        logger.info("  CONFLUENCE AI BOT v4.0 — MetaTrader 5")
        logger.info("=" * 60)
        logger.info(f"  Mode        : {'PAPER (simulation)' if self.paper_mode else '🔴 LIVE TRADING'}")
        logger.info(f"  Symboles    : {', '.join(self.symbols)}")
        logger.info(f"  Timeframe   : {self.timeframe}")
        logger.info(f"  Risque/trade: {self.risk_pct*100:.1f}%")
        logger.info(f"  SL/TP       : {self.sl_atr_mult}× / {self.tp_atr_mult}× ATR")
        logger.info(f"  Confluence  : ≥ {self.conf_pct*100:.0f}% du score max")
        logger.info(f"  Cooldown    : {self.cooldown_bars} barres")

        if not self.paper_mode or self.force_mt5_data:
            logger.warning("⚠️  MODE LIVE — Des positions RÉELLES vont être ouvertes!")

        # Connexion MT5
        if not self.paper_mode or self.force_mt5_data:
            self.broker = MT5Broker(
                account=account, password=password,
                server=server, magic=self.magic
            )
            if not self.broker.connect():
                logger.error("Impossible de se connecter à MT5. Arrêt.")
                return
            if not self.paper_mode:
                logger.info(self.broker.portfolio_summary())

        # Signal handlers pour arrêt propre
        signal.signal(signal.SIGINT,  self._handle_stop)
        signal.signal(signal.SIGTERM, self._handle_stop)

        self._running = True
        iteration = 0

        logger.info(f"\nBoucle principale démarrée (intervalle: {self.loop_seconds}s)\n")

        while self._running:
            iteration += 1
            loop_start = time.monotonic()

            logger.info(f"─── Itération {iteration} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ───")

            for symbol in self.symbols:
                try:
                    self._process_symbol(symbol)
                except Exception as e:
                    logger.error(f"Erreur sur {symbol}: {e}", exc_info=True)

            # Statut portfolio
            if iteration % 5 == 0:
                if self.broker:
                    logger.info(self.broker.portfolio_summary())
                else:
                    self._log_paper_status()

            # Attendre
            elapsed = time.monotonic() - loop_start
            sleep   = max(0, self.loop_seconds - elapsed)
            if sleep > 0 and self._running:
                time.sleep(sleep)

        # Arrêt propre
        logger.info("Arrêt du bot...")
        if self.broker:
            if not self.paper_mode:
                logger.info("Fermeture de toutes les positions...")
                self.broker.close_all()
            self.broker.disconnect()
        logger.info("Bot arrêté.")

    # ─────────────────────────────────────────
    # Traitement par symbole
    # ─────────────────────────────────────────

    def _process_symbol(self, symbol: str):
        """
        Traite un symbole: récupère les données, calcule les indicateurs,
        prédit et exécute le signal si nécessaire.
        """
        state = self.states[symbol]

        # ── 1. Récupérer les données ──
        df_raw = self._get_data(symbol)
        if df_raw is None or len(df_raw) < 230:
            logger.warning(f"{symbol}: données insuffisantes ({len(df_raw) if df_raw is not None else 0} barres)")
            return

        # Vérifier si une nouvelle barre est apparue
        last_bar = df_raw.index[-1]
        if state.last_bar_time == last_bar:
            logger.debug(f"{symbol}: pas de nouvelle barre, skip")
            return
        state.last_bar_time = last_bar
        state.n_bars_seen  += 1

        # ── 2. Calculer les indicateurs ──
        df = compute_all(df_raw)
        if df is None or len(df) == 0:
            return

        # ── 3. Pré-entraîner si première fois ──
        if not state.pretrained and len(df) >= self.pretrain_bars:
            logger.info(f"{symbol}: pré-entraînement AI sur {len(df)} barres...")
            norm_w = self._get_norm_w(df)
            pretrain_on_history(
                state.nn, df, norm_w,
                fwd_bars   = self.cooldown_bars,
                sl_atr     = self.sl_atr_mult,
                tp_atr     = self.tp_atr_mult,
                n_epochs   = 10,
                verbose    = True,
            )
            # Calibrer les seuils
            state.thresh_long, state.thresh_short = self._calibrate_thresholds(
                state.nn, df, norm_w)
            state.pretrained = True
            logger.info(f"{symbol}: seuils calibrés: LONG>{state.thresh_long:.3f} | "
                        f"SHORT<{state.thresh_short:.3f}")

        # ── 4. Traiter la dernière barre ──
        row      = df.iloc[-1]
        close    = float(row["close"])
        atr      = float(row["atr"])
        bar_idx  = len(df) - 1

        if np.isnan(atr) or atr <= 0:
            return

        # Mise à jour normalisation
        state.dv_win.append(abs(row.get("dist_vwap", 0.0)))
        state.et_win.append(abs(row.get("ema_trend", 0.0) * 100))
        state.gp_win.append(abs(row.get("gann_pos",  0.0)))
        state.mr_win.append(abs(row.get("mom_raw",   0.0)))

        norm_w   = state.get_norm_windows()
        row_dict = row.to_dict()
        row_dict["close"] = close
        row_dict["atr"]   = atr

        fv = build_features(row_dict, norm_w)

        # Online learning
        state.pending.append((fv, bar_idx))
        if len(state.pending) >= 5:
            old_fv, _ = state.pending[0]
            lbl = compute_label_continuous(close, old_fv.ref_close, old_fv.ref_atr)
            if lbl is not None:
                state.nn.train_step(old_fv, lbl)

        # Recalibrer les seuils toutes les 100 barres
        if state.n_bars_seen % 100 == 0 and state.pretrained:
            state.thresh_long, state.thresh_short = self._calibrate_thresholds(
                state.nn, df, norm_w)
            logger.info(f"{symbol}: recalibration seuils → "
                        f"LONG>{state.thresh_long:.3f} SHORT<{state.thresh_short:.3f}")

        # ── 5. Prédiction AI ──
        ai_proba = state.nn.predict(fv)

        # ── 6. Conditions de signal ──
        conf       = float(row.get("conf_score", 0.0))
        max_conf   = float(df["max_conf"].iloc[-1])
        is_chop    = bool(row.get("is_chop",    False))
        in_ob_bull = bool(row.get("in_ob_bull", False))
        in_ob_bear = bool(row.get("in_ob_bear", False))
        in_fvg_bull= bool(row.get("in_fvg_bull",False))
        in_fvg_bear= bool(row.get("in_fvg_bear",False))

        in_zone_bull = in_ob_bull or in_fvg_bull
        in_zone_bear = in_ob_bear or in_fvg_bear

        ai_long  = (ai_proba > state.thresh_long)  and in_zone_bull
        ai_short = (ai_proba < state.thresh_short) and in_zone_bear
        cf_long  = (conf >= max_conf * self.conf_pct)  and in_zone_bull
        cf_short = (conf <= -max_conf * self.conf_pct) and in_zone_bear

        # AI ET confluence requis ensemble
        long_cond  = ai_long  and cf_long
        short_cond = ai_short and cf_short

        if self.use_chop and is_chop:
            long_cond = short_cond = False

        cooldown_ok  = (bar_idx - state.last_signal_bar) >= self.cooldown_bars
        long_signal  = long_cond  and cooldown_ok
        short_signal = short_cond and cooldown_ok

        # ── 7. Récupérer position actuelle ──
        current_dir = self._get_current_direction(symbol)

        # ── 8. Fermer si signal opposé ──
        if current_dir == "LONG"  and short_signal:
            self._close(symbol, reason="REVERSE→SHORT", exit_price=close)
            current_dir = None
        elif current_dir == "SHORT" and long_signal:
            self._close(symbol, reason="REVERSE→LONG", exit_price=close)
            current_dir = None

        # ── 9. Ouvrir nouveau trade ──
        if current_dir is None and (long_signal or short_signal):
            direction = "LONG" if long_signal else "SHORT"
            sl = close - self.sl_atr_mult * atr if direction == "LONG" else close + self.sl_atr_mult * atr
            tp = close + self.tp_atr_mult * atr if direction == "LONG" else close - self.tp_atr_mult * atr

            cfg      = ASSET_CONFIG.get(symbol, ASSET_CONFIG["DEFAULT"])
            risk_usd = self.capital * self.risk_pct

            logger.info(
                f"🎯 SIGNAL {direction} | {symbol} | "
                f"Prix={close:.5f} | ATR={atr:.5f} | "
                f"AI={ai_proba:.3f} | Conf={conf:.2f} | "
                f"SL={sl:.5f} | TP={tp:.5f}"
            )

            self._open(symbol, direction, sl, tp, risk_usd,
                       conf_score=conf, ai_proba=ai_proba, entry_price=close)
            state.last_signal_bar = bar_idx

        else:
            logger.info(
                f"{symbol} | {close:.5f} | "
                f"AI={ai_proba:.3f} (L>{state.thresh_long:.3f} S<{state.thresh_short:.3f}) | "
                f"Conf={conf:.2f} | Pos={current_dir or 'FLAT'} | "
                f"Zone={'Bull' if in_zone_bull else 'Bear' if in_zone_bear else 'None'} | "
                f"Chop={'OUI' if is_chop else 'non'}"
            )

    # ─────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────

    def _get_data(self, symbol: str):
        """Récupère les données OHLCV — depuis MT5 ou synthétiques (paper)."""
        if self.paper_mode and not self.force_mt5_data:
            # En paper mode: générer des données de test
            from data_loader import generate_sample_data
            return generate_sample_data(n_bars=self.n_bars, seed=hash(symbol) % 999)
        elif self.broker is not None:
            try:
                return self.broker.get_ohlcv(symbol, self.tf_mt5, self.n_bars)
            except Exception as e:
                logger.error(f"Erreur récupération données {symbol}: {e}")
                return None
        else:
            from data_loader import generate_sample_data
            return generate_sample_data(n_bars=self.n_bars, seed=hash(symbol) % 999)

    def _get_current_direction(self, symbol: str) -> Optional[str]:
        """Retourne la direction de la position ouverte (LONG/SHORT/None)."""
        if self.paper_mode:
            trade = self.states[symbol].paper_trade
            return trade["direction"] if trade else None
        if self.broker is None:
            return None
        return self.broker.get_position_direction(symbol)

    def _open(self, symbol: str, direction: str, sl: float, tp: float,
              risk_usd: float, conf_score: float = 0.0, ai_proba: float = 0.5,
              entry_price: float | None = None):
        """Ouvre un trade (réel ou simulé)."""
        if self.paper_mode:
            state = self.states[symbol]
            entry = float(entry_price if entry_price is not None else (sl + tp) / 2)
            state.paper_trade = {
                "direction": direction,
                "entry_price": entry,
                "sl": sl,
                "tp": tp,
                "risk_usd": risk_usd,
                "conf_score": conf_score,
                "ai_proba": ai_proba,
                "entry_time": datetime.now(),
            }
            logger.info(f"[PAPER] OPEN {direction} {symbol} | "
                        f"Entry={entry:.5f} SL={sl:.5f} TP={tp:.5f} | Risk=${risk_usd:.2f}")
            notify_trade_open(
                symbol=symbol,
                direction=direction,
                price=entry,
                sl=sl,
                tp=tp,
                lot=0.0,
                risk_usd=risk_usd,
                ai_proba=ai_proba,
                conf_score=conf_score,
                timeframe=self.timeframe,
            )
        else:
            return self.broker.open_trade(
                symbol=symbol, direction=direction,
                sl=sl, tp=tp, risk_usd=risk_usd,
                conf_score=conf_score, ai_proba=ai_proba,
            )

    def _close(self, symbol: str, reason: str = "", exit_price: float | None = None):
        """Ferme un trade (réel ou simulé)."""
        if self.paper_mode:
            state = self.states[symbol]
            trade = state.paper_trade
            if not trade:
                logger.info(f"[PAPER] CLOSE {symbol} | raison: {reason}")
                return

            entry = float(trade["entry_price"])
            exit_px = float(exit_price if exit_price is not None else entry)
            sl = float(trade["sl"])
            direction = trade["direction"]
            risk_usd = float(trade.get("risk_usd", 0.0))
            sl_dist = max(abs(entry - sl), 1e-9)
            if direction == "LONG":
                pnl_usd = risk_usd * ((exit_px - entry) / sl_dist)
            else:
                pnl_usd = risk_usd * ((entry - exit_px) / sl_dist)
            pnl_pct = (pnl_usd / risk_usd * 100.0) if risk_usd else 0.0

            notify_trade_close(
                symbol=symbol,
                direction=direction,
                entry=entry,
                exit_price=exit_px,
                pnl_usd=pnl_usd,
                pnl_pct=pnl_pct,
                reason=reason,
                duration=str(datetime.now() - trade["entry_time"]),
            )
            logger.info(f"[PAPER] CLOSE {symbol} | raison: {reason} | Exit={exit_px:.5f}")
            state.paper_trade = None
        else:
            self.broker.close_position(symbol, reason=reason)

    def _get_norm_w(self, df) -> dict:
        """Calcule les fenêtres de normalisation depuis un DataFrame."""
        return {
            "vwap_std": max(df["dist_vwap"].abs().std(),  0.05),
            "ema_std":  max((df["ema_trend"]*100).abs().std(), 0.05),
            "gann_std": max(df["gann_pos"].abs().std(),   0.10),
            "mom_std":  max(df["mom_raw"].abs().std(),    0.10),
        }

    def _calibrate_thresholds(self, nn: NeuralNet, df, norm_w: dict) -> tuple[float, float]:
        """Calcule les seuils adaptatifs sur les données disponibles."""
        probas = []
        for _, row in df.iloc[220:].iterrows():
            rd = row.to_dict(); rd["close"] = row["close"]; rd["atr"] = row["atr"]
            if np.isnan(rd.get("atr", np.nan)) or rd.get("atr", 0) <= 0:
                continue
            fv = build_features(rd, norm_w)
            probas.append(nn.predict(fv))

        if len(probas) < 30:
            return 0.62, 0.38

        probas = np.array(probas)
        frac   = np.clip(self.target_tpd / self.bars_per_day / 2.0, 0.003, 0.15)
        tl     = float(np.percentile(probas, (1.0 - frac) * 100))
        ts     = float(np.percentile(probas, frac * 100))
        return max(tl, 0.51), min(ts, 0.49)

    def _log_paper_status(self):
        """Log le statut en paper mode."""
        logger.info("=== Paper Mode Status ===")
        for sym, state in self.states.items():
            logger.info(f"  {sym}: {state.n_bars_seen} barres | "
                        f"pretrained={state.pretrained} | "
                        f"nn_samples={state.nn.n_samples}")

    def _handle_stop(self, signum, frame):
        logger.info("Signal d'arrêt reçu...")
        self._running = False


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Confluence AI Bot — MetaTrader 5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Paper trading (simulation, pas de position réelle)
  python mt5_live_bot.py --paper

  # Live sur BTC et Or uniquement
  python mt5_live_bot.py --symbols BTCUSD XAUUSD --account 12345 --password xxx --server MonBroker-MT5

  # Multi-forex + métaux, timeframe 1h
  python mt5_live_bot.py --symbols EURUSD GBPUSD USDJPY XAUUSD --tf H1

  # Tous les symboles configurés
  python mt5_live_bot.py --symbols BTCUSD XAUUSD EURUSD GBPUSD USDJPY USDCHF AUDUSD

Symboles disponibles:
  Crypto : BTCUSD, ETHUSD
  Métaux : XAUUSD, XAGUSD
  Forex  : EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, USDCAD, NZDUSD, EURGBP, EURJPY, GBPJPY
        """
    )

    parser.add_argument("--symbols",  nargs="+",
                        default=["BTCUSD", "XAUUSD", "EURUSD"],
                        help="Symboles à trader")
    parser.add_argument("--tf",       default="M15",
                        choices=list(TF_MAP.keys()),
                        help="Timeframe MT5")
    parser.add_argument("--bars",     type=int,   default=500,
                        help="Barres historiques à charger")
    parser.add_argument("--account",  type=int,   default=0,
                        help="Numéro de compte MT5")
    parser.add_argument("--password", type=str,   default="",
                        help="Mot de passe MT5")
    parser.add_argument("--server",   type=str,   default="",
                        help="Serveur broker (ex: ICMarkets-MT5)")
    parser.add_argument("--capital",  type=float, default=10_000.0,
                        help="Capital de référence pour le sizing")
    parser.add_argument("--risk",     type=float, default=0.01,
                        help="Risque par trade (fraction, ex: 0.01 = 1%%)")
    parser.add_argument("--sl",       type=float, default=2.0,
                        help="Stop loss en multiples d'ATR")
    parser.add_argument("--tp",       type=float, default=4.0,
                        help="Take profit en multiples d'ATR")
    parser.add_argument("--conf",     type=float, default=0.30,
                        help="Seuil confluence (fraction du score max)")
    parser.add_argument("--cooldown", type=int,   default=10,
                        help="Barres minimum entre deux trades")
    parser.add_argument("--loop",     type=int,   default=60,
                        help="Intervalle de la boucle en secondes")
    parser.add_argument("--target-tpd", type=float, default=1.0,
                        help="Trades par jour cible (pour les seuils adaptatifs)")
    parser.add_argument("--no-chop",  action="store_true",
                        help="Désactiver le filtre chop")
    parser.add_argument("--paper",    action="store_true",
                        help="Mode simulation (aucun ordre réel)")
    parser.add_argument("--magic",    type=int,   default=20240101,
                        help="Magic number MT5 (identifiant du bot)")

    args = parser.parse_args()

    if not args.paper and not args.account:
        print("\n⚠️  ATTENTION: Pas de numéro de compte fourni.")
        print("   Si MT5 est déjà connecté, le bot utilisera le compte actif.")
        print("   Pour spécifier: --account 12345 --password xxx --server NomBroker")
        print("   Pour la simulation: ajouter --paper\n")
        resp = input("Continuer quand même? (oui/non): ").strip().lower()
        if resp != "oui":
            sys.exit(0)

    if not args.paper:
        print(f"\n⚠️  MODE LIVE — Positions RÉELLES sur {', '.join(args.symbols)}")
        print(f"   Risque par trade: {args.risk*100:.1f}%")
        print(f"   Appuyer sur Ctrl+C pour arrêter proprement\n")
        resp = input("Confirmer le trading live (oui/non): ").strip().lower()
        if resp != "oui":
            sys.exit(0)

    os.makedirs("logs", exist_ok=True)

    bot = MT5LiveBot(
        symbols       = args.symbols,
        timeframe     = args.tf,
        n_bars        = args.bars,
        loop_seconds  = args.loop,
        capital       = args.capital,
        risk_pct      = args.risk,
        sl_atr_mult   = args.sl,
        tp_atr_mult   = args.tp,
        target_tpd    = args.target_tpd,
        conf_pct      = args.conf,
        cooldown_bars = args.cooldown,
        use_chop      = not args.no_chop,
        paper_mode    = args.paper,
        magic         = args.magic,
    )

    bot.run(
        account  = args.account,
        password = args.password,
        server   = args.server,
    )


if __name__ == "__main__":
    main()
