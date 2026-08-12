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

# Forcer l'UTF-8 sur la console Windows — sinon les emojis (🔴, ⚠️, ─, ≥, etc.)
# dans les print()/logs font planter le script (UnicodeEncodeError, cp1252)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from indicators import compute_all
from ai_model import NeuralNet, FeatVec, build_features, compute_label_continuous, pretrain_on_history
from entry_filters import check_hard_rules
import trade_memory
from mt5_broker import MT5Broker
from dashboard import bot_data, start_server
from telegram_notify import (
    notify_trade_open, notify_trade_close, notify_bot_start,
    notify_bot_stop, notify_error, notify_daily_summary, notify_signal_detected
)

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
        timeframe:       str   = "H1",
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
        pretrain_bars:   int   = 300,
        magic:           int   = 20240101,
        dashboard_port:  int   = 5000,
        start_dashboard: bool  = True,
        memory_dir:      str   = "memory",
        base_sim_bars:   int   = 5000,
        save_every_iters: int  = 50,
        replay_every_bars: int = 200,
        replay_n:        int   = 20,
        long_only:       bool  = True,
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
        self.long_only     = long_only
        self.paper_mode    = paper_mode
        self.pretrain_bars = pretrain_bars
        self.bars_per_day  = BARS_PER_DAY.get(timeframe, 96)
        # Breakeven
        self.use_breakeven  = True
        self.be_trigger_pct = 0.50   # déclencher à 50% du chemin vers le TP
        self.be_offset_pct  = 0.001  # buffer de 0.1% au-dessus du prix d'entrée
        # Trailing stop
        self.use_trailing   = False
        self.trail_atr_mult = 1.0
        # Tracker les positions et leur prix d'entrée/BE
        self.position_data: dict = {}   # symbol → {entry, sl, tp, direction, be_triggered}
        # Dashboard web
        self.dashboard_port  = dashboard_port
        self.start_dashboard = start_dashboard

        # Mémoire persistante (poids AI + base de trades)
        self.memory_dir        = memory_dir
        self.base_sim_bars     = base_sim_bars
        self.save_every_iters  = save_every_iters
        self.replay_every_bars = replay_every_bars
        self.replay_n          = replay_n

        # État par symbole
        self.states: dict[str, SymbolState] = {
            s: SymbolState(symbol=s) for s in self.symbols
        }

        # Broker (sera initialisé dans run())
        self.broker: Optional[MT5Broker] = None
        self.magic  = magic

        # Historique trades
        self.trades: list = []
        # Pour arrêt propre
        self._running = False

    # ─────────────────────────────────────────
    # Mémoire persistante (Phases 1-3)
    # ─────────────────────────────────────────

    def _weights_path(self, symbol: str) -> str:
        return os.path.join(self.memory_dir, f"{symbol}_weights.npz")

    def _trades_db_path(self) -> str:
        return os.path.join(self.memory_dir, "trades_memory.db")

    def _init_symbol_memory(self, symbol: str):
        """
        Démarrage d'un symbole:
          1. Charge W_live depuis disque (mémoire accumulée entre sessions).
          2. Simule sur un historique long (base_sim_bars) pour obtenir W_base,
             écrasé à chaque démarrage — l'ancre reflète le régime actuel.
          3. Si aucune mémoire live n'existait, part de W_base plutôt que de
             poids aléatoires.
        """
        state = self.states[symbol]

        loaded = NeuralNet.load(self._weights_path(symbol), lr=state.nn.lr)
        if loaded is not None:
            state.nn = loaded
            logger.info(f"{symbol}: mémoire W_live chargée "
                        f"({loaded.n_samples} échantillons appris précédemment)")
        else:
            logger.info(f"{symbol}: pas de mémoire W_live compatible — démarrage à neuf")

        df_raw = None
        if self.broker:
            try:
                df_raw = self.broker.get_ohlcv(symbol, self.tf_mt5, self.base_sim_bars)
            except Exception as e:
                logger.warning(f"{symbol}: échec récupération historique d'ancrage: {e}")
        if df_raw is None or len(df_raw) < 230:
            from data_loader import generate_sample_data
            df_raw = generate_sample_data(n_bars=self.base_sim_bars, seed=hash(symbol) % 999)

        df = compute_all(df_raw)
        if df is None or len(df) < 230:
            logger.warning(f"{symbol}: historique insuffisant pour l'ancrage W_base — "
                            f"le pré-entraînement lazy prendra le relais")
            return

        norm_w  = self._get_norm_w(df)
        base_nn = NeuralNet(lr=state.nn.lr)
        logger.info(f"{symbol}: simulation d'ancrage sur {len(df)} barres...")
        pretrain_on_history(
            base_nn, df, norm_w,
            fwd_bars = self.cooldown_bars, sl_atr = self.sl_atr_mult,
            tp_atr   = self.tp_atr_mult,   n_epochs = 10, verbose = True,
        )
        state.nn.set_anchor(base_nn.W1, base_nn.W2)
        if loaded is None:
            state.nn.W1 = base_nn.W1.copy()
            state.nn.W2 = base_nn.W2.copy()

        state.thresh_long, state.thresh_short = self._calibrate_thresholds(state.nn, df, norm_w)
        state.pretrained = True
        logger.info(f"{symbol}: ancrage prêt — seuils LONG>{state.thresh_long:.3f} "
                    f"SHORT<{state.thresh_short:.3f}")

    def _save_all_memory(self):
        for symbol, state in self.states.items():
            try:
                state.nn.save(self._weights_path(symbol))
            except Exception as e:
                logger.warning(f"{symbol}: échec sauvegarde mémoire AI: {e}")

    def _record_trade_from_close(self, symbol: str, direction: str,
                                  pnl_usd: float, reason: str):
        """Enregistre le trade fermé dans trades_memory.db et sauvegarde W_live."""
        pos = self.position_data.get(symbol)
        if not pos or "features" not in pos:
            return
        outcome = trade_memory.classify_outcome(reason, pnl_usd, pos.get("be_triggered", False))
        risk_usd = pos.get("risk_usd") or 0.0
        r_multiple = (pnl_usd / risk_usd) if risk_usd else None
        try:
            trade_memory.record_trade(
                self._trades_db_path(), symbol, direction, pos["features"], outcome,
                pnl_usd, pos.get("ai_proba_entry", 0.5), pos.get("conf_entry", 0.0),
                r_multiple=r_multiple,
            )
            logger.info(f"{symbol}: trade enregistré en mémoire ({outcome}, pnl={pnl_usd:+.2f}$)")
        except Exception as e:
            logger.warning(f"{symbol}: échec enregistrement trade en mémoire: {e}")
        try:
            self.states[symbol].nn.save(self._weights_path(symbol))
        except Exception as e:
            logger.warning(f"{symbol}: échec sauvegarde mémoire AI après clôture: {e}")

    def _replay_trades(self, symbol: str, state: "SymbolState"):
        """Rejeu périodique de trades réels, équilibré 50/50 gagnants/perdants.
        Garde-fous: pas avant MIN_TRADES_FOR_REPLAY trades, surveille la saturation."""
        db_path = self._trades_db_path()
        n = trade_memory.count_trades(db_path, symbol)
        if n < trade_memory.MIN_TRADES_FOR_REPLAY:
            return
        sample = trade_memory.sample_balanced(db_path, symbol, n=self.replay_n)
        if not sample:
            return
        for features, label, weight in sample:
            fv = FeatVec(*features)
            state.nn.train_step(fv, label, weight=weight)
        logger.info(f"{symbol}: rejeu de {len(sample)} trades (DB={n} trades)")

        if len(state.nn.proba_history) >= 50:
            recent = np.array(state.nn.proba_history[-50:])
            saturated = float(np.mean((recent < 0.03) | (recent > 0.97)))
            if saturated > 0.5:
                logger.warning(f"{symbol}: probas saturées ({saturated*100:.0f}% près de 0/1) "
                                f"— signe d'overfit sur le rejeu, réduire replay_n/fréquence")

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
        logger.info(f"  Directions  : {'LONG uniquement' if self.long_only else 'LONG + SHORT'}")

        if not self.paper_mode:
            logger.warning("⚠️  MODE LIVE — Des positions RÉELLES vont être ouvertes!")

        # Dashboard web — expose bot_data en direct sur http://IP:PORT
        if self.start_dashboard:
            start_server(self.dashboard_port)
            logger.info(f"Dashboard web démarré sur le port {self.dashboard_port}")

        # Connexion MT5 — même en paper mode pour avoir les vraies données
        self.broker = MT5Broker(
            account=account, password=password,
            server=server, magic=self.magic
        )
        if self.broker.connect():
            logger.info("MT5 connecté — données réelles disponibles")
            logger.info(self.broker.portfolio_summary())
            if not self.broker.is_demo and not self.paper_mode:
                notify_error("SÉCURITÉ", "⚠️ Le bot vient de démarrer en mode LIVE "
                             "sur un compte MT5 RÉEL (pas un compte demo). "
                             "Vérifiez que c'est bien intentionnel.")
        else:
            logger.warning("MT5 non connecté — utilisation données synthétiques")
            self.broker = None

        # Mémoire persistante: charger W_live + ancrer W_base pour chaque symbole
        os.makedirs(self.memory_dir, exist_ok=True)
        logger.info("Initialisation de la mémoire AI (W_live + ancrage W_base)...")
        for symbol in self.symbols:
            try:
                self._init_symbol_memory(symbol)
            except Exception as e:
                logger.error(f"{symbol}: échec init mémoire AI: {e}", exc_info=True)

        # Signal handlers pour arrêt propre
        signal.signal(signal.SIGINT,  self._handle_stop)
        signal.signal(signal.SIGTERM, self._handle_stop)

        self._running = True
        iteration = 0

        logger.info(f"\nBoucle principale démarrée (intervalle: {self.loop_seconds}s)\n")
        notify_bot_start(self.symbols, self.timeframe, self.paper_mode)

        while self._running:
            iteration += 1
            loop_start = time.monotonic()

            logger.info(f"─── Itération {iteration} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ───")

            for symbol in self.symbols:
                try:
                    self._process_symbol(symbol)
                except Exception as e:
                    logger.error(f"Erreur sur {symbol}: {e}", exc_info=True)
                    notify_error(symbol, str(e))

            # Détecter les positions fermées côté broker (TP/SL touché, fermeture
            # manuelle) que le bot n'a pas fermées lui-même via _close()
            if self.broker and not self.paper_mode:
                try:
                    for closed in self.broker.sync_closed_trades():
                        self.trades.append(closed)
                        self._record_trade_from_close(
                            closed.symbol, closed.direction, closed.pnl_usd, "TP/SL/manuel")
                        self.position_data.pop(closed.symbol, None)
                except Exception as e:
                    logger.error(f"Erreur sync_closed_trades: {e}", exc_info=True)

            # Sauvegarde périodique de la mémoire AI (toutes les save_every_iters itérations)
            if iteration % self.save_every_iters == 0:
                self._save_all_memory()

            # Résumé quotidien à minuit
            now = datetime.now()
            if now.hour == 0 and now.minute < 2 and iteration > 1:
                balance = self.broker.get_balance() if self.broker else 0
                daily_pnl = self.broker.realized_pnl if self.broker else 0
                notify_daily_summary({
                    'n_trades': len(self.trades),
                    'winners': sum(1 for t in self.trades if getattr(t,'pnl_usd',0)>0),
                    'losers':  sum(1 for t in self.trades if getattr(t,'pnl_usd',0)<=0),
                    'win_rate': sum(1 for t in self.trades if getattr(t,'pnl_usd',0)>0)/max(len(self.trades),1)*100,
                    'pnl_usd': daily_pnl, 'balance': balance
                })
                self.trades.clear()  # reset compteur journalier

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
        logger.info("Sauvegarde de la mémoire AI (W_live)...")
        self._save_all_memory()
        if self.broker and not self.paper_mode:
            logger.info("Fermeture de toutes les positions...")
            self.broker.close_all()
            self.broker.disconnect()
        total_pnl = self.broker.realized_pnl if self.broker else 0.0
        total_trades = sum(len(s.nn.train_losses) for s in self.states.values())
        notify_bot_stop(total_pnl, len(self.trades) if hasattr(self, "trades") else 0)
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
            old_fv, _ = state.pending.popleft()
            lbl = compute_label_continuous(close, old_fv.ref_close, old_fv.ref_atr)
            if lbl is not None:
                state.nn.train_step(old_fv, lbl)

        # Recalibrer les seuils toutes les 100 barres
        if state.n_bars_seen % 100 == 0 and state.pretrained:
            state.thresh_long, state.thresh_short = self._calibrate_thresholds(
                state.nn, df, norm_w)
            logger.info(f"{symbol}: recalibration seuils → "
                        f"LONG>{state.thresh_long:.3f} SHORT<{state.thresh_short:.3f}")

        # Rejeu périodique de la base de trades (Phase 2)
        if state.n_bars_seen % self.replay_every_bars == 0:
            self._replay_trades(symbol, state)

        # ── 5. Prédiction AI ──
        ai_proba = state.nn.predict(fv)

        # ── 6. Conditions de signal ──
        conf       = float(row.get("conf_score", 0.0))
        ich_score  = float(row.get("ich_score",  0.0))
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

        # Règles dures (Niveau 1) — véto déterministe, ex: LONG avec confluence
        # négative ou Ichimoku baissier n'est jamais autorisé
        hard_block_long  = check_hard_rules("LONG",  conf, ich_score, max_conf) if in_zone_bull else "no_zone"
        hard_block_short = check_hard_rules("SHORT", conf, ich_score, max_conf) if in_zone_bear else "no_zone"

        # AI ET confluence ET règles dures requis ensemble
        long_cond  = ai_long  and cf_long  and hard_block_long  is None
        short_cond = ai_short and cf_short and hard_block_short is None

        if self.use_chop and is_chop:
            long_cond = short_cond = False

        if self.long_only:
            short_cond = False

        cooldown_ok  = (bar_idx - state.last_signal_bar) >= self.cooldown_bars
        long_signal  = long_cond  and cooldown_ok
        short_signal = short_cond and cooldown_ok

        # ── 6b. Diagnostic: pourquoi le trade est bloqué (pour le dashboard) ──
        blocked_reasons = []
        if self.use_chop and is_chop:
            blocked_reasons.append("chop")
        if not in_zone_bull and not in_zone_bear:
            blocked_reasons.append("no_zone")
        else:
            proba_ok = (in_zone_bull and ai_proba > state.thresh_long) or \
                       (in_zone_bear and ai_proba < state.thresh_short)
            confluence_ok = (in_zone_bull and conf >= max_conf * self.conf_pct) or \
                            (in_zone_bear and conf <= -max_conf * self.conf_pct)
            if not proba_ok:
                blocked_reasons.append("proba")
            if not confluence_ok:
                blocked_reasons.append("confluence")
            if in_zone_bull and hard_block_long:
                blocked_reasons.append(hard_block_long)
            if in_zone_bear and hard_block_short:
                blocked_reasons.append(hard_block_short)
        if not cooldown_ok:
            blocked_reasons.append("cooldown")
        if self.long_only and in_zone_bear and ai_short and cf_short and hard_block_short is None:
            blocked_reasons.append("long_only")

        # ── 7. Récupérer position actuelle ──
        current_dir = self._get_current_direction(symbol)

        # ── 7b. Gestion Breakeven / Trailing Stop ──
        if current_dir is not None and symbol in self.position_data:
            pos = self.position_data[symbol]
            entry = pos.get("entry", close)
            tp    = pos.get("tp", close)
            sl    = pos.get("sl", close)
            be_triggered = pos.get("be_triggered", False)
            direction = pos.get("direction", current_dir)

            if self.use_breakeven and not be_triggered:
                if direction == "LONG":
                    total_dist   = tp - entry
                    progress     = close - entry
                    if total_dist > 0 and progress >= total_dist * self.be_trigger_pct:
                        new_sl = entry * (1.0 + self.be_offset_pct)
                        if new_sl > sl:
                            logger.info(f"🔒 BREAKEVEN {symbol} | Entrée={entry:.5f} | Nouveau SL={new_sl:.5f} | Prix={close:.5f}")
                            self.position_data[symbol]["sl"] = new_sl
                            self.position_data[symbol]["be_triggered"] = True
                            if not self.paper_mode and self.broker:
                                self.broker.modify_sl_tp(symbol, sl=new_sl, tp=tp)
                            notify_trade_close(symbol, direction, entry, close,
                                               (close-entry)/entry*100, (close-entry)/entry*100,
                                               "BREAKEVEN_SET")
                else:  # SHORT
                    total_dist   = entry - tp
                    progress     = entry - close
                    if total_dist > 0 and progress >= total_dist * self.be_trigger_pct:
                        new_sl = entry * (1.0 - self.be_offset_pct)
                        if new_sl < sl:
                            logger.info(f"🔒 BREAKEVEN {symbol} | Entrée={entry:.5f} | Nouveau SL={new_sl:.5f} | Prix={close:.5f}")
                            self.position_data[symbol]["sl"] = new_sl
                            self.position_data[symbol]["be_triggered"] = True
                            if not self.paper_mode and self.broker:
                                self.broker.modify_sl_tp(symbol, sl=new_sl, tp=tp)

            # Trailing stop (après breakeven)
            if self.use_trailing and be_triggered:
                if direction == "LONG":
                    trail_sl = close - self.trail_atr_mult * atr
                    if trail_sl > self.position_data[symbol]["sl"]:
                        self.position_data[symbol]["sl"] = trail_sl
                        if not self.paper_mode and self.broker:
                            self.broker.modify_sl_tp(symbol, sl=trail_sl, tp=tp)
                        logger.info(f"📈 TRAILING SL {symbol} → {trail_sl:.5f}")
                else:
                    trail_sl = close + self.trail_atr_mult * atr
                    if trail_sl < self.position_data[symbol]["sl"]:
                        self.position_data[symbol]["sl"] = trail_sl
                        if not self.paper_mode and self.broker:
                            self.broker.modify_sl_tp(symbol, sl=trail_sl, tp=tp)
                        logger.info(f"📉 TRAILING SL {symbol} → {trail_sl:.5f}")

        # ── 8. Fermer si signal opposé ──
        if current_dir == "LONG"  and short_signal:
            closed = self._close(symbol, reason="REVERSE→SHORT", close_price=close)
            if closed:
                self.trades.append(closed)
            self.position_data.pop(symbol, None)
            current_dir = None
        elif current_dir == "SHORT" and long_signal:
            closed = self._close(symbol, reason="REVERSE→LONG", close_price=close)
            if closed:
                self.trades.append(closed)
            self.position_data.pop(symbol, None)
            current_dir = None

        # ── 9. Ouvrir nouveau trade ──
        if current_dir is None and (long_signal or short_signal):
            direction = "LONG" if long_signal else "SHORT"
            sl = close - self.sl_atr_mult * atr if direction == "LONG" else close + self.sl_atr_mult * atr
            tp = close + self.tp_atr_mult * atr if direction == "LONG" else close - self.tp_atr_mult * atr

            risk_usd = self.capital * self.risk_pct

            logger.info(
                f"🎯 SIGNAL {direction} | {symbol} | "
                f"Prix={close:.5f} | ATR={atr:.5f} | "
                f"AI={ai_proba:.3f} | Conf={conf:.2f} | "
                f"SL={sl:.5f} | TP={tp:.5f}"
            )
            notify_signal_detected(symbol, direction, ai_proba, conf,
                "OB Bull" if in_ob_bull else "FVG Bull" if in_fvg_bull else
                "OB Bear" if in_ob_bear else "FVG Bear", is_chop)

            # broker.open_trade() (live) envoie déjà sa propre notification Telegram —
            # on ne renotifie ici que pour le mode paper (pas de trade broker sous-jacent)
            trade_info = self._open(symbol, direction, sl, tp, risk_usd, close,
                       conf_score=conf, ai_proba=ai_proba)
            if trade_info:
                self.position_data[symbol] = {
                    "entry": trade_info["entry"], "sl": trade_info["sl"],
                    "tp": trade_info["tp"], "volume": trade_info["volume"],
                    "direction": direction, "be_triggered": False,
                    "features": fv.to_array().tolist(),
                    "ai_proba_entry": float(ai_proba), "conf_entry": float(conf),
                    "risk_usd": risk_usd,
                }
                if self.paper_mode:
                    notify_trade_open(
                        symbol=symbol, direction=direction,
                        price=trade_info["entry"], sl=sl, tp=tp,
                        lot=trade_info["volume"],
                        risk_usd=risk_usd, ai_proba=ai_proba,
                        conf_score=conf, timeframe=self.timeframe
                    )
                state.last_signal_bar = bar_idx

        else:
            logger.info(
                f"{symbol} | {close:.5f} | "
                f"AI={ai_proba:.3f} (L>{state.thresh_long:.3f} S<{state.thresh_short:.3f}) | "
                f"Conf={conf:.2f} | Pos={current_dir or 'FLAT'} | "
                f"Zone={'Bull' if in_zone_bull else 'Bear' if in_zone_bear else 'None'} | "
                f"Chop={'OUI' if is_chop else 'non'}"
            )

        # Mettre à jour le dashboard
        try:
            bot_data.update_symbol(symbol, {
                "close":        close,
                "ai_proba":     round(float(ai_proba), 4),
                "conf_score":   round(float(conf), 3),
                "thresh_long":  round(float(state.thresh_long), 3),
                "thresh_short": round(float(state.thresh_short), 3),
                "is_chop":      bool(is_chop),
                "zone":         "Bull" if in_zone_bull else "Bear" if in_zone_bear else "None",
                "position":     current_dir or "FLAT",
                "atr":          round(float(atr), 5),
            })
            bot_data.capital = self.broker.get_balance() if self.broker else bot_data.capital
            bot_data.add_equity(bot_data.capital)
            bot_data.add_diag(symbol, {
                "ai_proba":     round(float(ai_proba), 4),
                "thresh_long":  round(float(state.thresh_long), 3),
                "thresh_short": round(float(state.thresh_short), 3),
                "conf":         round(float(conf), 3),
                "max_conf":     round(float(max_conf), 3),
                "is_chop":      bool(is_chop),
                "in_zone_bull": bool(in_zone_bull),
                "in_zone_bear": bool(in_zone_bear),
                "cooldown_ok":  bool(cooldown_ok),
                "blocked":      blocked_reasons,
            })
        except Exception as _e:
            pass

    # ─────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────

    def _get_data(self, symbol: str):
        """Récupère les données OHLCV depuis MT5 (paper ou live).
        En paper mode: utilise les VRAIES données MT5 mais n'envoie pas d'ordres réels.
        """
        # Toujours essayer MT5 d'abord (paper ou live)
        if self.broker is not None:
            try:
                df = self.broker.get_ohlcv(symbol, self.tf_mt5, self.n_bars)
                if df is not None and len(df) > 50:
                    return df
            except Exception as e:
                logger.warning(f"MT5 données {symbol} échoué: {e} — fallback synthétique")

        # Fallback: données synthétiques si MT5 non disponible
        logger.warning(f"{symbol}: utilisation données synthétiques (MT5 non connecté)")
        from data_loader import generate_sample_data
        return generate_sample_data(n_bars=self.n_bars, seed=hash(symbol) % 999)

    def _get_current_direction(self, symbol: str) -> Optional[str]:
        """Retourne la direction de la position ouverte (LONG/SHORT/None)."""
        if self.paper_mode or self.broker is None:
            pos = self.position_data.get(symbol)
            return pos["direction"] if pos else None
        return self.broker.get_position_direction(symbol)

    def _open(self, symbol: str, direction: str, sl: float, tp: float,
              risk_usd: float, close_price: float,
              conf_score: float = 0.0, ai_proba: float = 0.5) -> Optional[dict]:
        """Ouvre un trade (réel ou simulé). Retourne {entry, sl, tp, volume} ou None si échec."""
        if self.paper_mode:
            volume = round(risk_usd / close_price, 4) if close_price else 0.0
            logger.info(f"[PAPER] OPEN {direction} {symbol} | "
                        f"SL={sl:.5f} TP={tp:.5f} | Risk=${risk_usd:.2f}")
            return {"entry": close_price, "sl": sl, "tp": tp, "volume": volume}

        trade = self.broker.open_trade(
            symbol=symbol, direction=direction,
            sl=sl, tp=tp, risk_usd=risk_usd,
            conf_score=conf_score, ai_proba=ai_proba,
        )
        if trade is None:
            return None
        return {"entry": trade.entry_price, "sl": sl, "tp": tp, "volume": trade.volume}

    def _close(self, symbol: str, reason: str = "", close_price: float = None) -> Optional[dict]:
        """Ferme un trade (réel ou simulé). Retourne le trade fermé (dict pour le dashboard) ou None."""
        if self.paper_mode:
            pos = self.position_data.get(symbol)
            if pos is None or close_price is None:
                logger.info(f"[PAPER] CLOSE {symbol} | raison: {reason}")
                return None
            entry, direction = pos["entry"], pos["direction"]
            volume = pos.get("volume", 0.0)
            pnl_pct = ((close_price - entry) / entry * 100 if direction == "LONG"
                       else (entry - close_price) / entry * 100)
            pnl_usd = volume * (close_price - entry) * (1 if direction == "LONG" else -1)
            notify_trade_close(symbol, direction, entry, close_price, pnl_usd, pnl_pct, reason)
            logger.info(f"[PAPER] CLOSE {symbol} | raison: {reason} | P&L={pnl_usd:+.2f}$")
            trade = {"symbol": symbol, "direction": direction, "pnl_usd": pnl_usd,
                     "exit_reason": reason, "exit_time": datetime.now().isoformat()}
            bot_data.add_trade(trade)
            self._record_trade_from_close(symbol, direction, pnl_usd, reason)
            return trade

        if not self.broker.close_position(symbol, reason=reason):
            return None
        closed = next((t for t in reversed(self.broker.trades)
                       if t.symbol.upper() == symbol.upper() and t.exit_time is not None), None)
        if closed is None:
            return None
        self._record_trade_from_close(symbol, closed.direction, closed.pnl_usd, reason)
        return {"symbol": symbol, "direction": closed.direction, "pnl_usd": closed.pnl_usd,
                "exit_reason": reason, "exit_time": closed.exit_time.isoformat()}

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
    parser.add_argument("--tf",       default="H1",
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
    parser.add_argument("--allow-short", action="store_true",
                        help="Autoriser les positions SHORT (désactivé par défaut — "
                             "le backtest montre l'edge concentré côté LONG)")
    parser.add_argument("--paper",    action="store_true",
                        help="Mode simulation (aucun ordre réel)")
    parser.add_argument("--magic",    type=int,   default=20240101,
                        help="Magic number MT5 (identifiant du bot)")
    parser.add_argument("--dashboard-port", type=int, default=5000,
                        help="Port du dashboard web (défaut: 5000)")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Désactiver le dashboard web")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Ne pas demander de confirmation (pour lancement automatisé)")
    parser.add_argument("--memory-dir", type=str, default="memory",
                        help="Dossier de mémoire persistante (poids AI + base de trades)")
    parser.add_argument("--base-bars", type=int, default=5000,
                        help="Barres historiques pour la simulation d'ancrage W_base au démarrage")
    parser.add_argument("--save-every", type=int, default=50,
                        help="Sauvegarder la mémoire AI toutes les N itérations")
    parser.add_argument("--replay-every", type=int, default=200,
                        help="Rejouer des trades de la DB toutes les N barres")
    parser.add_argument("--replay-n", type=int, default=20,
                        help="Nombre de trades échantillonnés par rejeu")

    args = parser.parse_args()

    if not args.paper and not args.account and not args.yes:
        print("\n⚠️  ATTENTION: Pas de numéro de compte fourni.")
        print("   Si MT5 est déjà connecté, le bot utilisera le compte actif.")
        print("   Pour spécifier: --account 12345 --password xxx --server NomBroker")
        print("   Pour la simulation: ajouter --paper\n")
        resp = input("Continuer quand même? (oui/non): ").strip().lower()
        if resp != "oui":
            sys.exit(0)

    if not args.paper and not args.yes:
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
        long_only     = not args.allow_short,
        paper_mode    = args.paper,
        magic         = args.magic,
        dashboard_port  = args.dashboard_port,
        start_dashboard = not args.no_dashboard,
        memory_dir        = args.memory_dir,
        base_sim_bars     = args.base_bars,
        save_every_iters  = args.save_every,
        replay_every_bars = args.replay_every,
        replay_n          = args.replay_n,
    )

    bot.run(
        account  = args.account,
        password = args.password,
        server   = args.server,
    )


if __name__ == "__main__":
    main()
