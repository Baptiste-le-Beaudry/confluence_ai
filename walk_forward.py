"""
walk_forward.py
Simulation Walk-Forward: entraîner sur N jours, tester sur 2 jours suivants.
Répéter sur toute la période historique et concaténer les résultats.

Logique:
  - Fenêtre d'entraînement: 20 jours (pré-entraînement AI)
  - Fenêtre de test: 2 jours (simulation réelle sans regarder l'avenir)
  - Glissement: 2 jours à chaque itération
  - Critères plus resserrés: moins de trades mais plus certains

Paramètres resserrés vs backtest standard:
  - target_trades_per_day: 0.5 (au lieu de 1.0) → moins de trades, plus sélectifs
  - conf_long_pct: 0.40 (au lieu de 0.25) → confluence plus forte requise
  - cooldown_bars: 16 (au lieu de 6) → plus d'espace entre les trades
  - pretrain_epochs: 15 (au lieu de 10) → meilleur pré-entraînement
  - sl_atr_mult: 1.5 (SL plus serré)
  - tp_atr_mult: 4.5 (TP plus large = meilleur ratio R/R)
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from collections import deque

from indicators import compute_all
from ai_model import (NeuralNet, FeatVec, build_features,
                      compute_label_continuous, pretrain_on_history)
from backtest import Trade


# ─────────────────────────────────────────────
# Paramètres resserrés
# ─────────────────────────────────────────────

@dataclass
class WFParams:
    # Fenêtres walk-forward
    train_days:       int   = 20    # jours pour pré-entraîner l'AI
    test_days:        int   = 2     # jours simulés réellement
    step_days:        int   = 2     # glissement entre fenêtres

    # AI — plus strict
    target_trades_per_day: float = 0.5   # 1 trade tous les 2 jours
    ai_norm_len:      int   = 50
    lr:               float = 0.025
    grad_clip:        float = 0.4
    use_ai:           bool  = True
    pretrain_epochs:  int   = 15
    use_adaptive_thresh: bool = True
    ai_thresh_long:   float = 0.62   # seuil par défaut si pas assez de données
    ai_thresh_short:  float = 0.38

    # Confluence — plus strict
    conf_long_pct:    float = 0.40   # 40% du score max (vs 25% avant)
    conf_short_pct:   float = 0.40

    # Filtres
    use_chop_filter:  bool  = True
    use_candle_confirm: bool = False
    use_htf:          bool  = False
    cooldown_bars:    int   = 16     # ~4h sur 15min (vs 6 avant)

    # Risque — ratio R/R amélioré
    capital:          float = 10_000.0
    risk_pct:         float = 0.008  # 0.8% par trade (vs 1%)
    sl_atr_mult:      float = 1.5    # SL plus serré
    tp_atr_mult:      float = 4.5    # TP plus large
    indicator_params: dict  = field(default_factory=dict)


# ─────────────────────────────────────────────
# Moteur Walk-Forward
# ─────────────────────────────────────────────

class WalkForwardEngine:
    def __init__(self, params: WFParams = None):
        self.p = params or WFParams()
        self.all_trades:      list[Trade] = []
        self.all_equity:      list[dict]  = []
        self.window_results:  list[dict]  = []
        self.capital = self.p.capital

    def run(self, df_raw: pd.DataFrame) -> dict:
        """
        Exécute le walk-forward complet sur df_raw.
        """
        print("=" * 64)
        print("  WALK-FORWARD BACKTEST — Confluence AI v4.0")
        print("  Critères resserrés: moins de trades, plus de certitude")
        print("=" * 64)

        # Calcul des indicateurs une seule fois
        print(f"\nCalcul des indicateurs sur {len(df_raw)} barres...")
        df = compute_all(df_raw, self.p.indicator_params)

        # Barres par jour
        if len(df) > 1:
            total_secs   = (df.index[-1] - df.index[0]).total_seconds()
            bar_secs     = total_secs / len(df)
            bars_per_day = max(86400.0 / bar_secs, 1.0)
        else:
            bars_per_day = 96.0

        bars_train = int(self.p.train_days * bars_per_day)
        bars_test  = int(self.p.test_days  * bars_per_day)
        bars_step  = int(self.p.step_days  * bars_per_day)

        total_bars  = len(df)
        total_days  = (df.index[-1] - df.index[0]).days
        n_windows   = max((total_bars - bars_train) // bars_step, 1)

        print(f"  {bars_per_day:.0f} barres/jour | {total_days} jours totaux")
        print(f"  Fenêtre entraînement: {self.p.train_days} jours ({bars_train} barres)")
        print(f"  Fenêtre test:         {self.p.test_days} jours ({bars_test} barres)")
        print(f"  Nombre de fenêtres:   {n_windows}")
        print(f"  Cible: {self.p.target_trades_per_day} trade/jour")
        print(f"  SL: {self.p.sl_atr_mult}× ATR | TP: {self.p.tp_atr_mult}× ATR")
        print(f"  Confluence min: {self.p.conf_long_pct*100:.0f}% du score max")
        print(f"  Cooldown: {self.p.cooldown_bars} barres ({self.p.cooldown_bars/bars_per_day*24:.1f}h)")
        print()

        # Normalisation globale
        norm_w_global = self._compute_norm_windows(df)

        window_idx = 0
        train_start = 0

        while train_start + bars_train + bars_test <= total_bars:
            window_idx += 1
            train_end  = train_start + bars_train
            test_end   = min(train_end + bars_test, total_bars)

            df_train = df.iloc[train_start:train_end]
            df_test  = df.iloc[train_end:test_end]

            if len(df_test) < 10:
                break

            t_start = df_test.index[0].strftime("%Y-%m-%d %H:%M")
            t_end   = df_test.index[-1].strftime("%Y-%m-%d %H:%M")
            print(f"Fenêtre {window_idx:3d}/{n_windows} | "
                  f"Test: {t_start} → {t_end} "
                  f"({len(df_test)} barres)", end="  ")

            # Pré-entraîner l'AI sur la fenêtre d'entraînement
            nn = NeuralNet(lr=self.p.lr, grad_clip=self.p.grad_clip)
            pretrain_on_history(
                nn, df_train, norm_w_global,
                fwd_bars   = int(bars_per_day * 0.25),  # 6h d'horizon
                sl_atr     = self.p.sl_atr_mult,
                tp_atr     = self.p.tp_atr_mult,
                n_epochs   = self.p.pretrain_epochs,
                verbose    = False,
            )

            # Calibrer les seuils sur la fenêtre d'entraînement
            thresh_long, thresh_short = self._calibrate_thresholds(
                nn, df_train, norm_w_global, bars_per_day)

            # Simuler sur la fenêtre de test
            w_trades, w_equity, capital_end = self._simulate_window(
                nn, df_test, norm_w_global, bars_per_day,
                thresh_long, thresh_short)

            n_w = len(w_trades)
            pnl = sum(t.pnl_usd for t in w_trades)
            wr  = sum(1 for t in w_trades if t.pnl_usd > 0) / n_w * 100 if n_w > 0 else 0
            print(f"| Trades: {n_w:3d} | P&L: {pnl:+8.2f}$ | WR: {wr:4.1f}%")

            self.all_trades.extend(w_trades)
            self.all_equity.extend(w_equity)
            self.window_results.append({
                "window":      window_idx,
                "test_start":  df_test.index[0],
                "test_end":    df_test.index[-1],
                "n_trades":    n_w,
                "pnl_usd":     pnl,
                "win_rate":    wr,
                "capital_end": capital_end,
            })

            self.capital = capital_end
            train_start += bars_step

        print()
        return self._compute_global_metrics(total_days)

    def _compute_norm_windows(self, df: pd.DataFrame) -> dict:
        return {
            "vwap_std": max(df["dist_vwap"].abs().std(), 0.05),
            "ema_std":  max((df["ema_trend"] * 100).abs().std(), 0.05),
            "gann_std": max(df["gann_pos"].abs().std(), 0.10),
            "mom_std":  max(df["mom_raw"].abs().std(), 0.10),
        }

    def _calibrate_thresholds(self, nn: NeuralNet, df_train: pd.DataFrame,
                               norm_w: dict, bars_per_day: float) -> tuple[float, float]:
        """Calcule les seuils adaptatifs sur la fenêtre d'entraînement."""
        probas = []
        for _, row in df_train.iterrows():
            rd = row.to_dict(); rd["close"] = row["close"]; rd["atr"] = row["atr"]
            if pd.isna(rd.get("atr", np.nan)) or rd.get("atr", 0) <= 0:
                continue
            fv = build_features(rd, norm_w)
            probas.append(nn.predict(fv))

        if len(probas) < 20:
            return self.p.ai_thresh_long, self.p.ai_thresh_short

        probas = np.array(probas)
        frac   = np.clip(self.p.target_trades_per_day / bars_per_day / 2.0, 0.002, 0.15)
        tl     = float(np.percentile(probas, (1.0 - frac) * 100))
        ts     = float(np.percentile(probas, frac * 100))
        tl     = max(tl, 0.51)
        ts     = min(ts, 0.49)
        return tl, ts

    def _simulate_window(self, nn: NeuralNet, df_test: pd.DataFrame,
                          norm_w: dict, bars_per_day: float,
                          thresh_long: float, thresh_short: float
                          ) -> tuple[list[Trade], list[dict], float]:
        """Simule les trades sur la fenêtre de test."""
        capital     = self.capital
        open_trade  = None
        trades:     list[Trade] = []
        equity:     list[dict]  = []
        pending     = deque()
        last_signal = -self.p.cooldown_bars
        max_conf    = float(df_test["max_conf"].iloc[-1])

        # Fenêtres de normalisation dynamique
        dv = deque(maxlen=self.p.ai_norm_len)
        et = deque(maxlen=self.p.ai_norm_len)
        gp = deque(maxlen=self.p.ai_norm_len)
        mr = deque(maxlen=self.p.ai_norm_len)

        for i, (ts, row) in enumerate(df_test.iterrows()):
            close = row["close"]; atr = row["atr"]
            if pd.isna(atr) or atr <= 0 or pd.isna(close):
                continue

            dv.append(abs(row.get("dist_vwap", 0.0)))
            et.append(abs(row.get("ema_trend", 0.0) * 100))
            gp.append(abs(row.get("gann_pos", 0.0)))
            mr.append(abs(row.get("mom_raw", 0.0)))
            nw = {
                "vwap_std": max(np.std(dv), 0.05) if len(dv) > 3 else norm_w["vwap_std"],
                "ema_std":  max(np.std(et), 0.05) if len(et) > 3 else norm_w["ema_std"],
                "gann_std": max(np.std(gp), 0.10) if len(gp) > 3 else norm_w["gann_std"],
                "mom_std":  max(np.std(mr), 0.10) if len(mr) > 3 else norm_w["mom_std"],
            }
            rd = row.to_dict(); rd["close"] = close; rd["atr"] = atr
            fv = build_features(rd, nw)

            # Online learning pendant le test
            pending.append((fv, i))
            if len(pending) > 5:
                old_fv, _ = pending.popleft()
                lbl = compute_label_continuous(close, old_fv.ref_close, old_fv.ref_atr)
                if lbl is not None:
                    nn.train_step(old_fv, lbl)

            ai_proba = nn.predict(fv)

            # Conditions — critères resserrés
            conf       = float(row.get("conf_score", 0.0))
            is_chop    = bool(row.get("is_chop", False))
            in_ob_bull = bool(row.get("in_ob_bull", False))
            in_ob_bear = bool(row.get("in_ob_bear", False))
            in_fvg_bull= bool(row.get("in_fvg_bull", False))
            in_fvg_bear= bool(row.get("in_fvg_bear", False))
            bull_conf  = bool(row.get("bull_confirm", False))
            bear_conf  = bool(row.get("bear_confirm", False))

            in_zone_bull = in_ob_bull or in_fvg_bull
            in_zone_bear = in_ob_bear or in_fvg_bear

            # Signal: AI ET confluence (les deux requis — plus strict)
            ai_long  = (ai_proba > thresh_long)  and in_zone_bull
            ai_short = (ai_proba < thresh_short) and in_zone_bear
            cf_long  = (conf >= max_conf * self.p.conf_long_pct)  and in_zone_bull
            cf_short = (conf <= -max_conf * self.p.conf_short_pct) and in_zone_bear

            # RESSERRÉ: AI ET confluence doivent être d'accord (AND au lieu de OR)
            long_cond  = (ai_long  and cf_long)  or (cf_long  and not self.p.use_ai)
            short_cond = (ai_short and cf_short) or (cf_short and not self.p.use_ai)

            if self.p.use_chop_filter and is_chop:
                long_cond = short_cond = False
            if self.p.use_candle_confirm:
                long_cond  = long_cond  and bull_conf
                short_cond = short_cond and bear_conf

            cooldown_ok  = (i - last_signal) >= self.p.cooldown_bars
            long_signal  = long_cond  and cooldown_ok and open_trade is None
            short_signal = short_cond and cooldown_ok and open_trade is None

            # Gestion position ouverte
            if open_trade is not None:
                closed = False; reason = ""
                if open_trade.direction == "LONG":
                    if   close <= open_trade.sl: closed, reason = True, "SL"
                    elif close >= open_trade.tp: closed, reason = True, "TP"
                    elif short_signal:           closed, reason = True, "REVERSE"
                else:
                    if   close >= open_trade.sl: closed, reason = True, "SL"
                    elif close <= open_trade.tp: closed, reason = True, "TP"
                    elif long_signal:            closed, reason = True, "REVERSE"

                if closed:
                    sign    = 1 if open_trade.direction == "LONG" else -1
                    pnl_pct = sign * (close - open_trade.entry_price) / open_trade.entry_price
                    pnl_usd = open_trade.size_usd * pnl_pct
                    open_trade.exit_price  = close; open_trade.exit_bar    = i
                    open_trade.exit_time   = ts;    open_trade.exit_reason  = reason
                    open_trade.pnl_usd     = pnl_usd; open_trade.pnl_pct   = pnl_pct
                    capital += pnl_usd
                    trades.append(open_trade)
                    open_trade = None

            # Ouverture
            if open_trade is None and (long_signal or short_signal):
                direction = "LONG" if long_signal else "SHORT"
                risk_usd  = capital * self.p.risk_pct
                denom     = max(self.p.sl_atr_mult * atr / close, 0.001)
                size_usd  = min(risk_usd / denom, capital * 0.15)
                sl = close - self.p.sl_atr_mult * atr if direction == "LONG" else close + self.p.sl_atr_mult * atr
                tp = close + self.p.tp_atr_mult * atr if direction == "LONG" else close - self.p.tp_atr_mult * atr
                open_trade = Trade(direction=direction, entry_price=close,
                                   entry_bar=i, entry_time=ts, size_usd=size_usd,
                                   sl=sl, tp=tp, conf_score=conf, ai_proba=ai_proba)
                last_signal = i

            # Équité
            unr = 0.0
            if open_trade:
                sign = 1 if open_trade.direction == "LONG" else -1
                unr  = open_trade.size_usd * sign * (close - open_trade.entry_price) / open_trade.entry_price

            equity.append({"time": ts, "equity": capital + unr, "cash": capital,
                           "ai_proba": ai_proba, "close": close,
                           "thresh_long": thresh_long, "thresh_short": thresh_short})

        # Fermer si fin de fenêtre
        if open_trade and len(df_test) > 0:
            lc  = df_test["close"].iloc[-1]
            sign = 1 if open_trade.direction == "LONG" else -1
            pnl_pct = sign * (lc - open_trade.entry_price) / open_trade.entry_price
            pnl_usd = open_trade.size_usd * pnl_pct
            open_trade.exit_price = lc; open_trade.exit_time = df_test.index[-1]
            open_trade.exit_reason = "END_WINDOW"; open_trade.pnl_usd = pnl_usd
            open_trade.pnl_pct = pnl_pct; capital += pnl_usd
            trades.append(open_trade)

        return trades, equity, capital

    def _compute_global_metrics(self, total_days: int) -> dict:
        n  = len(self.all_trades)
        if n == 0:
            return {"error": "Aucun trade.", "n_trades": 0}

        wins    = [t for t in self.all_trades if t.pnl_usd > 0]
        losses  = [t for t in self.all_trades if t.pnl_usd <= 0]
        gp      = sum(t.pnl_usd for t in wins)
        gl      = abs(sum(t.pnl_usd for t in losses))
        pf      = gp / gl if gl > 0 else np.inf
        wr      = len(wins) / n
        avg_w   = np.mean([t.pnl_usd for t in wins])   if wins   else 0
        avg_l   = np.mean([t.pnl_usd for t in losses]) if losses else 0
        exp     = wr * avg_w + (1 - wr) * avg_l

        eq   = pd.DataFrame(self.all_equity).set_index("time")
        ev   = eq["equity"].values
        peak = ev[0]; max_dd = 0.0
        for v in ev:
            peak   = max(peak, v)
            max_dd = max(max_dd, (peak - v) / peak)

        dr  = eq["equity"].resample("D").last().dropna().pct_change().dropna()
        shr = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0.0

        reasons = {}
        for t in self.all_trades: reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

        final = self.capital
        init  = self.p.capital

        return {
            "capital_initial":    init,
            "capital_final":      final,
            "total_return_pct":   (final - init) / init * 100,
            "n_trades":           n,
            "trades_per_day":     n / max(total_days, 1),
            "target_per_day":     self.p.target_trades_per_day,
            "total_days":         total_days,
            "n_windows":          len(self.window_results),
            "n_long":             sum(1 for t in self.all_trades if t.direction=="LONG"),
            "n_short":            sum(1 for t in self.all_trades if t.direction=="SHORT"),
            "win_rate_pct":       wr * 100,
            "profit_factor":      pf,
            "expectancy_usd":     exp,
            "gross_profit":       gp,
            "gross_loss":         gl,
            "avg_win_usd":        avg_w,
            "avg_loss_usd":       avg_l,
            "max_drawdown_pct":   max_dd * 100,
            "sharpe_ratio":       shr,
            "avg_duration_bars":  np.mean([t.exit_bar - t.entry_bar for t in self.all_trades]),
            "exit_reasons":       reasons,
            "ratio_rr":           abs(avg_w / avg_l) if avg_l != 0 else 0,
        }

    def print_report(self, m: dict):
        print("\n" + "═"*64)
        print("    RAPPORT WALK-FORWARD — Confluence AI v4.0")
        print("    Critères resserrés: AI + Confluence REQUIS ensemble")
        print("═"*64)
        print(f"  Capital initial     : ${m['capital_initial']:>12,.2f}")
        print(f"  Capital final       : ${m['capital_final']:>12,.2f}")
        icon = "✅" if m['total_return_pct'] > 0 else "❌"
        print(f"  Rendement total     : {m['total_return_pct']:>+12.2f}%  {icon}")
        print(f"  Max Drawdown        : {m['max_drawdown_pct']:>12.2f}%")
        print(f"  Sharpe Ratio        : {m['sharpe_ratio']:>12.3f}")
        print("─"*64)
        tpd_ok = "✅" if m['trades_per_day'] >= m['target_per_day']*0.5 else "⚠️"
        print(f"  Fenêtres testées    : {m['n_windows']:>12}")
        print(f"  Trades totaux       : {m['n_trades']:>12}  ({m['total_days']} jours)")
        print(f"  Trades/jour         : {m['trades_per_day']:>12.2f}  (cible: {m['target_per_day']:.1f}) {tpd_ok}")
        print(f"  LONG / SHORT        : {m['n_long']:>5} / {m['n_short']:<5}")
        print(f"  Win Rate            : {m['win_rate_pct']:>12.1f}%")
        wr_icon = "✅" if m['win_rate_pct'] > 45 else "⚠️"
        print(f"  Profit Factor       : {m['profit_factor']:>12.3f}  {wr_icon}")
        print(f"  Ratio R/R           : {m['ratio_rr']:>12.2f}")
        print(f"  Espérance/trade     : ${m['expectancy_usd']:>+12.2f}")
        print(f"  Gain moyen          : ${m['avg_win_usd']:>+12.2f}")
        print(f"  Perte moyenne       : ${m['avg_loss_usd']:>+12.2f}")
        print(f"  Durée moy.(barres)  : {m['avg_duration_bars']:>12.1f}")
        print("─"*64)
        print("  Clôtures par raison :")
        for r, c in sorted(m["exit_reasons"].items(), key=lambda x: -x[1]):
            pct = c / m["n_trades"] * 100
            print(f"    {r:<16}: {c:>4}  ({pct:.1f}%)")
        print("═"*64)

    def plot_results(self, m: dict, output_path: str = "wf_report.png"):
        eq  = pd.DataFrame(self.all_equity).set_index("time")
        BG  = "#161b22"; G = "#00ff88"; R = "#ff4466"
        B   = "#4488ff"; Y = "#ffcc00"; GR = "#888888"

        fig = plt.figure(figsize=(20, 16), facecolor="#0d1117")
        gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.32)

        def sa(ax, t=""):
            ax.set_facecolor(BG); ax.tick_params(colors=GR, labelsize=8)
            ax.spines[:].set_color("#333")
            if t: ax.set_title(t, color="white", fontsize=10, pad=5)

        # ── Courbe d'équité complète ──
        ax1 = fig.add_subplot(gs[0, :])
        sa(ax1, "Courbe Équité — Walk-Forward (fenêtres de 2 jours bout à bout)")
        ax1.plot(eq.index, eq["equity"], color=B, lw=1.5, label="Équité")
        ax1.axhline(self.p.capital, color=GR, ls="--", lw=0.8, alpha=0.5)

        # Colorer les fenêtres alternativement
        wr_list = self.window_results
        for k, wr in enumerate(wr_list):
            color = "#1a2a1a" if wr["pnl_usd"] >= 0 else "#2a1a1a"
            ax1.axvspan(wr["test_start"], wr["test_end"], alpha=0.3, color=color)

        # Marquer les trades
        for t in self.all_trades:
            c = G if t.pnl_usd > 0 else R
            if t.exit_time:
                ax1.axvspan(t.entry_time, t.exit_time, alpha=0.12, color=c)
        ax1.set_ylabel("USD", color=GR, fontsize=8)
        ax1.legend(fontsize=8, facecolor=BG, labelcolor="white")

        # ── P&L par fenêtre ──
        ax2 = fig.add_subplot(gs[1, :2])
        sa(ax2, "P&L par fenêtre de 2 jours")
        pnls_w  = [w["pnl_usd"] for w in wr_list]
        colors_w = [G if p >= 0 else R for p in pnls_w]
        ax2.bar(range(len(pnls_w)), pnls_w, color=colors_w, alpha=0.8)
        ax2.axhline(0, color=GR, lw=0.8)
        ax2.set_xlabel("Fenêtre #", color=GR, fontsize=8)
        ax2.set_ylabel("P&L USD", color=GR, fontsize=8)
        # Courbe cumulative
        ax2b = ax2.twinx(); ax2b.set_facecolor(BG)
        ax2b.plot(range(len(pnls_w)), np.cumsum(pnls_w), color=Y, lw=1.5, alpha=0.9)
        ax2b.set_ylabel("Cumulatif", color=Y, fontsize=7)
        ax2b.tick_params(colors=Y, labelsize=7)
        ax2b.spines[:].set_color("#333")

        # ── AI Proba dans le temps ──
        ax3 = fig.add_subplot(gs[1, 2])
        sa(ax3, "Distribution AI Proba (tous trades)")
        ai_probas = [t.ai_proba for t in self.all_trades]
        if ai_probas:
            long_p  = [t.ai_proba for t in self.all_trades if t.direction == "LONG"]
            short_p = [t.ai_proba for t in self.all_trades if t.direction == "SHORT"]
            if long_p:  ax3.hist(long_p,  bins=15, color=G, alpha=0.7, label="LONG")
            if short_p: ax3.hist(short_p, bins=15, color=R, alpha=0.7, label="SHORT")
            ax3.legend(fontsize=8, facecolor=BG, labelcolor="white")
            ax3.set_xlabel("AI Proba", color=GR, fontsize=8)
            ax3.set_ylabel("Fréquence", color=GR, fontsize=8)

        # ── P&L par trade ──
        ax4 = fig.add_subplot(gs[2, :2])
        sa(ax4, "P&L par trade (barres) + Cumulatif (ligne)")
        pnls = [t.pnl_usd for t in self.all_trades]
        cols = [G if p > 0 else R for p in pnls]
        ax4.bar(range(len(pnls)), pnls, color=cols, alpha=0.8, width=0.8)
        ax4.axhline(0, color=GR, lw=0.8)
        ax4b = ax4.twinx(); ax4b.set_facecolor(BG)
        ax4b.plot(range(len(pnls)), np.cumsum(pnls), color=B, lw=2.0, alpha=0.9)
        ax4b.set_ylabel("P&L cumulatif", color=B, fontsize=7)
        ax4b.tick_params(colors=B, labelsize=7)
        ax4b.spines[:].set_color("#333")
        ax4.set_xlabel("# Trade", color=GR, fontsize=8)
        ax4.set_ylabel("P&L / trade", color=GR, fontsize=8)

        # ── Métriques texte ──
        ax5 = fig.add_subplot(gs[2, 2])
        sa(ax5, "Résumé")
        ax5.axis("off")
        lines = [
            ("Rendement",    f"{m['total_return_pct']:+.2f}%",
             G if m["total_return_pct"] > 0 else R),
            ("Max Drawdown", f"{m['max_drawdown_pct']:.2f}%", R),
            ("Sharpe",       f"{m['sharpe_ratio']:.3f}",
             G if m["sharpe_ratio"] > 1 else Y),
            ("Win Rate",     f"{m['win_rate_pct']:.1f}%",
             G if m["win_rate_pct"] > 45 else Y),
            ("Profit Factor",f"{m['profit_factor']:.3f}",
             G if m["profit_factor"] > 1 else R),
            ("Ratio R/R",    f"{m['ratio_rr']:.2f}",
             G if m["ratio_rr"] > 1.5 else Y),
            ("Espérance",    f"${m['expectancy_usd']:+.2f}",
             G if m["expectancy_usd"] > 0 else R),
            ("Trades/jour",  f"{m['trades_per_day']:.2f}", "white"),
            ("# Fenêtres",   str(m["n_windows"]), B),
        ]
        for j, (label, value, color) in enumerate(lines):
            ax5.text(0.02, 0.93 - j*0.10, label+":", color=GR,
                     fontsize=9, transform=ax5.transAxes)
            ax5.text(0.52, 0.93 - j*0.10, value, color=color,
                     fontsize=9, fontweight="bold", transform=ax5.transAxes)

        ret = m["total_return_pct"]
        fig.suptitle(
            f"Walk-Forward BTC 15min  |  Rendement: {ret:+.2f}%  |  "
            f"{m['n_trades']} trades ({m['trades_per_day']:.1f}/jour)  |  "
            f"WR: {m['win_rate_pct']:.1f}%  |  PF: {m['profit_factor']:.2f}  |  "
            f"Sharpe: {m['sharpe_ratio']:.2f}  |  MaxDD: {m['max_drawdown_pct']:.1f}%",
            color="white", fontsize=11, fontweight="bold", y=0.995
        )

        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor="#0d1117")
        plt.close()
        print(f"Graphique sauvegardé: {output_path}")
