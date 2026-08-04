"""
backtest.py - Moteur de backtest Confluence AI v4.0
Stratégie:
  1. Phase de pré-analyse: scanner l'historique, identifier les setups gagnants
  2. Pré-entraîner le réseau de neurones sur ces setups
  3. Calculer les seuils adaptatifs pour atteindre ~1 trade/jour
  4. Simuler les trades avec l'AI et le score de confluence combinés
  5. Apprentissage online barre par barre pendant la simulation
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass, field
from indicators import compute_all
from ai_model import NeuralNet, FeatVec, build_features, compute_label_continuous, pretrain_on_history


@dataclass
class BacktestParams:
    # Fréquence cible
    target_trades_per_day: float = 1.0   # objectif: 1 trade par jour
    # AI
    ai_fwd:           int   = 5
    ai_norm_len:      int   = 50
    lr:               float = 0.02
    grad_clip:        float = 0.5
    use_ai:           bool  = True
    pretrain_epochs:  int   = 8          # epochs de pré-entraînement
    # Seuils (calculés automatiquement si use_adaptive_thresh=True)
    use_adaptive_thresh: bool = True
    ai_thresh_long:   float = 0.60
    ai_thresh_short:  float = 0.40
    # Confluence fallback
    conf_long_pct:    float = 0.30       # seuil bas pour avoir assez de trades
    conf_short_pct:   float = 0.30
    # Filtres (désactivés par défaut pour avoir plus de trades)
    use_htf:          bool  = False
    use_chop_filter:  bool  = True
    use_candle_confirm: bool = False
    cooldown_bars:    int   = 8
    # Risque
    capital:          float = 10_000.0
    risk_pct:         float = 0.01
    sl_atr_mult:      float = 2.0
    tp_atr_mult:      float = 4.0
    indicator_params: dict  = field(default_factory=dict)


@dataclass
class Trade:
    direction:   str
    entry_price: float
    entry_bar:   int
    entry_time:  pd.Timestamp
    size_usd:    float
    sl:          float
    tp:          float
    exit_price:  float = 0.0
    exit_bar:    int   = 0
    exit_time:   pd.Timestamp = None
    exit_reason: str   = ""
    pnl_usd:     float = 0.0
    pnl_pct:     float = 0.0
    conf_score:  float = 0.0
    ai_proba:    float = 0.5

    @property
    def is_winner(self): return self.pnl_usd > 0


class Backtester:
    def __init__(self, params: BacktestParams = None):
        self.p = params or BacktestParams()
        self.nn = NeuralNet(lr=self.p.lr, grad_clip=self.p.grad_clip)
        self.trades: list[Trade] = []
        self.equity_curve: list[dict] = []

    def run(self, df_raw: pd.DataFrame) -> dict:
        # ── Étape 1: Calcul des indicateurs ──
        print(f"\nÉtape 1/4 — Calcul des indicateurs ({len(df_raw)} barres)...")
        df = compute_all(df_raw, self.p.indicator_params)

        # Barres par jour
        if len(df) > 1:
            total_secs = (df.index[-1] - df.index[0]).total_seconds()
            bar_secs   = total_secs / len(df)
            bars_per_day = max(86400.0 / bar_secs, 1.0)
        else:
            bars_per_day = 96.0
        total_days = max((df.index[-1] - df.index[0]).days, 1)

        print(f"  → {bars_per_day:.0f} barres/jour | {total_days} jours")
        print(f"  → Objectif: {self.p.target_trades_per_day} trade(s)/jour "
              f"= {int(self.p.target_trades_per_day * total_days)} trades total")

        # Normalisation globale (pour le pré-entraînement)
        norm_w_global = {
            "vwap_std": max(df["dist_vwap"].abs().std(), 0.05),
            "ema_std":  max((df["ema_trend"] * 100).abs().std(), 0.05),
            "gann_std": max(df["gann_pos"].abs().std(), 0.10),
            "mom_std":  max(df["mom_raw"].abs().std(), 0.10),
        }

        # ── Étape 2: Pré-entraînement ──
        if self.p.use_ai:
            print(f"\nÉtape 2/4 — Pré-entraînement sur les setups historiques...")
            n_pretrain = pretrain_on_history(
                self.nn, df, norm_w_global,
                fwd_bars   = self.p.ai_fwd * 3,   # horizon plus long pour le pré-entraînement
                sl_atr     = self.p.sl_atr_mult,
                tp_atr     = self.p.tp_atr_mult,
                n_epochs   = self.p.pretrain_epochs,
            )

            # Calculer les probas sur tout l'historique pour calibrer les seuils
            print("  Calibration des seuils adaptatifs...")
            all_probas = []
            for i, (_, row) in enumerate(df.iterrows()):
                if i < 220: continue
                rd = row.to_dict(); rd["close"] = row["close"]; rd["atr"] = row["atr"]
                fv = build_features(rd, norm_w_global)
                p  = self.nn.predict(fv)
                all_probas.append(p)

            all_probas = np.array(all_probas)
            print(f"  Distribution AI: min={all_probas.min():.3f} | "
                  f"max={all_probas.max():.3f} | mean={all_probas.mean():.3f} | "
                  f"std={all_probas.std():.3f}")

        # ── Étape 3: Calcul des seuils ──
        print(f"\nÉtape 3/4 — Calcul des seuils...")

        if self.p.use_ai and self.p.use_adaptive_thresh and len(all_probas) > 50:
            thresh_long, thresh_short = self.nn.get_adaptive_thresholds(
                self.p.target_trades_per_day, bars_per_day)
            print(f"  Seuils adaptatifs: LONG > {thresh_long:.3f} | SHORT < {thresh_short:.3f}")
        else:
            thresh_long  = self.p.ai_thresh_long
            thresh_short = self.p.ai_thresh_short
            print(f"  Seuils fixes: LONG > {thresh_long:.3f} | SHORT < {thresh_short:.3f}")

        max_conf = float(df["max_conf"].iloc[-1])

        # ── Étape 4: Simulation des trades ──
        print(f"\nÉtape 4/4 — Simulation des trades...")
        capital = self.p.capital
        open_trade = None
        pending = deque()
        last_signal_bar = -self.p.cooldown_bars

        # Fenêtres de normalisation dynamique
        dv_win = deque(maxlen=self.p.ai_norm_len)
        et_win = deque(maxlen=self.p.ai_norm_len)
        gp_win = deque(maxlen=self.p.ai_norm_len)
        mr_win = deque(maxlen=self.p.ai_norm_len)

        # HTF
        htf_ema200 = None
        if self.p.use_htf:
            daily   = df["close"].resample("D").last().dropna()
            htf_ema = daily.ewm(span=200, adjust=False).mean()
            htf_ema200 = htf_ema.reindex(df.index, method="ffill")

        for i, (ts, row) in enumerate(df.iterrows()):
            if i < 220: continue
            close = row["close"]; atr = row["atr"]
            if pd.isna(atr) or atr <= 0 or pd.isna(close): continue

            # Normalisation dynamique
            dv_win.append(abs(row.get("dist_vwap", 0.0)))
            et_win.append(abs(row.get("ema_trend", 0.0) * 100))
            gp_win.append(abs(row.get("gann_pos", 0.0)))
            mr_win.append(abs(row.get("mom_raw", 0.0)))
            norm_w = {
                "vwap_std": max(np.std(dv_win), 0.05) if len(dv_win) > 5 else 0.5,
                "ema_std":  max(np.std(et_win), 0.05) if len(et_win) > 5 else 0.05,
                "gann_std": max(np.std(gp_win), 0.10) if len(gp_win) > 5 else 0.5,
                "mom_std":  max(np.std(mr_win), 0.10) if len(mr_win) > 5 else 1.0,
            }

            row_dict = row.to_dict(); row_dict["close"] = close; row_dict["atr"] = atr
            fv = build_features(row_dict, norm_w)

            # Online learning
            pending.append((fv, i))
            if len(pending) > self.p.ai_fwd:
                old_fv, _ = pending.popleft()
                lbl = compute_label_continuous(close, old_fv.ref_close, old_fv.ref_atr)
                if lbl is not None:
                    self.nn.train_step(old_fv, lbl)

            ai_proba = self.nn.predict(fv)

            # Mise à jour des seuils adaptatifs toutes les 500 barres
            if self.p.use_adaptive_thresh and i % 500 == 0 and len(self.nn.proba_history) > 100:
                thresh_long, thresh_short = self.nn.get_adaptive_thresholds(
                    self.p.target_trades_per_day, bars_per_day)

            # ── Conditions de signal ──
            conf       = float(row.get("conf_score", 0.0))
            is_chop    = bool(row.get("is_chop", False))
            in_ob_bull = bool(row.get("in_ob_bull", False))
            in_ob_bear = bool(row.get("in_ob_bear", False))
            in_fvg_bull= bool(row.get("in_fvg_bull", False))
            in_fvg_bear= bool(row.get("in_fvg_bear", False))
            bull_confirm = bool(row.get("bull_confirm", False))
            bear_confirm = bool(row.get("bear_confirm", False))

            in_zone_bull = in_ob_bull or in_fvg_bull
            in_zone_bear = in_ob_bear or in_fvg_bear

            # Signal AI (primary) ou confluence (fallback si zone présente)
            if self.p.use_ai:
                ai_long  = (ai_proba > thresh_long)  and in_zone_bull
                ai_short = (ai_proba < thresh_short) and in_zone_bear
                # Fallback confluence si AI n'a pas de signal fort mais conf est élevée
                conf_long_fb  = (conf  >= max_conf * self.p.conf_long_pct)  and in_zone_bull
                conf_short_fb = (conf  <= -max_conf * self.p.conf_short_pct) and in_zone_bear
                long_cond  = ai_long  or conf_long_fb
                short_cond = ai_short or conf_short_fb
            else:
                long_cond  = (conf >= max_conf * self.p.conf_long_pct)   and in_zone_bull
                short_cond = (conf <= -max_conf * self.p.conf_short_pct) and in_zone_bear

            # Filtres optionnels
            if self.p.use_chop_filter and is_chop:
                long_cond = short_cond = False
            if self.p.use_htf and htf_ema200 is not None:
                htf_val = htf_ema200.iloc[i] if i < len(htf_ema200) else np.nan
                if not pd.isna(htf_val):
                    if long_cond  and close < htf_val * 0.95: long_cond  = False
                    if short_cond and close > htf_val * 1.05: short_cond = False
            if self.p.use_candle_confirm:
                if long_cond  and not bull_confirm: long_cond  = False
                if short_cond and not bear_confirm: short_cond = False

            cooldown_ok = (i - last_signal_bar) >= self.p.cooldown_bars
            long_signal  = long_cond  and cooldown_ok and open_trade is None
            short_signal = short_cond and cooldown_ok and open_trade is None

            # ── Gestion position ouverte ──
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
                    sign = 1 if open_trade.direction == "LONG" else -1
                    pnl_pct = sign * (close - open_trade.entry_price) / open_trade.entry_price
                    pnl_usd = open_trade.size_usd * pnl_pct
                    open_trade.exit_price  = close; open_trade.exit_bar   = i
                    open_trade.exit_time   = ts;    open_trade.exit_reason = reason
                    open_trade.pnl_usd     = pnl_usd; open_trade.pnl_pct = pnl_pct
                    capital += pnl_usd
                    self.trades.append(open_trade)
                    open_trade = None

            # ── Ouverture trade ──
            if open_trade is None and (long_signal or short_signal):
                direction = "LONG" if long_signal else "SHORT"
                risk_usd  = capital * self.p.risk_pct
                size_usd  = min(risk_usd / max(self.p.sl_atr_mult * atr / close, 0.001),
                                capital * 0.20)
                sl = close - self.p.sl_atr_mult * atr if direction == "LONG" else close + self.p.sl_atr_mult * atr
                tp = close + self.p.tp_atr_mult * atr if direction == "LONG" else close - self.p.tp_atr_mult * atr
                open_trade = Trade(direction=direction, entry_price=close,
                                   entry_bar=i, entry_time=ts, size_usd=size_usd,
                                   sl=sl, tp=tp, conf_score=conf, ai_proba=ai_proba)
                last_signal_bar = i

            # Équité
            unr = 0.0
            if open_trade:
                sign = 1 if open_trade.direction == "LONG" else -1
                unr  = open_trade.size_usd * sign * (close - open_trade.entry_price) / open_trade.entry_price

            self.equity_curve.append({
                "time": ts, "equity": capital + unr, "cash": capital,
                "ai_proba": ai_proba, "conf_score": conf, "close": close,
                "thresh_long": thresh_long, "thresh_short": thresh_short,
                "in_trade": open_trade is not None,
            })

        # Fermer trade final
        if open_trade and len(df) > 0:
            lc = df["close"].iloc[-1]
            sign = 1 if open_trade.direction == "LONG" else -1
            pnl_pct = sign * (lc - open_trade.entry_price) / open_trade.entry_price
            pnl_usd = open_trade.size_usd * pnl_pct
            open_trade.exit_price = lc; open_trade.exit_bar = len(df)-1
            open_trade.exit_time  = df.index[-1]; open_trade.exit_reason = "END"
            open_trade.pnl_usd = pnl_usd; open_trade.pnl_pct = pnl_pct
            capital += pnl_usd; self.trades.append(open_trade)

        return self._compute_metrics(capital, total_days, bars_per_day)

    def _compute_metrics(self, final_capital, total_days, bars_per_day):
        initial  = self.p.capital
        n_trades = len(self.trades)
        if n_trades == 0:
            return {"error": "Aucun trade. Réduire cooldown_bars ou conf_long_pct.", "n_trades": 0}

        winners = [t for t in self.trades if t.pnl_usd > 0]
        losers  = [t for t in self.trades if t.pnl_usd <= 0]
        gross_p = sum(t.pnl_usd for t in winners)
        gross_l = abs(sum(t.pnl_usd for t in losers))
        pf      = gross_p / gross_l if gross_l > 0 else np.inf
        wr      = len(winners) / n_trades
        avg_w   = np.mean([t.pnl_usd for t in winners]) if winners else 0
        avg_l   = np.mean([t.pnl_usd for t in losers])  if losers  else 0
        exp     = wr * avg_w + (1 - wr) * avg_l

        eq      = pd.DataFrame(self.equity_curve).set_index("time")
        ev      = eq["equity"].values
        peak    = ev[0]; max_dd = 0.0
        for v in ev:
            peak   = max(peak, v)
            max_dd = max(max_dd, (peak - v) / peak)

        daily_eq  = eq["equity"].resample("D").last().dropna()
        daily_ret = daily_eq.pct_change().dropna()
        sharpe    = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0

        reasons = {}
        for t in self.trades: reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

        trades_per_day = n_trades / max(total_days, 1)

        return {
            "capital_initial":    initial,
            "capital_final":      final_capital,
            "total_return_pct":   (final_capital - initial) / initial * 100,
            "n_trades":           n_trades,
            "trades_per_day":     trades_per_day,
            "target_per_day":     self.p.target_trades_per_day,
            "total_days":         total_days,
            "n_long":             sum(1 for t in self.trades if t.direction == "LONG"),
            "n_short":            sum(1 for t in self.trades if t.direction == "SHORT"),
            "win_rate_pct":       wr * 100,
            "profit_factor":      pf,
            "expectancy_usd":     exp,
            "gross_profit":       gross_p,
            "gross_loss":         gross_l,
            "avg_win_usd":        avg_w,
            "avg_loss_usd":       avg_l,
            "max_drawdown_pct":   max_dd * 100,
            "sharpe_ratio":       sharpe,
            "avg_duration_bars":  np.mean([t.exit_bar - t.entry_bar for t in self.trades]),
            "exit_reasons":       reasons,
            "avg_ai_proba_long":  np.mean([t.ai_proba for t in self.trades if t.direction=="LONG"] or [0.5]),
            "avg_ai_proba_short": np.mean([t.ai_proba for t in self.trades if t.direction=="SHORT"] or [0.5]),
            "nn_weights":         self.nn.get_weights_summary(),
            "n_nn_samples":       self.nn.n_samples,
        }

    def print_report(self, m: dict):
        print("\n" + "═"*62)
        print("       RAPPORT DE BACKTEST — Confluence AI v4.0")
        print("═"*62)
        print(f"  Capital initial     : ${m['capital_initial']:>12,.2f}")
        print(f"  Capital final       : ${m['capital_final']:>12,.2f}")
        icon = "✅" if m['total_return_pct'] > 0 else "❌"
        print(f"  Rendement total     : {m['total_return_pct']:>+12.2f}%  {icon}")
        print(f"  Max Drawdown        : {m['max_drawdown_pct']:>12.2f}%")
        print(f"  Sharpe Ratio        : {m['sharpe_ratio']:>12.3f}")
        print("─"*62)
        target_icon = "✅" if m['trades_per_day'] >= m['target_per_day'] * 0.8 else "⚠️"
        print(f"  Trades totaux       : {m['n_trades']:>12}  ({m['total_days']} jours)")
        print(f"  Trades/jour         : {m['trades_per_day']:>12.2f}  (cible: {m['target_per_day']:.1f}) {target_icon}")
        print(f"  LONG / SHORT        : {m['n_long']:>5} / {m['n_short']:<5}")
        print(f"  Win Rate            : {m['win_rate_pct']:>12.1f}%")
        print(f"  Profit Factor       : {m['profit_factor']:>12.3f}")
        print(f"  Espérance/trade     : ${m['expectancy_usd']:>+12.2f}")
        print(f"  Gain moyen          : ${m['avg_win_usd']:>+12.2f}")
        print(f"  Perte moyenne       : ${m['avg_loss_usd']:>+12.2f}")
        print(f"  Durée moy.(barres)  : {m['avg_duration_bars']:>12.1f}")
        print("─"*62)
        print(f"  Échantillons AI     : {m['n_nn_samples']:>12}")
        print(f"  AI proba moy. LONG  : {m['avg_ai_proba_long']:>12.3f}")
        print(f"  AI proba moy. SHORT : {m['avg_ai_proba_short']:>12.3f}")
        print("─"*62)
        print("  Clôtures par raison :")
        for r, c in sorted(m["exit_reasons"].items(), key=lambda x:-x[1]):
            print(f"    {r:<12}: {c:>4}")
        print("─"*62)
        print("  Importance features AI :")
        ws = sorted(m["nn_weights"].items(), key=lambda x: -x[1])
        mx = max((v for _,v in ws), default=1e-9); mx = max(mx, 1e-9)
        for name, w in ws:
            bar = "█" * max(1, int(w * 25 / mx))
            print(f"    {name:<8}: {bar:<25} {w:.4f}")
        print("═"*62)
