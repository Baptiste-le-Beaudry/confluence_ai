"""
backtest.py - Moteur de backtest Confluence AI v5.0
Intègre les filtres d'entrée avancés (entry_filters.py):
  - Setup Squeeze Release
  - Setup Zone Pullback
  - Setup Combiné (recommandé)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass, field
from indicators import compute_all
from ai_model import NeuralNet, FeatVec, build_features, compute_label_continuous, pretrain_on_history
from entry_filters import get_entry_signal


@dataclass
class BacktestParams:
    # Setup d'entrée
    setup_mode:     str   = "combined"  # "squeeze", "pullback", "combined"
    min_quality:    float = 0.55        # score qualité minimum (0-1)
    ai_thresh:      float = 0.52        # seuil AI minimum (adaptatif prend le dessus)
    conf_pct:       float = 0.25        # fraction du score max pour confluence

    # AI
    ai_fwd:         int   = 5
    ai_norm_len:    int   = 50
    lr:             float = 0.02
    grad_clip:      float = 0.5
    use_ai:         bool  = True
    pretrain_epochs:int   = 10
    use_adaptive_thresh: bool = True
    target_trades_per_day: float = 5.0  # 5 trades/jour sur 1min

    # Filtres
    use_htf:        bool  = True
    use_chop_filter:bool  = True
    cooldown_bars:  int   = 10

    # Risque
    capital:        float = 10_000.0
    risk_pct:       float = 0.01
    sl_atr_mult:    float = 2.0
    tp_atr_mult:    float = 4.0
    # Breakeven automatique
    use_breakeven:  bool  = True    # activer le breakeven
    be_trigger_pct: float = 0.50    # déclencher à 50% du chemin vers le TP
    be_offset_pct:  float = 0.001   # SL = entrée + 0.1% (léger buffer au-dessus du BE)
    # Trailing stop (optionnel)
    use_trailing:   bool  = False   # trailing stop après breakeven
    trail_atr_mult: float = 1.0     # distance du trailing = 1× ATR
    # Take Profit partiel (50%/50%)
    use_partial_tp: bool  = True    # activer TP partiel
    tp1_mult:       float = 1.5     # TP1 = 1.5× ATR (50% de la position)
    tp2_mult:       float = 2.5     # TP2 = 2.5× ATR (50% restant)
    tp1_pct:        float = 0.50    # fraction fermée au TP1
    indicator_params: dict = field(default_factory=dict)


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
    quality:     float = 0.0   # score qualité du setup
    setup_type:  str   = ""    # raison/type d'entrée
    be_triggered:bool  = False  # breakeven activé
    be_price:    float = 0.0    # prix du breakeven
    max_profit:  float = 0.0    # profit max atteint (pour trailing)
    tp1_hit:     bool  = False  # TP1 partiel atteint
    tp1_pnl:     float = 0.0    # P&L réalisé au TP1
    remaining_pct: float = 1.0  # fraction restante de la position

    @property
    def is_winner(self): return self.pnl_usd > 0


class Backtester:
    def __init__(self, params: BacktestParams = None):
        self.p = params or BacktestParams()
        self.nn = NeuralNet(lr=self.p.lr, grad_clip=self.p.grad_clip)
        self.trades: list[Trade] = []
        self.equity_curve: list[dict] = []

    def run(self, df_raw: pd.DataFrame) -> dict:
        print(f"\nSetup: {self.p.setup_mode.upper()} | Qualité min: {self.p.min_quality} | AI: {self.p.ai_thresh}")
        print(f"Étape 1/4 — Calcul des indicateurs ({len(df_raw)} barres)...")
        df = compute_all(df_raw, self.p.indicator_params)

        # Barres par jour
        if len(df) > 1:
            total_secs   = (df.index[-1] - df.index[0]).total_seconds()
            bar_secs     = total_secs / len(df)
            bars_per_day = max(86400.0 / bar_secs, 1.0)
        else:
            bars_per_day = 96.0
        total_days = max((df.index[-1] - df.index[0]).days, 1)

        # Normalisation globale
        norm_w = {
            "vwap_std": max(df["dist_vwap"].abs().std(), 0.05),
            "ema_std":  max((df["ema_trend"]*100).abs().std(), 0.05),
            "gann_std": max(df["gann_pos"].abs().std(), 0.10),
            "mom_std":  max(df["mom_raw"].abs().std(), 0.10),
        }

        # Pré-entraînement
        if self.p.use_ai:
            print(f"Étape 2/4 — Pré-entraînement AI...")
            pretrain_on_history(
                self.nn, df, norm_w,
                fwd_bars   = self.p.ai_fwd * 3,
                sl_atr     = self.p.sl_atr_mult,
                tp_atr     = self.p.tp_atr_mult,
                n_epochs   = self.p.pretrain_epochs,
                verbose    = True,
            )
            # Calibrer les seuils adaptatifs
            all_probas = []
            for i, (_, row) in enumerate(df.iterrows()):
                if i < 220: continue
                rd = row.to_dict(); rd["close"] = row["close"]; rd["atr"] = row["atr"]
                if pd.isna(rd.get("atr", np.nan)) or rd.get("atr", 0) <= 0: continue
                fv = build_features(rd, norm_w)
                all_probas.append(self.nn.predict(fv))

            print(f"  Distribution AI: min={min(all_probas):.3f} max={max(all_probas):.3f} mean={np.mean(all_probas):.3f}")

        print(f"Étape 3/4 — Calcul des seuils adaptatifs...")
        thresh_long = thresh_short = None
        if self.p.use_adaptive_thresh and all_probas:
            probas = np.array(all_probas)
            frac   = np.clip(self.p.target_trades_per_day / bars_per_day / 2.0, 0.001, 0.10)
            thresh_long  = max(float(np.percentile(probas, (1-frac)*100)), self.p.ai_thresh)
            thresh_short = min(float(np.percentile(probas, frac*100)), 1.0 - self.p.ai_thresh)
            print(f"  Seuils: LONG > {thresh_long:.3f} | SHORT < {thresh_short:.3f}")
        else:
            thresh_long  = self.p.ai_thresh
            thresh_short = 1.0 - self.p.ai_thresh

        max_conf = float(df["max_conf"].iloc[-1])

        # Simulation
        print(f"Étape 4/4 — Simulation ({self.p.setup_mode} setup)...")
        capital    = self.p.capital
        open_trade = None
        pending    = deque()
        last_sig   = -self.p.cooldown_bars

        # HTF
        htf_ema200 = None
        if self.p.use_htf:
            daily   = df["close"].resample("D").last().dropna()
            htf_ema = daily.ewm(span=200, adjust=False).mean()
            htf_ema200 = htf_ema.reindex(df.index, method="ffill")

        dv = deque(maxlen=self.p.ai_norm_len)
        et = deque(maxlen=self.p.ai_norm_len)
        gp = deque(maxlen=self.p.ai_norm_len)
        mr = deque(maxlen=self.p.ai_norm_len)

        rows_list = list(df.iterrows())

        for idx, (ts, row) in enumerate(rows_list):
            if idx < 220: continue
            close = row["close"]; atr = row["atr"]
            if pd.isna(atr) or atr <= 0 or pd.isna(close): continue

            dv.append(abs(row.get("dist_vwap", 0.0)))
            et.append(abs(row.get("ema_trend", 0.0)*100))
            gp.append(abs(row.get("gann_pos", 0.0)))
            mr.append(abs(row.get("mom_raw", 0.0)))
            nw = {
                "vwap_std": max(np.std(dv), 0.05) if len(dv)>5 else norm_w["vwap_std"],
                "ema_std":  max(np.std(et), 0.05) if len(et)>5 else norm_w["ema_std"],
                "gann_std": max(np.std(gp), 0.10) if len(gp)>5 else norm_w["gann_std"],
                "mom_std":  max(np.std(mr), 0.10) if len(mr)>5 else norm_w["mom_std"],
            }

            rd = row.to_dict(); rd["close"] = close; rd["atr"] = atr
            # Ajouter les EMA au row_dict pour les filtres
            rd["ema20"]  = float(row.get("ema20", close))
            rd["ema50"]  = float(row.get("ema50", close))
            rd["ema200"] = float(row.get("ema200", close))
            fv = build_features(rd, nw)

            # Online learning
            pending.append((fv, idx))
            if len(pending) > self.p.ai_fwd:
                old_fv, _ = pending.popleft()
                lbl = compute_label_continuous(close, old_fv.ref_close, old_fv.ref_atr)
                if lbl is not None:
                    self.nn.train_step(old_fv, lbl)

            ai_proba = self.nn.predict(fv)

            # Recalibrer toutes les 500 barres
            if self.p.use_adaptive_thresh and idx % 500 == 0 and len(self.nn.proba_history) > 50:
                th_l, th_s = self.nn.get_adaptive_thresholds(
                    self.p.target_trades_per_day, bars_per_day)
                thresh_long  = max(th_l, 0.505)
                thresh_short = min(th_s, 0.495)

            # Filtre HTF
            if self.p.use_htf and htf_ema200 is not None:
                htf_val = htf_ema200.iloc[idx] if idx < len(htf_ema200) else np.nan
                if not pd.isna(htf_val):
                    rd["ema200_htf"] = htf_val

            # Ligne précédente pour squeeze release
            prev_rd = rows_list[idx-1][1].to_dict() if idx > 0 else {}

            # ── Filtre d'entrée avancé ──
            long_sig, short_sig, reason, quality = get_entry_signal(
                row         = rd,
                prev_row    = prev_rd,
                ai_proba    = ai_proba,
                conf_score  = float(row.get("conf_score", 0.0)),
                max_conf    = max_conf,
                setup_mode  = self.p.setup_mode,
                ai_thresh   = thresh_long,
                conf_pct    = self.p.conf_pct,
                min_quality = self.p.min_quality,
            )

            # Cooldown
            cooldown_ok = (idx - last_sig) >= self.p.cooldown_bars
            long_sig  = long_sig  and cooldown_ok and open_trade is None
            short_sig = short_sig and cooldown_ok and open_trade is None

            # Gestion position ouverte
            if open_trade is not None:

                # ── Breakeven automatique ──
                if self.p.use_breakeven and not open_trade.be_triggered:
                    if open_trade.direction == "LONG":
                        total_dist   = open_trade.tp - open_trade.entry_price
                        progress     = close - open_trade.entry_price
                        be_threshold = total_dist * self.p.be_trigger_pct
                        if progress >= be_threshold:
                            # Monter le SL au breakeven + petit buffer
                            be_sl = open_trade.entry_price * (1.0 + self.p.be_offset_pct)
                            if be_sl > open_trade.sl:  # seulement si ça monte le SL
                                open_trade.sl           = be_sl
                                open_trade.be_triggered = True
                                open_trade.be_price     = be_sl
                    else:  # SHORT
                        total_dist   = open_trade.entry_price - open_trade.tp
                        progress     = open_trade.entry_price - close
                        be_threshold = total_dist * self.p.be_trigger_pct
                        if progress >= be_threshold:
                            be_sl = open_trade.entry_price * (1.0 - self.p.be_offset_pct)
                            if be_sl < open_trade.sl:  # seulement si ça baisse le SL
                                open_trade.sl           = be_sl
                                open_trade.be_triggered = True
                                open_trade.be_price     = be_sl

                # ── Trailing stop (optionnel, après breakeven) ──
                if self.p.use_trailing and open_trade.be_triggered:
                    if open_trade.direction == "LONG":
                        # Trailing: SL suit le prix à trail_atr_mult × ATR en dessous
                        trail_sl = close - self.p.trail_atr_mult * atr
                        if trail_sl > open_trade.sl:  # seulement si ça monte
                            open_trade.sl = trail_sl
                    else:  # SHORT
                        trail_sl = close + self.p.trail_atr_mult * atr
                        if trail_sl < open_trade.sl:  # seulement si ça descend
                            open_trade.sl = trail_sl

                # ── TP Partiel ──
                tp1_price = getattr(open_trade, '_tp1', open_trade.tp)
                if self.p.use_partial_tp and not open_trade.tp1_hit:
                    if open_trade.direction == "LONG" and close >= tp1_price:
                        # Fermer 50% de la position au TP1
                        partial_size = open_trade.size_usd * self.p.tp1_pct
                        pnl_tp1      = partial_size * (close - open_trade.entry_price) / open_trade.entry_price
                        open_trade.tp1_hit      = True
                        open_trade.tp1_pnl      = pnl_tp1
                        open_trade.remaining_pct = 1.0 - self.p.tp1_pct
                        open_trade.size_usd     -= partial_size   # réduire la position
                        capital                 += pnl_tp1
                        # Monter le SL au BE après TP1
                        be_sl = open_trade.entry_price * (1.0 + self.p.be_offset_pct)
                        if be_sl > open_trade.sl:
                            open_trade.sl           = be_sl
                            open_trade.be_triggered = True
                    elif open_trade.direction == "SHORT" and close <= tp1_price:
                        partial_size = open_trade.size_usd * self.p.tp1_pct
                        pnl_tp1      = partial_size * (open_trade.entry_price - close) / open_trade.entry_price
                        open_trade.tp1_hit      = True
                        open_trade.tp1_pnl      = pnl_tp1
                        open_trade.remaining_pct = 1.0 - self.p.tp1_pct
                        open_trade.size_usd     -= partial_size
                        capital                 += pnl_tp1
                        be_sl = open_trade.entry_price * (1.0 - self.p.be_offset_pct)
                        if be_sl < open_trade.sl:
                            open_trade.sl           = be_sl
                            open_trade.be_triggered = True

                closed = False; exit_reason = ""
                if open_trade.direction == "LONG":
                    if   close <= open_trade.sl: closed, exit_reason = True, "SL_BE" if open_trade.be_triggered else "SL"
                    elif close >= open_trade.tp: closed, exit_reason = True, "TP2" if open_trade.tp1_hit else "TP"
                    elif short_sig:              closed, exit_reason = True, "REVERSE"
                else:
                    if   close >= open_trade.sl: closed, exit_reason = True, "SL_BE" if open_trade.be_triggered else "SL"
                    elif close <= open_trade.tp: closed, exit_reason = True, "TP2" if open_trade.tp1_hit else "TP"
                    elif long_sig:               closed, exit_reason = True, "REVERSE"

                if closed:
                    sign    = 1 if open_trade.direction == "LONG" else -1
                    pnl_pct = sign * (close - open_trade.entry_price) / open_trade.entry_price
                    pnl_usd = open_trade.size_usd * pnl_pct
                    open_trade.exit_price  = close; open_trade.exit_bar    = idx
                    open_trade.exit_time   = ts;    open_trade.exit_reason  = exit_reason
                    open_trade.pnl_usd     = pnl_usd; open_trade.pnl_pct   = pnl_pct
                    capital += pnl_usd
                    self.trades.append(open_trade)
                    open_trade = None

            # Ouverture
            if open_trade is None and (long_sig or short_sig):
                direction = "LONG" if long_sig else "SHORT"
                risk_usd  = capital * self.p.risk_pct
                denom     = max(self.p.sl_atr_mult * atr / close, 0.001)
                size_usd  = min(risk_usd / denom, capital * 0.20)
                sl = close - self.p.sl_atr_mult*atr if direction=="LONG" else close + self.p.sl_atr_mult*atr
                tp = close + self.p.tp_atr_mult*atr if direction=="LONG" else close - self.p.tp_atr_mult*atr
                # TP partiel: TP1 à tp1_mult ATR, TP2 à tp2_mult ATR
                if self.p.use_partial_tp:
                    tp1 = close + self.p.tp1_mult*atr if direction=="LONG" else close - self.p.tp1_mult*atr
                    tp2 = close + self.p.tp2_mult*atr if direction=="LONG" else close - self.p.tp2_mult*atr
                else:
                    tp1 = tp  # TP unique
                    tp2 = tp

                open_trade = Trade(
                    direction=direction, entry_price=close,
                    entry_bar=idx, entry_time=ts, size_usd=size_usd,
                    sl=sl, tp=tp2,  # tp final = TP2
                    conf_score=float(row.get("conf_score",0)),
                    ai_proba=ai_proba, quality=quality, setup_type=reason,
                    remaining_pct=1.0,
                )
                open_trade._tp1 = tp1  # stocker TP1 temporairement
                open_trade._tp2 = tp2
                last_sig = idx

            unr = 0.0
            if open_trade:
                sign = 1 if open_trade.direction=="LONG" else -1
                unr  = open_trade.size_usd*sign*(close-open_trade.entry_price)/open_trade.entry_price

            self.equity_curve.append({
                "time": ts, "equity": capital+unr, "cash": capital,
                "ai_proba": ai_proba, "conf_score": float(row.get("conf_score",0)),
                "close": close, "thresh_long": thresh_long, "thresh_short": thresh_short,
                "quality": quality,
            })

        # Fermer trade final
        if open_trade and len(df)>0:
            lc = df["close"].iloc[-1]
            sign = 1 if open_trade.direction=="LONG" else -1
            pnl_pct = sign*(lc-open_trade.entry_price)/open_trade.entry_price
            pnl_usd = open_trade.size_usd*pnl_pct
            open_trade.exit_price=lc; open_trade.exit_time=df.index[-1]
            open_trade.exit_reason="END"; open_trade.pnl_usd=pnl_usd
            open_trade.pnl_pct=pnl_pct; capital+=pnl_usd
            self.trades.append(open_trade)

        return self._compute_metrics(capital, total_days, bars_per_day)

    def _compute_metrics(self, final_capital, total_days, bars_per_day):
        initial  = self.p.capital
        n        = len(self.trades)
        if n == 0:
            return {"error": "Aucun trade.", "n_trades": 0}

        wins    = [t for t in self.trades if t.pnl_usd > 0]
        losses  = [t for t in self.trades if t.pnl_usd <= 0]
        gp      = sum(t.pnl_usd for t in wins)
        gl      = abs(sum(t.pnl_usd for t in losses))
        wr      = len(wins)/n
        avg_w   = np.mean([t.pnl_usd for t in wins])   if wins   else 0
        avg_l   = np.mean([t.pnl_usd for t in losses]) if losses else 0
        pf      = gp/gl if gl>0 else np.inf
        exp     = wr*avg_w+(1-wr)*avg_l
        rr      = abs(avg_w/avg_l) if avg_l!=0 else 0

        eq   = pd.DataFrame(self.equity_curve).set_index("time")
        ev   = eq["equity"].values
        peak = ev[0]; max_dd=0.0
        for v in ev:
            peak=max(peak,v); max_dd=max(max_dd,(peak-v)/peak)

        dr  = eq["equity"].resample("D").last().dropna().pct_change().dropna()
        shr = float(dr.mean()/dr.std()*np.sqrt(252)) if dr.std()>0 else 0.0

        reasons = {}
        for t in self.trades: reasons[t.exit_reason]=reasons.get(t.exit_reason,0)+1

        # Stats breakeven
        be_triggered = [t for t in self.trades if t.be_triggered]
        be_saved     = [t for t in be_triggered if t.exit_reason == "SL_BE"]
        be_won       = [t for t in be_triggered if t.pnl_usd > 0]
        tp1_trades   = [t for t in self.trades if t.tp1_hit]
        tp2_trades   = [t for t in self.trades if t.exit_reason == "TP2"]
        tp1_pnl_total= sum(t.tp1_pnl for t in tp1_trades)

        # Analyse des setups déclencheurs
        setup_types = {}
        for t in self.trades:
            # Extraire le type de setup depuis la raison
            s = t.setup_type.split("|")[0] if t.setup_type else "UNKNOWN"
            setup_types[s] = setup_types.get(s, 0) + 1

        # Qualité moyenne des trades gagnants vs perdants
        avg_quality_wins   = np.mean([t.quality for t in wins])   if wins   else 0
        avg_quality_losses = np.mean([t.quality for t in losses]) if losses else 0

        return {
            "capital_initial":     initial,
            "capital_final":       final_capital,
            "total_return_pct":    (final_capital-initial)/initial*100,
            "n_trades":            n,
            "trades_per_day":      n/max(total_days,1),
            "target_per_day":      self.p.target_trades_per_day,
            "total_days":          total_days,
            "n_long":              sum(1 for t in self.trades if t.direction=="LONG"),
            "n_short":             sum(1 for t in self.trades if t.direction=="SHORT"),
            "win_rate_pct":        wr*100,
            "profit_factor":       pf,
            "expectancy_usd":      exp,
            "gross_profit":        gp,
            "gross_loss":          gl,
            "avg_win_usd":         avg_w,
            "avg_loss_usd":        avg_l,
            "max_drawdown_pct":    max_dd*100,
            "sharpe_ratio":        shr,
            "avg_duration_bars":   np.mean([t.exit_bar-t.entry_bar for t in self.trades]),
            "exit_reasons":        reasons,
            "setup_types":         setup_types,
            "avg_quality_wins":    avg_quality_wins,
            "avg_quality_losses":  avg_quality_losses,
            "n_nn_samples":        self.nn.n_samples,
            "n_be_triggered":      len(be_triggered),
            "n_be_saved":          len(be_saved),
            "n_be_won":            len(be_won),
            "be_trigger_pct":      self.p.be_trigger_pct,
            "n_tp1_hit":           len(tp1_trades),
            "n_tp2_hit":           len(tp2_trades),
            "tp1_pnl_total":       tp1_pnl_total,
        }

    def print_report(self, m: dict):
        print("\n"+"═"*62)
        print(f"  RAPPORT — Setup: {self.p.setup_mode.upper()} | Qualité: {self.p.min_quality}")
        print("═"*62)
        print(f"  Capital initial    : ${m['capital_initial']:>12,.2f}")
        print(f"  Capital final      : ${m['capital_final']:>12,.2f}")
        icon = "✅" if m['total_return_pct']>0 else "❌"
        print(f"  Rendement total    : {m['total_return_pct']:>+12.2f}%  {icon}")
        print(f"  Max Drawdown       : {m['max_drawdown_pct']:>12.2f}%")
        print(f"  Sharpe Ratio       : {m['sharpe_ratio']:>12.3f}")
        print("─"*62)
        print(f"  Trades totaux      : {m['n_trades']:>12}")
        print(f"  Trades/jour        : {m['trades_per_day']:>12.2f}  (cible: {m['target_per_day']:.1f})")
        print(f"  LONG / SHORT       : {m['n_long']:>5} / {m['n_short']:<5}")
        print(f"  Win Rate           : {m['win_rate_pct']:>12.1f}%")
        print(f"  Profit Factor      : {m['profit_factor']:>12.3f}")
        print(f"  Espérance/trade    : ${m['expectancy_usd']:>+12.2f}")
        print(f"  Qualité moy wins   : {m['avg_quality_wins']:>12.3f}")
        print(f"  Qualité moy losses : {m['avg_quality_losses']:>12.3f}")
        print("─"*62)
        print("  Raisons de sortie:")
        for r,c in sorted(m["exit_reasons"].items(),key=lambda x:-x[1]):
            print(f"    {r:<14}: {c:>4}")
        print("─"*62)
        print("  Setups déclencheurs:")
        for s,c in sorted(m.get("setup_types",{}).items(),key=lambda x:-x[1]):
            print(f"    {s:<20}: {c:>4}")
        print("─"*62)
        n_be    = m.get("n_be_triggered", 0)
        n_saved = m.get("n_be_saved", 0)
        n_won   = m.get("n_be_won", 0)
        n_tp1   = m.get("n_tp1_hit", 0)
        n_tp2   = m.get("n_tp2_hit", 0)
        tp1_pnl = m.get("tp1_pnl_total", 0)
        if n_tp1 > 0:
            print(f"  TP Partiel stats:")
            print(f"    TP1 atteints: {n_tp1:>4}  ({n_tp1/m['n_trades']*100:.1f}% des trades)")
            print(f"    TP2 atteints: {n_tp2:>4}  ({n_tp2/m['n_trades']*100:.1f}% des trades)")
            print(f"    P&L TP1     : ${tp1_pnl:>+8.2f}  (profits déjà sécurisés)")
        if n_be > 0:
            print(f"  Breakeven stats:")
            print(f"    Déclenchés  : {n_be:>4}  ({n_be/m['n_trades']*100:.1f}% des trades)")
            print(f"    Sauvés (BE) : {n_saved:>4}  (auraient été SL)")
        print("═"*62)
