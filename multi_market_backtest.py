"""
multi_market_backtest.py
Backtest multi-marchés sur données 1 minute depuis Binance.

Marchés testés simultanément:
  Crypto : BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT
  Forex* : EUR/USD, GBP/USD, USD/JPY (via Binance si disponible)
  Or*    : XAU/USDT (via Binance si disponible)

*Note: Binance ne propose pas tous les forex. Pour forex/or en 1min
on utilise les paires disponibles sur Binance ou un CSV externe.

Usage:
    python multi_market_backtest.py
    python multi_market_backtest.py --bars 5000
    python multi_market_backtest.py --markets BTCUSDT ETHUSDT SOLUSDT
    python multi_market_backtest.py --from-csv  (utilise des CSV locaux)
"""

import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque

sys.path.insert(0, os.path.dirname(__file__))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from indicators import compute_all
from ai_model   import NeuralNet, build_features, compute_label_continuous, pretrain_on_history


# ─────────────────────────────────────────────
# Marchés disponibles
# ─────────────────────────────────────────────

MARKETS_BINANCE = {
    # Crypto (disponibles sur Binance en 1min)
    "BTCUSDT":  {"name": "Bitcoin",   "type": "crypto", "pip": 0.01},
    "ETHUSDT":  {"name": "Ethereum",  "type": "crypto", "pip": 0.01},
    "BNBUSDT":  {"name": "BNB",       "type": "crypto", "pip": 0.001},
    "SOLUSDT":  {"name": "Solana",    "type": "crypto", "pip": 0.001},
    "XRPUSDT":  {"name": "Ripple",    "type": "crypto", "pip": 0.0001},
    "ADAUSDT":  {"name": "Cardano",   "type": "crypto", "pip": 0.0001},
    "DOGEUSDT": {"name": "Dogecoin",  "type": "crypto", "pip": 0.00001},
    "AVAXUSDT": {"name": "Avalanche", "type": "crypto", "pip": 0.001},
}

# Symboles à tester par défaut
DEFAULT_MARKETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]


# ─────────────────────────────────────────────
# Téléchargement données 1min Binance
# ─────────────────────────────────────────────

def fetch_1min_binance(symbol: str, n_bars: int = 5000) -> pd.DataFrame:
    """
    Télécharge les données 1 minute depuis Binance (API publique gratuite).
    Binance limite à 1000 barres par requête → pagination automatique.

    Args:
        symbol: ex "BTCUSDT"
        n_bars: nombre de barres à récupérer (max ~1000 par appel)

    Returns:
        DataFrame OHLCV avec DatetimeIndex UTC
    """
    import ccxt
    exchange = ccxt.binance({"enableRateLimit": True})

    all_ohlcv = []
    limit     = 1000
    end_ts    = None
    remaining = n_bars

    print(f"  Téléchargement {symbol} 1min ({n_bars} barres)...", end="", flush=True)

    while remaining > 0:
        fetch_n = min(limit, remaining)
        params  = {}
        if end_ts:
            params["endTime"] = end_ts

        try:
            ohlcv = exchange.fetch_ohlcv(
                symbol.replace("USDT", "/USDT"),
                timeframe = "1m",
                limit     = fetch_n,
                params    = params,
            )
        except Exception as e:
            print(f" ERREUR: {e}")
            break

        if not ohlcv:
            break

        all_ohlcv = ohlcv + all_ohlcv
        end_ts    = ohlcv[0][0] - 1  # avant la première barre
        remaining -= len(ohlcv)

        if len(ohlcv) < fetch_n:
            break

        time.sleep(0.1)  # respecter le rate limit

    if not all_ohlcv:
        print(" 0 barres")
        return pd.DataFrame()

    df = pd.DataFrame(all_ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index().drop_duplicates()
    df = df.astype(float)

    print(f" {len(df)} barres ({df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')})")
    return df


def load_or_download(symbol: str, n_bars: int, cache_dir: str = "data_1min") -> pd.DataFrame:
    """
    Charge depuis le cache si disponible, sinon télécharge.
    Cache dans data_1min/{symbol}_1min.csv
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{symbol}_1min.csv")

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        # Si le cache a assez de barres récentes, l'utiliser
        if len(df) >= n_bars * 0.9:
            print(f"  {symbol}: chargé depuis cache ({len(df)} barres)")
            return df.tail(n_bars)

    # Télécharger
    df = fetch_1min_binance(symbol, n_bars)
    if not df.empty:
        df.to_csv(cache_path)

    return df


# ─────────────────────────────────────────────
# Backtest sur un seul marché
# ─────────────────────────────────────────────

@dataclass
class MarketResult:
    symbol:          str
    name:            str
    n_bars:          int
    n_trades:        int
    n_long:          int
    n_short:         int
    win_rate:        float
    profit_factor:   float
    total_return:    float
    max_drawdown:    float
    sharpe:          float
    expectancy:      float
    avg_win:         float
    avg_loss:        float
    ratio_rr:        float
    capital_final:   float
    trades_per_day:  float
    total_days:      int
    exit_reasons:    dict
    equity_curve:    list
    trades:          list


def run_single_market(
    symbol:       str,
    df_raw:       pd.DataFrame,
    capital:      float = 10_000.0,
    risk_pct:     float = 0.01,
    sl_atr:       float = 2.0,
    tp_atr:       float = 4.0,
    cooldown:     int   = 3,      # 3 barres 1min = 3 minutes entre trades
    conf_pct:     float = 0.25,
    target_tpd:   float = 5.0,   # 5 trades/jour sur 1min
    pretrain_ep:  int   = 8,
    use_chop:     bool  = True,
    train_end:    pd.Timestamp = None,
    test_end:     pd.Timestamp = None,
    recalib_bars: int   = 1440,   # recalibrer les seuils tous les N barres (1440 = 1x/jour en 1min)
    base_alloc:   float = 0.0,    # 0 = bot pur (in/out). >0 = coeur long + overlay tactique
    long_only:    bool  = False,  # n'ouvrir que des LONG (pas de SHORT)
) -> MarketResult:
    """
    Backteste la stratégie sur un marché donné.

    Par défaut, découpe 70% entraînement / 30% test par fraction du nombre
    de barres. Si train_end est fourni, découpe par date à la place (ex:
    3 mois d'entraînement / 3 mois de test) — nécessaire pour une vraie
    évaluation out-of-sample sur une période calendaire précise.
    """

    if len(df_raw) < 300:
        return None

    # Calcul indicateurs
    df = compute_all(df_raw)

    # Barres par jour (1min → 1440 barres/jour théorique, moins les week-ends)
    total_secs   = (df.index[-1] - df.index[0]).total_seconds()
    bars_per_day = max(86400.0 / max(total_secs / len(df), 1), 1.0)

    # Découpage entraînement / test
    if train_end is not None:
        df_train = df[df.index < train_end]
        df_test  = df[df.index >= train_end]
        if test_end is not None:
            df_test = df_test[df_test.index < test_end]
    else:
        split    = int(len(df) * 0.70)
        df_train = df.iloc[:split]
        df_test  = df.iloc[split:]

    if len(df_train) < 300 or len(df_test) < 50:
        print(f"  ⚠️  {symbol}: données train/test insuffisantes "
              f"(train={len(df_train)}, test={len(df_test)})")
        return None

    # Normalisation calculée UNIQUEMENT sur la période d'entraînement — sinon
    # les stats de la période de test (future) fuient dans le pré-entraînement.
    norm_w = {
        "vwap_std": max(df_train["dist_vwap"].abs().std(), 0.05),
        "ema_std":  max((df_train["ema_trend"]*100).abs().std(), 0.05),
        "gann_std": max(df_train["gann_pos"].abs().std(), 0.10),
        "mom_std":  max(df_train["mom_raw"].abs().std(), 0.10),
    }

    # Réseau de neurones
    nn = NeuralNet(lr=0.02, grad_clip=0.5)

    pretrain_on_history(
        nn, df_train, norm_w,
        fwd_bars = max(cooldown, 3),
        sl_atr   = sl_atr,
        tp_atr   = tp_atr,
        n_epochs = pretrain_ep,
        verbose  = False,
    )

    # Calibrer les seuils adaptatifs
    probas_cal = []
    for _, row in df_train.iloc[220:].iterrows():
        rd = row.to_dict(); rd["close"] = row["close"]; rd["atr"] = row["atr"]
        if not pd.isna(rd.get("atr", np.nan)) and rd.get("atr", 0) > 0:
            fv = build_features(rd, norm_w)
            probas_cal.append(nn.predict(fv))

    frac = np.clip(target_tpd / bars_per_day / 2.0, 0.001, 0.15)
    if len(probas_cal) > 30:
        probas_cal = np.array(probas_cal)
        thresh_l   = max(float(np.percentile(probas_cal, (1-frac)*100)), 0.51)
        thresh_s   = min(float(np.percentile(probas_cal, frac*100)),     0.49)
    else:
        thresh_l, thresh_s = 0.62, 0.38

    # Simulation out-of-sample sur df_test (déjà découpé plus haut)
    max_conf = float(df["max_conf"].iloc[-1])

    # Coeur long + overlay tactique: une fraction du capital reste en position
    # longue permanente (capte la dérive de fond de l'actif) pendant que le
    # reste ("cap") trade tactiquement avec la même logique de signal que le
    # bot pur ci-dessous. base_alloc=0 => comportement identique au bot pur.
    base_capital = capital * base_alloc
    base_units   = (base_capital / float(df_test["close"].iloc[0])) if (base_alloc > 0 and len(df_test) > 0) else 0.0

    cap         = capital - base_capital
    open_trade  = None
    trades      = []
    equity      = []
    pending     = deque()
    last_sig    = -cooldown

    dv = deque(maxlen=50); et = deque(maxlen=50)
    gp = deque(maxlen=50); mr = deque(maxlen=50)

    # Recalibration périodique des seuils pendant le test — indispensable dès
    # que l'apprentissage online (train_step ci-dessous) continue pendant la
    # simulation: un seuil figé au moment du pré-entraînement devient obsolète
    # en quelques jours et laisse passer bien plus de signaux que la cible
    # (constaté: ~6x le volume visé, présent dès le 1er mois de test).
    recent_probas   = deque(maxlen=max(recalib_bars * 4, 2000))
    n_recalibs      = 0
    thresh_history  = [(0, thresh_l, thresh_s)]

    for i, (ts, row) in enumerate(df_test.iterrows()):
        close = row["close"]; atr = row["atr"]
        if pd.isna(atr) or atr <= 0 or pd.isna(close): continue

        dv.append(abs(row.get("dist_vwap", 0.0)))
        et.append(abs(row.get("ema_trend", 0.0)*100))
        gp.append(abs(row.get("gann_pos",  0.0)))
        mr.append(abs(row.get("mom_raw",   0.0)))
        nw = {
            "vwap_std": max(np.std(dv), 0.05) if len(dv)>3 else norm_w["vwap_std"],
            "ema_std":  max(np.std(et), 0.05) if len(et)>3 else norm_w["ema_std"],
            "gann_std": max(np.std(gp), 0.10) if len(gp)>3 else norm_w["gann_std"],
            "mom_std":  max(np.std(mr), 0.10) if len(mr)>3 else norm_w["mom_std"],
        }
        rd = row.to_dict(); rd["close"]=close; rd["atr"]=atr
        fv = build_features(rd, nw)

        # Online learning
        pending.append((fv, i))
        if len(pending) > 5:
            old_fv, _ = pending.popleft()
            lbl = compute_label_continuous(close, old_fv.ref_close, old_fv.ref_atr)
            if lbl is not None:
                nn.train_step(old_fv, lbl)

        ai_proba = nn.predict(fv)
        recent_probas.append(ai_proba)

        # Recalibration périodique (voir commentaire plus haut)
        if i > 0 and i % recalib_bars == 0 and len(recent_probas) >= 500:
            arr = np.array(recent_probas)
            new_tl = max(float(np.percentile(arr, (1-frac)*100)), 0.51)
            new_ts = min(float(np.percentile(arr, frac*100)),     0.49)
            thresh_l, thresh_s = new_tl, new_ts
            n_recalibs += 1
            thresh_history.append((i, thresh_l, thresh_s))

        # Conditions
        conf       = float(row.get("conf_score", 0.0))
        is_chop    = bool(row.get("is_chop", False))
        in_zone_b  = bool(row.get("in_ob_bull",False)) or bool(row.get("in_fvg_bull",False))
        in_zone_s  = bool(row.get("in_ob_bear",False)) or bool(row.get("in_fvg_bear",False))

        ai_l  = (ai_proba > thresh_l) and in_zone_b
        ai_s  = (ai_proba < thresh_s) and in_zone_s
        cf_l  = (conf >= max_conf * conf_pct) and in_zone_b
        cf_s  = (conf <= -max_conf * conf_pct) and in_zone_s

        long_sig  = (ai_l and cf_l) and not (use_chop and is_chop) and (i-last_sig)>=cooldown
        short_sig = (ai_s and cf_s) and not (use_chop and is_chop) and (i-last_sig)>=cooldown and not long_only

        # Gestion position
        if open_trade:
            dir_, entry, size, sl, tp, entry_ai, entry_conf = open_trade
            closed=False; reason=""
            if dir_=="LONG":
                if close<=sl: closed,reason=True,"SL"
                elif close>=tp: closed,reason=True,"TP"
                elif short_sig: closed,reason=True,"REV"
            else:
                if close>=sl: closed,reason=True,"SL"
                elif close<=tp: closed,reason=True,"TP"
                elif long_sig: closed,reason=True,"REV"
            if closed:
                sign  = 1 if dir_=="LONG" else -1
                pnl   = size * sign * (close-entry)/entry
                cap  += pnl
                trades.append({"dir":dir_,"entry":entry,"exit":close,
                                "pnl":pnl,"reason":reason,"time":ts,
                                "ai":entry_ai,"conf":entry_conf})
                open_trade = None

        if not open_trade and (long_sig or short_sig):
            dir_ = "LONG" if long_sig else "SHORT"
            size = min(cap*risk_pct/(sl_atr*atr/close), cap*0.20) if sl_atr*atr>0 else cap*0.01
            sl   = close-sl_atr*atr if dir_=="LONG" else close+sl_atr*atr
            tp   = close+tp_atr*atr if dir_=="LONG" else close-tp_atr*atr
            open_trade = (dir_, close, size, sl, tp, ai_proba, conf)
            last_sig   = i

        unr = 0.0
        if open_trade:
            d,e,s,_,_,_,_ = open_trade
            sign = 1 if d=="LONG" else -1
            unr  = s*sign*(close-e)/e
        base_val = base_units * close
        equity.append({"time":ts,"equity":cap+unr+base_val,"close":close})

    # Fermer trade final
    if open_trade and len(df_test)>0:
        d,e,s,_,_,entry_ai,entry_conf = open_trade
        lc = df_test["close"].iloc[-1]
        sign = 1 if d=="LONG" else -1
        pnl  = s*sign*(lc-e)/e
        cap += pnl
        trades.append({"dir":d,"entry":e,"exit":lc,"pnl":pnl,
                        "reason":"END","time":df_test.index[-1],
                        "ai":entry_ai,"conf":entry_conf})

    # Métriques
    n = len(trades)
    test_days_empty = max((df_test.index[-1] - df_test.index[0]).days, 1) if len(df_test) else 1
    final_base_val = base_units * float(df_test["close"].iloc[-1]) if len(df_test) else 0.0
    if n == 0:
        return MarketResult(symbol=symbol, name=MARKETS_BINANCE.get(symbol,{}).get("name",symbol),
                            n_bars=len(df), n_trades=0, n_long=0, n_short=0,
                            win_rate=0, profit_factor=0,
                            total_return=(cap+final_base_val-capital)/capital*100,
                            max_drawdown=0, sharpe=0, expectancy=0,
                            avg_win=0, avg_loss=0, ratio_rr=0,
                            capital_final=cap+final_base_val, trades_per_day=0,
                            total_days=test_days_empty, exit_reasons={},
                            equity_curve=equity, trades=trades)

    wins   = [t for t in trades if t["pnl"]>0]
    losses = [t for t in trades if t["pnl"]<=0]
    gp_    = sum(t["pnl"] for t in wins)
    gl_    = abs(sum(t["pnl"] for t in losses))
    wr     = len(wins)/n
    avg_w  = np.mean([t["pnl"] for t in wins])   if wins   else 0
    avg_l  = np.mean([t["pnl"] for t in losses]) if losses else 0
    exp    = wr*avg_w + (1-wr)*avg_l
    pf     = gp_/gl_ if gl_>0 else np.inf
    rr     = abs(avg_w/avg_l) if avg_l!=0 else 0

    eq_vals = [e["equity"] for e in equity]
    peak=eq_vals[0]; max_dd=0.0
    for v in eq_vals:
        peak=max(peak,v); max_dd=max(max_dd,(peak-v)/peak)

    eq_df = pd.DataFrame(equity).set_index("time")
    dr    = eq_df["equity"].resample("D").last().dropna().pct_change().dropna()
    sharpe= float(dr.mean()/dr.std()*np.sqrt(252)) if dr.std()>0 else 0

    reasons = {}
    for t in trades: reasons[t["reason"]] = reasons.get(t["reason"],0)+1

    test_days = max((df_test.index[-1]-df_test.index[0]).days,1)

    if n_recalibs > 0:
        t0, tl0, ts0 = thresh_history[0]
        t1, tl1, ts1 = thresh_history[-1]
        print(f"\n  Recalibrations: {n_recalibs} | "
              f"seuils initiaux L>{tl0:.3f}/S<{ts0:.3f} → "
              f"seuils finaux L>{tl1:.3f}/S<{ts1:.3f}", end="")

    return MarketResult(
        symbol=symbol,
        name=MARKETS_BINANCE.get(symbol,{}).get("name",symbol),
        n_bars=len(df), n_trades=n, n_long=sum(1 for t in trades if t["dir"]=="LONG"),
        n_short=sum(1 for t in trades if t["dir"]=="SHORT"),
        win_rate=wr*100, profit_factor=pf,
        total_return=(cap+final_base_val-capital)/capital*100,
        max_drawdown=max_dd*100, sharpe=sharpe, expectancy=exp,
        avg_win=avg_w, avg_loss=avg_l, ratio_rr=rr,
        capital_final=cap+final_base_val, trades_per_day=n/test_days,
        total_days=test_days, exit_reasons=reasons,
        equity_curve=equity, trades=trades,
    )


# ─────────────────────────────────────────────
# Rapport multi-marchés
# ─────────────────────────────────────────────

def print_summary(results: list[MarketResult]):
    """Affiche le tableau comparatif des marchés."""
    valid = [r for r in results if r and r.n_trades > 0]
    if not valid:
        print("Aucun résultat valide.")
        return

    print("\n" + "═"*100)
    print("   BACKTEST MULTI-MARCHÉS — Confluence AI v5.0 — Données 1 minute")
    print("═"*100)
    header = f"{'Marché':<12} {'Barres':>7} {'Trades':>7} {'T/jour':>7} {'WR%':>7} {'PF':>7} {'R/R':>6} {'Return%':>9} {'MaxDD%':>8} {'Sharpe':>8} {'Esp/$':>8}"
    print(header)
    print("─"*100)

    # Trier par rendement décroissant
    valid_sorted = sorted(valid, key=lambda r: -r.total_return)

    for r in valid_sorted:
        ret_icon = "✅" if r.total_return > 0 else "❌"
        pf_str   = f"{r.profit_factor:.2f}" if r.profit_factor != np.inf else "∞"
        print(
            f"{r.symbol:<12} {r.n_bars:>7,} {r.n_trades:>7} "
            f"{r.trades_per_day:>7.1f} {r.win_rate:>7.1f} {pf_str:>7} "
            f"{r.ratio_rr:>6.2f} {r.total_return:>+9.2f} {r.max_drawdown:>8.2f} "
            f"{r.sharpe:>8.3f} {r.expectancy:>+8.2f}  {ret_icon}"
        )

    print("─"*100)
    # Meilleur marché
    best = valid_sorted[0]
    print(f"\n🏆 Meilleur rendement : {best.symbol} ({best.name}) → {best.total_return:+.2f}%")
    best_sharpe = max(valid, key=lambda r: r.sharpe)
    print(f"🏆 Meilleur Sharpe   : {best_sharpe.symbol} → {best_sharpe.sharpe:.3f}")
    best_pf = max(valid, key=lambda r: r.profit_factor if r.profit_factor != np.inf else 0)
    print(f"🏆 Meilleur PF       : {best_pf.symbol} → {best_pf.profit_factor:.2f}")
    print("═"*100)


def plot_multi_results(results: list[MarketResult], output: str = "multi_backtest.png"):
    """Graphique comparatif multi-marchés."""
    valid = [r for r in results if r and r.n_trades > 0]
    if not valid:
        return

    n = len(valid)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols + 2   # +2 pour le graphique comparatif en haut

    fig = plt.figure(figsize=(20, 5*rows), facecolor="#0d1117")
    gs  = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.45, wspace=0.30)

    BG="#161b22"; G="#00ff88"; R="#ff4466"; B="#4488ff"; Y="#ffcc00"; GR="#888888"

    def sa(ax, t=""):
        ax.set_facecolor(BG); ax.tick_params(colors=GR, labelsize=8)
        ax.spines[:].set_color("#333")
        if t: ax.set_title(t, color="white", fontsize=9, pad=4)

    # ── Graphique comparatif des rendements ──
    ax_ret = fig.add_subplot(gs[0, :])
    sa(ax_ret, "Rendements comparés — Backtest 1 minute")
    symbols = [r.symbol for r in valid]
    returns = [r.total_return for r in valid]
    cols_bar = [G if r>=0 else R for r in returns]
    bars = ax_ret.bar(symbols, returns, color=cols_bar, alpha=0.85, width=0.6)
    ax_ret.axhline(0, color=GR, lw=0.8)
    for bar, val in zip(bars, returns):
        ax_ret.text(bar.get_x()+bar.get_width()/2, val+(0.05 if val>=0 else -0.15),
                    f"{val:+.2f}%", ha="center", va="bottom" if val>=0 else "top",
                    color="white", fontsize=9, fontweight="bold")
    ax_ret.set_ylabel("Rendement %", color=GR, fontsize=9)

    # ── Tableau métriques ──
    ax_tbl = fig.add_subplot(gs[1, :])
    sa(ax_tbl, "")
    ax_tbl.axis("off")
    col_labels = ["Marché","Trades","T/jour","Win%","PF","R/R","MaxDD%","Sharpe","Espérance"]
    tbl_data = []
    for r in valid:
        pf_str = f"{r.profit_factor:.2f}" if r.profit_factor!=np.inf else "∞"
        tbl_data.append([
            r.symbol, str(r.n_trades), f"{r.trades_per_day:.1f}",
            f"{r.win_rate:.1f}%", pf_str, f"{r.ratio_rr:.2f}",
            f"{r.max_drawdown:.1f}%", f"{r.sharpe:.3f}", f"${r.expectancy:+.2f}"
        ])
    tbl = ax_tbl.table(cellText=tbl_data, colLabels=col_labels,
                       loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    for (row_idx, col_idx), cell in tbl.get_celld().items():
        cell.set_facecolor("#0d1117" if row_idx==0 else "#1a1f2e")
        cell.set_edgecolor("#333")
        cell.set_text_props(color="white" if row_idx==0 else GR)
        if row_idx > 0:
            try:
                val_str = tbl_data[row_idx-1][col_idx]
                if col_idx==7 and float(val_str.replace("$","").replace("+",""))>0:
                    cell.set_facecolor("#0d2a1a")
                elif col_idx==7:
                    cell.set_facecolor("#2a0d1a")
            except: pass

    # ── Courbe d'équité par marché ──
    for idx, r in enumerate(valid):
        row_idx = 2 + idx // cols
        col_idx = idx % cols
        ax = fig.add_subplot(gs[row_idx, col_idx])
        ret_icon = "✅" if r.total_return>0 else "❌"
        sa(ax, f"{r.symbol} | {r.total_return:+.2f}% | WR:{r.win_rate:.0f}% | {r.n_trades}T {ret_icon}")

        if r.equity_curve:
            eq_df = pd.DataFrame(r.equity_curve)
            ax.plot(range(len(eq_df)), eq_df["equity"], color=B, lw=1.2)
            ax.axhline(10000, color=GR, ls="--", lw=0.7, alpha=0.5)

            # P&L par trade
            for t in r.trades:
                c = G if t["pnl"]>0 else R
                # Pas de axvspan car on n'a pas les indices exactement — on mark simplement
            ax.set_ylabel("Équité $", color=GR, fontsize=7)

            # Sous-graphique P&L
            ax2 = ax.twinx()
            ax2.set_facecolor(BG)
            pnls = [t["pnl"] for t in r.trades]
            cum  = np.cumsum(pnls)
            if len(cum) > 0:
                x_trade = np.linspace(0, len(eq_df)-1, len(cum))
                ax2.plot(x_trade, cum, color=Y, lw=1.0, alpha=0.7)
            ax2.set_ylabel("P&L cum.", color=Y, fontsize=6)
            ax2.tick_params(colors=Y, labelsize=6)
            ax2.spines[:].set_color("#333")

    fig.suptitle(
        f"Backtest Multi-Marchés 1min — Confluence AI v5.0 — "
        f"{len(valid)} marchés testés",
        color="white", fontsize=13, fontweight="bold", y=0.995
    )
    plt.savefig(output, dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"\nGraphique sauvegardé: {output}")


def export_csv(results: list[MarketResult], output: str = "multi_backtest_trades.csv"):
    """Export tous les trades de tous les marchés en CSV."""
    all_trades = []
    for r in results:
        if not r: continue
        for t in r.trades:
            all_trades.append({
                "symbol":    r.symbol,
                "direction": t["dir"],
                "entry":     t["entry"],
                "exit":      t["exit"],
                "pnl_usd":   t["pnl"],
                "reason":    t["reason"],
                "ai_proba":  t.get("ai", 0),
                "conf":      t.get("conf", 0),
                "time":      t["time"],
            })
    if all_trades:
        df = pd.DataFrame(all_trades)
        df.to_csv(output, index=False)
        print(f"Trades exportés: {output} ({len(df)} lignes)")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backtest Multi-Marchés 1min — Confluence AI")
    parser.add_argument("--markets", nargs="+", default=DEFAULT_MARKETS,
                        help="Marchés à tester ex: BTCUSDT ETHUSDT SOLUSDT")
    parser.add_argument("--bars",    type=int,   default=5000,
                        help="Barres 1min à télécharger par marché")
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--risk",    type=float, default=0.01,   help="Risque/trade")
    parser.add_argument("--sl",      type=float, default=2.0,    help="SL en ATR")
    parser.add_argument("--tp",      type=float, default=4.0,    help="TP en ATR")
    parser.add_argument("--cooldown",type=int,   default=3,      help="Barres entre signaux")
    parser.add_argument("--conf",    type=float, default=0.25,   help="Seuil confluence")
    parser.add_argument("--tpd",     type=float, default=5.0,    help="Trades/jour cible")
    parser.add_argument("--no-cache", action="store_true",       help="Forcer le téléchargement")
    parser.add_argument("--from-csv", action="store_true",       help="Utiliser les CSV dans data_1min/")
    parser.add_argument("--from",     dest="date_from", default=None, help="Date début YYYY-MM-DD")
    parser.add_argument("--to",       dest="date_to",   default=None, help="Date fin YYYY-MM-DD (défaut: aujourd'hui)")
    parser.add_argument("--train-months", type=float, default=None,
                        help="Mois d'historique dédiés à l'entraînement AI (ex: 3). "
                             "Si fourni, découpe train/test par date calendaire au lieu "
                             "de la fraction 70/30 par défaut.")
    parser.add_argument("--test-months", type=float, default=None,
                        help="Mois d'historique dédiés au test out-of-sample après "
                             "--train-months (ex: 3). Sans cette option, le test "
                             "utilise tout ce qui reste après la période d'entraînement.")
    parser.add_argument("--interval", default="1m", choices=["1m", "1h", "4h"],
                        help="Intervalle des bougies à charger depuis data_1min/ (--from-csv)")
    parser.add_argument("--recalib-bars", type=int, default=1440,
                        help="Recalibrer les seuils AI tous les N barres pendant le test "
                             "(défaut 1440 = 1x/jour en 1min; réduire pour H1/H4)")
    parser.add_argument("--base-alloc", type=float, default=0.0,
                        help="Fraction du capital gardée en position longue permanente "
                             "(coeur long + overlay tactique). 0 = bot pur in/out (défaut).")
    parser.add_argument("--long-only", action="store_true",
                        help="N'ouvrir que des positions LONG (pas de SHORT)")
    parser.add_argument("--output",   default="multi_backtest.png")
    args = parser.parse_args()

    print("="*70)
    print("  BACKTEST MULTI-MARCHÉS 1 MINUTE — Confluence AI v5.0")
    print("="*70)
    print(f"  Marchés    : {', '.join(args.markets)}")
    print(f"  Barres     : {args.bars} par marché")
    print(f"  Capital    : ${args.capital:,.0f}")
    print(f"  SL/TP      : {args.sl}× / {args.tp}× ATR  (R/R ~{args.tp/args.sl:.1f})")
    print(f"  Cooldown   : {args.cooldown} barres (={args.cooldown} min)")
    print(f"  Conf. min  : {args.conf*100:.0f}%")
    print()

    results = []

    for symbol in args.markets:
        sym_up = symbol.upper()
        if not sym_up.endswith("USDT"):
            sym_up += "USDT"

        print(f"\n{'─'*50}")
        print(f"Marché: {sym_up}")

        # Charger les données
        if args.from_csv:
            # Chercher un CSV local dans data_1min/
            import glob
            from datetime import timezone
            interval_label = {"1m": "1min", "1h": "1h", "4h": "4h"}[args.interval]
            csv_files = sorted(glob.glob(f"data_1min/{sym_up}_{interval_label}*.csv"))
            if not csv_files:
                print(f"  ⚠️  Aucun CSV trouvé pour {sym_up} ({args.interval}) dans data_1min/")
                print(f"     Lancer d'abord: python download_history.py --symbol {sym_up} "
                      f"--interval {args.interval} --days 30")
                continue
            # Charger le fusionné si présent, sinon le fichier couvrant la plus
            # large période (le tri alphabétique par nom de fichier ne reflète
            # PAS la taille de l'historique — un chunk récent de 30 jours peut
            # trier après un fichier d'1 an commençant plus tôt).
            merged = [f for f in csv_files if "MERGED" in f]
            if merged:
                csv_path = merged[0]
            else:
                csv_path = max(csv_files, key=lambda p: os.path.getsize(p))
            print(f"  Chargement CSV: {os.path.basename(csv_path)}")
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, utc=True)
            df = df.astype(float).sort_index().dropna()
            # Filtrer sur la période si spécifiée
            if args.date_from:
                from datetime import timezone as tz
                start = pd.Timestamp(args.date_from, tz="UTC")
                df = df[df.index >= start]
            if args.date_to:
                end = pd.Timestamp(args.date_to, tz="UTC")
                df = df[df.index <= end]
            print(f"  {len(df):,} barres | {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")
        elif args.no_cache:
            df = fetch_1min_binance(sym_up, args.bars)
        else:
            df = load_or_download(sym_up, args.bars)

        if df.empty or len(df) < 300:
            print(f"  ⚠️  Pas assez de données pour {sym_up}, skip")
            continue

        # Découpage train/test par date calendaire si demandé
        train_end = test_end = None
        if args.train_months is not None:
            train_end = df.index[0] + pd.Timedelta(days=args.train_months * 30)
            if args.test_months is not None:
                test_end = train_end + pd.Timedelta(days=args.test_months * 30)
            te_str = test_end.strftime('%Y-%m-%d') if test_end is not None else df.index[-1].strftime('%Y-%m-%d')
            print(f"  Entraînement: {df.index[0].strftime('%Y-%m-%d')} → {train_end.strftime('%Y-%m-%d')} "
                  f"| Test: {train_end.strftime('%Y-%m-%d')} → {te_str}")

        # Backtest
        print(f"  Calcul indicateurs...", end="", flush=True)
        result = run_single_market(
            symbol    = sym_up,
            df_raw    = df,
            capital   = args.capital,
            risk_pct  = args.risk,
            sl_atr    = args.sl,
            tp_atr    = args.tp,
            cooldown  = args.cooldown,
            conf_pct  = args.conf,
            target_tpd= args.tpd,
            train_end = train_end,
            test_end  = test_end,
            recalib_bars = args.recalib_bars,
            base_alloc   = args.base_alloc,
            long_only    = args.long_only,
        )
        print(" OK")

        if result:
            results.append(result)
            ret_icon = "✅" if result.total_return > 0 else "❌"
            print(
                f"  {ret_icon} Trades={result.n_trades} | "
                f"WR={result.win_rate:.1f}% | "
                f"PF={result.profit_factor:.2f} | "
                f"Return={result.total_return:+.2f}% | "
                f"Sharpe={result.sharpe:.3f}"
            )

    if not results:
        print("\nAucun résultat. Vérifier la connexion internet.")
        return

    # Rapport
    print_summary(results)
    plot_multi_results(results, args.output)
    export_csv(results, args.output.replace(".png", "_trades.csv"))

    # Résumé CSV des métriques
    metrics_df = pd.DataFrame([{
        "symbol":          r.symbol,
        "name":            r.name,
        "n_trades":        r.n_trades,
        "trades_per_day":  r.trades_per_day,
        "win_rate_pct":    r.win_rate,
        "profit_factor":   r.profit_factor,
        "ratio_rr":        r.ratio_rr,
        "total_return_pct":r.total_return,
        "max_drawdown_pct":r.max_drawdown,
        "sharpe":          r.sharpe,
        "expectancy_usd":  r.expectancy,
        "capital_final":   r.capital_final,
        "total_days":      r.total_days,
    } for r in results if r])

    metrics_csv = args.output.replace(".png", "_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Métriques exportées: {metrics_csv}")


if __name__ == "__main__":
    main()
