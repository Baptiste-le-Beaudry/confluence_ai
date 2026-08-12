"""
seq_multiasset_backtest.py

Variante expérimentale par rapport à multi_market_backtest.py:
  - Fenêtre de contexte: le réseau voit les N dernières bougies aplaties
    (SeqFeatVec) au lieu d'une bougie isolée, pour pouvoir apprendre des
    séquences/patterns plutôt qu'un instantané.
  - Réseau PARTAGÉ entraîné simultanément sur plusieurs actifs (BTC, ETH,
    SOL par défaut) — pré-entraînement joint sur les setups combinés des
    trois historiques, puis apprentissage online entrelacé chronologiquement
    pendant le test (le même cerveau apprend de BTC, ETH et SOL en même
    temps, dans l'ordre réel du calendrier).

Sert à répondre à deux questions:
  1. L'edge H1 mesuré sur BTC seul se reproduit-il sur d'autres actifs?
  2. Voir plusieurs bougies + plus de données aide-t-il le réseau à
     apprendre des patterns plus robustes (mesuré via PF/Sharpe/drawdown)?

Usage:
    python seq_multiasset_backtest.py
    python seq_multiasset_backtest.py --seq-len 8 --hidden 16
    python seq_multiasset_backtest.py --symbols BTCUSDT ETHUSDT
"""

import argparse
import glob
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque

sys.path.insert(0, os.path.dirname(__file__))
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from indicators import compute_all
from ai_model import (NeuralNet, N_FEAT, build_features, build_seq_featvec,
                       generate_seq_samples, compute_label_continuous)

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DATA_DIR = "data_1min"


def load_symbol(symbol: str, data_dir: str = DATA_DIR) -> pd.DataFrame:
    files = glob.glob(os.path.join(data_dir, f"{symbol}_1h_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"Pas de CSV H1 pour {symbol} dans {data_dir}/ — "
            f"lancer: python download_history.py --symbol {symbol} --interval 1h --from 2017-08-17")
    path = max(files, key=os.path.getsize)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df.astype(float).sort_index()


def norm_w_from(df_train: pd.DataFrame) -> dict:
    return {
        "vwap_std": max(df_train["dist_vwap"].abs().std(), 0.05),
        "ema_std":  max((df_train["ema_trend"] * 100).abs().std(), 0.05),
        "gann_std": max(df_train["gann_pos"].abs().std(), 0.10),
        "mom_std":  max(df_train["mom_raw"].abs().std(), 0.10),
    }


def calibrate_thresholds_seq(nn, df_train, norm_w, seq_len, target_tpd, bars_per_day):
    probas = []
    window = deque(maxlen=seq_len)
    for _, row in df_train.iloc[220:].iterrows():
        rd = row.to_dict(); rd["close"] = row["close"]; rd["atr"] = row["atr"]
        if pd.isna(rd.get("atr", np.nan)) or rd.get("atr", 0) <= 0:
            continue
        window.append(build_features(rd, norm_w))
        if len(window) < seq_len:
            continue
        fv = build_seq_featvec(window, seq_len)
        probas.append(nn.predict(fv))

    if len(probas) < 30:
        return 0.62, 0.38
    probas = np.array(probas)
    frac = np.clip(target_tpd / bars_per_day / 2.0, 0.001, 0.15)
    tl = max(float(np.percentile(probas, (1 - frac) * 100)), 0.51)
    ts = min(float(np.percentile(probas, frac * 100)), 0.49)
    return tl, ts


def compute_metrics(trades: list, equity: list, capital: float, test_days: int) -> dict:
    n = len(trades)
    if n == 0:
        return {"n_trades": 0, "win_rate": 0, "profit_factor": 0, "total_return": 0,
                "max_drawdown": 0, "sharpe": 0, "capital_final": capital,
                "trades_per_day": 0, "expectancy": 0, "ratio_rr": 0}

    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gp = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))
    wr = len(wins) / n
    avg_w = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_l = np.mean([t["pnl"] for t in losses]) if losses else 0
    pf = gp / gl if gl > 0 else np.inf
    rr = abs(avg_w / avg_l) if avg_l != 0 else 0
    cap_final = capital + sum(t["pnl"] for t in trades)

    eq_vals = [e["equity"] for e in equity]
    peak = eq_vals[0] if eq_vals else capital
    max_dd = 0.0
    for v in eq_vals:
        peak = max(peak, v)
        max_dd = max(max_dd, (peak - v) / peak if peak > 0 else 0)

    if equity:
        eq_df = pd.DataFrame(equity).set_index("time")
        dr = eq_df["equity"].resample("D").last().dropna().pct_change().dropna()
        sharpe = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0
    else:
        sharpe = 0

    return {
        "n_trades": n, "win_rate": wr * 100, "profit_factor": pf,
        "total_return": (cap_final - capital) / capital * 100,
        "max_drawdown": max_dd * 100, "sharpe": sharpe, "capital_final": cap_final,
        "trades_per_day": n / max(test_days, 1), "expectancy": wr * avg_w + (1 - wr) * avg_l,
        "ratio_rr": rr,
    }


def main():
    parser = argparse.ArgumentParser(description="Backtest multi-actifs + fenêtre de contexte — Confluence AI")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--seq-len", type=int, default=5, help="Bougies dans la fenêtre de contexte")
    parser.add_argument("--hidden", type=int, default=12, help="Neurones cachés")
    parser.add_argument("--capital", type=float, default=10_000.0, help="Capital initial PAR actif")
    parser.add_argument("--risk", type=float, default=0.01)
    parser.add_argument("--sl", type=float, default=2.0)
    parser.add_argument("--tp", type=float, default=4.0)
    parser.add_argument("--cooldown", type=int, default=3)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--tpd", type=float, default=1.0)
    parser.add_argument("--recalib-bars", type=int, default=168)
    parser.add_argument("--pretrain-epochs", type=int, default=8)
    parser.add_argument("--train-end", default="2022-01-23")
    parser.add_argument("--test-end", default=None)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--no-chop", action="store_true")
    parser.add_argument("--output", default="backtest_seq_multiasset.png")
    args = parser.parse_args()

    long_only = not args.allow_short
    use_chop  = not args.no_chop
    seq_len   = args.seq_len
    train_end = pd.Timestamp(args.train_end, tz="UTC")
    test_end  = pd.Timestamp(args.test_end, tz="UTC") if args.test_end else None

    print("=" * 70)
    print("  BACKTEST MULTI-ACTIFS + FENÊTRE DE CONTEXTE — Confluence AI")
    print("=" * 70)
    print(f"  Actifs      : {', '.join(args.symbols)}")
    print(f"  Fenêtre     : {seq_len} bougies ({seq_len * N_FEAT} features en entrée, {args.hidden} neurones cachés)")
    print(f"  Cerveau     : PARTAGÉ entre tous les actifs")
    print(f"  Directions  : {'LONG uniquement' if long_only else 'LONG + SHORT'}")
    print()

    data = {}
    for symbol in args.symbols:
        try:
            df_raw = load_symbol(symbol)
        except FileNotFoundError as e:
            print(f"  ⚠️  {e}")
            continue
        df = compute_all(df_raw)
        df_train = df[df.index < train_end]
        df_test  = df[df.index >= train_end]
        if test_end is not None:
            df_test = df_test[df_test.index < test_end]
        if len(df_train) < 300 or len(df_test) < 50:
            print(f"  ⚠️  {symbol}: données insuffisantes (train={len(df_train)}, test={len(df_test)}), ignoré")
            continue
        total_secs = (df.index[-1] - df.index[0]).total_seconds()
        bars_per_day = max(86400.0 / max(total_secs / len(df), 1), 1.0)
        data[symbol] = {
            "df_train": df_train, "df_test": df_test,
            "norm_w": norm_w_from(df_train), "bars_per_day": bars_per_day,
            "max_conf": float(df["max_conf"].iloc[-1]),
        }
        print(f"  {symbol}: train {df_train.index[0].date()}→{df_train.index[-1].date()} "
              f"({len(df_train):,} barres) | test {df_test.index[0].date()}→{df_test.index[-1].date()} "
              f"({len(df_test):,} barres)")

    if not data:
        print("Aucun actif utilisable.")
        return

    # ── Pré-entraînement joint sur les setups combinés des N actifs ──
    nn = NeuralNet(lr=0.02, grad_clip=0.5, n_hidden=args.hidden, n_feat=seq_len * N_FEAT)
    all_samples = []
    print()
    for symbol, d in data.items():
        s = generate_seq_samples(d["df_train"], d["norm_w"], seq_len,
                                  fwd_bars=max(args.cooldown, 3), sl_atr=args.sl, tp_atr=args.tp)
        print(f"  {symbol}: {len(s):,} setups d'entraînement")
        all_samples.extend(s)

    print(f"\n  Entraînement JOINT: {len(all_samples):,} setups combinés, {args.pretrain_epochs} epochs...")
    nn.batch_train(all_samples, n_epochs=args.pretrain_epochs)
    print(f"  Loss finale: {np.mean(nn.train_losses[-200:]):.4f}\n")

    # ── Calibration des seuils (par actif, sur le réseau partagé déjà entraîné) ──
    for symbol, d in data.items():
        tl, ts = calibrate_thresholds_seq(nn, d["df_train"], d["norm_w"], seq_len,
                                           args.tpd, d["bars_per_day"])
        d["thresh_l"], d["thresh_s"] = tl, ts
        print(f"  {symbol}: seuils LONG>{tl:.3f} SHORT<{ts:.3f}")

    # ── Simulation chronologique entrelacée (le même réseau apprend des 3
    #    actifs dans l'ordre réel du calendrier, pas actif par actif) ──
    combined = []
    for symbol, d in data.items():
        tdf = d["df_test"].copy()
        tdf["__symbol__"] = symbol
        combined.append(tdf)
    stream = pd.concat(combined).sort_index()

    state = {symbol: {
        "window": deque(maxlen=seq_len), "pending": deque(),
        "open_trade": None, "cap": args.capital, "last_sig": -args.cooldown, "i": 0,
        "recent_probas": deque(maxlen=max(args.recalib_bars * 4, 2000)),
        "thresh_l": d["thresh_l"], "thresh_s": d["thresh_s"],
        "trades": [], "equity": [], "n_recalibs": 0,
    } for symbol, d in data.items()}

    print(f"\n  Simulation entrelacée sur {len(stream):,} barres combinées...")

    for ts, row in stream.iterrows():
        symbol = row["__symbol__"]
        d, st = data[symbol], state[symbol]

        close, atr = row["close"], row["atr"]
        if pd.isna(atr) or atr <= 0 or pd.isna(close):
            continue

        rd = row.to_dict(); rd["close"] = close; rd["atr"] = atr
        st["window"].append(build_features(rd, d["norm_w"]))
        st["i"] += 1
        if len(st["window"]) < seq_len:
            continue
        fv = build_seq_featvec(st["window"], seq_len)

        # Online learning — chaque actif contribue au même cerveau
        st["pending"].append((fv, st["i"]))
        if len(st["pending"]) > 5:
            old_fv, _ = st["pending"].popleft()
            lbl = compute_label_continuous(close, old_fv.ref_close, old_fv.ref_atr)
            if lbl is not None:
                nn.train_step(old_fv, lbl)

        ai_proba = nn.predict(fv)
        st["recent_probas"].append(ai_proba)

        if st["i"] % args.recalib_bars == 0 and len(st["recent_probas"]) >= 500:
            arr = np.array(st["recent_probas"])
            frac = np.clip(args.tpd / d["bars_per_day"] / 2.0, 0.001, 0.15)
            st["thresh_l"] = max(float(np.percentile(arr, (1 - frac) * 100)), 0.51)
            st["thresh_s"] = min(float(np.percentile(arr, frac * 100)), 0.49)
            st["n_recalibs"] += 1

        conf = float(row.get("conf_score", 0.0))
        is_chop = bool(row.get("is_chop", False))
        in_zone_b = bool(row.get("in_ob_bull", False)) or bool(row.get("in_fvg_bull", False))
        in_zone_s = bool(row.get("in_ob_bear", False)) or bool(row.get("in_fvg_bear", False))

        ai_l = (ai_proba > st["thresh_l"]) and in_zone_b
        ai_s = (ai_proba < st["thresh_s"]) and in_zone_s
        cf_l = (conf >= d["max_conf"] * args.conf) and in_zone_b
        cf_s = (conf <= -d["max_conf"] * args.conf) and in_zone_s

        long_sig  = (ai_l and cf_l) and not (use_chop and is_chop) and (st["i"] - st["last_sig"]) >= args.cooldown
        short_sig = ((ai_s and cf_s) and not (use_chop and is_chop)
                     and (st["i"] - st["last_sig"]) >= args.cooldown and not long_only)

        if st["open_trade"]:
            dir_, entry, size, sl, tp = st["open_trade"]
            closed, reason = False, ""
            if dir_ == "LONG":
                if close <= sl: closed, reason = True, "SL"
                elif close >= tp: closed, reason = True, "TP"
                elif short_sig: closed, reason = True, "REV"
            else:
                if close >= sl: closed, reason = True, "SL"
                elif close <= tp: closed, reason = True, "TP"
                elif long_sig: closed, reason = True, "REV"
            if closed:
                sign = 1 if dir_ == "LONG" else -1
                pnl = size * sign * (close - entry) / entry
                st["cap"] += pnl
                st["trades"].append({"symbol": symbol, "dir": dir_, "entry": entry, "exit": close,
                                      "pnl": pnl, "reason": reason, "time": ts})
                st["open_trade"] = None

        if not st["open_trade"] and (long_sig or short_sig):
            dir_ = "LONG" if long_sig else "SHORT"
            denom = max(args.sl * atr / close, 1e-9)
            size = min(st["cap"] * args.risk / denom, st["cap"] * 0.20)
            sl = close - args.sl * atr if dir_ == "LONG" else close + args.sl * atr
            tp = close + args.tp * atr if dir_ == "LONG" else close - args.tp * atr
            st["open_trade"] = (dir_, close, size, sl, tp)
            st["last_sig"] = st["i"]

        unr = 0.0
        if st["open_trade"]:
            d_, e_, s_, _, _ = st["open_trade"]
            sign = 1 if d_ == "LONG" else -1
            unr = s_ * sign * (close - e_) / e_
        st["equity"].append({"time": ts, "equity": st["cap"] + unr})

    # Fermer les trades encore ouverts en fin de test
    for symbol, d in data.items():
        st = state[symbol]
        if st["open_trade"] and len(d["df_test"]) > 0:
            dir_, entry, size, sl, tp = st["open_trade"]
            lc = float(d["df_test"]["close"].iloc[-1])
            sign = 1 if dir_ == "LONG" else -1
            pnl = size * sign * (lc - entry) / entry
            st["cap"] += pnl
            st["trades"].append({"symbol": symbol, "dir": dir_, "entry": entry, "exit": lc,
                                  "pnl": pnl, "reason": "END", "time": d["df_test"].index[-1]})

    # ── Rapport ──
    print("\n" + "═" * 100)
    print("  RÉSULTATS — Réseau partagé, fenêtre de contexte")
    print("═" * 100)
    header = f"{'Actif':<10} {'Trades':>7} {'T/jour':>7} {'WR%':>7} {'PF':>7} {'R/R':>6} {'Return%':>9} {'MaxDD%':>8} {'Sharpe':>8} {'Recalibs':>9}"
    print(header)
    print("─" * 100)

    all_metrics = {}
    all_trades_export = []
    total_days_ref = max((next(iter(data.values()))["df_test"].index[-1]
                           - next(iter(data.values()))["df_test"].index[0]).days, 1)

    for symbol, d in data.items():
        st = state[symbol]
        test_days = max((d["df_test"].index[-1] - d["df_test"].index[0]).days, 1)
        m = compute_metrics(st["trades"], st["equity"], args.capital, test_days)
        all_metrics[symbol] = m
        pf_str = f"{m['profit_factor']:.2f}" if m["profit_factor"] != np.inf else "∞"
        icon = "✅" if m["total_return"] > 0 else "❌"
        print(f"{symbol:<10} {m['n_trades']:>7} {m['trades_per_day']:>7.2f} {m['win_rate']:>7.1f} "
              f"{pf_str:>7} {m['ratio_rr']:>6.2f} {m['total_return']:>+9.2f} {m['max_drawdown']:>8.2f} "
              f"{m['sharpe']:>8.3f} {st['n_recalibs']:>9}  {icon}")
        for t in st["trades"]:
            all_trades_export.append(t)

    print("─" * 100)
    combined_final = sum(m["capital_final"] for m in all_metrics.values())
    combined_initial = args.capital * len(all_metrics)
    combined_return = (combined_final - combined_initial) / combined_initial * 100
    print(f"  Portefeuille combiné ({len(all_metrics)} actifs, ${combined_initial:,.0f} initial): "
          f"${combined_final:,.0f}  ({combined_return:+.2f}%)")
    print("═" * 100)

    if all_trades_export:
        pd.DataFrame(all_trades_export).to_csv(args.output.replace(".png", "_trades.csv"), index=False)
    pd.DataFrame(all_metrics).T.to_csv(args.output.replace(".png", "_metrics.csv"))

    # ── Graphique: équité par actif + portefeuille combiné ──
    BG = "#161b22"; G = "#00ff88"; R = "#ff4466"; GR = "#888888"
    fig = plt.figure(figsize=(16, 5 * (len(data) + 1)), facecolor="#0d1117")
    gs = gridspec.GridSpec(len(data) + 1, 1, figure=fig, hspace=0.4)

    def style_ax(ax, title=""):
        ax.set_facecolor(BG); ax.tick_params(colors=GR, labelsize=8)
        for s in ax.spines.values(): s.set_color("#333")
        if title: ax.set_title(title, color="white", fontsize=10)

    combined_eq = None
    for idx, (symbol, d) in enumerate(data.items()):
        st = state[symbol]
        ax = fig.add_subplot(gs[idx])
        m = all_metrics[symbol]
        style_ax(ax, f"{symbol} | Return={m['total_return']:+.2f}% | WR={m['win_rate']:.1f}% | "
                     f"PF={m['profit_factor']:.2f} | Sharpe={m['sharpe']:.2f} | {m['n_trades']} trades")
        if st["equity"]:
            eq = pd.DataFrame(st["equity"]).set_index("time")
            ax.plot(eq.index, eq["equity"], color=G if m["total_return"] > 0 else R, lw=1.3)
            ax.axhline(args.capital, color=GR, ls="--", lw=0.7, alpha=0.5)
            eq_r = eq.rename(columns={"equity": symbol})
            combined_eq = eq_r if combined_eq is None else combined_eq.join(eq_r, how="outer")

    ax = fig.add_subplot(gs[len(data)])
    style_ax(ax, f"Portefeuille combiné | Return={combined_return:+.2f}%")
    if combined_eq is not None:
        combined_eq = combined_eq.ffill().bfill()
        total = combined_eq.sum(axis=1)
        ax.plot(total.index, total, color="#4488ff", lw=1.6)
        ax.axhline(combined_initial, color=GR, ls="--", lw=0.7, alpha=0.5)

    fig.suptitle(f"Fenêtre de contexte ({seq_len} bougies) + réseau partagé {', '.join(data.keys())}",
                 color="white", fontsize=12, fontweight="bold")
    plt.savefig(args.output, dpi=140, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"\nGraphique sauvegardé: {args.output}")
    print(f"Trades exportés: {args.output.replace('.png', '_trades.csv')}")
    print(f"Métriques exportées: {args.output.replace('.png', '_metrics.csv')}")


if __name__ == "__main__":
    main()
