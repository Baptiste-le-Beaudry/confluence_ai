"""
main.py
Point d'entrée du backtest.

Usage:
    python main.py                          # données synthétiques
    python main.py --source binance         # données Binance temps réel
    python main.py --source binance --tf 1h --bars 2000
    python main.py --source csv --file mon_fichier.csv
    python main.py --no-ai                  # backtest sans AI (score de confluence)
    python main.py --optimize               # optimisation des hyperparamètres
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data_loader import load_from_binance, load_from_csv, generate_sample_data
from backtest import Backtester, BacktestParams


# ─────────────────────────────────────────────
# Génération du rapport graphique
# ─────────────────────────────────────────────

def plot_results(backtester: Backtester, metrics: dict, output_path: str = "backtest_report.png"):
    """Génère un rapport visuel complet."""
    eq = pd.DataFrame(backtester.equity_curve).set_index("time")
    trades = backtester.trades

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Couleurs
    GREEN  = "#00ff88"
    RED    = "#ff4466"
    BLUE   = "#4488ff"
    YELLOW = "#ffcc00"
    GRAY   = "#888888"
    BG     = "#161b22"

    def style_ax(ax, title=""):
        ax.set_facecolor(BG)
        ax.tick_params(colors=GRAY, labelsize=8)
        ax.spines[:].set_color("#333")
        if title:
            ax.set_title(title, color="white", fontsize=10, pad=6)

    # ── 1. Courbe d'équité ──
    ax1 = fig.add_subplot(gs[0, :2])
    style_ax(ax1, "Courbe d'équité")
    ax1.plot(eq.index, eq["equity"], color=BLUE, linewidth=1.5, label="Équité")
    ax1.axhline(backtester.p.capital, color=GRAY, linestyle="--", linewidth=0.8, alpha=0.6)

    for t in trades:
        color = GREEN if t.pnl_usd > 0 else RED
        if t.exit_time and t.entry_time:
            ax1.axvspan(t.entry_time, t.exit_time, alpha=0.08, color=color)

    ax1.set_ylabel("USD", color=GRAY, fontsize=8)
    ax1.yaxis.label.set_color(GRAY)
    ax1.legend(fontsize=8, facecolor=BG, labelcolor="white")

    # ── 2. Prix + signaux ──
    ax2 = fig.add_subplot(gs[1, :2])
    style_ax(ax2, "Prix BTC + Signaux")
    ax2.plot(eq.index, eq["close"], color="#aaaaaa", linewidth=0.8, alpha=0.8)

    long_trades  = [t for t in trades if t.direction == "LONG"]
    short_trades = [t for t in trades if t.direction == "SHORT"]

    if long_trades:
        ax2.scatter([t.entry_time for t in long_trades],
                    [t.entry_price for t in long_trades],
                    color=GREEN, marker="^", s=60, zorder=5, label="Long")
    if short_trades:
        ax2.scatter([t.entry_time for t in short_trades],
                    [t.entry_price for t in short_trades],
                    color=RED, marker="v", s=60, zorder=5, label="Short")
    ax2.legend(fontsize=8, facecolor=BG, labelcolor="white")
    ax2.set_ylabel("Prix USD", color=GRAY, fontsize=8)

    # ── 3. AI Proba ──
    ax3 = fig.add_subplot(gs[2, :2])
    style_ax(ax3, "Probabilité AI dans le temps")
    ax3.plot(eq.index, eq["ai_proba"], color=YELLOW, linewidth=0.8, alpha=0.9)
    ax3.axhline(0.65, color=GREEN, linestyle="--", linewidth=0.8, alpha=0.7, label="Seuil Long")
    ax3.axhline(0.35, color=RED,   linestyle="--", linewidth=0.8, alpha=0.7, label="Seuil Short")
    ax3.axhline(0.5,  color=GRAY,  linestyle=":",  linewidth=0.6, alpha=0.5)
    ax3.set_ylim(0, 1)
    ax3.legend(fontsize=7, facecolor=BG, labelcolor="white")
    ax3.set_ylabel("P(LONG)", color=GRAY, fontsize=8)

    # ── 4. Métriques texte ──
    ax4 = fig.add_subplot(gs[0, 2])
    ax4.set_facecolor(BG)
    ax4.axis("off")
    ax4.set_title("Métriques", color="white", fontsize=10, pad=6)

    lines = [
        ("Rendement",    f"{metrics['total_return_pct']:+.2f}%",
         GREEN if metrics["total_return_pct"] > 0 else RED),
        ("Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%", RED),
        ("Sharpe",       f"{metrics['sharpe_ratio']:.3f}",
         GREEN if metrics["sharpe_ratio"] > 1 else YELLOW),
        ("Win Rate",     f"{metrics['win_rate_pct']:.1f}%",
         GREEN if metrics["win_rate_pct"] > 50 else RED),
        ("Profit Factor",f"{metrics['profit_factor']:.3f}",
         GREEN if metrics["profit_factor"] > 1 else RED),
        ("Espérance",    f"${metrics['expectancy_usd']:+.2f}",
         GREEN if metrics["expectancy_usd"] > 0 else RED),
        ("# Trades",     str(metrics["n_trades"]), "white"),
        ("Long / Short", f"{metrics['n_long']} / {metrics['n_short']}", "white"),
        ("AI Samples",   str(metrics["n_nn_samples"]), BLUE),
    ]
    for j, (label, value, color) in enumerate(lines):
        ax4.text(0.05, 0.92 - j * 0.10, label + ":", color=GRAY,
                 fontsize=9, transform=ax4.transAxes)
        ax4.text(0.55, 0.92 - j * 0.10, value, color=color,
                 fontsize=9, fontweight="bold", transform=ax4.transAxes)

    # ── 5. Distribution des PnL ──
    ax5 = fig.add_subplot(gs[1, 2])
    style_ax(ax5, "Distribution P&L par trade")
    pnls = [t.pnl_usd for t in trades]
    if pnls:
        colors = [GREEN if p > 0 else RED for p in pnls]
        ax5.bar(range(len(pnls)), pnls, color=colors, alpha=0.8, width=0.8)
        ax5.axhline(0, color=GRAY, linewidth=0.8)
        ax5.set_xlabel("# Trade", color=GRAY, fontsize=8)
        ax5.set_ylabel("P&L USD", color=GRAY, fontsize=8)

    # ── 6. Importance des features AI ──
    ax6 = fig.add_subplot(gs[2, 2])
    style_ax(ax6, "Importance Features AI")
    weights = metrics.get("nn_weights", {})
    if weights:
        sorted_w = sorted(weights.items(), key=lambda x: x[1])
        names  = [x[0] for x in sorted_w]
        values = [x[1] for x in sorted_w]
        bars = ax6.barh(names, values, color=BLUE, alpha=0.8)
        for bar, val in zip(bars, values):
            ax6.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", color="white", fontsize=7)
        ax6.set_xlabel("Importance", color=GRAY, fontsize=8)

    fig.suptitle("Backtest — Confluence AI v4.0",
                 color="white", fontsize=14, fontweight="bold", y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nRapport graphique sauvegardé: {output_path}")


# ─────────────────────────────────────────────
# Optimisation des hyperparamètres
# ─────────────────────────────────────────────

def optimize(df: pd.DataFrame, n_trials=20):
    """
    Teste plusieurs combinaisons de paramètres et retourne les meilleurs.
    Optimise: lr, ai_thresh_long, ai_thresh_short, sl_atr_mult, tp_atr_mult
    """
    print("\n" + "═" * 60)
    print("  OPTIMISATION DES HYPERPARAMÈTRES")
    print("═" * 60)

    results = []
    rng = np.random.default_rng(42)

    for trial in range(n_trials):
        params = BacktestParams(
            lr              = float(rng.choice([0.005, 0.01, 0.02, 0.05])),
            ai_thresh_long  = float(rng.uniform(0.60, 0.75)),
            ai_thresh_short = float(rng.uniform(0.25, 0.40)),
            sl_atr_mult     = float(rng.uniform(1.5, 3.0)),
            tp_atr_mult     = float(rng.uniform(2.0, 6.0)),
            cooldown_bars   = int(rng.integers(5, 20)),
            capital         = 10_000.0,
        )
        bt = Backtester(params)
        m  = bt.run(df)
        if "error" not in m and m["n_trades"] >= 5:
            score = m["sharpe_ratio"] - m["max_drawdown_pct"] * 0.1
            results.append({
                "score":           score,
                "sharpe":          m["sharpe_ratio"],
                "return_pct":      m["total_return_pct"],
                "max_dd":          m["max_drawdown_pct"],
                "win_rate":        m["win_rate_pct"],
                "n_trades":        m["n_trades"],
                "lr":              params.lr,
                "thresh_long":     params.ai_thresh_long,
                "thresh_short":    params.ai_thresh_short,
                "sl_atr":          params.sl_atr_mult,
                "tp_atr":          params.tp_atr_mult,
                "cooldown":        params.cooldown_bars,
            })
            print(f"  Trial {trial+1:2d}: Sharpe={m['sharpe_ratio']:.3f} | "
                  f"Return={m['total_return_pct']:+.1f}% | "
                  f"DD={m['max_drawdown_pct']:.1f}% | "
                  f"Trades={m['n_trades']}")

    if not results:
        print("Aucun résultat valide.")
        return None

    best = sorted(results, key=lambda x: -x["score"])[0]
    print("\n  ─── MEILLEURS PARAMÈTRES ───")
    for k, v in best.items():
        print(f"    {k:<18}: {v:.4f}" if isinstance(v, float) else f"    {k:<18}: {v}")
    return best


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backtest Confluence AI v4.0")
    parser.add_argument("--source",   default="synthetic",
                        choices=["synthetic", "binance", "csv"],
                        help="Source des données")
    parser.add_argument("--symbol",   default="BTC/USDT",    help="Symbole (Binance)")
    parser.add_argument("--tf",       default="15m",
                        choices=["1m","5m","15m","30m","1h","4h","1d"],
                        help="Timeframe")
    parser.add_argument("--bars",     type=int, default=2000, help="Nombre de barres")
    parser.add_argument("--file",     default="data.csv",    help="Fichier CSV")
    parser.add_argument("--capital",  type=float, default=10000.0)
    parser.add_argument("--risk",     type=float, default=0.01,  help="% capital par trade")
    parser.add_argument("--sl",       type=float, default=2.0,   help="SL en ATR")
    parser.add_argument("--tp",       type=float, default=4.0,   help="TP en ATR")
    parser.add_argument("--lr",       type=float, default=0.01,  help="Learning rate AI")
    parser.add_argument("--no-ai",    action="store_true",        help="Désactiver l'AI")
    parser.add_argument("--no-htf",   action="store_true",        help="Désactiver le filtre HTF")
    parser.add_argument("--no-chop",  action="store_true",        help="Désactiver le filtre chop")
    parser.add_argument("--no-candle",action="store_true",        help="Désactiver confirmation bougie")
    parser.add_argument("--optimize", action="store_true",        help="Optimiser les paramètres")
    parser.add_argument("--output",   default="backtest_report.png")
    args = parser.parse_args()

    # ── Chargement des données ──
    if args.source == "synthetic":
        df = generate_sample_data(n_bars=args.bars, freq=args.tf.replace("m", "min").replace("h", "h").replace("1d","D"))
    elif args.source == "binance":
        df = load_from_binance(args.symbol, args.tf, args.bars)
    else:
        df = load_from_csv(args.file)

    # ── Optimisation ──
    if args.optimize:
        optimize(df, n_trials=25)
        return

    # ── Configuration du backtest ──
    params = BacktestParams(
        use_ai           = not args.no_ai,
        use_htf          = not args.no_htf,
        use_chop_filter  = not args.no_chop,
        use_candle_confirm = not args.no_candle,
        capital          = args.capital,
        risk_pct         = args.risk,
        sl_atr_mult      = args.sl,
        tp_atr_mult      = args.tp,
        lr               = args.lr,
    )

    print(f"\nConfiguration:")
    print(f"  AI activée         : {params.use_ai}")
    print(f"  Filtre HTF         : {params.use_htf}")
    print(f"  Filtre Chop        : {params.use_chop_filter}")
    print(f"  Confirmation bougie: {params.use_candle_confirm}")
    print(f"  Capital            : ${params.capital:,.0f}")
    print(f"  Risque/trade       : {params.risk_pct*100:.1f}%")
    print(f"  SL / TP            : {params.sl_atr_mult}× / {params.tp_atr_mult}× ATR")

    # ── Exécution ──
    bt = Backtester(params)
    metrics = bt.run(df)

    if "error" in metrics:
        print(f"\nErreur: {metrics['error']}")
        return

    bt.print_report(metrics)
    plot_results(bt, metrics, args.output)

    # Sauvegarder les trades en CSV
    trades_df = pd.DataFrame([{
        "direction":    t.direction,
        "entry_time":   t.entry_time,
        "entry_price":  t.entry_price,
        "exit_time":    t.exit_time,
        "exit_price":   t.exit_price,
        "exit_reason":  t.exit_reason,
        "pnl_usd":      t.pnl_usd,
        "pnl_pct":      t.pnl_pct * 100,
        "conf_score":   t.conf_score,
        "ai_proba":     t.ai_proba,
    } for t in bt.trades])

    csv_path = args.output.replace(".png", "_trades.csv")
    trades_df.to_csv(csv_path, index=False)
    print(f"Trades exportés: {csv_path}")


if __name__ == "__main__":
    main()
