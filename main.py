"""
main.py — Compare les 3 setups d'entrée sur les données réelles
Usage:
    python main.py --source csv --file data_1min/BNBUSDT_1min_*.csv
    python main.py --compare-setups
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data_loader import generate_sample_data, load_from_csv
from backtest import Backtester, BacktestParams


def compare_setups(df: pd.DataFrame, output: str = "setup_comparison.png"):
    """Compare les 3 setups sur les mêmes données."""
    BG="#161b22"; G="#00ff88"; R="#ff4466"; B="#4488ff"; Y="#ffcc00"; GR="#888888"

    configs = [
        {"setup_mode": "squeeze",  "min_quality": 0.40, "label": "Setup A\nSqueeze Release",  "color": Y},
        {"setup_mode": "pullback", "min_quality": 0.40, "label": "Setup B\nZone Pullback",    "color": B},
        {"setup_mode": "combined", "min_quality": 0.45, "label": "Setup C\nCombiné (recommandé)", "color": G},
    ]

    results = []
    for cfg in configs:
        print(f"\n{'─'*50}")
        print(f"Test: {cfg['label'].replace(chr(10),' ')}")
        params = BacktestParams(
            setup_mode   = cfg["setup_mode"],
            min_quality  = cfg["min_quality"],
            ai_thresh    = 0.52,
            conf_pct     = 0.25,
            target_trades_per_day = 5.0,
            sl_atr_mult  = 2.0,
            tp_atr_mult  = 4.0,
            cooldown_bars= 10,
        )
        bt = Backtester(params)
        m  = bt.run(df)
        if "error" not in m:
            bt.print_report(m)
            results.append({"cfg": cfg, "metrics": m, "bt": bt})
        else:
            print(f"  ERREUR: {m['error']}")
            results.append({"cfg": cfg, "metrics": None, "bt": bt})

    # Graphique comparatif
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor="#0d1117")
    fig.subplots_adjust(hspace=0.40, wspace=0.30)

    def sa(ax, t=""):
        ax.set_facecolor(BG); ax.tick_params(colors=GR, labelsize=8)
        ax.spines[:].set_color("#333")
        if t: ax.set_title(t, color="white", fontsize=9, pad=4)

    # Ligne 1: courbes d'équité
    for i, res in enumerate(results):
        ax = axes[0][i]
        cfg = res["cfg"]; m = res["metrics"]; bt = res["bt"]
        ret_str = f"{m['total_return_pct']:+.2f}%" if m else "N/A"
        wr_str  = f"WR:{m['win_rate_pct']:.0f}%" if m else ""
        n_str   = f"{m['n_trades']}T" if m else ""
        icon    = "✅" if m and m['total_return_pct']>0 else "❌"
        sa(ax, f"{cfg['label'].replace(chr(10),' ')} | {ret_str} | {wr_str} | {n_str} {icon}")

        if bt.equity_curve:
            eq = pd.DataFrame(bt.equity_curve)
            ax.plot(range(len(eq)), eq["equity"], color=cfg["color"], lw=1.5)
            ax.axhline(10000, color=GR, ls="--", lw=0.7, alpha=0.5)
            ax.set_ylabel("Équité $", color=GR, fontsize=7)

    # Ligne 2: métriques comparées
    ax_bar = axes[1][0]
    sa(ax_bar, "Rendement % par setup")
    labels = [r["cfg"]["label"].replace("\n"," ") for r in results]
    returns= [r["metrics"]["total_return_pct"] if r["metrics"] else 0 for r in results]
    colors = [r["cfg"]["color"] for r in results]
    bars = ax_bar.bar(range(len(labels)), returns, color=colors, alpha=0.85)
    ax_bar.axhline(0, color=GR, lw=0.8)
    for b, v in zip(bars, returns):
        ax_bar.text(b.get_x()+b.get_width()/2, v+(0.01 if v>=0 else -0.03),
                    f"{v:+.2f}%", ha="center", color="white", fontsize=9, fontweight="bold")
    ax_bar.set_xticks(range(len(labels)))
    ax_bar.set_xticklabels(["Squeeze","Pullback","Combiné"], color=GR, fontsize=8)

    ax_wr = axes[1][1]
    sa(ax_wr, "Win Rate & Profit Factor")
    wrs = [r["metrics"]["win_rate_pct"] if r["metrics"] else 0 for r in results]
    pfs = [min(r["metrics"]["profit_factor"],5) if r["metrics"] else 0 for r in results]
    x = np.arange(3)
    ax_wr.bar(x-0.2, wrs, 0.35, color=[r["cfg"]["color"] for r in results], alpha=0.8, label="WR%")
    ax_wr2 = ax_wr.twinx()
    ax_wr2.bar(x+0.2, pfs, 0.35, color=[r["cfg"]["color"] for r in results], alpha=0.4, label="PF")
    ax_wr2.set_facecolor(BG); ax_wr2.tick_params(colors=GR, labelsize=7)
    ax_wr2.spines[:].set_color("#333")
    ax_wr2.set_ylabel("Profit Factor", color=GR, fontsize=7)
    ax_wr.set_ylabel("Win Rate %", color=GR, fontsize=7)
    ax_wr.set_xticks(x)
    ax_wr.set_xticklabels(["Squeeze","Pullback","Combiné"], color=GR, fontsize=8)
    ax_wr.axhline(33.3, color=R, ls="--", lw=0.8, alpha=0.6)

    ax_q = axes[1][2]
    sa(ax_q, "Qualité moyenne: Wins vs Losses")
    qw = [r["metrics"]["avg_quality_wins"]   if r["metrics"] else 0 for r in results]
    ql = [r["metrics"]["avg_quality_losses"] if r["metrics"] else 0 for r in results]
    x = np.arange(3)
    ax_q.bar(x-0.2, qw, 0.35, color=G, alpha=0.8, label="Wins")
    ax_q.bar(x+0.2, ql, 0.35, color=R, alpha=0.8, label="Losses")
    ax_q.set_xticks(x)
    ax_q.set_xticklabels(["Squeeze","Pullback","Combiné"], color=GR, fontsize=8)
    ax_q.legend(fontsize=7, facecolor=BG, labelcolor="white")
    ax_q.set_ylabel("Score qualité moyen", color=GR, fontsize=7)

    fig.suptitle("Comparaison des 3 Setups d'Entrée — Confluence AI v5.0",
                 color="white", fontsize=12, fontweight="bold")
    plt.savefig(output, dpi=140, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"\nGraphique comparatif: {output}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="synthetic", choices=["synthetic","csv"])
    parser.add_argument("--file",   default="")
    parser.add_argument("--bars",   type=int, default=3000)
    parser.add_argument("--setup",  default="combined",
                        choices=["squeeze","pullback","combined","all"])
    parser.add_argument("--quality",type=float, default=0.55)
    parser.add_argument("--sl",     type=float, default=2.0)
    parser.add_argument("--tp",     type=float, default=4.0)
    parser.add_argument("--output", default="backtest_result.png")
    args = parser.parse_args()

    if args.source == "csv" and args.file:
        df = load_from_csv(args.file)
    else:
        df = generate_sample_data(n_bars=args.bars, seed=42)

    if args.setup == "all":
        compare_setups(df, args.output)
    else:
        params = BacktestParams(
            setup_mode    = args.setup,
            min_quality   = args.quality,
            ai_thresh     = 0.52,
            conf_pct      = 0.25,
            sl_atr_mult   = args.sl,
            tp_atr_mult   = args.tp,
            target_trades_per_day = 5.0,
            cooldown_bars = 10,
        )
        bt = Backtester(params)
        m  = bt.run(df)
        if "error" not in m:
            bt.print_report(m)
        else:
            print(f"Erreur: {m['error']}")


if __name__ == "__main__":
    main()
