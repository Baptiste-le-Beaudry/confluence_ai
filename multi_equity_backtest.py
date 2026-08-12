"""
multi_equity_backtest.py
Backtest multi-marchés pour indices, ETFs, forex, or — données Yahoo Finance.
Compare les résultats avec le backtest crypto.

Compatible avec: QQQ, SPY, GLD, NQ=F, EURUSD=X, etc.

Usage:
    python multi_equity_backtest.py
    python multi_equity_backtest.py --symbols QQQ SPY GLD NQ=F EURUSD=X
    python multi_equity_backtest.py --interval 5m --sl 1.5 --tp 4.0
    python multi_equity_backtest.py --compare-crypto  (compare avec résultats crypto)
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

sys.path.insert(0, os.path.dirname(__file__))

from indicators import compute_all
from ai_model   import NeuralNet, build_features, compute_label_continuous, pretrain_on_history
from multi_market_backtest import run_single_market, MarketResult, print_summary

SAVE_DIR    = "data_equity"
DEFAULT_SYM = ["QQQ", "SPY", "GLD", "NQ=F", "EURUSD=X"]

# Mapping ticker → nom lisible
EQUITY_NAMES = {
    "QQQ":      "NASDAQ-100",
    "SPY":      "S&P 500",
    "DIA":      "Dow Jones",
    "IWM":      "Russell 2000",
    "NQ=F":     "NASDAQ Fut.",
    "ES=F":     "S&P500 Fut.",
    "GLD":      "Gold ETF",
    "GC=F":     "Gold Fut.",
    "SLV":      "Silver ETF",
    "USO":      "Oil ETF",
    "CL=F":     "Oil Fut.",
    "EURUSD=X": "EUR/USD",
    "GBPUSD=X": "GBP/USD",
    "USDJPY=X": "USD/JPY",
    "USDCHF=X": "USD/CHF",
    "BTC-USD":  "Bitcoin",
    "ETH-USD":  "Ethereum",
}


def load_equity_csv(symbol: str, interval: str, save_dir: str) -> pd.DataFrame:
    """
    Charge le CSV d'un symbole équité depuis data_equity/.

    Args:
        symbol:   Ticker ex "QQQ", "EURUSD=X"
        interval: "1m", "5m", etc.
        save_dir: Dossier où chercher

    Returns:
        DataFrame OHLCV ou DataFrame vide
    """
    safe_sym = symbol.replace("=", "_").replace("-", "_")
    pattern  = os.path.join(save_dir, f"{safe_sym}_{interval}_*.csv")
    files    = sorted(glob.glob(pattern))

    if not files:
        print(f"  ⚠️  Aucun CSV trouvé pour {symbol} ({interval})")
        print(f"     Lancer d'abord: python download_equity_markets.py --symbols {symbol} --interval {interval}")
        return pd.DataFrame()

    # Prendre le plus récent
    fpath = files[-1]
    df = pd.read_csv(fpath, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.astype(float).sort_index().dropna()

    print(f"  {symbol}: {len(df):,} barres | "
          f"{df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")
    return df


def run_equity_backtest(
    symbols:    list[str],
    interval:   str   = "1m",
    save_dir:   str   = SAVE_DIR,
    capital:    float = 10_000.0,
    risk_pct:   float = 0.01,
    sl_atr:     float = 2.0,
    tp_atr:     float = 4.0,
    cooldown:   int   = 3,
    conf_pct:   float = 0.25,
    target_tpd: float = 5.0,
) -> list[MarketResult]:
    """Lance le backtest sur tous les symboles équités."""

    results = []
    for symbol in symbols:
        print(f"\n{'─'*55}")
        print(f"Marché: {symbol} ({EQUITY_NAMES.get(symbol, symbol)})")

        df = load_equity_csv(symbol, interval, save_dir)
        if df.empty or len(df) < 300:
            print(f"  Pas assez de données, skip")
            continue

        # Adapter le cooldown selon l'intervalle
        # 1min: cooldown=3 barres = 3min
        # 5min: cooldown=3 barres = 15min
        # 1h:   cooldown=3 barres = 3h
        adapted_cooldown = cooldown

        print(f"  Calcul indicateurs...", end="", flush=True)
        r = run_single_market(
            symbol     = symbol,
            df_raw     = df,
            capital    = capital,
            risk_pct   = risk_pct,
            sl_atr     = sl_atr,
            tp_atr     = tp_atr,
            cooldown   = adapted_cooldown,
            conf_pct   = conf_pct,
            target_tpd = target_tpd,
        )
        print(" OK")

        if r:
            # Corriger le nom
            r.name = EQUITY_NAMES.get(symbol, symbol)
            results.append(r)
            icon = "✅" if r.total_return > 0 else "❌"
            print(
                f"  {icon} Trades={r.n_trades} | WR={r.win_rate:.1f}% | "
                f"PF={r.profit_factor:.2f} | Return={r.total_return:+.2f}% | "
                f"Sharpe={r.sharpe:.3f}"
            )

    return results


def plot_comparison(
    equity_results: list[MarketResult],
    crypto_results: list[MarketResult] = None,
    output: str = "equity_backtest.png",
):
    """
    Graphique comparatif: indices/forex vs crypto (optionnel).
    """
    all_results = equity_results + (crypto_results or [])
    valid       = [r for r in all_results if r and r.n_trades > 0]

    if not valid:
        print("Aucun résultat à afficher")
        return

    BG="#161b22"; G="#00ff88"; R="#ff4466"; B="#4488ff"
    Y="#ffcc00"; GR="#888888"; PU="#bb88ff"; OR="#ffaa44"

    # Couleurs par type de marché
    def get_color(symbol):
        if any(x in symbol for x in ["QQQ","SPY","DIA","IWM","NQ","ES"]):
            return PU   # violet pour indices
        elif any(x in symbol for x in ["GLD","GC","SLV","XAU"]):
            return OR   # orange pour or
        elif "USD" in symbol or "JPY" in symbol or "CHF" in symbol:
            return Y    # jaune pour forex
        else:
            return B    # bleu pour crypto

    n = len(valid)
    cols = min(n, 3)
    rows_curves = (n + cols - 1) // cols
    fig_height  = 6 + rows_curves * 5

    fig = plt.figure(figsize=(20, fig_height), facecolor="#0d1117")
    gs  = gridspec.GridSpec(2 + rows_curves, cols,
                            figure=fig, hspace=0.50, wspace=0.32)

    def sa(ax, t=""):
        ax.set_facecolor(BG); ax.tick_params(colors=GR, labelsize=8)
        ax.spines[:].set_color("#333")
        if t: ax.set_title(t, color="white", fontsize=9, pad=4)

    # ── Graphique comparatif rendements ──
    ax_ret = fig.add_subplot(gs[0, :])
    sa(ax_ret, "Rendements comparés — Indices / Forex / Or vs Crypto")
    symbols  = [r.symbol for r in valid]
    names    = [r.name   for r in valid]
    returns  = [r.total_return for r in valid]
    colors_b = [get_color(s) for s in symbols]
    bars = ax_ret.bar(names, returns, color=colors_b, alpha=0.85, width=0.6)
    ax_ret.axhline(0, color=GR, lw=0.8)
    for bar, val, sym in zip(bars, returns, symbols):
        ax_ret.text(bar.get_x()+bar.get_width()/2,
                    val + (0.02 if val >= 0 else -0.05),
                    f"{val:+.2f}%",
                    ha="center", va="bottom" if val >= 0 else "top",
                    color="white", fontsize=9, fontweight="bold")
    ax_ret.set_ylabel("Rendement %", color=GR, fontsize=9)
    ax_ret.tick_params(axis="x", rotation=20)

    # Légende des couleurs
    from matplotlib.patches import Patch
    legend_items = [
        Patch(color=PU, label="Indices (NASDAQ, S&P500)"),
        Patch(color=OR, label="Or / Métaux"),
        Patch(color=Y,  label="Forex"),
        Patch(color=B,  label="Crypto"),
    ]
    ax_ret.legend(handles=legend_items, loc="upper right",
                  facecolor=BG, labelcolor="white", fontsize=8)

    # ── Tableau comparatif ──
    ax_tbl = fig.add_subplot(gs[1, :])
    sa(ax_tbl, "")
    ax_tbl.axis("off")

    col_labels = ["Marché","Type","Trades","T/jour","Win%","PF","R/R","Return%","MaxDD%","Sharpe","Esp/$"]
    tbl_data = []
    for r in sorted(valid, key=lambda x: -x.total_return):
        def mtype(s):
            if any(x in s for x in ["QQQ","SPY","DIA","NQ","ES"]): return "Index"
            elif any(x in s for x in ["GLD","GC","SLV"]): return "Or/Métal"
            elif "USD" in s or "JPY" in s: return "Forex"
            else: return "Crypto"
        pf_str = f"{r.profit_factor:.2f}" if r.profit_factor != float("inf") else "∞"
        tbl_data.append([
            r.symbol, mtype(r.symbol), str(r.n_trades),
            f"{r.trades_per_day:.1f}", f"{r.win_rate:.1f}%",
            pf_str, f"{r.ratio_rr:.2f}",
            f"{r.total_return:+.2f}%", f"{r.max_drawdown:.2f}%",
            f"{r.sharpe:.3f}", f"${r.expectancy:+.2f}"
        ])

    tbl = ax_tbl.table(cellText=tbl_data, colLabels=col_labels,
                       loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.5)
    for (ri, ci), cell in tbl.get_celld().items():
        cell.set_facecolor("#0d1117" if ri==0 else "#1a1f2e")
        cell.set_edgecolor("#333")
        cell.set_text_props(color="white" if ri==0 else GR)
        if ri > 0:
            try:
                val = tbl_data[ri-1][7]  # Return%
                pct = float(val.replace("%","").replace("+",""))
                cell_color = "#0d2a1a" if pct > 0 and ci==7 else "#2a0d1a" if pct < 0 and ci==7 else "#1a1f2e"
                cell.set_facecolor(cell_color)
            except: pass

    # ── Courbes d'équité ──
    for idx, r in enumerate(valid):
        row_i = 2 + idx // cols
        col_i = idx % cols
        ax = fig.add_subplot(gs[row_i, col_i])
        color = get_color(r.symbol)
        icon  = "✅" if r.total_return > 0 else "❌"
        sa(ax, f"{r.name} | {r.total_return:+.2f}% | WR:{r.win_rate:.0f}% | {r.n_trades}T {icon}")

        if r.equity_curve:
            eq_df = pd.DataFrame(r.equity_curve)
            ax.plot(range(len(eq_df)), eq_df["equity"], color=color, lw=1.2)
            ax.axhline(10000, color=GR, ls="--", lw=0.7, alpha=0.5)
            ax.set_ylabel("Équité $", color=GR, fontsize=7)

            # P&L cumulatif
            pnls = [t["pnl"] for t in r.trades]
            if pnls:
                cum  = np.cumsum(pnls)
                x_t  = np.linspace(0, len(eq_df)-1, len(cum))
                ax2  = ax.twinx()
                ax2.set_facecolor(BG)
                ax2.plot(x_t, cum, color=Y, lw=0.8, alpha=0.7)
                ax2.set_ylabel("P&L cum.", color=Y, fontsize=6)
                ax2.tick_params(colors=Y, labelsize=6)
                ax2.spines[:].set_color("#333")

    interval_label = "1min" if "1m" in (equity_results[0].symbol if equity_results else "") else "?"
    fig.suptitle(
        f"Backtest Multi-Marchés — Confluence AI v5.0 — "
        f"Indices + Forex + Or vs Crypto",
        color="white", fontsize=12, fontweight="bold", y=0.998
    )
    plt.savefig(output, dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"\nGraphique sauvegardé: {output}")


def export_comparison_csv(results: list[MarketResult], output: str):
    """Export CSV comparatif."""
    rows = []
    for r in results:
        if not r: continue
        def mtype(s):
            if any(x in s for x in ["QQQ","SPY","DIA","NQ","ES"]): return "Index"
            elif any(x in s for x in ["GLD","GC","SLV"]): return "Or/Metal"
            elif "USD" in s or "JPY" in s: return "Forex"
            else: return "Crypto"
        rows.append({
            "symbol":           r.symbol,
            "name":             r.name,
            "type":             mtype(r.symbol),
            "n_trades":         r.n_trades,
            "trades_per_day":   round(r.trades_per_day, 2),
            "win_rate_pct":     round(r.win_rate, 2),
            "profit_factor":    round(r.profit_factor, 3) if r.profit_factor != float("inf") else 999,
            "ratio_rr":         round(r.ratio_rr, 2),
            "total_return_pct": round(r.total_return, 3),
            "max_drawdown_pct": round(r.max_drawdown, 3),
            "sharpe":           round(r.sharpe, 3),
            "expectancy_usd":   round(r.expectancy, 2),
            "capital_final":    round(r.capital_final, 2),
        })
    df = pd.DataFrame(rows).sort_values("total_return_pct", ascending=False)
    df.to_csv(output, index=False)
    print(f"Comparaison exportée: {output}")
    return df


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Backtest Multi-Marchés Équités/Forex/Or — Confluence AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Étapes:
  1. Télécharger les données:
     python download_equity_markets.py --symbols QQQ SPY GLD NQ=F EURUSD=X --interval 1m

  2. Lancer le backtest:
     python multi_equity_backtest.py

  3. Comparer avec le crypto:
     python multi_equity_backtest.py --compare-crypto

Exemples:
  python multi_equity_backtest.py --symbols QQQ SPY GLD
  python multi_equity_backtest.py --interval 5m --sl 1.5 --tp 4.5
  python multi_equity_backtest.py --symbols QQQ SPY GLD EURUSD=X --compare-crypto
        """
    )
    parser.add_argument("--symbols",       nargs="+",  default=DEFAULT_SYM)
    parser.add_argument("--interval",      default="1m")
    parser.add_argument("--capital",       type=float, default=10_000.0)
    parser.add_argument("--risk",          type=float, default=0.01)
    parser.add_argument("--sl",            type=float, default=2.0)
    parser.add_argument("--tp",            type=float, default=4.0)
    parser.add_argument("--cooldown",      type=int,   default=3)
    parser.add_argument("--conf",          type=float, default=0.25)
    parser.add_argument("--tpd",           type=float, default=5.0)
    parser.add_argument("--compare-crypto",action="store_true",
                        help="Inclure les résultats crypto du dernier backtest")
    parser.add_argument("--output",        default="equity_backtest.png")
    args = parser.parse_args()

    print("="*65)
    print("  BACKTEST MULTI-MARCHÉS ÉQUITÉS — Confluence AI v5.0")
    print("="*65)
    print(f"  Symboles  : {', '.join(args.symbols)}")
    print(f"  Intervalle: {args.interval}")
    print(f"  SL/TP     : {args.sl}× / {args.tp}× ATR")
    print(f"  Capital   : ${args.capital:,.0f}")
    print()

    # Backtest équités
    equity_results = run_equity_backtest(
        symbols    = args.symbols,
        interval   = args.interval,
        save_dir   = SAVE_DIR,
        capital    = args.capital,
        risk_pct   = args.risk,
        sl_atr     = args.sl,
        tp_atr     = args.tp,
        cooldown   = args.cooldown,
        conf_pct   = args.conf,
        target_tpd = args.tpd,
    )

    # Charger résultats crypto si demandé
    crypto_results = []
    if args.compare_crypto:
        crypto_csv = "multi_backtest_metrics.csv"
        if os.path.exists(crypto_csv):
            print(f"\nChargement résultats crypto depuis {crypto_csv}...")
            crypto_df = pd.read_csv(crypto_csv)
            print(f"  {len(crypto_df)} marchés crypto trouvés")
        else:
            print(f"\n⚠️  Pas de résultats crypto trouvés ({crypto_csv})")
            print("   Lancer d'abord: python multi_market_backtest.py --from-csv")

    # Rapport complet
    all_results = equity_results + crypto_results
    if all_results:
        print_summary(all_results)
        plot_comparison(equity_results, crypto_results, args.output)
        df_comp = export_comparison_csv(all_results, args.output.replace(".png","_comparison.csv"))

        print("\n" + "─"*65)
        print("CLASSEMENT PAR RENDEMENT:")
        print("─"*65)
        for _, row in df_comp.iterrows():
            icon = "✅" if row["total_return_pct"] > 0 else "❌"
            print(f"  {icon} {row['symbol']:<12} {row['name']:<15} "
                  f"Return: {row['total_return_pct']:+.2f}% | "
                  f"Sharpe: {row['sharpe']:.3f} | "
                  f"Type: {row['type']}")
    else:
        print("\nAucun résultat. Télécharger les données d'abord:")
        print(f"  python download_equity_markets.py --symbols {' '.join(args.symbols)} --interval {args.interval}")


if __name__ == "__main__":
    main()
