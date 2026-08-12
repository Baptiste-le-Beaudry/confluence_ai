"""
trade_memory.py - Mémoire long terme des trades fermés (Phase 2).

Un trade réel (TP/SL/BE) reste instructif beaucoup plus longtemps qu'une
barre: on l'enregistre en SQLite avec un poids qui vieillit lentement
(0.99^age, plancher 0.10) et on le rejoue périodiquement dans le NeuralNet
du symbole, en échantillonnant 50/50 entre gagnants et perdants pour éviter
que le réseau n'apprenne juste "prédis toujours la baisse".
"""

import json
import sqlite3
import time
from pathlib import Path

import numpy as np

SCHEMA_VERSION       = 1
CAPACITY_PER_SYMBOL  = 500     # FIFO: on garde les 500 trades les plus récents
AGE_DECAY            = 0.99    # poids_trade = poids_initial * 0.99^(trades écoulés)
WEIGHT_FLOOR         = 0.10    # une leçon ne disparaît jamais complètement
MIN_TRADES_FOR_REPLAY = 30     # garde-fou anti-overfit: pas de rejeu avant 30 trades

# outcome -> (label_si_LONG, label_si_SHORT, poids_initial)
OUTCOME_TABLE = {
    "TP2":          (1.00, 0.00, 3.0),
    "TP1":          (0.85, 0.15, 2.0),
    "BE":           (0.55, 0.45, 1.0),
    "SL":           (0.05, 0.95, 2.5),
    # Le bot live ferme aussi sur signal inverse — pas de TP/SL au sens
    # strict, donc mapping séparé selon que la clôture a été profitable.
    "REVERSE_WIN":  (0.75, 0.25, 1.5),
    "REVERSE_LOSS": (0.20, 0.80, 1.5),
}

WINNING_OUTCOMES = {"TP1", "TP2", "REVERSE_WIN"}


def classify_outcome(reason: str, pnl_usd: float, be_triggered: bool) -> str:
    """Best-effort: le bot ne distingue pas toujours TP1/TP2/BE pour les
    positions fermées côté broker (sync_closed_trades ne donne pas la raison
    exacte, ex: "TP/SL/manuel") — on retombe sur le signe du PnL dans ce cas."""
    r = (reason or "").upper()
    if "REVERSE" in r:
        return "REVERSE_WIN" if pnl_usd > 0 else "REVERSE_LOSS"
    has_tp, has_sl = "TP" in r, "SL" in r
    if be_triggered and abs(pnl_usd) < 1e-6:
        return "BE"
    if has_sl and not has_tp:
        return "SL"
    if has_tp and not has_sl:
        return "TP2"
    # raison ambiguë ou générique (ex: "TP/SL/manuel") -> signe du PnL
    return "TP2" if pnl_usd > 0 else "SL"


def label_and_weight(direction: str, outcome: str) -> tuple[float, float]:
    label_long, label_short, weight = OUTCOME_TABLE.get(outcome, (0.5, 0.5, 1.0))
    label = label_long if direction == "LONG" else label_short
    return label, weight


def connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol          TEXT    NOT NULL,
            timestamp       TEXT    NOT NULL,
            direction       TEXT    NOT NULL,
            features        TEXT    NOT NULL,
            outcome         TEXT    NOT NULL,
            pnl_usd         REAL    NOT NULL,
            r_multiple      REAL,
            ai_proba_entry  REAL,
            conf_entry      REAL,
            label           REAL    NOT NULL,
            weight_initial  REAL    NOT NULL,
            schema_version  INTEGER NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
    conn.commit()
    return conn


def record_trade(db_path: str, symbol: str, direction: str, features: list,
                  outcome: str, pnl_usd: float, ai_proba_entry: float,
                  conf_entry: float, r_multiple: float = None) -> None:
    label, weight = label_and_weight(direction, outcome)
    conn = connect(db_path)
    try:
        conn.execute(
            "INSERT INTO trades (symbol, timestamp, direction, features, outcome, "
            "pnl_usd, r_multiple, ai_proba_entry, conf_entry, label, weight_initial, "
            "schema_version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (symbol, time.strftime("%Y-%m-%dT%H:%M:%S"), direction,
             json.dumps(list(features)), outcome, float(pnl_usd),
             None if r_multiple is None else float(r_multiple),
             float(ai_proba_entry), float(conf_entry), float(label),
             float(weight), SCHEMA_VERSION),
        )
        conn.commit()
        _enforce_capacity(conn, symbol)
    finally:
        conn.close()


def _enforce_capacity(conn: sqlite3.Connection, symbol: str,
                       capacity: int = CAPACITY_PER_SYMBOL):
    ids = [r[0] for r in conn.execute(
        "SELECT id FROM trades WHERE symbol=? ORDER BY id DESC", (symbol,))]
    if len(ids) > capacity:
        old_ids = ids[capacity:]
        conn.executemany("DELETE FROM trades WHERE id=?", [(i,) for i in old_ids])
        conn.commit()


def count_trades(db_path: str, symbol: str) -> int:
    conn = connect(db_path)
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM trades WHERE symbol=?", (symbol,)).fetchone()[0]
    finally:
        conn.close()


def sample_balanced(db_path: str, symbol: str, n: int = 20) -> list[tuple]:
    """
    Échantillonne jusqu'à n trades pour rejeu, équilibrés 50/50 entre
    gagnants et perdants, avec un poids vieilli (0.99^age, plancher 0.10).

    Retourne une liste de (features: list[float], label: float, weight: float).
    """
    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, direction, features, outcome, label, weight_initial "
            "FROM trades WHERE symbol=? ORDER BY id ASC", (symbol,)
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return []

    max_id = rows[-1][0]
    winners, losers = [], []
    for rid, direction, feat_json, outcome, label, w0 in rows:
        age = max_id - rid
        weight = max(w0 * (AGE_DECAY ** age), WEIGHT_FLOOR)
        entry = (json.loads(feat_json), label, weight)
        (winners if outcome in WINNING_OUTCOMES else losers).append(entry)

    rng = np.random.default_rng()

    def pick(pool, k):
        if not pool:
            return []
        idx = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
        return [pool[i] for i in idx]

    half = n // 2
    sample = pick(winners, half) + pick(losers, n - half)
    rng.shuffle(sample)
    return sample
