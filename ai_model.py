"""
ai_model.py - Réseau de neurones + apprentissage supervisé sur patterns gagnants.

NOUVELLE APPROCHE:
  Phase 1 (analyse historique): Scanner tout l'historique, identifier les setups
  qui ont réellement généré un profit. Extraire les features de ces setups.
  Entraîner le réseau sur CES exemples avant de trader.

  Phase 2 (online learning): Continuer à apprendre barre par barre.

  Objectif: au moins 1 trade/jour → seuils adaptatifs basés sur la distribution
  réelle des probabilités produites par le réseau.
"""

import numpy as np
from dataclasses import dataclass, field
from collections import deque


def safe_tanh(x):
    return np.tanh(np.clip(x, -20.0, 20.0))

def sigmoid(z):
    z = np.clip(z, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-z))

def norm_ratio(val, ref, eps=1e-9):
    return safe_tanh(val / max(abs(ref), eps))


N_FEAT   = 13
N_HIDDEN = 6   # Augmenté: plus de capacité

FEAT_NAMES = ["vwap","fib","gann","ich","ema","ob","fvg","sr","vol","bos","bb","cvd","mom"]


@dataclass
class FeatVec:
    vwap: float = 0.0
    fib:  float = 0.0
    gann: float = 0.0
    ich:  float = 0.0
    ema:  float = 0.0
    ob:   float = 0.0
    fvg:  float = 0.0
    sr:   float = 0.0
    vol:  float = 0.0
    bos:  float = 0.0
    bb:   float = 0.0
    cvd:  float = 0.0
    mom:  float = 0.0
    ref_close: float = 0.0
    ref_atr:   float = 0.0
    direction: int = 0   # +1 LONG, -1 SHORT, 0 inconnu

    def to_array(self) -> np.ndarray:
        return np.array([self.vwap, self.fib, self.gann, self.ich, self.ema,
                         self.ob, self.fvg, self.sr, self.vol,
                         self.bos, self.bb, self.cvd, self.mom], dtype=np.float64)


class NeuralNet:
    """
    Réseau 13 → 6 (tanh) → 1 (sigmoid).
    Pré-entraîné sur les setups gagnants historiques, puis apprentissage online.
    """
    def __init__(self, lr=0.02, grad_clip=0.5, decay=0.0005, n_hidden=N_HIDDEN):
        self.lr        = lr
        self.grad_clip = grad_clip
        self.decay     = decay
        self.n_hidden  = n_hidden

        # Initialisation Xavier pour une meilleure convergence
        scale1 = np.sqrt(2.0 / (N_FEAT + n_hidden))
        scale2 = np.sqrt(2.0 / (n_hidden + 1))
        self.W1 = np.random.randn(n_hidden, N_FEAT + 1) * scale1
        self.W2 = np.random.randn(n_hidden + 1) * scale2

        self.train_losses: list[float] = []
        self.n_samples = 0
        self.proba_history: list[float] = []  # historique des probas pour seuils adaptatifs

    def forward(self, x: np.ndarray):
        # Remplacer les nan/inf dans les features
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        x_bias = np.append(x, 1.0)
        z1 = self.W1 @ x_bias
        z1 = np.nan_to_num(z1, nan=0.0, posinf=20.0, neginf=-20.0)
        a1 = safe_tanh(z1)
        a1_b = np.append(a1, 1.0)
        z2 = float(np.dot(self.W2, a1_b))
        z2 = 0.0 if np.isnan(z2) or np.isinf(z2) else z2
        return sigmoid(z2), a1

    def predict(self, fv: FeatVec) -> float:
        p, _ = self.forward(fv.to_array())
        self.proba_history.append(float(p))
        if len(self.proba_history) > 500:
            self.proba_history.pop(0)
        return float(p)

    def train_step(self, fv: FeatVec, label: float, weight: float = 1.0):
        """
        weight: pondération de l'échantillon (>1 pour les exemples importants)
        """
        x = fv.to_array()
        proba, a1 = self.forward(x)

        g_out = np.clip((proba - label) * weight, -self.grad_clip, self.grad_clip)

        self.W2 *= (1.0 - self.decay)
        self.W1 *= (1.0 - self.decay)

        a1_b = np.append(a1, 1.0)
        self.W2 -= self.lr * g_out * a1_b
        self.W2 = np.nan_to_num(self.W2, nan=0.0, posinf=1.0, neginf=-1.0)

        delta1 = np.clip(
            (g_out * self.W2[:self.n_hidden]) * (1.0 - a1 ** 2),
            -self.grad_clip, self.grad_clip
        )
        x_bias = np.append(x, 1.0)
        self.W1 -= self.lr * np.outer(delta1, x_bias)
        self.W1 = np.nan_to_num(self.W1, nan=0.0, posinf=1.0, neginf=-1.0)

        eps = 1e-9
        loss = -(label * np.log(proba + eps) + (1 - label) * np.log(1 - proba + eps))
        if not np.isnan(loss):
            self.train_losses.append(float(loss))
        self.n_samples += 1

    def batch_train(self, samples: list[tuple], n_epochs: int = 3):
        """
        Entraînement par batch sur une liste de (FeatVec, label, weight).
        Utilisé pour la Phase 1 (pré-entraînement sur l'historique).
        """
        for epoch in range(n_epochs):
            np.random.shuffle(samples)
            for fv, lbl, w in samples:
                self.train_step(fv, lbl, w)

    def get_adaptive_thresholds(self, target_trades_per_day: float,
                                 bars_per_day: float) -> tuple[float, float]:
        """
        Calcule les seuils LONG/SHORT adaptatifs pour atteindre la fréquence de trade cible.

        Si on veut 1 trade/jour sur 96 barres 15min:
        → on veut que ~1/96 = 1% des barres déclenchent un signal
        → on prend le percentile 99 des probas pour le seuil LONG
        → et le percentile 1 pour le seuil SHORT

        Args:
            target_trades_per_day: nombre de trades visés par jour
            bars_per_day: nombre de barres par jour (96 pour 15min, 24 pour 1h)

        Returns:
            (thresh_long, thresh_short)
        """
        if len(self.proba_history) < 50:
            return 0.60, 0.40  # valeurs par défaut

        probas = np.array(self.proba_history)

        # Fraction de barres devant déclencher un signal
        # On divise par 2 car on a LONG et SHORT
        frac = target_trades_per_day / bars_per_day / 2.0
        frac = np.clip(frac, 0.005, 0.20)  # entre 0.5% et 20%

        thresh_long  = float(np.percentile(probas, (1.0 - frac) * 100))
        thresh_short = float(np.percentile(probas, frac * 100))

        # Sécurité: garder un minimum de différence avec 0.5
        thresh_long  = max(thresh_long,  0.50)
        thresh_short = min(thresh_short, 0.50)

        return thresh_long, thresh_short

    def get_weights_summary(self) -> dict:
        importance = {}
        for i, name in enumerate(FEAT_NAMES):
            col_norm = np.linalg.norm(self.W1[:, i])
            importance[name] = float(col_norm * np.linalg.norm(self.W2[:self.n_hidden]))
        return importance


# ─────────────────────────────────────────────
# Construction des features
# ─────────────────────────────────────────────

def build_features(row: dict, norm_w: dict) -> FeatVec:
    vwap_std = max(norm_w.get("vwap_std", 0.5),  0.05)
    ema_std  = max(norm_w.get("ema_std",  0.05), 0.05)
    gann_std = max(norm_w.get("gann_std", 0.5),  0.10)
    mom_std  = max(norm_w.get("mom_std",  1.0),  0.10)

    ob_feat  = norm_ratio(row.get("ob_dist",  3.0), 3.0)
    fvg_feat = norm_ratio(row.get("fvg_dist", 3.0), 3.0)

    return FeatVec(
        vwap = norm_ratio(row.get("dist_vwap", 0.0),        vwap_std),
        fib  = safe_tanh(row.get("fib_pos", 0.5) * 2.0 - 1.0),
        gann = norm_ratio(row.get("gann_pos", 0.0),         gann_std),
        ich  = float(row.get("ich_score", 0)),
        ema  = norm_ratio(row.get("ema_trend", 0.0) * 100,  ema_std),
        ob   = ob_feat,
        fvg  = fvg_feat,
        sr   = float(row.get("sr_score", 0)),
        vol  = float(row.get("vol_score", 0)),
        bos  = safe_tanh(float(row.get("bos_score", 0)) * 0.5),
        bb   = safe_tanh(float(row.get("bb_pos", 0.5)) * 2.0 - 1.0),
        cvd  = norm_ratio(float(row.get("cvd_score", 0)), 1.0),
        mom  = norm_ratio(row.get("mom_raw", 0.0),          mom_std),
        ref_close = row.get("close", 0.0),
        ref_atr   = row.get("atr", 1.0),
    )


def compute_label_continuous(current_close: float, ref_close: float,
                              ref_atr: float, direction: int = 0) -> float | None:
    """
    Label continu: sigmoid(rendement en ATR × sensibilité).
    direction: +1 pour forcer vers label bull, -1 vers bear, 0 neutre.
    """
    if ref_atr <= 0 or ref_close <= 0:
        return None
    move = (current_close - ref_close) / ref_atr
    # Sensibilité plus forte pour mieux différencier les cas
    return float(sigmoid(move * 3.0))


# ─────────────────────────────────────────────
# Phase 1: Pré-entraînement sur setups gagnants
# ─────────────────────────────────────────────

def pretrain_on_history(nn: NeuralNet, df_indicators, norm_w: dict,
                         verbose: bool = True,
                         fwd_bars: int = 5, sl_atr: float = 2.0,
                         tp_atr: float = 4.0, n_epochs: int = 5):
    """
    Analyse tout l'historique et entraîne le réseau sur les setups réels.

    Pour chaque barre dans une zone (OB/FVG):
      - Simuler un trade LONG ou SHORT
      - Vérifier si TP ou SL est touché dans les fwd_bars barres suivantes
      - Créer un label 1.0 (TP touché = bon) ou 0.0 (SL touché = mauvais)
      - Pondérer les exemples: trades TP = poids élevé, SL = poids normal

    Cela pré-oriente les poids vers les patterns qui GAGNENT vraiment.
    """
    samples = []
    close = df_indicators["close"].values
    atr   = df_indicators["atr"].values
    in_ob_bull  = df_indicators["in_ob_bull"].values
    in_ob_bear  = df_indicators["in_ob_bear"].values
    in_fvg_bull = df_indicators["in_fvg_bull"].values
    in_fvg_bear = df_indicators["in_fvg_bear"].values
    rows = df_indicators.to_dict("records")

    n = len(close)

    for i in range(220, n - fwd_bars - 1):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        row = rows[i]
        row["close"] = close[i]
        row["atr"]   = atr[i]
        fv = build_features(row, norm_w)

        in_bull = bool(in_ob_bull[i]) or bool(in_fvg_bull[i])
        in_bear = bool(in_ob_bear[i]) or bool(in_fvg_bear[i])

        if not in_bull and not in_bear:
            continue

        # Simuler LONG si dans zone bull
        if in_bull:
            entry = close[i]
            sl_price = entry - sl_atr * atr[i]
            tp_price = entry + tp_atr * atr[i]
            label = 0.5  # neutre par défaut
            weight = 1.0
            for j in range(i + 1, min(i + fwd_bars + 1, n)):
                if close[j] >= tp_price:
                    label  = 1.0   # TP touché → bon signal LONG
                    weight = 2.5   # fortement pondéré
                    break
                elif close[j] <= sl_price:
                    label  = 0.1   # SL touché → mauvais signal
                    weight = 1.5
                    break
            samples.append((fv, label, weight))

        # Simuler SHORT si dans zone bear
        if in_bear:
            entry = close[i]
            sl_price = entry + sl_atr * atr[i]
            tp_price = entry - tp_atr * atr[i]
            label = 0.5
            weight = 1.0
            for j in range(i + 1, min(i + fwd_bars + 1, n)):
                if close[j] <= tp_price:
                    label  = 0.0   # TP touché → bon signal SHORT (proba basse = SHORT)
                    weight = 2.5
                    break
                elif close[j] >= sl_price:
                    label  = 0.9   # SL touché → mauvais SHORT
                    weight = 1.5
                    break
            samples.append((fv, label, weight))

    if samples:
        if verbose: print(f"  Pré-entraînement: {len(samples)} setups historiques, {n_epochs} epochs...")
        tp_count = sum(1 for _, l, _ in samples if l >= 0.9 or l <= 0.1)
        print(f"  TP hits: {sum(1 for _,l,_ in samples if l==1.0)} | "
              f"SL hits: {sum(1 for _,l,_ in samples if l==0.1)} | "
              f"Neutres: {sum(1 for _,l,_ in samples if l==0.5)}")
        nn.batch_train(samples, n_epochs=n_epochs)
        print(f"  Pré-entraînement terminé. Loss finale: {np.mean(nn.train_losses[-100:]):.4f}")

    return len(samples)