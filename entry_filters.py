"""
entry_filters.py
Filtres d'entrée avancés — 3 setups de qualité institutionnelle.

Setup 1 — SQUEEZE RELEASE:
  Bollinger Squeeze → relâchement + volume surge + AI + confluence

Setup 2 — ZONE PULLBACK:
  Prix retourne dans OB/FVG → bougie de rejet + EMA 200 + AI + confluence

Setup 3 — COMBINÉ (recommandé):
  Zone OB/FVG + EMA 200 + AI + (Squeeze OU Bougie OU BOS)

Chaque setup retourne:
  (signal_long: bool, signal_short: bool, reason: str, quality_score: float)
"""

import numpy as np
import pandas as pd


def check_ema200_trend(row: dict) -> tuple[bool, bool]:
    """
    Vérifie la tendance EMA 200.
    Returns: (ema_bull, ema_bear)
    """
    close   = row.get("close", 0)
    ema200  = row.get("ema200", 0)
    ema50   = row.get("ema50", 0)
    ema20   = row.get("ema20", 0)

    # Haussier: prix > EMA200 ET EMA50 > EMA200 ET EMA20 > EMA50
    ema_bull = (close > ema200) and (ema50 > ema200)
    # Baissier: prix < EMA200 ET EMA50 < EMA200
    ema_bear = (close < ema200) and (ema50 < ema200)

    return ema_bull, ema_bear


def check_candle_confirm(row: dict, pin_ratio: float = 0.30) -> tuple[bool, bool]:
    """
    Vérifie la confirmation de bougie (engulfing ou pin bar).
    Returns: (bull_confirm, bear_confirm)
    """
    bull = bool(row.get("bull_confirm", False))
    bear = bool(row.get("bear_confirm", False))
    return bull, bear


def check_volume_surge(row: dict, vol_mult: float = 1.5) -> bool:
    """Vérifie si le volume est supérieur à la moyenne."""
    return bool(row.get("vol_score", 0) != 0) or bool(row.get("vol_score", 0) == 1)


def check_squeeze_release(row: dict, prev_row: dict) -> tuple[bool, str]:
    """
    Détecte la sortie d'un Bollinger Squeeze.
    squeeze=True sur prev_row et squeeze=False sur row = relâchement.

    Returns: (squeeze_released, direction_hint)
    """
    was_squeeze = bool(prev_row.get("squeeze", False)) if prev_row else False
    is_squeeze  = bool(row.get("squeeze", False))

    if was_squeeze and not is_squeeze:
        # Direction du relâchement selon la position dans les BB
        bb_pos = float(row.get("bb_pos", 0.5))
        if bb_pos > 0.6:
            return True, "LONG"
        elif bb_pos < 0.4:
            return True, "SHORT"
        else:
            return True, "NEUTRAL"
    return False, "NONE"


def check_bos_choch(row: dict) -> tuple[bool, bool]:
    """
    Vérifie BOS ou CHoCH.
    Returns: (bull_structure, bear_structure)
    """
    bos_score = int(row.get("bos_score", 0))
    bull = bos_score > 0  # BOS↑ ou CHoCH↑
    bear = bos_score < 0  # BOS↓ ou CHoCH↓
    return bull, bear


# ─────────────────────────────────────────────
# SETUP 1 — BOLLINGER SQUEEZE RELEASE
# ─────────────────────────────────────────────

def setup_squeeze_release(
    row:        dict,
    prev_row:   dict,
    ai_proba:   float,
    conf_score: float,
    max_conf:   float,
    ai_thresh:  float = 0.65,
    conf_pct:   float = 0.30,
) -> tuple[bool, bool, str, float]:
    """
    Setup Squeeze Release:
    1. Bollinger Squeeze vient de se relâcher
    2. Volume supérieur à la moyenne
    3. AI > seuil
    4. Confluence > seuil
    5. Prix dans zone OB/FVG (optionnel mais bonus)

    Returns: (long, short, reason, quality_score 0-1)
    """
    squeeze_released, sq_dir = check_squeeze_release(row, prev_row)
    vol_surge  = bool(row.get("vol_score", 0) != 0)
    ai_long    = ai_proba > ai_thresh
    ai_short   = ai_proba < (1.0 - ai_thresh)
    cf_long    = conf_score >= max_conf * conf_pct
    cf_short   = conf_score <= -max_conf * conf_pct
    in_zone_b  = bool(row.get("in_ob_bull", False)) or bool(row.get("in_fvg_bull", False))
    in_zone_s  = bool(row.get("in_ob_bear", False)) or bool(row.get("in_fvg_bear", False))
    ema_bull, ema_bear = check_ema200_trend(row)

    if not squeeze_released or not vol_surge:
        return False, False, "", 0.0

    # Score de qualité
    quality = 0.0
    long_ok = short_ok = False

    if sq_dir in ("LONG", "NEUTRAL") and ai_long and cf_long and ema_bull:
        quality = 0.5
        quality += 0.2 if in_zone_b else 0.0
        quality += 0.15 if ai_proba > 0.72 else 0.0
        quality += 0.15 if conf_score >= max_conf * 0.45 else 0.0
        long_ok = True

    if sq_dir in ("SHORT", "NEUTRAL") and ai_short and cf_short and ema_bear:
        quality = 0.5
        quality += 0.2 if in_zone_s else 0.0
        quality += 0.15 if ai_proba < 0.28 else 0.0
        quality += 0.15 if conf_score <= -max_conf * 0.45 else 0.0
        short_ok = True

    reason = f"SQUEEZE_RELEASE|vol={vol_surge}|ai={ai_proba:.3f}|conf={conf_score:.2f}|q={quality:.2f}"
    return long_ok, short_ok, reason, quality


# ─────────────────────────────────────────────
# SETUP 2 — RETOUR EN ZONE (PULLBACK)
# ─────────────────────────────────────────────

def setup_zone_pullback(
    row:        dict,
    ai_proba:   float,
    conf_score: float,
    max_conf:   float,
    ai_thresh:  float = 0.65,
    conf_pct:   float = 0.30,
) -> tuple[bool, bool, str, float]:
    """
    Setup Zone Pullback:
    1. Prix dans une zone OB ou FVG
    2. EMA 200 haussière (long) ou baissière (short)
    3. Bougie de confirmation (engulfing ou pin bar)
    4. Volume supérieur à la moyenne
    5. AI > seuil ET Confluence > seuil

    Returns: (long, short, reason, quality_score 0-1)
    """
    in_zone_b    = bool(row.get("in_ob_bull", False)) or bool(row.get("in_fvg_bull", False))
    in_zone_s    = bool(row.get("in_ob_bear", False)) or bool(row.get("in_fvg_bear", False))
    bull_candle, bear_candle = check_candle_confirm(row)
    ema_bull, ema_bear = check_ema200_trend(row)
    vol_ok       = bool(row.get("vol_score", 0) != 0) or True  # volume optionnel
    vol_surge    = bool(row.get("vol_score", 0) == 1)
    is_chop      = bool(row.get("is_chop", False))

    # Bloquer en zone de chop
    if is_chop:
        return False, False, "", 0.0

    long_ok = short_ok = False
    quality = 0.0

    # Signal LONG
    if in_zone_b and bull_candle and ema_bull and ai_proba > ai_thresh and conf_score >= max_conf * conf_pct:
        quality = 0.40  # base
        quality += 0.20 if vol_surge else 0.0           # volume confirme
        quality += 0.15 if ai_proba > 0.72 else 0.0    # AI très confiante
        quality += 0.15 if conf_score >= max_conf * 0.45 else 0.0  # conf élevée
        quality += 0.10 if check_bos_choch(row)[0] else 0.0  # structure haussière
        long_ok = True

    # Signal SHORT
    if in_zone_s and bear_candle and ema_bear and ai_proba < (1.0 - ai_thresh) and conf_score <= -max_conf * conf_pct:
        quality = 0.40
        quality += 0.20 if vol_surge else 0.0
        quality += 0.15 if ai_proba < 0.28 else 0.0
        quality += 0.15 if conf_score <= -max_conf * 0.45 else 0.0
        quality += 0.10 if check_bos_choch(row)[1] else 0.0
        short_ok = True

    reason = (f"ZONE_PULLBACK|zone={'Bull' if in_zone_b else 'Bear' if in_zone_s else 'None'}"
              f"|candle={'Bull' if bull_candle else 'Bear' if bear_candle else 'None'}"
              f"|ema={'Bull' if ema_bull else 'Bear' if ema_bear else 'Neut'}"
              f"|ai={ai_proba:.3f}|conf={conf_score:.2f}|q={quality:.2f}")

    return long_ok, short_ok, reason, quality


# ─────────────────────────────────────────────
# SETUP 3 — COMBINÉ (RECOMMANDÉ)
# ─────────────────────────────────────────────

def setup_combined(
    row:        dict,
    prev_row:   dict,
    ai_proba:   float,
    conf_score: float,
    max_conf:   float,
    ai_thresh:  float = 0.65,
    conf_pct:   float = 0.30,
    min_quality: float = 0.45,
) -> tuple[bool, bool, str, float]:
    """
    Setup Combiné — le plus sélectif et le plus efficace:

    OBLIGATOIRE:
      ✅ Prix dans zone OB ou FVG
      ✅ EMA 200 dans la bonne direction
      ✅ AI > seuil ET Confluence > seuil
      ✅ PAS en zone de chop

    + AU MOINS UN de:
      a) Squeeze relâché + Volume surge       (+0.30 qualité)
      b) Bougie engulfing ou pin bar          (+0.25 qualité)
      c) BOS ou CHoCH dans la bonne direction (+0.20 qualité)
      d) CVD divergence                       (+0.15 qualité)

    Le trade n'est pris que si quality >= min_quality (défaut 0.55)

    Returns: (long, short, reason, quality_score 0-1)
    """
    # ── Conditions obligatoires ──
    in_zone_b = bool(row.get("in_ob_bull", False)) or bool(row.get("in_fvg_bull", False))
    in_zone_s = bool(row.get("in_ob_bear", False)) or bool(row.get("in_fvg_bear", False))
    is_chop   = bool(row.get("is_chop", False))
    ema_bull, ema_bear = check_ema200_trend(row)
    ai_long   = ai_proba > ai_thresh
    ai_short  = ai_proba < (1.0 - ai_thresh)
    cf_long   = conf_score >= max_conf * conf_pct
    cf_short  = conf_score <= -max_conf * conf_pct

    if is_chop:
        return False, False, "", 0.0

    # ── Conditions optionnelles (score de qualité) ──
    bull_candle, bear_candle = check_candle_confirm(row)
    bos_bull, bos_bear       = check_bos_choch(row)
    sq_released, sq_dir      = check_squeeze_release(row, prev_row)
    vol_surge                = bool(row.get("vol_score", 0) == 1)
    cvd_div_bull             = bool(row.get("cvd_div_bull", False))
    cvd_div_bear             = bool(row.get("cvd_div_bear", False))

    long_quality = short_quality = 0.0
    reasons_long = []; reasons_short = []

    if in_zone_b and ema_bull and ai_long and cf_long:
        long_quality = 0.40  # base augmentée (zone + EMA + AI + conf)

        # Bonus optionnels
        if sq_released and sq_dir in ("LONG", "NEUTRAL") and vol_surge:
            long_quality += 0.30
            reasons_long.append("SQUEEZE")
        if bull_candle:
            long_quality += 0.25
            reasons_long.append("CANDLE")
        if bos_bull:
            long_quality += 0.20
            reasons_long.append("BOS")
        if cvd_div_bull:
            long_quality += 0.15
            reasons_long.append("CVD_DIV")
        if vol_surge and "SQUEEZE" not in reasons_long:
            long_quality += 0.10
            reasons_long.append("VOL")
        # Bonus AI très confiante
        if ai_proba > 0.75:
            long_quality += 0.10
        # Bonus confluence élevée
        if conf_score >= max_conf * 0.50:
            long_quality += 0.10

    if in_zone_s and ema_bear and ai_short and cf_short:
        short_quality = 0.40

        if sq_released and sq_dir in ("SHORT", "NEUTRAL") and vol_surge:
            short_quality += 0.30
            reasons_short.append("SQUEEZE")
        if bear_candle:
            short_quality += 0.25
            reasons_short.append("CANDLE")
        if bos_bear:
            short_quality += 0.20
            reasons_short.append("BOS")
        if cvd_div_bear:
            short_quality += 0.15
            reasons_short.append("CVD_DIV")
        if vol_surge and "SQUEEZE" not in reasons_short:
            short_quality += 0.10
            reasons_short.append("VOL")
        if ai_proba < 0.25:
            short_quality += 0.10
        if conf_score <= -max_conf * 0.50:
            short_quality += 0.10

    # Appliquer le seuil de qualité minimum
    long_ok  = long_quality  >= min_quality and long_quality  > 0
    short_ok = short_quality >= min_quality and short_quality > 0

    # Pas les deux en même temps
    if long_ok and short_ok:
        if long_quality >= short_quality:
            short_ok = False
        else:
            long_ok = False

    quality = long_quality if long_ok else short_quality if short_ok else 0.0
    triggers = reasons_long if long_ok else reasons_short

    reason = (f"COMBINED|{'LONG' if long_ok else 'SHORT' if short_ok else 'NONE'}"
              f"|triggers={'+'.join(triggers) if triggers else 'BASE'}"
              f"|ai={ai_proba:.3f}|conf={conf_score:.2f}"
              f"|ema={'Bull' if ema_bull else 'Bear' if ema_bear else 'N'}"
              f"|q={quality:.2f}")

    return long_ok, short_ok, reason, quality


# ─────────────────────────────────────────────
# Règles dures (Niveau 1) — refus déterministe, gratuit, instantané
# ─────────────────────────────────────────────

def check_hard_rules(direction: str, conf_score: float, ich_score: float,
                      max_conf: float, min_conf_frac: float = 0.25) -> str | None:
    """
    Véto non négociable, appliqué avant tout signal LONG/SHORT.
    Corrige le cas observé où un LONG était ouvert avec confluence négative
    et Ichimoku sous le nuage (ex: GBPUSD, confluence -1.8): ces conditions
    ne devraient jamais permettre un LONG, quels que soient l'AI et les
    autres filtres.

    Returns:
        Une raison de refus (str) ou None si le trade passe les règles dures.
    """
    if direction == "LONG":
        if conf_score < 0:
            return "hard_conf_negative"
        if ich_score < 0:
            return "hard_ich_bearish"
    else:  # SHORT
        if conf_score > 0:
            return "hard_conf_positive"
        if ich_score > 0:
            return "hard_ich_bullish"
    if abs(conf_score) < max_conf * min_conf_frac:
        return "hard_conf_too_weak"
    return None


# ─────────────────────────────────────────────
# Point d'entrée principal
# ─────────────────────────────────────────────

def get_entry_signal(
    row:         dict,
    prev_row:    dict,
    ai_proba:    float,
    conf_score:  float,
    max_conf:    float,
    setup_mode:  str   = "combined",   # "squeeze", "pullback", "combined"
    ai_thresh:   float = 0.65,
    conf_pct:    float = 0.30,
    min_quality: float = 0.45,
) -> tuple[bool, bool, str, float]:
    """
    Point d'entrée unifié.

    Args:
        row:         Ligne du DataFrame indicateurs (barre actuelle)
        prev_row:    Barre précédente (pour détecter squeeze release)
        ai_proba:    Probabilité AI (0-1)
        conf_score:  Score de confluence
        max_conf:    Score max possible
        setup_mode:  "squeeze", "pullback", "combined"
        ai_thresh:   Seuil AI minimum pour LONG (ex: 0.65)
        conf_pct:    Fraction du score max pour confluence (ex: 0.30)
        min_quality: Score qualité minimum pour prendre le trade (ex: 0.55)

    Returns:
        (long_signal, short_signal, reason, quality_score)
    """
    if setup_mode == "squeeze":
        return setup_squeeze_release(
            row, prev_row, ai_proba, conf_score, max_conf, ai_thresh, conf_pct)

    elif setup_mode == "pullback":
        return setup_zone_pullback(
            row, ai_proba, conf_score, max_conf, ai_thresh, conf_pct)

    else:  # combined (défaut)
        return setup_combined(
            row, prev_row, ai_proba, conf_score, max_conf,
            ai_thresh, conf_pct, min_quality)
