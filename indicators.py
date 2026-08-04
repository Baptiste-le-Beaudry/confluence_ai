"""
indicators.py
Traduction exacte des indicateurs du Pine Script v4.0 en Python.
Chaque fonction retourne une Series pandas alignée sur le DataFrame OHLCV.
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# Fonctions mathématiques (identiques au Pine)
# ─────────────────────────────────────────────

def safe_tanh(x: float | np.ndarray) -> float | np.ndarray:
    x = np.clip(x, -20.0, 20.0)
    return np.tanh(x)

def sigmoid(z: float | np.ndarray) -> float | np.ndarray:
    z = np.clip(z, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-z))

def norm_ratio(val, ref):
    return safe_tanh(val / ref) if ref > 0 else 0.0


# ─────────────────────────────────────────────
# 1. VWAP journalier
# ─────────────────────────────────────────────

def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP réinitialisé chaque jour."""
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3
    df2 = df.copy()
    df2["date"] = df2.index.normalize()
    df2["pv"] = hlc3 * df2["volume"]
    df2["cum_pv"]  = df2.groupby("date")["pv"].cumsum()
    df2["cum_vol"] = df2.groupby("date")["volume"].cumsum()
    vwap = df2["cum_pv"] / df2["cum_vol"]
    return vwap.rename("vwap")


# ─────────────────────────────────────────────
# 2. FIBONACCI (session précédente)
# ─────────────────────────────────────────────

def compute_fib(df: pd.DataFrame):
    """Retourne fib0, fib0_5, fib618, fib786, fib_position."""
    daily = df["close"].resample("D").ohlc()
    daily_high = df["high"].resample("D").max()
    daily_low  = df["low"].resample("D").min()

    prev_high = daily_high.shift(1).reindex(df.index, method="ffill")
    prev_low  = daily_low.shift(1).reindex(df.index, method="ffill")

    fib_range   = prev_high - prev_low
    fib0        = prev_low
    fib0_5      = fib0 + fib_range * 0.5
    fib618      = fib0 + fib_range * 0.618
    fib786      = fib0 + fib_range * 0.786
    fib_pos     = np.where(fib_range != 0, (df["close"] - fib0) / fib_range, 0.5)

    return fib0, fib0_5, fib618, fib786, pd.Series(fib_pos, index=df.index)


def compute_fib_score(df: pd.DataFrame, rsi_len=14, fib_tol=0.3):
    fib0, fib0_5, fib618, fib786, fib_pos = compute_fib(df)
    rsi = compute_rsi(df["close"], rsi_len)

    def near(lvl):
        return (lvl != 0) & (np.abs(df["close"] - lvl) / lvl * 100 < fib_tol)

    fib_near    = near(fib0) | near(fib0_5) | near(fib618) | near(fib786)
    rsi_rising  = rsi > rsi.shift(1)
    rsi_falling = rsi < rsi.shift(1)

    score = np.where(fib_near & rsi_rising, 1, np.where(fib_near & rsi_falling, -1, 0))
    return pd.Series(score, index=df.index, name="fib_score"), fib_pos


# ─────────────────────────────────────────────
# 3. RSI
# ─────────────────────────────────────────────

def compute_rsi(close: pd.Series, length=14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=length - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=length - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename("rsi")


# ─────────────────────────────────────────────
# 4. ATR
# ─────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, length=14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.ewm(com=length - 1, adjust=False).mean().rename("atr")


# ─────────────────────────────────────────────
# 5. GANN 1x1
# ─────────────────────────────────────────────

def compute_gann(df: pd.DataFrame, atr: pd.Series, gann_len=20):
    swing_hi     = df["high"].rolling(gann_len).max()
    swing_lo     = df["low"].rolling(gann_len).min()
    hi_bars      = df["high"].rolling(gann_len).apply(lambda x: gann_len - 1 - np.argmax(x[::-1]), raw=True)
    lo_bars      = df["low"].rolling(gann_len).apply(lambda x: gann_len - 1 - np.argmin(x[::-1]), raw=True)
    use_low      = lo_bars <= hi_bars
    gann_1x1     = np.where(use_low,
                            swing_lo + atr * lo_bars.clip(lower=1),
                            swing_hi - atr * hi_bars.clip(lower=1))
    gann_1x1     = pd.Series(gann_1x1, index=df.index)
    gann_pos     = np.where(atr != 0, (df["close"] - gann_1x1) / atr, 0.0)
    score        = np.where(df["close"] > gann_1x1, 1, np.where(df["close"] < gann_1x1, -1, 0))
    return (pd.Series(score, index=df.index, name="gann_score"),
            pd.Series(gann_pos, index=df.index, name="gann_pos"))


# ─────────────────────────────────────────────
# 6. ICHIMOKU
# ─────────────────────────────────────────────

def compute_ichimoku(df: pd.DataFrame, tenkan=9, kijun=26, senkou_b=52):
    def mid(h, l, n): return (h.rolling(n).max() + l.rolling(n).min()) / 2
    tk  = mid(df["high"], df["low"], tenkan)
    kj  = mid(df["high"], df["low"], kijun)
    sA  = (tk + kj) / 2
    sB  = mid(df["high"], df["low"], senkou_b)
    cA  = sA.shift(kijun)
    cB  = sB.shift(kijun)

    above = (df["close"] > cA) & (df["close"] > cB)
    below = (df["close"] < cA) & (df["close"] < cB)
    chikou_bull = df["close"] > df["close"].shift(kijun)
    chikou_bear = df["close"] < df["close"].shift(kijun)

    score = np.where(above & (tk > kj) & chikou_bull,  1,
            np.where(below & (tk < kj) & chikou_bear, -1, 0))
    return pd.Series(score, index=df.index, name="ich_score")


# ─────────────────────────────────────────────
# 7. EMA
# ─────────────────────────────────────────────

def compute_ema_scores(df: pd.DataFrame, fast=20, mid=50, slow=200):
    e20  = df["close"].ewm(span=fast, adjust=False).mean()
    e50  = df["close"].ewm(span=mid,  adjust=False).mean()
    e200 = df["close"].ewm(span=slow, adjust=False).mean()

    trend = np.where(e200 != 0, (e20 - e200) / e200, 0.0)
    struct_bull = (e50 > e200) & (df["close"] > e200)
    struct_bear = (e50 < e200) & (df["close"] < e200)
    score_base  = np.where((e20 > e50) & (e50 > e200),  1,
                  np.where((e20 < e50) & (e50 < e200), -1, 0))

    return (e20, e50, e200,
            pd.Series(score_base, index=df.index, name="ema_score"),
            pd.Series(trend,      index=df.index, name="ema_trend"),
            pd.Series(struct_bull, index=df.index),
            pd.Series(struct_bear, index=df.index))


# ─────────────────────────────────────────────
# 8. ORDER BLOCKS
# ─────────────────────────────────────────────

def compute_ob(df: pd.DataFrame, atr: pd.Series, ob_impulse=1.5):
    impulse_up = (df["close"] - df["close"].shift(1)) > ob_impulse * atr
    impulse_dn = (df["close"].shift(1) - df["close"]) > ob_impulse * atr

    ob_bull_top = pd.Series(np.nan, index=df.index)
    ob_bull_bot = pd.Series(np.nan, index=df.index)
    ob_bear_top = pd.Series(np.nan, index=df.index)
    ob_bear_bot = pd.Series(np.nan, index=df.index)

    for i in range(16, len(df)):
        if impulse_up.iloc[i]:
            for j in range(i - 1, max(i - 16, 0), -1):
                if df["close"].iloc[j] < df["open"].iloc[j]:
                    ob_bull_top.iloc[i] = df["high"].iloc[j]
                    ob_bull_bot.iloc[i] = df["low"].iloc[j]
                    break
        if impulse_dn.iloc[i]:
            for j in range(i - 1, max(i - 16, 0), -1):
                if df["close"].iloc[j] > df["open"].iloc[j]:
                    ob_bear_top.iloc[i] = df["high"].iloc[j]
                    ob_bear_bot.iloc[i] = df["low"].iloc[j]
                    break

    # Propager la dernière zone valide
    ob_bull_top = ob_bull_top.ffill()
    ob_bull_bot = ob_bull_bot.ffill()
    ob_bear_top = ob_bear_top.ffill()
    ob_bear_bot = ob_bear_bot.ffill()

    in_ob_bull = (df["close"] <= ob_bull_top) & (df["close"] >= ob_bull_bot)
    in_ob_bear = (df["close"] <= ob_bear_top) & (df["close"] >= ob_bear_bot)
    score      = np.where(in_ob_bull, 1, np.where(in_ob_bear, -1, 0))

    return (pd.Series(score, index=df.index, name="ob_score"),
            in_ob_bull, in_ob_bear,
            ob_bull_top, ob_bull_bot,
            ob_bear_top, ob_bear_bot)


# ─────────────────────────────────────────────
# 9. FVG
# ─────────────────────────────────────────────

def compute_fvg(df: pd.DataFrame):
    fvg_bull = df["low"] > df["high"].shift(2)
    fvg_bear = df["high"] < df["low"].shift(2)

    fvg_bull_top = df["low"].where(fvg_bull).ffill()
    fvg_bull_bot = df["high"].shift(2).where(fvg_bull).ffill()
    fvg_bear_top = df["low"].shift(2).where(fvg_bear).ffill()
    fvg_bear_bot = df["high"].where(fvg_bear).ffill()

    in_fvg_bull = (df["close"] <= fvg_bull_top) & (df["close"] >= fvg_bull_bot)
    in_fvg_bear = (df["close"] <= fvg_bear_top) & (df["close"] >= fvg_bear_bot)
    score       = np.where(in_fvg_bull, 1, np.where(in_fvg_bear, -1, 0))

    return pd.Series(score, index=df.index, name="fvg_score"), in_fvg_bull, in_fvg_bear


# ─────────────────────────────────────────────
# 10. SUPPORT / RÉSISTANCE
# ─────────────────────────────────────────────

def compute_sr(df: pd.DataFrame, left=15, right=15, tol=0.15):
    ph = df["high"].rolling(left + right + 1, center=True).apply(
        lambda x: x[left] if x[left] == x.max() else np.nan, raw=True)
    pl = df["low"].rolling(left + right + 1, center=True).apply(
        lambda x: x[left] if x[left] == x.min() else np.nan, raw=True)

    ph_last = ph.ffill()
    pl_last = pl.ffill()

    dist_res = np.abs(df["close"] - ph_last) / ph_last * 100
    dist_sup = np.abs(df["close"] - pl_last) / pl_last * 100
    score    = np.where(dist_sup < tol, 1, np.where(dist_res < tol, -1, 0))
    return pd.Series(score, index=df.index, name="sr_score")


# ─────────────────────────────────────────────
# 11. VOLUME
# ─────────────────────────────────────────────

def compute_vol_score(df: pd.DataFrame, vol_len=20, vol_mult=1.5):
    vol_sma   = df["volume"].rolling(vol_len).mean()
    surge     = df["volume"] > vol_sma * vol_mult
    score     = np.where(surge & (df["close"] > df["open"]),  1,
                np.where(surge & (df["close"] < df["open"]), -1, 0))
    return pd.Series(score, index=df.index, name="vol_score")


# ─────────────────────────────────────────────
# 12. MARKET STRUCTURE BOS/CHoCH
# ─────────────────────────────────────────────

def compute_bos(df: pd.DataFrame, ema_struct_bull: pd.Series, ema_struct_bear: pd.Series, ms_len=10):
    ms_ph = df["high"].rolling(ms_len * 2 + 1, center=True).apply(
        lambda x: x[ms_len] if x[ms_len] == x.max() else np.nan, raw=True).ffill()
    ms_pl = df["low"].rolling(ms_len * 2 + 1, center=True).apply(
        lambda x: x[ms_len] if x[ms_len] == x.min() else np.nan, raw=True).ffill()

    bos_bull  = (df["close"] > ms_ph) & (df["close"].shift(1) <= ms_ph)
    bos_bear  = (df["close"] < ms_pl) & (df["close"].shift(1) >= ms_pl)
    choch_bull = bos_bull & ema_struct_bear
    choch_bear = bos_bear & ema_struct_bull

    score = np.where(choch_bull, 2, np.where(choch_bear, -2,
            np.where(bos_bull, 1, np.where(bos_bear, -1, 0))))
    return (pd.Series(score,      index=df.index, name="bos_score"),
            pd.Series(bos_bull,   index=df.index),
            pd.Series(bos_bear,   index=df.index),
            pd.Series(choch_bull, index=df.index),
            pd.Series(choch_bear, index=df.index))


# ─────────────────────────────────────────────
# 13. BOLLINGER / SQUEEZE / CHOP
# ─────────────────────────────────────────────

def compute_bb_squeeze(df: pd.DataFrame, atr: pd.Series, bb_len=20, bb_mult=2.0, kc_mult=1.5, chop_thresh=55.0):
    bb_mid   = df["close"].rolling(bb_len).mean()
    bb_std   = df["close"].rolling(bb_len).std()
    bb_upper = bb_mid + bb_mult * bb_std
    bb_lower = bb_mid - bb_mult * bb_std
    bb_pos   = np.where((bb_upper - bb_lower) > 0,
                        (df["close"] - bb_lower) / (bb_upper - bb_lower), 0.5)
    bb_pos   = pd.Series(bb_pos, index=df.index)

    kc_upper = bb_mid + kc_mult * atr
    kc_lower = bb_mid - kc_mult * atr
    squeeze  = (bb_upper < kc_upper) & (bb_lower > kc_lower)

    score = np.where(squeeze, 0, np.where(bb_pos > 0.8, 1, np.where(bb_pos < 0.2, -1, 0)))

    # Chop index
    high_max = df["high"].rolling(14).max()
    low_min  = df["low"].rolling(14).min()
    atr_sum  = (df["high"] - df["low"]).rolling(14).sum()
    chop_range = (high_max - low_min).replace(0, np.nan)
    chop_val = 100 * np.log10(atr_sum / chop_range) / np.log10(14)
    is_chop  = chop_val >= chop_thresh

    return (pd.Series(score,    index=df.index, name="bb_score"),
            pd.Series(squeeze,  index=df.index),
            pd.Series(is_chop,  index=df.index),
            bb_pos, chop_val)


# ─────────────────────────────────────────────
# 14. CVD
# ─────────────────────────────────────────────

def compute_cvd(df: pd.DataFrame, cvd_len=20):
    bull_vol = np.where(df["close"] >= df["open"], df["volume"], 0.0)
    bear_vol = np.where(df["close"] <  df["open"], df["volume"], 0.0)
    delta    = bull_vol - bear_vol
    cvd      = pd.Series(delta, index=df.index).cumsum()
    cvd_ma   = cvd.rolling(cvd_len).mean()

    cvd_rising   = cvd > cvd_ma
    price_rising = df["close"] > df["close"].rolling(cvd_len).mean()

    cvd_div_bull = cvd_rising & ~price_rising
    cvd_div_bear = ~cvd_rising & price_rising

    score = np.where(cvd_div_bull, 1, np.where(cvd_div_bear, -1,
            np.where(cvd_rising & price_rising, 1,
            np.where(~cvd_rising & ~price_rising, -1, 0))))

    return (pd.Series(score,        index=df.index, name="cvd_score"),
            pd.Series(cvd_div_bull, index=df.index),
            pd.Series(cvd_div_bear, index=df.index))


# ─────────────────────────────────────────────
# 15. MOMENTUM
# ─────────────────────────────────────────────

def compute_momentum(df: pd.DataFrame):
    mom_5   = df["close"] - df["close"].shift(5)
    mom_20  = df["close"] - df["close"].shift(20)
    mom_raw = mom_5 * 0.6 + mom_20 * 0.4
    score   = np.where(mom_raw > 0, 1, np.where(mom_raw < 0, -1, 0))
    return pd.Series(score, index=df.index, name="mom_score"), mom_raw


# ─────────────────────────────────────────────
# 16. CONFIRMATION BOUGIES
# ─────────────────────────────────────────────

def compute_candle_confirm(df: pd.DataFrame, pin_ratio=0.3):
    body   = (df["close"] - df["open"]).abs()
    rng    = df["high"] - df["low"]
    u_wick = df["high"] - df[["close", "open"]].max(axis=1)
    l_wick = df[["close", "open"]].min(axis=1) - df["low"]

    bull_engulf = ((df["close"] > df["open"]) &
                   (df["close"] > df["high"].shift(1)) &
                   (df["open"]  < df["close"].shift(1)) &
                   (body > body.shift(1)))
    bear_engulf = ((df["close"] < df["open"]) &
                   (df["close"] < df["low"].shift(1)) &
                   (df["open"]  > df["close"].shift(1)) &
                   (body > body.shift(1)))

    pin_bull = (rng > 0) & (l_wick / rng > 1.0 - pin_ratio) & (df["close"] > df["open"])
    pin_bear = (rng > 0) & (u_wick / rng > 1.0 - pin_ratio) & (df["close"] < df["open"])

    bull_confirm = bull_engulf | pin_bull
    bear_confirm = bear_engulf | pin_bear
    return bull_confirm, bear_confirm


# ─────────────────────────────────────────────
# CALCUL COMPLET DE TOUS LES INDICATEURS
# ─────────────────────────────────────────────

def compute_all(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """
    Calcule tous les indicateurs et retourne un DataFrame enrichi.

    params: dict optionnel pour surcharger les paramètres par défaut.
    """
    p = {
        "rsi_len": 14, "fib_tol": 0.3,
        "gann_len": 20, "tenkan": 9, "kijun": 26, "senkou_b": 52,
        "ema_fast": 20, "ema_mid": 50, "ema_slow": 200,
        "atr_len": 14, "ob_impulse": 1.5,
        "sr_left": 15, "sr_right": 15, "sr_tol": 0.15,
        "vol_len": 20, "vol_mult": 1.5,
        "ms_len": 10, "bb_len": 20, "bb_mult": 2.0,
        "kc_mult": 1.5, "chop_thresh": 55.0, "cvd_len": 20,
        "pin_ratio": 0.3,
        # Poids confluence
        "w_vwap": 1.0, "w_fib": 1.0, "w_gann": 1.0, "w_ich": 1.0,
        "w_ema": 1.0, "w_ob": 1.0, "w_fvg": 1.0, "w_sr": 1.0,
        "w_vol": 1.0, "w_bos": 1.2, "w_bb": 0.8, "w_cvd": 1.0, "w_mom": 0.8,
    }
    if params:
        p.update(params)

    out = df.copy()

    # ATR (utilisé partout)
    atr = compute_atr(df, p["atr_len"])
    out["atr"] = atr

    # 1. VWAP
    out["vwap"]        = compute_vwap(df)
    out["vwap_slope"]  = out["vwap"].diff()
    out["dist_vwap"]   = np.where(out["vwap"] != 0,
                                  (df["close"] - out["vwap"]) / out["vwap"] * 100, 0.0)
    out["vwap_score"]  = np.where((df["close"] > out["vwap"]) & (out["vwap_slope"] > 0),  1,
                         np.where((df["close"] < out["vwap"]) & (out["vwap_slope"] < 0), -1, 0))

    # 2. Fibonacci
    out["fib_score"], out["fib_pos"] = compute_fib_score(df, p["rsi_len"], p["fib_tol"])

    # 3. Gann
    out["gann_score"], out["gann_pos"] = compute_gann(df, atr, p["gann_len"])

    # 4. Ichimoku
    out["ich_score"] = compute_ichimoku(df, p["tenkan"], p["kijun"], p["senkou_b"])

    # 5. EMA
    e20, e50, e200, ema_score, ema_trend, struct_bull, struct_bear = compute_ema_scores(
        df, p["ema_fast"], p["ema_mid"], p["ema_slow"])
    out["ema20"]       = e20
    out["ema50"]       = e50
    out["ema200"]      = e200
    out["ema_score"]   = ema_score
    out["ema_trend"]   = ema_trend
    out["ema_struct_bull"] = struct_bull
    out["ema_struct_bear"] = struct_bear

    # 6. Order Blocks
    (out["ob_score"], out["in_ob_bull"], out["in_ob_bear"],
     out["ob_bull_top"], out["ob_bull_bot"],
     out["ob_bear_top"], out["ob_bear_bot"]) = compute_ob(df, atr, p["ob_impulse"])

    # 7. FVG
    out["fvg_score"], out["in_fvg_bull"], out["in_fvg_bear"] = compute_fvg(df)

    # 8. S/R
    out["sr_score"] = compute_sr(df, p["sr_left"], p["sr_right"], p["sr_tol"])

    # 9. Volume
    out["vol_score"] = compute_vol_score(df, p["vol_len"], p["vol_mult"])

    # 10. BOS / CHoCH
    (out["bos_score"], out["bos_bull"], out["bos_bear"],
     out["choch_bull"], out["choch_bear"]) = compute_bos(
        df, out["ema_struct_bull"], out["ema_struct_bear"], p["ms_len"])

    # 11. BB / Squeeze / Chop
    (out["bb_score"], out["squeeze"], out["is_chop"],
     out["bb_pos"], out["chop_val"]) = compute_bb_squeeze(
        df, atr, p["bb_len"], p["bb_mult"], p["kc_mult"], p["chop_thresh"])

    # 12. CVD
    out["cvd_score"], out["cvd_div_bull"], out["cvd_div_bear"] = compute_cvd(df, p["cvd_len"])

    # 13. Momentum
    out["mom_score"], out["mom_raw"] = compute_momentum(df)

    # 14. Confirmation bougies
    out["bull_confirm"], out["bear_confirm"] = compute_candle_confirm(df, p["pin_ratio"])

    # SCORE DE CONFLUENCE TOTAL
    w = p
    out["conf_score"] = (
        w["w_vwap"] * out["vwap_score"] +
        w["w_fib"]  * out["fib_score"]  +
        w["w_gann"] * out["gann_score"] +
        w["w_ich"]  * out["ich_score"]  +
        w["w_ema"]  * out["ema_score"]  +
        w["w_ob"]   * out["ob_score"]   +
        w["w_fvg"]  * out["fvg_score"]  +
        w["w_sr"]   * out["sr_score"]   +
        w["w_vol"]  * out["vol_score"]  +
        w["w_bos"]  * out["bos_score"]  +
        w["w_bb"]   * out["bb_score"]   +
        w["w_cvd"]  * out["cvd_score"]  +
        w["w_mom"]  * out["mom_score"]
    )

    out["max_conf"] = sum([w["w_vwap"], w["w_fib"], w["w_gann"], w["w_ich"],
                           w["w_ema"], w["w_ob"], w["w_fvg"], w["w_sr"],
                           w["w_vol"], w["w_bos"], w["w_bb"], w["w_cvd"], w["w_mom"]])

    return out
