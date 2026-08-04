"""
mt5_broker.py
Broker MetaTrader 5 pour le bot Confluence AI v4.0.

Compatible: BTC/USD, XAU/USD, EUR/USD, GBP/USD, USD/JPY, etc.
Nécessite: pip install MetaTrader5 (Windows uniquement)

IMPORTANT: MT5 doit être ouvert et connecté à votre compte avant de lancer le bot.
"""

import time
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from telegram_notify import notify_trade_open, notify_trade_close

logger = logging.getLogger("confluence_bot")

# Import conditionnel — MT5 est Windows only
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 non installé. Lancer: pip install MetaTrader5")


# ─────────────────────────────────────────────
# Configuration des symboles
# ─────────────────────────────────────────────

# Mapping nom court → symbole MT5 exact (selon votre broker)
SYMBOL_MAP = {
    # Crypto
    "BTCUSD":  "BTCUSD",       # ou "BTC/USD", "Bitcoin" selon votre broker
    "ETHUSD":  "ETHUSD",
    # Métaux
    "XAUUSD":  "XAUUSD",       # Or
    "XAGUSD":  "XAGUSD",       # Argent
    # Forex majeurs
    "EURUSD":  "EURUSD",
    "GBPUSD":  "GBPUSD",
    "USDJPY":  "USDJPY",
    "USDCHF":  "USDCHF",
    "AUDUSD":  "AUDUSD",
    "USDCAD":  "USDCAD",
    "NZDUSD":  "NZDUSD",
    # Forex croisés
    "EURGBP":  "EURGBP",
    "EURJPY":  "EURJPY",
    "GBPJPY":  "GBPJPY",
}

# Paramètres de risque par type d'actif
ASSET_CONFIG = {
    "BTCUSD": {
        "risk_pct":    0.01,    # 1% du capital par trade
        "sl_atr_mult": 2.0,
        "tp_atr_mult": 4.0,
        "min_lot":     0.01,
        "max_lot":     1.0,
        "comment":     "Confluence AI - BTC",
    },
    "XAUUSD": {
        "risk_pct":    0.01,
        "sl_atr_mult": 2.0,
        "tp_atr_mult": 4.0,
        "min_lot":     0.01,
        "max_lot":     5.0,
        "comment":     "Confluence AI - Gold",
    },
    "DEFAULT": {
        "risk_pct":    0.01,
        "sl_atr_mult": 1.5,
        "tp_atr_mult": 4.5,
        "min_lot":     0.01,
        "max_lot":     10.0,
        "comment":     "Confluence AI - Forex",
    },
}


@dataclass
class MT5Trade:
    """Enregistrement d'un trade MT5."""
    symbol:      str
    direction:   str          # "LONG" ou "SHORT"
    ticket:      int          # numéro de ticket MT5
    volume:      float
    entry_price: float
    sl:          float
    tp:          float
    risk_usd:    float = 0.0
    entry_time:  datetime = field(default_factory=datetime.now)
    exit_price:  float = 0.0
    exit_time:   datetime = None
    pnl_usd:     float = 0.0
    conf_score:  float = 0.0
    ai_proba:    float = 0.5


# ─────────────────────────────────────────────
# Broker MT5
# ─────────────────────────────────────────────

class MT5Broker:
    """
    Broker pour MetaTrader 5.

    Usage:
        broker = MT5Broker(account=123456, password="xxx", server="Broker-Server")
        broker.connect()
        broker.open_trade("BTCUSD", "LONG", sl=95000, tp=100000, risk_usd=100)
        broker.close_all("BTCUSD")
        broker.disconnect()
    """

    def __init__(
        self,
        account:  int   = 0,
        password: str   = "",
        server:   str   = "",
        magic:    int   = 20240101,   # identifiant unique du bot
        slippage: int   = 10,         # slippage max en points
    ):
        if not MT5_AVAILABLE:
            raise ImportError(
                "MetaTrader5 non disponible.\n"
                "Installer avec: pip install MetaTrader5\n"
                "Fonctionne uniquement sur Windows avec MT5 installé."
            )
        self.account  = account
        self.password = password
        self.server   = server
        self.magic    = magic
        self.slippage = slippage
        self.trades: list[MT5Trade] = []
        self.realized_pnl: float = 0.0
        self._connected = False

    # ─────────────────────────────────────────
    # Connexion
    # ─────────────────────────────────────────

    def connect(self) -> bool:
        """
        Initialise la connexion à MT5.
        MT5 doit être ouvert sur votre PC.
        """
        if not mt5.initialize():
            logger.error(f"MT5 initialize() échoué: {mt5.last_error()}")
            return False

        # Login si les credentials sont fournis
        if self.account and self.password:
            ok = mt5.login(self.account, password=self.password, server=self.server)
            if not ok:
                logger.error(f"MT5 login() échoué: {mt5.last_error()}")
                mt5.shutdown()
                return False

        info = mt5.account_info()
        if info is None:
            logger.error("Impossible de récupérer les infos du compte MT5")
            return False

        self._connected = True
        logger.info(f"MT5 connecté: compte={info.login}, broker={info.company}, "
                    f"balance=${info.balance:.2f}, levier=1:{info.leverage}")
        return True

    def disconnect(self):
        """Ferme la connexion MT5."""
        mt5.shutdown()
        self._connected = False
        logger.info("MT5 déconnecté")

    def is_connected(self) -> bool:
        return self._connected and mt5.terminal_info() is not None

    # ─────────────────────────────────────────
    # Infos compte
    # ─────────────────────────────────────────

    def get_balance(self) -> float:
        """Retourne le solde du compte en USD."""
        info = mt5.account_info()
        return float(info.balance) if info else 0.0

    def get_equity(self) -> float:
        """Retourne l'équité (balance + P&L non réalisé)."""
        info = mt5.account_info()
        return float(info.equity) if info else 0.0

    def get_free_margin(self) -> float:
        """Retourne la marge libre."""
        info = mt5.account_info()
        return float(info.margin_free) if info else 0.0

    def get_open_positions(self, symbol: str = None) -> list:
        """
        Retourne les positions ouvertes par ce bot (magic number).

        Args:
            symbol: Filtrer par symbole (optionnel)
        """
        if symbol:
            positions = mt5.positions_get(symbol=self._resolve_symbol(symbol))
        else:
            positions = mt5.positions_get()

        if positions is None:
            return []

        # Filtrer par magic number (seulement nos trades)
        return [p for p in positions if p.magic == self.magic]

    def has_open_position(self, symbol: str) -> bool:
        """Vérifie si une position est déjà ouverte sur ce symbole."""
        return len(self.get_open_positions(symbol)) > 0

    def get_position_direction(self, symbol: str) -> Optional[str]:
        """Retourne la direction de la position ouverte (LONG/SHORT/None)."""
        positions = self.get_open_positions(symbol)
        if not positions:
            return None
        # mt5.ORDER_TYPE_BUY = 0, mt5.ORDER_TYPE_SELL = 1
        return "LONG" if positions[0].type == 0 else "SHORT"

    # ─────────────────────────────────────────
    # Prix en temps réel
    # ─────────────────────────────────────────

    def get_price(self, symbol: str) -> tuple[float, float]:
        """
        Retourne (bid, ask) pour un symbole.

        Returns:
            (bid, ask) — bid pour vendre, ask pour acheter
        """
        sym = self._resolve_symbol(symbol)
        tick = mt5.symbol_info_tick(sym)
        if tick is None:
            raise ValueError(f"Impossible de récupérer le prix de {sym}: {mt5.last_error()}")
        return float(tick.bid), float(tick.ask)

    def get_atr(self, symbol: str, timeframe_mt5: int, period: int = 14) -> float:
        """
        Calcule l'ATR depuis les barres MT5.

        Args:
            symbol:       Symbole MT5
            timeframe_mt5: ex: mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1
            period:       Période ATR
        """
        import numpy as np
        sym   = self._resolve_symbol(symbol)
        rates = mt5.copy_rates_from_pos(sym, timeframe_mt5, 0, period + 5)
        if rates is None or len(rates) < period:
            return 0.0

        highs  = rates["high"]
        lows   = rates["low"]
        closes = rates["close"]
        trs    = []
        for i in range(1, len(rates)):
            tr = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i-1]),
                     abs(lows[i]  - closes[i-1]))
            trs.append(tr)

        return float(np.mean(trs[-period:]))

    def get_ohlcv(self, symbol: str, timeframe_mt5: int, n_bars: int = 500):
        """
        Récupère les barres OHLCV depuis MT5 sous forme de DataFrame.

        Args:
            symbol:        Symbole
            timeframe_mt5: ex mt5.TIMEFRAME_M15
            n_bars:        Nombre de barres

        Returns:
            pandas DataFrame avec colonnes open, high, low, close, volume
        """
        import pandas as pd
        sym   = self._resolve_symbol(symbol)
        rates = mt5.copy_rates_from_pos(sym, timeframe_mt5, 0, n_bars)
        if rates is None:
            raise ValueError(f"Impossible de récupérer les barres de {sym}")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time")
        df = df.rename(columns={
            "open": "open", "high": "high", "low": "low",
            "close": "close", "tick_volume": "volume"
        })
        return df[["open", "high", "low", "close", "volume"]]

    # ─────────────────────────────────────────
    # Calcul de la taille de position
    # ─────────────────────────────────────────

    def calculate_lot_size(
        self,
        symbol:   str,
        sl_price: float,
        entry_price: float,
        risk_usd: float,
    ) -> float:
        """
        Calcule la taille de lot pour risquer exactement risk_usd.

        Formule: lot = risk_usd / (|entry - sl| × contract_size × tick_value)

        Args:
            symbol:      Symbole MT5
            sl_price:    Niveau du stop loss
            entry_price: Prix d'entrée
            risk_usd:    Montant à risquer en USD

        Returns:
            Taille de lot normalisée selon les règles du broker
        """
        sym  = self._resolve_symbol(symbol)
        info = mt5.symbol_info(sym)
        if info is None:
            logger.error(f"Symbole inconnu: {sym}")
            return 0.0

        sl_distance = abs(entry_price - sl_price)
        if sl_distance <= 0:
            return info.volume_min

        # Valeur d'un pip / tick
        tick_size  = info.trade_tick_size
        tick_value = info.trade_tick_value  # valeur d'un tick pour 1 lot

        if tick_size <= 0 or tick_value <= 0:
            logger.warning(f"Impossible de calculer tick_value pour {sym}, utilisation lot min")
            return info.volume_min

        # Nombre de ticks dans le SL
        sl_ticks = sl_distance / tick_size

        # Lot correspondant au risque désiré
        lot = risk_usd / (sl_ticks * tick_value)

        # Normaliser selon les règles du broker
        lot = max(lot, info.volume_min)
        lot = min(lot, info.volume_max)

        # Arrondir au pas de lot
        step = info.volume_step
        lot  = round(lot / step) * step
        lot  = round(lot, 2)

        return lot

    # ─────────────────────────────────────────
    # Ouverture de trade
    # ─────────────────────────────────────────

    def open_trade(
        self,
        symbol:      str,
        direction:   str,       # "LONG" ou "SHORT"
        sl:          float,     # prix stop loss
        tp:          float,     # prix take profit
        risk_usd:    float,     # montant à risquer en USD
        comment:     str = "",
        conf_score:  float = 0.0,
        ai_proba:    float = 0.5,
    ) -> Optional[MT5Trade]:
        """
        Ouvre un trade sur MT5.

        Args:
            symbol:    ex "BTCUSD", "XAUUSD", "EURUSD"
            direction: "LONG" ou "SHORT"
            sl:        Prix du stop loss
            tp:        Prix du take profit
            risk_usd:  Montant en USD à risquer (détermine le lot size)

        Returns:
            MT5Trade ou None si échec
        """
        sym = self._resolve_symbol(symbol)

        # Vérifier que le symbole est dispo
        if not mt5.symbol_select(sym, True):
            logger.error(f"Symbole non disponible: {sym}")
            return None

        # Prix d'entrée
        bid, ask = self.get_price(symbol)
        if direction == "LONG":
            entry_price = ask
            order_type  = mt5.ORDER_TYPE_BUY
        else:
            entry_price = bid
            order_type  = mt5.ORDER_TYPE_SELL

        # Calcul du lot
        lot = self.calculate_lot_size(sym, sl, entry_price, risk_usd)
        if lot <= 0:
            logger.error(f"Lot size invalide pour {sym}")
            return None

        # Construction de la requête
        cfg     = ASSET_CONFIG.get(symbol.upper(), ASSET_CONFIG["DEFAULT"])
        comment = comment or cfg["comment"]

        request = {
            "action":      mt5.TRADE_ACTION_DEAL,
            "symbol":      sym,
            "volume":      lot,
            "type":        order_type,
            "price":       entry_price,
            "sl":          round(sl, self._get_digits(sym)),
            "tp":          round(tp, self._get_digits(sym)),
            "deviation":   self.slippage,
            "magic":       self.magic,
            "comment":     comment[:31],   # MT5 limite à 31 caractères
            "type_time":   mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        logger.info(f"Envoi ordre: {direction} {lot} {sym} @ {entry_price:.5f} | "
                    f"SL={sl:.5f} TP={tp:.5f}")

        result = mt5.order_send(request)

        if result is None:
            logger.error(f"order_send() retourné None: {mt5.last_error()}")
            return None

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Ordre refusé: retcode={result.retcode} "
                         f"({self._retcode_str(result.retcode)}) | "
                         f"comment='{result.comment}'")
            return None

        trade = MT5Trade(
            symbol      = symbol,
            direction   = direction,
            ticket      = result.order,
            volume      = lot,
            entry_price = result.price,
            sl          = sl,
            tp          = tp,
            risk_usd    = risk_usd,
            entry_time  = datetime.now(),
            conf_score  = conf_score,
            ai_proba    = ai_proba,
        )
        self.trades.append(trade)

        logger.info(f"Ordre exécuté: ticket=#{result.order} | "
                    f"{direction} {lot} {sym} @ {result.price:.5f}")

        notify_trade_open(
            symbol=symbol,
            direction=direction,
            price=result.price,
            sl=sl,
            tp=tp,
            lot=lot,
            risk_usd=risk_usd,
            ai_proba=ai_proba,
            conf_score=conf_score,
        )
        return trade

    # ─────────────────────────────────────────
    # Fermeture de trade
    # ─────────────────────────────────────────

    def close_position(self, symbol: str, reason: str = "") -> bool:
        """
        Ferme la position ouverte sur un symbole.

        Args:
            symbol: Symbole à fermer
            reason: Raison de la fermeture (pour les logs)

        Returns:
            True si succès
        """
        sym       = self._resolve_symbol(symbol)
        positions = self.get_open_positions(symbol)

        if not positions:
            logger.info(f"Aucune position ouverte sur {sym}")
            return True

        success = True
        for pos in positions:
            bid, ask = self.get_price(symbol)

            # Pour fermer: si on est LONG on vend, si SHORT on achète
            if pos.type == 0:  # BUY → fermer avec SELL
                close_type  = mt5.ORDER_TYPE_SELL
                close_price = bid
            else:              # SELL → fermer avec BUY
                close_type  = mt5.ORDER_TYPE_BUY
                close_price = ask

            request = {
                "action":    mt5.TRADE_ACTION_DEAL,
                "symbol":    sym,
                "volume":    pos.volume,
                "type":      close_type,
                "position":  pos.ticket,
                "price":     close_price,
                "deviation": self.slippage,
                "magic":     self.magic,
                "comment":   f"Close {reason}"[:31],
                "type_time":    mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)

            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Fermeture échouée pour ticket #{pos.ticket}: "
                             f"{mt5.last_error()}")
                success = False
                continue

            pnl = pos.profit
            self.realized_pnl += pnl
            trade = next(
                (
                    t for t in reversed(self.trades)
                    if t.symbol.upper() == symbol.upper() and t.exit_time is None
                ),
                None,
            )
            pnl_pct = (pnl / trade.risk_usd * 100.0) if trade and trade.risk_usd else 0.0
            if trade:
                trade.exit_price = close_price
                trade.exit_time = datetime.now()
                trade.pnl_usd = pnl

            notify_trade_close(
                symbol=symbol,
                direction="LONG" if pos.type == 0 else "SHORT",
                entry=pos.price_open,
                exit_price=close_price,
                pnl_usd=pnl,
                pnl_pct=pnl_pct,
                reason=reason,
            )
            logger.info(f"Position fermée ({reason}): ticket=#{pos.ticket} | "
                        f"{sym} | P&L={pnl:+.2f}$")

        return success

    def close_all(self, symbol: str = None) -> bool:
        """Ferme toutes les positions du bot (ou celles d'un symbole)."""
        positions = self.get_open_positions(symbol)
        if not positions:
            return True

        success = True
        for pos in positions:
            sym = pos.symbol
            result = self.close_position(sym, reason="CLOSE_ALL")
            success = success and result

        return success

    def modify_sl_tp(self, symbol: str, sl: float, tp: float) -> bool:
        """
        Modifie le SL et TP d'une position ouverte.

        Utile pour le trailing stop ou pour déplacer le SL en breakeven.
        """
        sym       = self._resolve_symbol(symbol)
        positions = self.get_open_positions(symbol)

        if not positions:
            return False

        digits = self._get_digits(sym)
        for pos in positions:
            request = {
                "action":   mt5.TRADE_ACTION_SLTP,
                "symbol":   sym,
                "position": pos.ticket,
                "sl":       round(sl, digits),
                "tp":       round(tp, digits),
                "magic":    self.magic,
            }
            result = mt5.order_send(request)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Modification SL/TP échouée: {mt5.last_error()}")
                return False

        logger.info(f"SL/TP modifiés: {sym} | SL={sl:.5f} TP={tp:.5f}")
        return True

    # ─────────────────────────────────────────
    # Portfolio
    # ─────────────────────────────────────────

    def portfolio_summary(self) -> str:
        info      = mt5.account_info()
        positions = self.get_open_positions()

        lines = [
            "=== MT5 Portfolio ===",
            f"  Balance    : ${info.balance:>12,.2f}" if info else "  Balance    : N/A",
            f"  Equity     : ${info.equity:>12,.2f}"  if info else "  Equity     : N/A",
            f"  Free Margin: ${info.margin_free:>12,.2f}" if info else "",
            f"  P&L réalisé: ${self.realized_pnl:>+12,.2f}",
            f"  Trades bot : {len(self.trades):>12}",
        ]

        if positions:
            lines.append(f"\n  Positions ouvertes ({len(positions)}):")
            for pos in positions:
                direction = "LONG" if pos.type == 0 else "SHORT"
                lines.append(
                    f"    {pos.symbol:<10} {direction:<5} "
                    f"lot={pos.volume:.2f} "
                    f"@ {pos.price_open:.5f} "
                    f"P&L={pos.profit:+.2f}$"
                )

        return "\n".join(lines)

    # ─────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────

    def _resolve_symbol(self, symbol: str) -> str:
        """Résout le nom du symbole selon le broker."""
        sym = SYMBOL_MAP.get(symbol.upper(), symbol)
        return sym

    def _get_digits(self, symbol: str) -> int:
        """Retourne le nombre de décimales du symbole."""
        info = mt5.symbol_info(symbol)
        return info.digits if info else 5

    @staticmethod
    def _retcode_str(retcode: int) -> str:
        """Traduit un retcode MT5 en texte lisible."""
        codes = {
            10004: "REQUOTE",
            10006: "REJECTED",
            10007: "CANCELED",
            10008: "PLACED",
            10009: "DONE",
            10010: "DONE_PARTIAL",
            10011: "ERROR",
            10012: "TIMEOUT",
            10013: "INVALID",
            10014: "INVALID_VOLUME",
            10015: "INVALID_PRICE",
            10016: "INVALID_STOPS",
            10017: "TRADE_DISABLED",
            10018: "MARKET_CLOSED",
            10019: "NO_MONEY",
            10020: "PRICE_CHANGED",
            10021: "PRICE_OFF",
            10022: "INVALID_EXPIRATION",
            10025: "TOO_MANY_REQUESTS",
        }
        return codes.get(retcode, f"CODE_{retcode}")
