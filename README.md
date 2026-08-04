# Confluence AI v3

Bot de trading MetaTrader 5 basé sur des indicateurs techniques, un moteur AI et des modes de backtest / walk-forward.

## Fonctionnalites

- Connexion MT5 via `run_bot.py` et `mt5_live_bot.py`
- Mode paper ou live
- Backtest et walk-forward
- Notifications Telegram
- Calcul d'indicateurs et features pour le modele

## Lancement

```powershell
python run_bot.py --paper --symbols BTCUSD XAUUSD
```

## Fichiers principaux

- `run_bot.py` : point d'entree du bot
- `mt5_live_bot.py` : logique du bot live/paper
- `mt5_broker.py` : couche MT5
- `backtest.py` : backtests
- `walk_forward.py` : validation walk-forward

## Configuration

Renseigner les variables MT5 et Telegram dans `.env` avant le lancement.
