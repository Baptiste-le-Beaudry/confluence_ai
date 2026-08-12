@echo off
title Confluence AI Bot
cd C:\confluence_ai_v3
:restart
python run_bot.py --yes
timeout /t 15
goto restart
