"""
dashboard.py
Dashboard web pour le bot Confluence AI — mis à jour toutes les 10 minutes.
Affiche: probabilité AI, trades, P&L, statut des symboles.

Lancement:
    python dashboard.py          # port 5000
    python dashboard.py --port 8080

Accès depuis votre navigateur:
    http://VOTRE_IP_AWS:5000

Sur AWS — ouvrir le port dans Security Groups:
    EC2 → Security Groups → Inbound Rules → Add Rule
    Type: Custom TCP | Port: 5000 | Source: Anywhere
"""

import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from collections import deque, defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler
import argparse

# ─────────────────────────────────────────────
# Stockage des données du bot
# ─────────────────────────────────────────────

class BotData:
    """Données partagées entre le bot et le dashboard."""
    def __init__(self):
        self.lock          = threading.Lock()
        self.symbols_state = {}   # symbol → dernière barre
        self.trades        = deque(maxlen=200)
        self.equity_points = deque(maxlen=500)
        self.capital       = 10_000.0
        self.start_time    = datetime.now(tz=timezone.utc)
        self.last_update   = datetime.now(tz=timezone.utc)
        self.n_iterations  = 0
        # Diagnostic: historique des raisons de blocage par symbole
        self.diag_history  = defaultdict(lambda: deque(maxlen=500))

    def update_symbol(self, symbol: str, data: dict):
        with self.lock:
            self.symbols_state[symbol] = {**data, "ts": datetime.now(tz=timezone.utc).isoformat()}
            self.last_update = datetime.now(tz=timezone.utc)
            self.n_iterations += 1

    def add_trade(self, trade: dict):
        with self.lock:
            self.trades.appendleft(trade)

    def add_equity(self, equity: float):
        with self.lock:
            self.equity_points.append({
                "ts":     datetime.now(tz=timezone.utc).isoformat(),
                "equity": round(equity, 2)
            })

    def add_diag(self, symbol: str, record: dict):
        """Enregistre pourquoi un trade a (ou n'a pas) été pris à cette itération."""
        with self.lock:
            self.diag_history[symbol].append({
                **record,
                "ts": datetime.now(tz=timezone.utc).isoformat(),
            })

    def to_json(self) -> str:
        with self.lock:
            uptime = str(datetime.now(tz=timezone.utc) - self.start_time).split(".")[0]
            return json.dumps({
                "symbols":     self.symbols_state,
                "trades":      list(self.trades),
                "equity":      list(self.equity_points),
                "capital":     round(self.capital, 2),
                "start_time":  self.start_time.isoformat(),
                "last_update": self.last_update.isoformat(),
                "n_iterations":self.n_iterations,
                "uptime":      uptime,
                "diag":        {sym: list(hist) for sym, hist in self.diag_history.items()},
            }, default=str)

# Instance globale
bot_data = BotData()


# ─────────────────────────────────────────────
# Serveur HTTP
# ─────────────────────────────────────────────

HTML_PAGE = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="refresh" content="600">
<title>Confluence AI Bot — Dashboard</title>
<style>
  :root {
    --bg:      #0d1117;
    --card:    #161b22;
    --border:  #30363d;
    --green:   #00ff88;
    --red:     #ff4466;
    --blue:    #4488ff;
    --yellow:  #ffcc00;
    --gray:    #8b949e;
    --white:   #e6edf3;
    --purple:  #bb86fc;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--white);
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-size: 14px;
    min-height: 100vh;
  }

  /* Header */
  .header {
    background: var(--card);
    border-bottom: 1px solid var(--border);
    padding: 16px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
  }
  .header h1 {
    font-size: 18px;
    font-weight: 700;
    color: var(--green);
    letter-spacing: -0.3px;
  }
  .header .meta {
    font-size: 12px;
    color: var(--gray);
    text-align: right;
    line-height: 1.6;
  }
  .status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--green);
    animation: pulse 2s infinite;
    margin-right: 6px;
  }
  @keyframes pulse {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.4; }
  }

  /* Layout */
  .container { max-width: 1400px; margin: 0 auto; padding: 20px 24px; }
  .grid-4 { display: grid; grid-template-columns: repeat(4,1fr); gap: 16px; margin-bottom: 20px; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }
  @media (max-width: 900px) {
    .grid-4 { grid-template-columns: repeat(2,1fr); }
    .grid-2 { grid-template-columns: 1fr; }
  }

  /* Cards */
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
  }
  .card-title {
    font-size: 11px;
    font-weight: 600;
    color: var(--gray);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 10px;
  }
  .card-value {
    font-size: 28px;
    font-weight: 700;
    line-height: 1;
  }
  .card-sub {
    font-size: 12px;
    color: var(--gray);
    margin-top: 4px;
  }
  .green  { color: var(--green); }
  .red    { color: var(--red); }
  .blue   { color: var(--blue); }
  .yellow { color: var(--yellow); }
  .gray   { color: var(--gray); }

  /* Symbols grid */
  .symbols-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
  }
  .symbol-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
    position: relative;
    overflow: hidden;
  }
  .symbol-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--border);
  }
  .symbol-card.bull::before { background: var(--green); }
  .symbol-card.bear::before { background: var(--red); }
  .symbol-card.chop::before { background: var(--gray); }

  .symbol-name { font-size: 13px; font-weight: 700; margin-bottom: 8px; }
  .symbol-price { font-size: 20px; font-weight: 600; margin-bottom: 8px; }
  .symbol-row {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: var(--gray);
    margin-bottom: 4px;
  }
  .symbol-row span:last-child { color: var(--white); font-weight: 500; }

  /* AI Proba bar */
  .proba-bar-wrap {
    margin-top: 10px;
    height: 6px;
    background: #21262d;
    border-radius: 3px;
    overflow: hidden;
    position: relative;
  }
  .proba-bar {
    height: 100%;
    border-radius: 3px;
    transition: width 0.5s ease;
  }
  .proba-label {
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: var(--gray);
    margin-top: 3px;
  }

  /* Badge */
  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 20px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.5px;
  }
  .badge-long  { background: rgba(0,255,136,0.15); color: var(--green); border: 1px solid rgba(0,255,136,0.3); }
  .badge-short { background: rgba(255,68,102,0.15); color: var(--red);   border: 1px solid rgba(255,68,102,0.3); }
  .badge-flat  { background: rgba(139,148,158,0.15); color: var(--gray); border: 1px solid rgba(139,148,158,0.3); }
  .badge-chop  { background: rgba(255,204,0,0.15);  color: var(--yellow); border: 1px solid rgba(255,204,0,0.3); }

  /* Equity chart */
  .chart-wrap { width: 100%; height: 200px; position: relative; }
  svg.chart { width: 100%; height: 100%; }

  /* Trades table */
  .trades-table { width: 100%; border-collapse: collapse; }
  .trades-table th {
    font-size: 11px;
    color: var(--gray);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 8px 12px;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }
  .trades-table td {
    padding: 10px 12px;
    border-bottom: 1px solid rgba(48,54,61,0.5);
    font-size: 12px;
  }
  .trades-table tr:last-child td { border-bottom: none; }
  .trades-table tr:hover td { background: rgba(255,255,255,0.02); }

  /* Refresh bar */
  .refresh-bar {
    height: 3px;
    background: rgba(68,136,255,0.2);
    border-radius: 2px;
    margin-top: 20px;
    overflow: hidden;
  }
  .refresh-bar-inner {
    height: 100%;
    background: var(--blue);
    animation: countdown 600s linear;
    transform-origin: left;
  }
  @keyframes countdown {
    from { width: 100%; }
    to   { width: 0%; }
  }
  .section-title {
    font-size: 13px;
    font-weight: 600;
    color: var(--gray);
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }
  .empty-state {
    text-align: center;
    padding: 30px;
    color: var(--gray);
    font-size: 13px;
  }
</style>
</head>
<body>

<div class="header">
  <h1><span class="status-dot"></span>Confluence AI Bot</h1>
  <div class="meta" id="meta-info">Chargement...</div>
</div>

<div class="container">

  <!-- KPIs -->
  <div class="grid-4" id="kpis">
    <div class="card">
      <div class="card-title">Capital</div>
      <div class="card-value blue" id="kpi-capital">—</div>
      <div class="card-sub" id="kpi-return">—</div>
    </div>
    <div class="card">
      <div class="card-title">Trades totaux</div>
      <div class="card-value" id="kpi-trades">—</div>
      <div class="card-sub" id="kpi-wr">Win Rate —</div>
    </div>
    <div class="card">
      <div class="card-title">P&L total</div>
      <div class="card-value" id="kpi-pnl">—</div>
      <div class="card-sub" id="kpi-expect">Espérance —</div>
    </div>
    <div class="card">
      <div class="card-title">Uptime</div>
      <div class="card-value green" id="kpi-uptime">—</div>
      <div class="card-sub" id="kpi-iter">— itérations</div>
    </div>
  </div>

  <!-- Symboles -->
  <div class="section-title">Symboles surveillés</div>
  <div class="symbols-grid" id="symbols-grid">
    <div class="empty-state">Chargement des données...</div>
  </div>

  <!-- Graphique équité + Trades récents -->
  <div class="grid-2">
    <div class="card">
      <div class="section-title">Courbe d'équité</div>
      <div class="chart-wrap" id="equity-chart-wrap">
        <svg class="chart" id="equity-svg" viewBox="0 0 600 200" preserveAspectRatio="none">
          <text x="300" y="110" text-anchor="middle" fill="#8b949e" font-size="13">
            En attente de données...
          </text>
        </svg>
      </div>
    </div>
    <div class="card">
      <div class="section-title">Trades récents</div>
      <div id="trades-wrap">
        <div class="empty-state">Aucun trade pour le moment</div>
      </div>
    </div>
  </div>

  <!-- Diagnostic: pourquoi pas de trade -->
  <div class="section-title">Diagnostic — Pourquoi pas de trade ?</div>
  <div class="grid-2">
    <div class="card">
      <div class="card-title" style="margin-bottom:14px">Raisons de blocage (toutes itérations, tous symboles)</div>
      <div id="diag-bars"><div class="empty-state">En attente de données...</div></div>
    </div>
    <div class="card">
      <div class="card-title" style="margin-bottom:14px">Détail par symbole</div>
      <div id="diag-table"><div class="empty-state">En attente de données...</div></div>
    </div>
  </div>
  <div class="symbols-grid" id="diag-sparklines"></div>

  <!-- Barre de refresh -->
  <div class="refresh-bar"><div class="refresh-bar-inner"></div></div>
  <div style="text-align:center; font-size:11px; color:var(--gray); margin-top:8px;">
    Mise à jour automatique toutes les 10 minutes
  </div>

</div>

<script>
async function loadData() {
  try {
    const resp = await fetch('/api/data');
    const data = await resp.json();
    renderDashboard(data);
  } catch(e) {
    console.error('Erreur chargement:', e);
  }
}

function fmt(n, dec=2) {
  return Number(n).toLocaleString('fr-CA', {minimumFractionDigits:dec, maximumFractionDigits:dec});
}

function renderDashboard(data) {
  // Meta
  const lastUp = new Date(data.last_update);
  document.getElementById('meta-info').innerHTML =
    `Dernière mise à jour: ${lastUp.toLocaleTimeString('fr-CA')}<br>` +
    `Uptime: ${data.uptime} | ${data.n_iterations} itérations`;

  // KPIs
  const capital = data.capital || 10000;
  const initCap = 10000;
  const ret = ((capital - initCap) / initCap * 100);
  document.getElementById('kpi-capital').textContent = '$' + fmt(capital);
  document.getElementById('kpi-return').textContent =
    (ret >= 0 ? '+' : '') + fmt(ret) + '% depuis le début';
  document.getElementById('kpi-return').className = 'card-sub ' + (ret>=0?'green':'red');

  const trades = data.trades || [];
  const wins   = trades.filter(t => t.pnl_usd > 0);
  const wr     = trades.length > 0 ? (wins.length / trades.length * 100) : 0;
  const pnl    = trades.reduce((s,t) => s + (t.pnl_usd||0), 0);
  const expect = trades.length > 0 ? pnl / trades.length : 0;

  document.getElementById('kpi-trades').textContent = trades.length;
  document.getElementById('kpi-wr').textContent = 'Win Rate ' + fmt(wr,1) + '%';
  document.getElementById('kpi-pnl').textContent = (pnl>=0?'+':'') + '$' + fmt(pnl);
  document.getElementById('kpi-pnl').className = 'card-value ' + (pnl>=0?'green':'red');
  document.getElementById('kpi-expect').textContent = 'Espérance $' + fmt(expect);
  document.getElementById('kpi-uptime').textContent = data.uptime || '—';
  document.getElementById('kpi-iter').textContent = (data.n_iterations||0) + ' itérations';

  // Symboles
  const syms = data.symbols || {};
  const grid = document.getElementById('symbols-grid');
  if (Object.keys(syms).length === 0) {
    grid.innerHTML = '<div class="empty-state">En attente des données des symboles...</div>';
  } else {
    grid.innerHTML = Object.entries(syms).map(([sym, s]) => {
      const proba    = parseFloat(s.ai_proba || 0.5);
      const conf     = parseFloat(s.conf_score || 0);
      const pos      = s.position || 'FLAT';
      const isChop   = s.is_chop;
      const zone     = s.zone || 'None';
      const tl       = parseFloat(s.thresh_long  || 0.61);
      const ts       = parseFloat(s.thresh_short || 0.39);
      const price    = parseFloat(s.close || 0);

      let cardClass  = 'symbol-card';
      let posBadge   = '<span class="badge badge-flat">FLAT</span>';
      if (isChop) {
        cardClass += ' chop';
        posBadge   = '<span class="badge badge-chop">CHOP</span>';
      } else if (pos === 'LONG' || proba > tl) {
        cardClass += ' bull';
        posbadge    = '<span class="badge badge-long">LONG</span>';
        posbadge = posBadge || posbadge;
      } else if (pos === 'SHORT' || proba < ts) {
        cardClass += ' bear';
        posBadge   = '<span class="badge badge-short">SHORT</span>';
      }

      // Couleur barre proba
      let probaColor = '#8b949e';
      if (proba > tl)      probaColor = '#00ff88';
      else if (proba < ts) probaColor = '#ff4466';
      else if (proba > 0.5)probaColor = '#4488ff';
      const probaWidth = (proba * 100).toFixed(1);

      return `
      <div class="${cardClass}">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
          <span class="symbol-name">${sym}</span>
          ${posBadge || posbadge || '<span class="badge badge-flat">FLAT</span>'}
        </div>
        <div class="symbol-price">${price > 100 ? fmt(price,2) : fmt(price,5)}</div>
        <div class="symbol-row">
          <span>AI Proba</span>
          <span style="color:${probaColor}">${(proba*100).toFixed(1)}%</span>
        </div>
        <div class="proba-bar-wrap">
          <div class="proba-bar" style="width:${probaWidth}%;background:${probaColor}"></div>
        </div>
        <div class="proba-label">
          <span>SHORT &lt;${(ts*100).toFixed(0)}%</span>
          <span>LONG &gt;${(tl*100).toFixed(0)}%</span>
        </div>
        <div class="symbol-row" style="margin-top:8px">
          <span>Confluence</span>
          <span>${fmt(conf,2)}</span>
        </div>
        <div class="symbol-row">
          <span>Zone</span>
          <span>${zone}</span>
        </div>
        <div class="symbol-row">
          <span>Chop</span>
          <span style="color:${isChop?'#ffcc00':'#8b949e'}">${isChop?'OUI ⚠️':'non'}</span>
        </div>
      </div>`;
    }).join('');
  }

  // Courbe équité
  const eqPoints = data.equity || [];
  if (eqPoints.length > 1) {
    const vals   = eqPoints.map(p => p.equity);
    const minV   = Math.min(...vals);
    const maxV   = Math.max(...vals);
    const range  = maxV - minV || 1;
    const W = 600; const H = 200; const PAD = 20;

    const pts = vals.map((v, i) => {
      const x = PAD + (i / (vals.length-1)) * (W - PAD*2);
      const y = H - PAD - ((v - minV) / range) * (H - PAD*2);
      return `${x},${y}`;
    }).join(' ');

    const lastV  = vals[vals.length-1];
    const firstV = vals[0];
    const lineCol= lastV >= firstV ? '#00ff88' : '#ff4466';

    const svg = document.getElementById('equity-svg');
    svg.innerHTML = `
      <defs>
        <linearGradient id="eg" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%"   stop-color="${lineCol}" stop-opacity="0.3"/>
          <stop offset="100%" stop-color="${lineCol}" stop-opacity="0"/>
        </linearGradient>
      </defs>
      <polyline points="${pts}" fill="none" stroke="${lineCol}" stroke-width="2"/>
      <text x="${W-PAD}" y="${PAD}" text-anchor="end" fill="${lineCol}" font-size="11" font-weight="600">
        $${fmt(lastV)}
      </text>
      <text x="${PAD}" y="${H-5}" fill="#8b949e" font-size="10">
        $${fmt(minV)}
      </text>
      <text x="${PAD}" y="${PAD+10}" fill="#8b949e" font-size="10">
        $${fmt(maxV)}
      </text>`;
  }

  // Trades récents
  const tWrap = document.getElementById('trades-wrap');
  if (trades.length === 0) {
    tWrap.innerHTML = '<div class="empty-state">Aucun trade pour le moment<br><small>Le bot cherche des setups...</small></div>';
  } else {
    const rows = trades.slice(0, 15).map(t => {
      const pnl   = t.pnl_usd || 0;
      const color = pnl > 0 ? '#00ff88' : pnl < 0 ? '#ff4466' : '#8b949e';
      const ts    = t.exit_time ? new Date(t.exit_time).toLocaleTimeString('fr-CA') : '—';
      return `<tr>
        <td>${t.symbol || '—'}</td>
        <td><span class="badge badge-${(t.direction||'').toLowerCase()}">${t.direction||'—'}</span></td>
        <td style="color:${color};font-weight:600">${pnl>=0?'+':''}$${fmt(Math.abs(pnl))}</td>
        <td>${t.exit_reason || '—'}</td>
        <td style="color:#8b949e">${ts}</td>
      </tr>`;
    }).join('');

    tWrap.innerHTML = `
    <table class="trades-table">
      <thead>
        <tr>
          <th>Symbole</th><th>Direction</th>
          <th>P&L</th><th>Raison</th><th>Heure</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;
  }

  renderDiagnostics(data);
}

const REASON_LABELS = {
  chop:       'Marché en range (chop)',
  no_zone:    'Hors zone OB / FVG',
  proba:      'AI proba sous le seuil',
  confluence: 'Confluence insuffisante',
  cooldown:   'Cooldown actif (trade récent)',
};

function renderDiagnostics(data) {
  const diag = data.diag || {};
  const syms = Object.keys(diag);
  const allRecords = syms.flatMap(s => diag[s]);

  // Barres agrégées des raisons de blocage
  const barsWrap = document.getElementById('diag-bars');
  if (allRecords.length === 0) {
    barsWrap.innerHTML = '<div class="empty-state">En attente de données...</div>';
  } else {
    const counts = {chop:0, no_zone:0, proba:0, confluence:0, cooldown:0};
    let blockedTotal = 0;
    allRecords.forEach(r => {
      const reasons = r.blocked || [];
      if (reasons.length > 0) blockedTotal++;
      reasons.forEach(reason => { if (reason in counts) counts[reason]++; });
    });
    const maxCount = Math.max(...Object.values(counts), 1);
    barsWrap.innerHTML = Object.entries(counts)
      .sort((a,b) => b[1]-a[1])
      .map(([key, count]) => {
        const pct   = (count / allRecords.length * 100);
        const width = (count / maxCount * 100).toFixed(1);
        return `
        <div style="margin-bottom:12px">
          <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px">
            <span>${REASON_LABELS[key]}</span>
            <span style="color:#8b949e">${fmt(pct,1)}% des itérations</span>
          </div>
          <div style="height:10px;background:#21262d;border-radius:4px;overflow:hidden">
            <div style="height:100%;width:${width}%;background:#ff4466;border-radius:4px"></div>
          </div>
        </div>`;
      }).join('') +
      `<div style="font-size:11px;color:#8b949e;margin-top:8px">${blockedTotal} / ${allRecords.length} itérations sans trade possible</div>`;
  }

  // Détail par symbole
  const tableWrap = document.getElementById('diag-table');
  if (syms.length === 0) {
    tableWrap.innerHTML = '<div class="empty-state">En attente de données...</div>';
  } else {
    const rows = syms.map(sym => {
      const recs = diag[sym];
      if (recs.length === 0) return '';
      const last = recs[recs.length - 1];
      const n    = recs.length;
      const pctOf = (key) => fmt(recs.filter(r => (r.blocked||[]).includes(key)).length / n * 100, 0);
      return `<tr>
        <td>${sym}</td>
        <td>${(last.ai_proba*100).toFixed(1)}%</td>
        <td style="color:#8b949e">L&gt;${(last.thresh_long*100).toFixed(0)}% S&lt;${(last.thresh_short*100).toFixed(0)}%</td>
        <td>${pctOf('chop')}%</td>
        <td>${pctOf('no_zone')}%</td>
        <td>${pctOf('proba')}%</td>
        <td>${pctOf('confluence')}%</td>
        <td>${pctOf('cooldown')}%</td>
      </tr>`;
    }).join('');

    tableWrap.innerHTML = `
    <div style="overflow-x:auto">
    <table class="trades-table">
      <thead><tr>
        <th>Symbole</th><th>AI Proba</th><th>Seuils</th>
        <th>Chop</th><th>Hors zone</th><th>Proba</th><th>Confl.</th><th>Cooldown</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>
    </div>`;
  }

  // Sparklines: proba AI vs seuils, par symbole
  const sparkWrap = document.getElementById('diag-sparklines');
  if (syms.length === 0) {
    sparkWrap.innerHTML = '';
  } else {
    sparkWrap.innerHTML = syms.map(sym => {
      const recs = diag[sym].slice(-100);
      if (recs.length < 2) return '';
      const probas = recs.map(r => r.ai_proba);
      const tl = recs[recs.length-1].thresh_long;
      const ts = recs[recs.length-1].thresh_short;
      const minV  = Math.min(...probas, ts);
      const maxV  = Math.max(...probas, tl);
      const range = (maxV - minV) || 1;
      const W = 220, H = 70, PAD = 6;
      const toXY = (v, i) => [
        PAD + (i/(probas.length-1)) * (W-PAD*2),
        H - PAD - ((v-minV)/range) * (H-PAD*2),
      ];
      const pts  = probas.map((v,i) => toXY(v,i).join(',')).join(' ');
      const ylTl = toXY(tl,0)[1];
      const ylTs = toXY(ts,0)[1];
      return `
      <div class="symbol-card">
        <div class="symbol-name" style="margin-bottom:6px">${sym}</div>
        <div class="symbol-row"><span>AI Proba (${recs.length} derniers pts)</span></div>
        <svg width="100%" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" style="margin-top:4px">
          <line x1="0" y1="${ylTl}" x2="${W}" y2="${ylTl}" stroke="#00ff88" stroke-width="1" stroke-dasharray="3,3"/>
          <line x1="0" y1="${ylTs}" x2="${W}" y2="${ylTs}" stroke="#ff4466" stroke-width="1" stroke-dasharray="3,3"/>
          <polyline points="${pts}" fill="none" stroke="#4488ff" stroke-width="1.5"/>
        </svg>
        <div class="proba-label">
          <span style="color:#ff4466">--- seuil SHORT</span>
          <span style="color:#00ff88">--- seuil LONG</span>
        </div>
      </div>`;
    }).join('');
  }
}

// Charger au démarrage
loadData();
// Recharger toutes les 10 minutes
setInterval(loadData, 600000);
// Recharger les données API toutes les 60s sans recharger la page
setInterval(loadData, 60000);
</script>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Silencer les logs HTTP

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))

        elif self.path == '/api/data':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(bot_data.to_json().encode('utf-8'))

        else:
            self.send_response(404)
            self.end_headers()


def start_server(port: int = 5000):
    """Lance le serveur dashboard dans un thread séparé."""
    server = HTTPServer(('0.0.0.0', port), DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Dashboard: http://0.0.0.0:{port}")
    return server


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()

    print(f"Dashboard standalone lancé sur http://0.0.0.0:{args.port}")
    print("Données de test...")

    # Données de démonstration
    import random
    symbols = ["BTCUSD", "XAUUSD", "EURUSD", "NAS100"]
    prices  = {"BTCUSD": 65000, "XAUUSD": 4300, "EURUSD": 1.155, "NAS100": 19800}

    for sym in symbols:
        bot_data.update_symbol(sym, {
            "close":       prices[sym],
            "ai_proba":    round(random.uniform(0.38, 0.72), 3),
            "conf_score":  round(random.uniform(-3, 4), 2),
            "thresh_long": 0.62,
            "thresh_short":0.38,
            "is_chop":     random.choice([True, False]),
            "zone":        random.choice(["OB Bull", "FVG Bear", "None"]),
            "position":    "FLAT",
        })

    eq = 10000.0
    for i in range(50):
        eq += random.uniform(-20, 30)
        bot_data.add_equity(eq)

    bot_data.capital = eq

    # Historique diagnostic de démo (simule un marché en chop qui bloque les trades)
    for sym in symbols:
        tl, ts = 0.62, 0.38
        for i in range(150):
            proba   = round(random.gauss(0.5, 0.05), 4)
            is_chop = random.random() < 0.55
            in_bull = random.random() < 0.25
            in_bear = random.random() < 0.25
            conf    = round(random.uniform(-2, 2), 3)
            max_conf= 4.0
            reasons = []
            if is_chop:
                reasons.append("chop")
            if not in_bull and not in_bear:
                reasons.append("no_zone")
            else:
                proba_ok = (in_bull and proba > tl) or (in_bear and proba < ts)
                conf_ok  = (in_bull and conf >= max_conf*0.30) or (in_bear and conf <= -max_conf*0.30)
                if not proba_ok:
                    reasons.append("proba")
                if not conf_ok:
                    reasons.append("confluence")
            bot_data.add_diag(sym, {
                "ai_proba": proba, "thresh_long": tl, "thresh_short": ts,
                "conf": conf, "max_conf": max_conf,
                "is_chop": is_chop, "in_zone_bull": in_bull, "in_zone_bear": in_bear,
                "cooldown_ok": True, "blocked": reasons,
            })

    server = HTTPServer(('0.0.0.0', args.port), DashboardHandler)
    print(f"Ouvrir: http://localhost:{args.port}")
    server.serve_forever()
