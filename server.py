"""
server.py – Premium UI. Clean 2-class predictions. Speculative tree. Live stats.
"""

import threading
import time
import json
import os
import logging
import copy
from flask import Flask, jsonify, render_template_string
import fetcher
import predictor as pred_module

logging.getLogger("werkzeug").setLevel(logging.WARNING)
app = Flask(__name__)

_state_lock = threading.Lock()
_current_prediction = None
_last_result = None
_predictor = None
_last_data_round = None
_is_ready = False
_prediction_tree = {}

CLASS_NAMES  = pred_module.CLASS_NAMES
CLASS_COLORS = pred_module.CLASS_COLORS

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>StarMaker Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet"/>
<style>
:root {
  --bg:        #080c14;
  --surface:   #0e1521;
  --card:      #111927;
  --border:    rgba(255,255,255,0.07);
  --border2:   rgba(255,255,255,0.12);
  --text:      #e8edf5;
  --muted:     #6b7a96;
  --accent:    #4f8ef7;
  --accent2:   #7b5cf0;
  --win:       #22d97a;
  --loss:      #f05a5a;
  --skip:      #f0b429;
  --glow-play: rgba(34,217,122,0.15);
  --glow-skip: rgba(240,180,41,0.15);
  --radius:    16px;
  --radius-sm: 10px;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Space Grotesk', sans-serif;
  min-height: 100vh;
  background-image:
    radial-gradient(ellipse 80% 50% at 20% -10%, rgba(79,142,247,0.08) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 110%, rgba(123,92,240,0.07) 0%, transparent 60%);
}

/* ── Header ── */
header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 18px 32px;
  border-bottom: 1px solid var(--border);
  background: rgba(14,21,33,0.8);
  backdrop-filter: blur(12px);
  position: sticky;
  top: 0;
  z-index: 100;
}
.header-left { display: flex; align-items: center; gap: 12px; }
.logo-dot {
  width: 10px; height: 10px; border-radius: 50%;
  background: var(--win);
  box-shadow: 0 0 10px var(--win);
  animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.6;transform:scale(0.85)} }
.logo-text { font-size: 0.85rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: var(--text); }
.logo-sub  { font-size: 0.65rem; color: var(--muted); letter-spacing: 1px; margin-top: 1px; }
.turbo-pill {
  background: linear-gradient(135deg, rgba(34,217,122,0.15), rgba(79,142,247,0.15));
  border: 1px solid rgba(34,217,122,0.3);
  color: var(--win);
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 1.5px;
  padding: 4px 10px;
  border-radius: 20px;
  display: none;
}
.turbo-pill.active { display: inline-block; }
.header-stats { display: flex; gap: 24px; }
.hstat { text-align: right; }
.hstat-val { font-family: 'JetBrains Mono', monospace; font-size: 1rem; font-weight: 600; }
.hstat-lbl { font-size: 0.6rem; color: var(--muted); letter-spacing: 1px; text-transform: uppercase; margin-top: 1px; }

/* ── Layout ── */
.container { max-width: 980px; margin: 0 auto; padding: 24px 20px; display: grid; gap: 18px; }

/* ── Cards ── */
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 24px;
  position: relative;
  overflow: hidden;
  transition: border-color 0.4s;
}
.card::before {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: var(--radius);
  opacity: 0;
  transition: opacity 0.4s;
  pointer-events: none;
}
.card.play-glow { border-color: rgba(34,217,122,0.35); }
.card.play-glow::before { background: radial-gradient(ellipse at 50% 0%, var(--glow-play), transparent 70%); opacity: 1; }
.card.skip-glow { border-color: rgba(240,180,41,0.35); }
.card.skip-glow::before { background: radial-gradient(ellipse at 50% 0%, var(--glow-skip), transparent 70%); opacity: 1; }

.card-label {
  font-size: 0.62rem;
  font-weight: 700;
  letter-spacing: 1.8px;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 16px;
}

/* ── Main prediction card ── */
.pred-top { display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 20px; }
.round-block { }
.round-label { font-size: 0.65rem; color: var(--muted); letter-spacing: 1px; text-transform: uppercase; }
.round-num {
  font-family: 'JetBrains Mono', monospace;
  font-size: 3.2rem;
  font-weight: 700;
  line-height: 1;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-top: 4px;
}
.action-block { text-align: right; }
.action-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 8px 18px;
  border-radius: 30px;
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 1px;
  border: 1.5px solid;
  transition: all 0.3s;
}
.action-badge.play {
  background: rgba(34,217,122,0.12);
  color: var(--win);
  border-color: rgba(34,217,122,0.4);
  box-shadow: 0 0 20px rgba(34,217,122,0.1);
}
.action-badge.skip {
  background: rgba(240,180,41,0.12);
  color: var(--skip);
  border-color: rgba(240,180,41,0.4);
}
.action-badge .dot2 {
  width: 7px; height: 7px; border-radius: 50%;
  background: currentColor;
  animation: pulse 1.5s infinite;
}
.skip-reason { font-size: 0.7rem; color: var(--skip); margin-top: 6px; font-family: 'JetBrains Mono', monospace; }
.streak-badge {
  font-size: 0.65rem; color: var(--loss);
  background: rgba(240,90,90,0.1);
  border: 1px solid rgba(240,90,90,0.25);
  border-radius: 6px;
  padding: 2px 8px;
  font-family: 'JetBrains Mono', monospace;
  margin-top: 4px;
  display: none;
}
.streak-badge.visible { display: inline-block; }
.warmup-badge {
  font-size: 0.65rem; color: var(--accent);
  background: rgba(79,142,247,0.1);
  border: 1px solid rgba(79,142,247,0.25);
  border-radius: 6px; padding: 2px 8px;
  font-family: 'JetBrains Mono', monospace;
  display: none;
}
.warmup-badge.visible { display: inline-block; }

/* ── Picks ── */
.picks { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.pick-card {
  border-radius: var(--radius-sm);
  padding: 18px;
  border: 1.5px solid transparent;
  position: relative;
  overflow: hidden;
  transition: transform 0.2s, border-color 0.3s;
}
.pick-card:hover { transform: translateY(-1px); }
.pick-rank {
  font-size: 0.58rem;
  font-weight: 700;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  opacity: 0.7;
  margin-bottom: 8px;
}
.pick-name {
  font-size: 1.35rem;
  font-weight: 700;
  margin-bottom: 10px;
}
.conf-bar-track {
  height: 4px;
  background: rgba(255,255,255,0.07);
  border-radius: 2px;
  overflow: hidden;
  margin-bottom: 6px;
}
.conf-bar-fill {
  height: 100%;
  border-radius: 2px;
  transition: width 0.6s cubic-bezier(0.4,0,0.2,1);
}
.conf-val {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.8rem;
  font-weight: 600;
}

/* ── Probability breakdown ── */
.prob-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 8px;
  margin-top: 4px;
}
.prob-item {
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 8px 10px;
  position: relative;
  overflow: hidden;
  transition: border-color 0.3s;
}
.prob-item.is-win { border-color: rgba(255,255,255,0.18); }
.prob-item-bg {
  position: absolute;
  bottom: 0; left: 0; right: 0;
  transition: height 0.6s cubic-bezier(0.4,0,0.2,1);
  opacity: 0.12;
}
.prob-name { font-size: 0.65rem; font-weight: 600; position: relative; }
.prob-val  { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 600; position: relative; margin-top: 2px; }
.win-dot   { width: 5px; height: 5px; border-radius: 50%; display: inline-block; vertical-align: middle; margin-left: 4px; margin-top: -1px; }

/* ── Lookahead matrix ── */
.matrix-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 10px;
}
@media (max-width: 600px) { .matrix-grid { grid-template-columns: repeat(2, 1fr); } }
.matrix-item {
  background: rgba(255,255,255,0.025);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 12px;
  border-left: 3px solid;
  transition: background 0.2s;
}
.matrix-item:hover { background: rgba(255,255,255,0.045); }
.mx-if   { font-size: 0.58rem; color: var(--muted); letter-spacing: 1px; text-transform: uppercase; margin-bottom: 6px; }
.mx-lbl  { font-size: 0.75rem; font-weight: 700; }
.mx-then { display: flex; flex-direction: column; gap: 3px; margin-top: 6px; }
.mx-pred { font-size: 0.68rem; font-weight: 600; }
.mx-act  { font-size: 0.58rem; letter-spacing: 0.5px; margin-top: 5px; padding: 2px 6px; border-radius: 4px; display: inline-block; }
.mx-act.play { background: rgba(34,217,122,0.15); color: var(--win); }
.mx-act.skip { background: rgba(240,180,41,0.12); color: var(--skip); }
.mx-computing { opacity: 0.25; font-size: 0.65rem; color: var(--muted); display: flex; align-items: center; justify-content: center; height: 80px; }

/* ── History ── */
.history-list { display: flex; flex-direction: column; gap: 6px; }
.history-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 14px;
  background: rgba(255,255,255,0.025);
  border-radius: 8px;
  border: 1px solid var(--border);
  animation: slideIn 0.3s ease;
}
@keyframes slideIn { from { opacity:0; transform:translateY(-4px); } to { opacity:1; transform:translateY(0); } }
.hist-round { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: var(--muted); min-width: 40px; }
.hist-preds { display: flex; gap: 6px; flex: 1; }
.hist-pred  { font-size: 0.72rem; font-weight: 600; padding: 2px 8px; border-radius: 5px; }
.hist-arrow { font-size: 0.65rem; color: var(--muted); }
.hist-true  { font-size: 0.8rem; font-weight: 700; }
.hist-result { font-size: 1rem; min-width: 20px; text-align: center; }

/* ── Stats bar ── */
.stats-row { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; }
.stat-block {
  background: rgba(255,255,255,0.025);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px;
  text-align: center;
}
.stat-val { font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 700; }
.stat-lbl { font-size: 0.6rem; color: var(--muted); letter-spacing: 1px; text-transform: uppercase; margin-top: 4px; }

/* ── Entropy bar ── */
.entropy-row { display: flex; align-items: center; gap: 12px; margin-top: 10px; }
.entropy-label { font-size: 0.65rem; color: var(--muted); width: 60px; }
.entropy-track { flex: 1; height: 3px; background: rgba(255,255,255,0.07); border-radius: 2px; overflow: hidden; }
.entropy-fill  { height: 100%; border-radius: 2px; transition: width 0.5s; background: linear-gradient(90deg, var(--win), var(--skip), var(--loss)); }
.entropy-val   { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: var(--muted); width: 40px; text-align: right; }

/* ── Two column layout ── */
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
@media (max-width: 680px) { .two-col { grid-template-columns: 1fr; } .stats-row { grid-template-columns: repeat(3, 1fr); } .prob-grid { grid-template-columns: repeat(2, 1fr); } }
</style>
</head>
<body>

<header>
  <div class="header-left">
    <div class="logo-dot" id="live-dot"></div>
    <div>
      <div class="logo-text">StarMaker Predictor</div>
      <div class="logo-sub">Live · 2-Class Engine</div>
    </div>
    <span class="turbo-pill" id="turbo-pill">⚡ TURBO</span>
  </div>
  <div class="header-stats">
    <div class="hstat">
      <div class="hstat-val" id="h-winpct" style="color:var(--win)">—</div>
      <div class="hstat-lbl">Win Rate</div>
    </div>
    <div class="hstat">
      <div class="hstat-val" id="h-played">—</div>
      <div class="hstat-lbl">Played</div>
    </div>
    <div class="hstat">
      <div class="hstat-val" id="h-streak" style="color:var(--loss)">—</div>
      <div class="hstat-lbl">Streak</div>
    </div>
  </div>
</header>

<div class="container">

  <!-- ── Main prediction ── -->
  <div class="card" id="main-card">
    <div class="pred-top">
      <div class="round-block">
        <div class="round-label">Predicting Round</div>
        <div class="round-num" id="round-num">—</div>
        <span class="warmup-badge" id="warmup-badge">WARMING UP</span>
        <span class="streak-badge" id="streak-badge">LOSS STREAK</span>
      </div>
      <div class="action-block">
        <div class="action-badge" id="action-badge">
          <span class="dot2"></span>
          <span id="action-text">WAITING</span>
        </div>
        <div class="skip-reason" id="skip-reason"></div>
      </div>
    </div>

    <div class="picks" id="picks-container">
      <div class="pick-card" id="pick-1" style="background:rgba(255,255,255,0.03)">
        <div class="pick-rank">Primary Pick</div>
        <div class="pick-name" id="p1-name">—</div>
        <div class="conf-bar-track"><div class="conf-bar-fill" id="p1-bar" style="width:0%"></div></div>
        <div class="conf-val" id="p1-conf">—</div>
      </div>
      <div class="pick-card" id="pick-2" style="background:rgba(255,255,255,0.03)">
        <div class="pick-rank">Secondary Pick</div>
        <div class="pick-name" id="p2-name">—</div>
        <div class="conf-bar-track"><div class="conf-bar-fill" id="p2-bar" style="width:0%"></div></div>
        <div class="conf-val" id="p2-conf">—</div>
      </div>
    </div>

    <div class="entropy-row">
      <div class="entropy-label">Entropy</div>
      <div class="entropy-track"><div class="entropy-fill" id="entropy-fill" style="width:0%"></div></div>
      <div class="entropy-val" id="entropy-val">—</div>
    </div>
  </div>

  <!-- ── Probability breakdown ── -->
  <div class="card">
    <div class="card-label">All Class Probabilities</div>
    <div class="prob-grid" id="prob-grid"></div>
  </div>

  <!-- ── Stats + History ── -->
  <div class="two-col">
    <div class="card">
      <div class="card-label">Session Stats</div>
      <div class="stats-row" style="grid-template-columns:repeat(3,1fr)">
        <div class="stat-block">
          <div class="stat-val" id="s-played">0</div>
          <div class="stat-lbl">Played</div>
        </div>
        <div class="stat-block">
          <div class="stat-val" id="s-won" style="color:var(--win)">0</div>
          <div class="stat-lbl">Won</div>
        </div>
        <div class="stat-block">
          <div class="stat-val" id="s-lost" style="color:var(--loss)">0</div>
          <div class="stat-lbl">Lost</div>
        </div>
        <div class="stat-block">
          <div class="stat-val" id="s-winpct" style="color:var(--win)">0%</div>
          <div class="stat-lbl">Win %</div>
        </div>
        <div class="stat-block">
          <div class="stat-val" id="s-skipped" style="color:var(--muted)">0</div>
          <div class="stat-lbl">Skipped</div>
        </div>
        <div class="stat-block">
          <div class="stat-val" id="s-playrate" style="color:var(--accent)">0%</div>
          <div class="stat-lbl">Play Rate</div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-label">Recent Results</div>
      <div class="history-list" id="history-list"></div>
    </div>
  </div>

  <!-- ── Lookahead matrix ── -->
  <div class="card">
    <div class="card-label">Lookahead — If Round <span id="lookahead-round">—</span> Is...</div>
    <div class="matrix-grid" id="matrix-grid"></div>
  </div>

</div>

<script>
const COLORS = {{ class_colors | tojson }};
const NAMES  = {{ class_names | tojson }};

let prevRound = null;

function hex2rgba(hex, a) {
  const r = parseInt(hex.slice(1,3),16);
  const g = parseInt(hex.slice(3,5),16);
  const b = parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${a})`;
}

function renderPicks(pred) {
  const picks = [
    { id: '1', rank: 'Primary Pick',   cls: pred.pred1_cls, conf: pred.pred1_conf },
    { id: '2', rank: 'Secondary Pick',  cls: pred.pred2_cls, conf: pred.pred2_conf },
  ];
  picks.forEach(p => {
    const col = COLORS[p.cls];
    document.getElementById(`pick-${p.id}`).style.background = hex2rgba(col, 0.07);
    document.getElementById(`pick-${p.id}`).style.borderColor = hex2rgba(col, p.id === '1' ? 0.5 : 0.25);
    document.getElementById(`p${p.id}-name`).textContent = NAMES[p.cls];
    document.getElementById(`p${p.id}-name`).style.color = col;
    document.getElementById(`p${p.id}-bar`).style.width = Math.min(p.conf, 100) + '%';
    document.getElementById(`p${p.id}-bar`).style.background = col;
    document.getElementById(`p${p.id}-conf`).textContent = p.conf.toFixed(1) + '%';
    document.getElementById(`p${p.id}-conf`).style.color = col;
  });
}

function renderProbs(pred) {
  if (!pred.all_probs) return;
  const winSet = new Set(pred.win_classes || []);
  const grid = document.getElementById('prob-grid');
  let html = '';
  for (let i = 1; i <= 8; i++) {
    const prob = pred.all_probs[i] ?? 0;
    const col  = COLORS[i];
    const isWin = winSet.has(i);
    html += `
      <div class="prob-item ${isWin ? 'is-win' : ''}" style="border-color:${isWin ? hex2rgba(col,0.35) : ''}">
        <div class="prob-item-bg" style="background:${col};height:${prob}%"></div>
        <div class="prob-name" style="color:${col}">${NAMES[i]}${isWin ? `<span class="win-dot" style="background:${col}"></span>` : ''}</div>
        <div class="prob-val"  style="color:${col}">${prob.toFixed(1)}%</div>
      </div>`;
  }
  grid.innerHTML = html;
}

function renderMatrix(tree, nextRound) {
  const branches = (tree || {})[nextRound] || {};
  document.getElementById('lookahead-round').textContent = nextRound;
  let html = '';
  for (let i = 1; i <= 8; i++) {
    const bp = branches[String(i)];
    const col = COLORS[i];
    if (bp) {
      const actCls = bp.action === 'PLAY' ? 'play' : 'skip';
      html += `
        <div class="matrix-item" style="border-left-color:${col}">
          <div class="mx-if">If comes</div>
          <div class="mx-lbl" style="color:${col}">${NAMES[i]}</div>
          <div class="mx-then">
            <div class="mx-pred" style="color:${COLORS[bp.pred1_cls]}">▶ ${NAMES[bp.pred1_cls]}</div>
            <div class="mx-pred" style="color:${COLORS[bp.pred2_cls]};opacity:0.7">▶ ${NAMES[bp.pred2_cls]}</div>
          </div>
          <span class="mx-act ${actCls}">${bp.action}</span>
        </div>`;
    } else {
      html += `<div class="matrix-item" style="border-left-color:${col}"><div class="mx-computing">Computing…</div></div>`;
    }
  }
  document.getElementById('matrix-grid').innerHTML = html;
}

function renderHistory(history) {
  const list = document.getElementById('history-list');
  const items = [...history].reverse().slice(0, 8);
  if (!items.length) { list.innerHTML = '<div style="color:var(--muted);font-size:0.75rem;text-align:center;padding:16px">No results yet</div>'; return; }
  list.innerHTML = items.map(h => {
    const p1c = COLORS[h.pred1], p2c = COLORS[h.pred2], tc = COLORS[h.true];
    const icon = h.won ? '✅' : '❌';
    return `
      <div class="history-item">
        <span class="hist-round">#${h.round}</span>
        <div class="hist-preds">
          <span class="hist-pred" style="background:${hex2rgba(p1c,0.15)};color:${p1c}">${NAMES[h.pred1]}</span>
          <span class="hist-pred" style="background:${hex2rgba(p2c,0.1)};color:${p2c};opacity:0.8">${NAMES[h.pred2]}</span>
        </div>
        <span class="hist-arrow">→</span>
        <span class="hist-true" style="color:${tc}">${NAMES[h.true]}</span>
        <span class="hist-result">${icon}</span>
      </div>`;
  }).join('');
}

async function update() {
  try {
    const res  = await fetch('/api/state');
    const data = await res.json();
    const pred = data.prediction;
    const stats = data.stats || {};

    // Header stats
    document.getElementById('h-winpct').textContent  = stats.win_pct != null ? stats.win_pct + '%' : '—';
    document.getElementById('h-played').textContent  = stats.played ?? '—';
    document.getElementById('h-streak').textContent  = stats.loss_streak ?? '—';
    document.getElementById('turbo-pill').classList.toggle('active', !!data.spec_ready);

    // Stats panel
    document.getElementById('s-played').textContent  = stats.played   ?? 0;
    document.getElementById('s-won').textContent      = stats.won      ?? 0;
    document.getElementById('s-lost').textContent     = stats.lost     ?? 0;
    document.getElementById('s-winpct').textContent   = (stats.win_pct ?? 0) + '%';
    document.getElementById('s-skipped').textContent  = stats.skipped  ?? 0;
    document.getElementById('s-playrate').textContent = (stats.play_rate ?? 0) + '%';

    if (!pred) return;

    // Flash on new round
    if (pred.next_round !== prevRound) {
      document.getElementById('main-card').animate(
        [{ boxShadow: '0 0 40px rgba(79,142,247,0.3)' }, { boxShadow: '0 0 0 transparent' }],
        { duration: 600, easing: 'ease-out' }
      );
      prevRound = pred.next_round;
    }

    // Round number
    document.getElementById('round-num').textContent = '#' + pred.next_round;

    // Action badge
    const badge = document.getElementById('action-badge');
    const isPlay = pred.action === 'PLAY';
    badge.className = 'action-badge ' + (isPlay ? 'play' : 'skip');
    document.getElementById('action-text').textContent = pred.action;
    document.getElementById('main-card').className = 'card ' + (isPlay ? 'play-glow' : 'skip-glow');
    document.getElementById('skip-reason').textContent = pred.skip_reason || '';

    // Streak + warmup
    const streakEl  = document.getElementById('streak-badge');
    const warmupEl  = document.getElementById('warmup-badge');
    streakEl.textContent = `${pred.loss_streak} LOSS STREAK`;
    streakEl.classList.toggle('visible', pred.loss_streak >= 2);
    warmupEl.classList.toggle('visible', !!stats.is_warmup);

    // Entropy
    const maxE = 3.0;
    const ePct = Math.min((pred.entropy / maxE) * 100, 100);
    document.getElementById('entropy-fill').style.width = ePct + '%';
    document.getElementById('entropy-val').textContent  = pred.entropy;

    // Picks
    renderPicks(pred);

    // All probs
    renderProbs(pred);

    // Matrix
    renderMatrix(data.tree, pred.next_round);

    // History
    renderHistory(data.history || []);

  } catch(e) { /* silent */ }
}

update();
setInterval(update, 1000);
</script>
</body>
</html>
"""


def _build_predictor():
    global _predictor, _is_ready
    _predictor = pred_module.LivePredictor()
    _is_ready = True
    print("[System] Engine Ready.")


def _generate_speculative_tree(base_rounds):
    global _prediction_tree
    if not _predictor:
        return

    with _state_lock:
        base_state = copy.deepcopy(_predictor)

    latest_round_num = base_rounds[-1]["round"]
    next_id   = latest_round_num + 1
    future_id = latest_round_num + 2
    new_tree  = {next_id: {}, future_id: {}}

    print(f"[Speculate] Generating paths for #{next_id}...")

    for i in range(1, 9):
        sim_1 = copy.deepcopy(base_rounds)
        sim_1.append({"round": next_id, "reward_index": i})

        temp_predictor = copy.deepcopy(base_state)
        temp_predictor.sync(sim_1)
        pred = temp_predictor.predict_next(provided_data=sim_1)
        new_tree[next_id][str(i)] = pred

        for j in range(1, 9):
            sim_2 = copy.deepcopy(sim_1)
            sim_2.append({"round": future_id, "reward_index": j})
            deep_predictor = copy.deepcopy(temp_predictor)
            deep_predictor.sync(sim_2)
            pred_deep = deep_predictor.predict_next(provided_data=sim_2)
            new_tree[future_id][f"{i}_{j}"] = pred_deep

    with _state_lock:
        _prediction_tree = new_tree
    print("[Turbo] Tree completed.")


def _prediction_loop():
    global _current_prediction, _last_result, _last_data_round
    if not _is_ready:
        return
    try:
        rounds = sorted(fetcher.load_existing_data()[0], key=lambda x: x["round"])
        if not rounds:
            return
        latest_round = rounds[-1]["round"]
        latest_val   = rounds[-1]["reward_index"]

        with _state_lock:
            if _last_data_round and latest_round <= _last_data_round:
                return

        new_pred = None
        with _state_lock:
            if latest_round in _prediction_tree:
                new_pred = _prediction_tree[latest_round].get(str(latest_val))

        if not new_pred:
            new_pred = _predictor.predict_next()
        else:
            _predictor.sync(rounds)

        with _state_lock:
            _current_prediction = new_pred
            _last_data_round = latest_round

        threading.Thread(
            target=_generate_speculative_tree, args=(rounds,), daemon=True
        ).start()
    except Exception as e:
        print(f"[Error] Loop: {e}")


def _fetch_loop():
    while True:
        try:
            fetcher.fetch_new_rounds()
        except Exception:
            pass
        time.sleep(5)


def _predict_loop():
    _build_predictor()
    while True:
        _prediction_loop()
        time.sleep(1)


@app.route("/")
def index():
    return render_template_string(
        HTML_TEMPLATE, class_names=CLASS_NAMES, class_colors=CLASS_COLORS
    )


@app.route("/api/state")
def api_state():
    with _state_lock:
        stats = _predictor.get_stats() if _predictor else {}
        return jsonify({
            "prediction": _current_prediction,
            "last_result": _last_result,
            "history":     _predictor.get_recent_history(10) if _predictor else [],
            "stats":       stats,
            "spec_ready":  len(_prediction_tree) > 0,
            "tree":        _prediction_tree,
        })


if __name__ == "__main__":
    threading.Thread(target=_fetch_loop, daemon=True).start()
    threading.Thread(target=_predict_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)