/**
 * main.js — AI Fraud Detection System
 * Handles: tabs, PCA sliders, fetch API calls, verdict rendering,
 *          score gauges, score bars, module breakdown, drag-and-drop uploads.
 */

'use strict';

// ── State ─────────────────────────────────────────────────────────
const state = {
  threshold: 0.30,
  mlReady:   false,
  nlpReady:  false,
  cvReady:   false,
};

// ── ULB Transaction Presets ──────────────────────────────────────
// Source: ULB Credit Card Fraud dataset — anonymised PCA projections.
const PRESETS = {
  // ── Fraud patterns ──────────────────────────────────────────────
  fraud_a: {
    label: 'Fraud A — Card-Not-Present (€0)',
    time: 406, amount: 0.0,
    v: [-2.31, 1.95, -1.61, 3.99, -0.52, -1.43, -2.54, 1.39,
        -2.77, -2.77, 3.20, -2.90, -0.60, -4.97, -1.68, -1.51,
        -3.32, -0.47, -0.58, -0.51, 0.36, 0.71, -0.51, -0.34,
         0.13, -0.68, 0.04, 0.09],
  },
  fraud_b: {
    label: 'Fraud B — High Value (€529)',
    time: 55842, amount: 529.0,
    v: [-1.78, 1.86, -1.85, 2.13, -0.05, -0.83, -1.43, 0.42,
        -2.51, -3.25, 3.62, -3.06, -0.24, -4.54, -1.44, -0.58,
        -3.21, -0.35, -0.44, -0.60, 0.25, 0.60, -0.38, -0.25,
         0.11, -0.47, 0.06, 0.07],
  },
  fraud_c: {
    label: 'Fraud C — Micro Transaction (€1.98)',
    time: 12503, amount: 1.98,
    v: [-3.04, 2.15, -3.23, 4.11, -0.81, -2.03, -3.12, 1.82,
        -3.10, -3.31, 3.87, -3.43, -0.71, -5.21, -2.01, -1.88,
        -3.55, -0.61, -0.70, -0.63, 0.41, 0.88, -0.64, -0.42,
         0.17, -0.82, 0.05, 0.11],
  },
  fraud_d: {
    label: 'Fraud D — Wire Transfer (€8,900)',
    time: 101234, amount: 8900.0,
    v: [-2.55, 2.42, -2.78, 3.61, -0.38, -1.89, -2.91, 1.61,
        -2.95, -3.02, 3.54, -3.17, -0.56, -5.08, -1.90, -1.73,
        -3.44, -0.53, -0.64, -0.57, 0.39, 0.81, -0.58, -0.39,
         0.15, -0.75, 0.05, 0.10],
  },
  // ── Legitimate patterns ──────────────────────────────────────────
  legit_a: {
    label: 'Normal — Everyday Purchase (€49.95)',
    time: 85000, amount: 49.95,
    v: [1.19, 0.27, 0.17, 0.45, -0.32, -0.07, 0.07, -0.05,
        0.10, 0.10, -0.17, 0.03, 0.02, -0.04, -0.03, 0.05,
        0.01, 0.09, -0.06, 0.04, -0.02, 0.01, 0.07, -0.03,
        0.06, 0.01, 0.05, -0.01],
  },
  legit_b: {
    label: 'Normal — Salary / Large Legit (€3,200)',
    time: 160000, amount: 3200.0,
    v: [1.52, 0.31, 0.22, 0.58, -0.25, -0.04, 0.09, -0.03,
        0.12, 0.14, -0.19, 0.04, 0.03, -0.05, -0.02, 0.06,
        0.02, 0.11, -0.05, 0.05, -0.01, 0.02, 0.08, -0.02,
        0.07, 0.02, 0.04, -0.01],
  },
  legit_c: {
    label: 'Normal — Contactless (€4.50)',
    time: 43200, amount: 4.50,
    v: [0.95, 0.21, 0.14, 0.38, -0.28, -0.06, 0.05, -0.04,
        0.08, 0.08, -0.14, 0.02, 0.01, -0.03, -0.02, 0.04,
        0.01, 0.07, -0.04, 0.03, -0.01, 0.01, 0.06, -0.02,
        0.05, 0.01, 0.04, -0.01],
  },
};

// ── Module metadata for verdict rendering ─────────────────────────
const MOD_META = {
  ml:  { iconId: 'ic-bar-chart', name: 'ML · XGBoost',     bg: '#FEF3C7', color: '#B45309' },
  nlp: { iconId: 'ic-message',   name: 'NLP · DistilBERT', bg: '#FFEDD5', color: '#C2410C' },
  cv:  { iconId: 'ic-pen',       name: 'CV · Siamese',     bg: '#ECFCCB', color: '#4D7C0F' },
};

// ── DOM ready ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  buildPcaSliders();
  initThresholdSlider();
  initPresetButtons();
  initNlpExamples();
  initDropZones();
  initAnalyzeButtons();
  fetchStatus();
});

// ── Tabs ──────────────────────────────────────────────────────────
function initTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(btn.dataset.tab).classList.add('active');
    });
  });
}

// ── PCA Sliders ───────────────────────────────────────────────────
function buildPcaSliders() {
  const grid = document.getElementById('pca-grid');
  if (!grid) return;
  for (let i = 1; i <= 28; i++) {
    const cell = document.createElement('div');
    cell.className = 'pca-cell';
    cell.innerHTML = `
      <label>V${i}</label>
      <input type="range" id="v${i}" min="-10" max="10" step="0.1" value="0">
      <span class="pca-val" id="v${i}-val">0.0</span>`;
    grid.appendChild(cell);
    const slider = cell.querySelector('input');
    const valEl  = cell.querySelector('.pca-val');
    slider.addEventListener('input', () => {
      valEl.textContent = parseFloat(slider.value).toFixed(1);
    });
  }
}

function getPcaValues() {
  return Array.from({length: 28}, (_, i) =>
    parseFloat(document.getElementById(`v${i+1}`).value)
  );
}

function setPcaValues(vals) {
  vals.forEach((v, i) => {
    const slider = document.getElementById(`v${i+1}`);
    const valEl  = document.getElementById(`v${i+1}-val`);
    if (slider) { slider.value = v; valEl.textContent = parseFloat(v).toFixed(1); }
  });
}

// ── Threshold ─────────────────────────────────────────────────────
function initThresholdSlider() {
  const slider  = document.getElementById('threshold-slider');
  const display = document.getElementById('threshold-val');
  if (!slider) return;
  slider.addEventListener('input', () => {
    state.threshold = parseFloat(slider.value);
    display.textContent = state.threshold.toFixed(2);
  });
}

// ── Presets ───────────────────────────────────────────────────────
function initPresetButtons() {
  // Dynamically wire every preset key to its button
  Object.keys(PRESETS).forEach(key => {
    const btn = document.getElementById(`btn-preset-${key.replace('_', '-')}`);
    if (btn) btn.addEventListener('click', () => loadPreset(key));
  });
}

function loadPreset(key) {
  const p = PRESETS[key]; if (!p) return;
  document.getElementById('txn-time').value   = p.time;
  document.getElementById('txn-amount').value = p.amount;
  setPcaValues(p.v);
}

// ── NLP Example Library ───────────────────────────────────────────
const NLP_EXAMPLES = [
  // ── Phishing / Fraud ──────────────────────────────────────────────
  {
    tag: 'fraud', label: 'Bank Account Lock',
    text: 'URGENT: Your HDFC Bank account has been locked due to suspicious activity! Verify your identity immediately at http://hdfc-secure-account.net or your account will be permanently disabled within 24 hours.',
  },
  {
    tag: 'fraud', label: 'OTP Harvest Scam',
    text: 'SBI Alert: A new device has attempted to access your account. Share your OTP 847291 with our agent on 9876543210 to block this device. Do NOT share with anyone else.',
  },
  {
    tag: 'fraud', label: 'Prize Winner Scam',
    text: 'Congratulations! You have been selected as the lucky winner of Rs.15,00,000 in the KBC Lottery 2026. To claim your prize, send your Aadhaar number and bank details to claim@kbc-lottery-india.com',
  },
  {
    tag: 'fraud', label: 'Account Takeover',
    text: 'Your Netflix account will be suspended today. Update your payment details now: http://netflix-billing-update.info/secure — Failure to do so will result in permanent account termination.',
  },
  {
    tag: 'fraud', label: 'Phishing Email Body',
    text: 'Dear Customer, We have detected unusual sign-in activity from IP 192.168.44.21 (Russia). Your account has been temporarily restricted. Click here to verify: http://paypal-secure-verify.com/restore and restore access within 48 hours.',
  },
  {
    tag: 'fraud', label: 'UPI Refund Scam',
    text: 'Your refund of Rs.4,999 from Amazon has been processed. To receive your refund scan the QR code sent to your email. If not received, call our fraud helpline 8800-XXXX-XXXX and share your UPI PIN.',
  },
  // ── Legitimate ────────────────────────────────────────────────────
  {
    tag: 'legit', label: 'Purchase Confirmation',
    text: 'Your Amazon order #112-4857291-0039412 has been delivered. Rate your experience: amazon.com/review',
  },
  {
    tag: 'legit', label: 'Bank Debit Alert',
    text: 'HDFC Bank: INR 2,450 spent on SWIGGY INSTAMART on 17-Jul-26. Avl Bal: INR 44,201. SMS BLOCK to 5676782 to disable card.',
  },
  {
    tag: 'legit', label: 'Google Security Alert',
    text: 'Google Account security alert: A new sign-in on Chrome, Windows, Mumbai. Time: 11:43 AM. If this was you, no action needed.',
  },
  {
    tag: 'legit', label: 'Flight Booking',
    text: 'Your IndiGo booking 6E-2341 is confirmed. DEL→BOM on 20-Jul-26 at 06:30. PNR: ABCXYZ. Check-in opens 48 hours before departure.',
  },
];

function initNlpExamples() {
  const grid = document.getElementById('nlp-examples-grid');
  if (!grid) return;
  grid.innerHTML = '';

  NLP_EXAMPLES.forEach((ex, idx) => {
    const btn = document.createElement('button');
    btn.className = ex.tag === 'fraud' ? 'btn btn-fraud' : 'btn btn-success';
    btn.style.cssText = 'font-size:.72rem;padding:.4rem .7rem;justify-content:flex-start;text-align:left;width:100%';
    btn.innerHTML = `
      <svg width="12" height="12"><use href="#${ex.tag === 'fraud' ? 'ic-alert' : 'ic-check'}"/></svg>
      ${ex.label}`;
    btn.addEventListener('click', () => {
      document.getElementById('nlp-text').value = ex.text;
      // Highlight active button
      grid.querySelectorAll('button').forEach(b => b.style.outline = 'none');
      btn.style.outline = `2px solid ${ex.tag === 'fraud' ? '#DC2626' : '#059669'}`;
    });
    grid.appendChild(btn);
  });
}

// ── Drop Zones ────────────────────────────────────────────────────
function initDropZones() {
  setupDropZone('dz-ref',  'file-ref',  'preview-ref');
  setupDropZone('dz-test', 'file-test', 'preview-test');
}

function setupDropZone(dzId, inputId, previewId) {
  const dz      = document.getElementById(dzId);
  const input   = document.getElementById(inputId);
  const preview = document.getElementById(previewId);
  if (!dz || !input) return;

  input.addEventListener('change', () => {
    if (input.files[0]) showPreview(dz, preview, input.files[0]);
  });
  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('dragover'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
  dz.addEventListener('drop', e => {
    e.preventDefault(); dz.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      const dt = new DataTransfer();
      dt.items.add(file);
      input.files = dt.files;
      showPreview(dz, preview, file);
    }
  });
}

function showPreview(dz, preview, file) {
  const reader = new FileReader();
  reader.onload = e => { preview.src = e.target.result; dz.classList.add('has-image'); };
  reader.readAsDataURL(file);
}

// ── Analyze Buttons ───────────────────────────────────────────────
function initAnalyzeButtons() {
  document.getElementById('btn-analyze-txn')?.addEventListener('click',  analyzeTransaction);
  document.getElementById('btn-analyze-text')?.addEventListener('click', analyzeText);
  document.getElementById('btn-verify-sig')?.addEventListener('click',   analyzeSignature);
}

// ── Status Fetch ──────────────────────────────────────────────────
async function fetchStatus() {
  try {
    const data = await fetch('/api/status').then(r => r.json());
    state.mlReady  = data.ml;
    state.nlpReady = data.nlp;
    state.cvReady  = data.cv;

    setStatus('status-ml',  data.ml);
    setStatus('status-nlp', data.nlp);
    setStatus('status-cv',  data.cv);

    if (!data.ml && !data.nlp && !data.cv) {
      document.getElementById('sidebar-warning')?.classList.add('visible');
    }
  } catch (e) { console.warn('Status check failed:', e); }
}

function setStatus(id, ready) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = ready ? 'Ready' : 'Not trained';
  el.className   = ready ? 'pill-status status-ready' : 'pill-status status-notready';
}

// ── Transaction Analysis ──────────────────────────────────────────
async function analyzeTransaction() {
  clearResults('txn-results', 'txn-msg');
  if (!state.mlReady) {
    showMsg('txn-msg', 'error', 'XGBoost model not trained. Run python train_all.py first.');
    return;
  }

  const pca     = getPcaValues();
  const time_val = parseFloat(document.getElementById('txn-time').value)   || 0;
  const amount   = parseFloat(document.getElementById('txn-amount').value) || 0;
  const text     = document.getElementById('txn-text').value.trim();
  const payload  = { time_val, amount, pca, threshold: state.threshold };
  if (text) payload.text = text;

  showLoading('Running multi-modal fraud analysis…');
  try {
    const res    = await fetch('/api/analyze/transaction', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    const result = await res.json();
    hideLoading();
    if (!res.ok) { showMsg('txn-msg', 'error', result.detail || 'Analysis failed.'); return; }
    renderVerdict('txn-results', result, false);
  } catch (e) { hideLoading(); showMsg('txn-msg', 'error', `Network error: ${e.message}`); }
}

// ── Text Analysis ─────────────────────────────────────────────────
async function analyzeText() {
  clearResults('text-results', 'text-msg');
  const text = document.getElementById('nlp-text').value.trim();
  if (!text) { showMsg('text-msg', 'warning', 'Please enter some text to analyse.'); return; }
  if (!state.nlpReady) { showMsg('text-msg', 'error', 'NLP model not trained. Run python train_all.py first.'); return; }

  showLoading('Running DistilBERT NLP analysis…');
  try {
    const res    = await fetch('/api/analyze/text', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ text }),
    });
    const result = await res.json();
    hideLoading();
    if (!res.ok) { showMsg('text-msg', 'error', result.detail || 'Analysis failed.'); return; }
    renderVerdict('text-results', result, false);
  } catch (e) { hideLoading(); showMsg('text-msg', 'error', `Network error: ${e.message}`); }
}

// ── Signature Verification ────────────────────────────────────────
async function analyzeSignature() {
  clearResults('sig-results', 'sig-msg');
  const refInput  = document.getElementById('file-ref');
  const testInput = document.getElementById('file-test');
  if (!refInput.files[0] || !testInput.files[0]) {
    showMsg('sig-msg', 'warning', 'Please upload both a reference and a test signature.'); return;
  }
  if (!state.cvReady) { showMsg('sig-msg', 'error', 'CV model not trained. Run python train_all.py first.'); return; }

  const formData = new FormData();
  formData.append('ref_image',  refInput.files[0]);
  formData.append('test_image', testInput.files[0]);

  showLoading('Running Siamese ResNet18 analysis…');
  try {
    const res    = await fetch('/api/analyze/signature', { method: 'POST', body: formData });
    const result = await res.json();
    hideLoading();
    if (!res.ok) { showMsg('sig-msg', 'error', result.detail || 'Analysis failed.'); return; }
    renderVerdict('sig-results', result, true);
  } catch (e) { hideLoading(); showMsg('sig-msg', 'error', `Network error: ${e.message}`); }
}

// ══════════════════════════════════════════════════════════════════
//  RENDERING
// ══════════════════════════════════════════════════════════════════

function renderVerdict(containerId, result, isSignature) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const isFraud   = result.verdict === 'FRAUD';
  const bannerCls = isFraud ? 'fraud' : 'legit';
  const label     = isSignature
    ? (isFraud ? 'FORGED SIGNATURE'     : 'GENUINE SIGNATURE')
    : (isFraud ? 'FRAUD DETECTED'       : 'LEGITIMATE TRANSACTION');
  const color     = isFraud ? 'var(--fraud)'   : 'var(--success)';
  const iconId    = isFraud ? 'ic-x-circle'    : 'ic-check';
  const modules   = result.module_results || {};

  // ── Verdict banner ──
  let html = `
    <div class="verdict-banner ${bannerCls}" style="animation:fadeUp .4s ease">
      <div class="verdict-icon-wrap">
        <svg width="26" height="26"><use href="#${iconId}"/></svg>
      </div>
      <div class="verdict-label">${label}</div>
      <div class="verdict-meta">
        Confidence: <strong>${result.confidence}%</strong>
        &nbsp;&bull;&nbsp;
        Fusion Score: <strong>${result.final_score.toFixed(4)}</strong>
      </div>
    </div>`;

  // ── Gauge row ──
  const gaugeItems = [
    { label: 'Fusion Score', score: result.final_score, color: isFraud ? '#B91C1C' : '#4D7C0F' }
  ];
  if (modules.ml)  gaugeItems.push({ label: 'ML Score',  score: modules.ml.score,  color: '#B45309' });
  if (modules.nlp) gaugeItems.push({ label: 'NLP Score', score: modules.nlp.score, color: '#C2410C' });
  if (modules.cv)  gaugeItems.push({ label: 'CV Score',  score: modules.cv.score,  color: '#4D7C0F' });

  html += `<div class="gauges-row">`;
  gaugeItems.forEach(g => { html += buildGauge(g.label, g.score, g.color); });

  if (isSignature && modules.cv?.distance !== undefined) {
    html += `
      <div class="distance-card">
        <div class="dist-label">Embedding Distance</div>
        <div class="dist-val">${parseFloat(modules.cv.distance).toFixed(4)}</div>
        <div class="dist-thr">Threshold: ${document.getElementById('sig-threshold')?.value ?? 0.5}</div>
      </div>`;
  }
  html += `</div>`;

  // ── Score bars (2-column grid) ──
  html += `<div class="score-bars-grid">`;
  gaugeItems.forEach(g => { html += buildScoreBar(g.label, g.score, g.color); });
  html += `</div>`;

  // ── Module breakdown ──
  if (Object.keys(modules).length > 0) {
    html += `
      <hr class="divider">
      <div class="breakdown-title">
        <svg width="13" height="13"><use href="#ic-activity"/></svg>
        Module Breakdown
      </div>
      <div class="module-breakdown">`;
    for (const [key, detail] of Object.entries(modules)) {
      html += buildModuleCard(key, detail);
    }
    html += `</div>`;
  }

  container.innerHTML = html;
  container.style.display = 'block';

  // Animate score bars after DOM insert
  requestAnimationFrame(() => {
    container.querySelectorAll('.score-bar-fill[data-width]').forEach(el => {
      el.style.width = el.dataset.width;
    });
  });
}

function buildGauge(label, score, color) {
  const r    = 34, cx = 42, cy = 42;
  const circ = 2 * Math.PI * r;
  const dash = score * circ;
  const pct  = Math.round(score * 100);
  return `
    <div class="gauge-card">
      <svg class="gauge-svg" width="84" height="84" viewBox="0 0 84 84">
        <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="#DDD5C4" stroke-width="7"/>
        <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${color}" stroke-width="7"
          stroke-dasharray="${dash.toFixed(1)} ${circ.toFixed(1)}"
          stroke-dashoffset="${(circ * 0.25).toFixed(1)}"
          stroke-linecap="round"
          style="transition:stroke-dasharray 1s cubic-bezier(.4,0,.2,1)"/>
        <text x="${cx}" y="${cy + 5}" text-anchor="middle" font-size="14" font-weight="800"
              fill="${color}" font-family="Inter,sans-serif">${pct}%</text>
      </svg>
      <div class="gauge-val" style="color:${color}">${score.toFixed(4)}</div>
      <div class="gauge-label">${label}</div>
    </div>`;
}

function buildScoreBar(label, score, color) {
  const pct = (score * 100).toFixed(1);
  return `
    <div class="score-bar-wrap">
      <div class="score-bar-header">
        <span class="score-bar-label">${label}</span>
        <span class="score-bar-val" style="color:${color}">${score.toFixed(4)}</span>
      </div>
      <div class="score-bar-bg">
        <div class="score-bar-fill" style="background:${color}" data-width="${pct}%"></div>
      </div>
    </div>`;
}

function buildModuleCard(key, detail) {
  const m       = MOD_META[key] || { iconId: 'ic-activity', name: key.toUpperCase(), bg: '#F1F5F9', color: '#6B7280' };
  const isFraud = ['FRAUD','PHISHING','FORGED'].some(w => (detail.verdict ?? '').toUpperCase().includes(w));
  const vColor  = isFraud ? '#DC2626' : '#16A34A';
  const vBg     = isFraud ? '#FEE2E2' : '#DCFCE7';
  return `
    <div class="module-card">
      <div class="mod-icon-wrap" style="background:${m.bg};color:${m.color}">
        <svg width="20" height="20"><use href="#${m.iconId}"/></svg>
      </div>
      <div class="mod-name" style="color:${m.color}">${m.name}</div>
      <div class="mod-score" style="color:${vColor}">${detail.score.toFixed(3)}</div>
      <div class="mod-verdict" style="color:${vColor};background:${vBg};
           display:inline-block;padding:.2rem .55rem;border-radius:20px;font-size:.68rem;margin-top:.2rem">
        ${detail.verdict}
      </div>
      ${detail.distance !== undefined
        ? `<div class="mod-dist">Distance: ${parseFloat(detail.distance).toFixed(4)}</div>`
        : ''}
    </div>`;
}

// ── Utilities ─────────────────────────────────────────────────────
function showLoading(text = 'Analysing…') {
  document.getElementById('loading-text').textContent = text;
  document.getElementById('loading-overlay').classList.add('active');
}

function hideLoading() {
  document.getElementById('loading-overlay').classList.remove('active');
}

function clearResults(resultId, msgId) {
  const el = document.getElementById(resultId);
  if (el) { el.innerHTML = ''; el.style.display = 'none'; }
  const msg = document.getElementById(msgId);
  if (msg) { msg.textContent = ''; msg.className = 'msg'; }
}

function showMsg(id, type, text) {
  const el = document.getElementById(id);
  if (!el) return;
  const iconMap = { error: 'ic-x-circle', warning: 'ic-alert', info: 'ic-info' };
  el.innerHTML = `<svg width="14" height="14"><use href="#${iconMap[type] || 'ic-info'}"/></svg>${text}`;
  el.className = `msg msg-${type} active`;
}
