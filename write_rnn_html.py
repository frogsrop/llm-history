#!/usr/bin/env python3
# Helper script to write module-2-rnn-lstm.html
content = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <title>RNN / LSTM \u2014 LLM Evolution</title>
  <link rel="stylesheet" href="/static/style.css">
  <style>
    .mode-toggle { display:flex; gap:8px; margin-bottom:20px; }
    .mode-btn {
      padding:8px 22px; border-radius:var(--radius); border:1px solid var(--border);
      background:var(--bg-card); color:var(--text-muted); font-size:14px; font-weight:600;
      cursor:pointer; transition:background var(--transition),color var(--transition),border-color var(--transition);
    }
    .mode-btn.active { background:var(--epoch-rnn); border-color:var(--epoch-rnn); color:#0d1117; }
    .rnn-svg-wrap { overflow-x:auto; margin:8px 0 4px; }
    .rnn-cell { fill:#1c1f2e; stroke:var(--epoch-rnn); stroke-width:1.5; }
    .rnn-cell-lstm { fill:#1c1f2e; stroke:#9b59b6; stroke-width:1.5; }
    .node-label { font-family:"Segoe UI",sans-serif; font-size:11px; fill:var(--epoch-rnn); font-weight:600; }
    .node-label-lstm { font-family:"Segoe UI",sans-serif; font-size:11px; fill:#c39bd3; font-weight:600; }
    .token-label { font-family:"Segoe UI",sans-serif; font-size:12px; fill:#a5d6ff; }
    .arr { fill:none; stroke:#30363d; stroke-width:1.5; marker-end:url(#arr); }
    .arr-h { fill:none; stroke:var(--epoch-rnn); stroke-width:1.8; marker-end:url(#arrH); }
    .arr-hl { fill:none; stroke:#9b59b6; stroke-width:1.8; marker-end:url(#arrHL); }
    .arr-c { fill:none; stroke:#f1948a; stroke-width:1.5; stroke-dasharray:4,3; marker-end:url(#arrC); }
    .gate-label { font-family:"Segoe UI",sans-serif; font-size:10px; fill:#c39bd3; }
    @keyframes fp { 0%,100%{stroke-opacity:.3} 50%{stroke-opacity:1} }
    .anim { animation:fp 1.2s ease-in-out infinite; }
    .gen-token { display:inline-block; padding:2px 8px; border-radius:4px; margin:2px 3px; font-size:15px; transition:background .25s,color .25s; }
    .gen-token.active { background:rgba(168,85,247,.3); color:var(--epoch-rnn); font-weight:700; }
    .gen-token.done { background:rgba(168,85,247,.08); color:var(--text); }
    .gen-token.seed { background:rgba(255,255,255,.06); color:var(--text-muted); }
    .word-input { background:var(--bg); border:1px solid var(--border); border-radius:var(--radius); color:var(--text); padding:8px 12px; font-size:14px; width:160px; outline:none; transition:border-color .2s; }
    .word-input:focus { border-color:var(--epoch-rnn); }
    .controls-row { display:flex; align-items:center; gap:16px; flex-wrap:wrap; margin-bottom:16px; }
    .cmp th { text-align:center; }
    .cmp td { text-align:center; font-size:13px; }
    .cmp td:first-child { text-align:left; font-weight:600; }
    .tg { color:#3fb950; font-weight:600; }
    .tb { color:var(--epoch-llm); }
    .tm { color:var(--epoch-embeddings); }
  </style>
</head>
<body>
  <nav id="sidebar"></nav>
  <main id="main-content">
    <h1 class="section-title"><span class="epoch-bar epoch-rnn"></span>RNN / LSTM</h1>
    <p class="section-sub">1986\u20131997 \u2014 \u043d\u0435\u0439\u0440\u043e\u043d\u043d\u0430\u044f \u0441\u0435\u0442\u044c \u0441 \u043f\u0430\u043c\u044f\u0442\u044c\u044e \u043e \u043f\u0440\u043e\u0448\u043b\u043e\u043c</p>

    <div class="mode-toggle">
      <button class="mode-btn active" id="btn-rnn">RNN</button>
      <button class="mode-btn" id="btn-lstm">LSTM</button>
    </div>

    <!-- Architecture diagram -->
    <div class="card">
      <div class="card-title" id="diagram-title">\u0410\u0440\u0445\u0438\u0442\u0435\u043a\u0442\u0443\u0440\u0430 RNN
        <span class="term" style="margin-left:8px;font-size:14px;font-weight:400;color:var(--text-muted);">hidden state
          <button class="hint-btn" data-hint="Hidden state \u2014 \u2018\u0440\u0430\u0431\u043e\u0447\u0430\u044f \u043f\u0430\u043c\u044f\u0442\u044c\u2019 \u0441\u0435\u0442\u0438. \u041f\u0435\u0440\u0435\u0434\u0430\u0451\u0442\u0441\u044f \u043e\u0442 \u0448\u0430\u0433\u0430 \u043a \u0448\u0430\u0433\u0443 \u043a\u0430\u043a \u044d\u0441\u0442\u0430\u0444\u0435\u0442\u043d\u0430\u044f \u043f\u0430\u043b\u043e\u0447\u043a\u0430. \u041d\u043e \u043f\u0430\u043b\u043e\u0447\u043a\u0430 \u043c\u0430\u043b\u0435\u043d\u044c\u043a\u0430\u044f \u2014 \u043c\u043d\u043e\u0433\u043e \u043d\u0435 \u0443\u043d\u0435\u0441\u0451\u0442.">?</button>
        </span>
      </div>
      <!-- RNN SVG -->
      <div class="rnn-svg-wrap" id="svg-rnn">
        <svg class="diagram-svg" viewBox="0 0 680 180" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <marker id="arr"  markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3z" fill="#30363d"/></marker>
            <marker id="arrH" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3z" fill="#A855F7"/></marker>
          </defs>
          <text x="60"  y="168" class="token-label" text-anchor="middle">\u043a\u0443\u0437\u043d\u0435\u0446</text>
          <text x="190" y="168" class="token-label" text-anchor="middle">\u043a\u043e\u0432\u0430\u043b</text>
          <text x="320" y="168" class="token-label" text-anchor="middle">\u0436\u0435\u043b\u0435\u0437\u043e</text>
          <text x="450" y="168" class="token-label" text-anchor="middle">\u043f\u043e\u043a\u0430</text>
          <text x="580" y="168" class="token-label" text-anchor="middle">\u0436\u0435\u043b\u0435\u0437\u043e</text>
          <line x1="60"  y1="156" x2="60"  y2="122" class="arr"/>
          <line x1="190" y1="156" x2="190" y2="122" class="arr"/>
          <line x1="320" y1="156" x2="320" y2="122" class="arr"/>
          <line x1="450" y1="156" x2="450" y2="122" class="arr"/>
          <line x1="580" y1="156" x2="580" y2="122" class="arr"/>
          <rect x="30"  y="80" width="60" height="40" rx="6" class="rnn-cell"/>
          <rect x="160" y="80" width="60" height="40" rx="6" class="rnn-cell"/>
          <rect x="290" y="80" width="60" height="40" rx="6" class="rnn-cell"/>
          <rect x="420" y="80" width="60" height="40" rx="6" class="rnn-cell"/>
          <rect x="550" y="80" width="60" height="40" rx="6" class="rnn-cell"/>
          <text x="60"  y="104" class="node-label" text-anchor="middle">h\u2081</text>
          <text x="190" y="104" class="node-label" text-anchor="middle">h\u2082</text>
          <text x="320" y="104" class="node-label" text-anchor="middle">h\u2083</text>
          <text x="450" y="104" class="node-label" text-anchor="middle">h\u2084</text>
          <text x="580" y="104" class="node-label" text-anchor="middle">h\u2085</text>
          <line x1="90"  y1="100" x2="158" y2="100" class="arr-h anim"/>
          <line x1="220" y1="100" x2="288" y2="100" class="arr-h anim"/>
          <line x1="350" y1="100" x2="418" y2="100" class="arr-h anim"/>
          <line x1="480" y1="100" x2="548" y2="100" class="arr-h anim"/>
          <line x1="580" y1="80" x2="580" y2="50" class="arr"/>
          <rect x="548" y="30" width="64" height="22" rx="4" fill="var(--bg-hover)" stroke="var(--border)" stroke-width="1"/>
          <text x="580" y="45" font-family="Segoe UI" font-size="11" fill="var(--text-muted)" text-anchor="middle">y (next)</text>
        </svg>
      </div>
      <!-- LSTM SVG -->
      <div class="rnn-svg-wrap" id="svg-lstm" style="display:none">
        <svg class="diagram-svg" viewBox="0 0 680 230" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <marker id="arrHL" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3z" fill="#9b59b6"/></marker>
            <marker id="arrC"  markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3z" fill="#f1948a"/></marker>
          </defs>
          <text x="60"  y="220" class="token-label" text-anchor="middle">\u043a\u0443\u0437\u043d\u0435\u0446</text>
          <text x="190" y="220" class="token-label" text-anchor="middle">\u043a\u043e\u0432\u0430\u043b</text>
          <text x="320" y="220" class="token-label" text-anchor="middle">\u0436\u0435\u043b\u0435\u0437\u043e</text>
          <text x="450" y="220" class="token-label" text-anchor="middle">\u043f\u043e\u043a\u0430</text>
          <text x="580" y="220" class="token-label" text-anchor="middle">\u0436\u0435\u043b\u0435\u0437\u043e</text>
          <line x1="60"  y1="210" x2="60"  y2="176" class="arr"/>
          <line x1="190" y1="210" x2="190" y2="176" class="arr"/>
          <line x1="320" y1="210" x2="320" y2="176" class="arr"/>
          <line x1="450" y1="210" x2="450" y2="176" class="arr"/>
          <line x1="580" y1="210" x2="580" y2="176" class="arr"/>
          <rect x="30"  y="120" width="60" height="55" rx="6" class="rnn-cell-lstm"/>
          <rect x="160" y="120" width="60" height="55" rx="6" class="rnn-cell-lstm"/>
          <rect x="290" y="120" width="60" height="55" rx="6" class="rnn-cell-lstm"/>
          <rect x="420" y="120" width="60" height="55" rx="6" class="rnn-cell-lstm"/>
          <rect x="550" y="120" width="60" height="55" rx="6" class="rnn-cell-lstm"/>
          <text x="60"  y="140" class="node-label-lstm" text-anchor="middle">h</text>
          <text x="60"  y="154" class="gate-label" text-anchor="middle">f\u00b7i\u00b7g\u00b7o</text>
          <text x="190" y="140" class="node-label-lstm" text-anchor="middle">h</text>
          <text x="190" y="154" class="gate-label" text-anchor="middle">f\u00b7i\u00b7g\u00b7o</text>
          <text x="320" y="140" class="node-label-lstm" text-anchor="middle">h</text>
          <text x="320" y="154" class="gate-label" text-anchor="middle">f\u00b7i\u00b7g\u00b7o</text>
          <text x="450" y="140" class="node-label-lstm" text-anchor="middle">h</text>
          <text x="450" y="154" class="gate-label" text-anchor="middle">f\u00b7i\u00b7g\u00b7o</text>
          <text x="580" y="140" class="node-label-lstm" text-anchor="middle">h</text>
          <text x="580" y="154" class="gate-label" text-anchor="middle">f\u00b7i\u00b7g\u00b7o</text>
          <line x1="90"  y1="150" x2="158" y2="150" class="arr-hl anim"/>
          <line x1="220" y1="150" x2="288" y2="150" class="arr-hl anim"/>
          <line x1="350" y1="150" x2="418" y2="150" class="arr-hl anim"/>
          <line x1="480" y1="150" x2="548" y2="150" class="arr-hl anim"/>
          <line x1="90"  y1="125" x2="158" y2="125" class="arr-c anim"/>
          <line x1="220" y1="125" x2="288" y2="125" class="arr-c anim"/>
          <line x1="350" y1="125" x2="418" y2="125" class="arr-c anim"/>
          <line x1="480" y1="125" x2="548" y2="125" class="arr-c anim"/>
          <text x="10" y="128" font-family="Segoe UI" font-size="10" fill="#f1948a">c</text>
          <text x="10" y="153" font-family="Segoe UI" font-size="10" fill="#9b59b6">h</text>
          <line x1="580" y1="120" x2="580" y2="85" class="arr"/>
          <rect x="548" y="64" width="64" height="22" rx="4" fill="var(--bg-hover)" stroke="var(--border)" stroke-width="1"/>
          <text x="580" y="79" font-family="Segoe UI" font-size="11" fill="var(--text-muted)" text-anchor="middle">y (next)</text>
          <line x1="20" y1="30" x2="50" y2="30" class="arr-c"/>
          <text x="55" y="34" font-family="Segoe UI" font-size="10" fill="#f1948a">cell state c (\u0434\u043e\u043b\u0433\u043e\u0432\u0440\u0435\u043c\u0435\u043d\u043d\u0430\u044f \u043f\u0430\u043c\u044f\u0442\u044c)</text>
          <line x1="20" y1="48" x2="50" y2="48" class="arr-hl"/>
          <text x="55" y="52" font-family="Segoe UI" font-size="10" fill="#9b59b6">hidden state h (\u043a\u0440\u0430\u0442\u043a\u043e\u0441\u0440\u043e\u0447\u043d\u0430\u044f)</text>
        </svg>
      </div>
    </div>

    <!-- Hidden size slider -->
    <div class="card">
      <div class="card-title">\u0421\u043b\u0430\u0439\u0434\u0435\u0440 \u0451\u043c\u043a\u043e\u0441\u0442\u0438
        <span class="term" style="margin-left:8px;font-size:14px;font-weight:400;color:var(--text-muted);">hidden size
          <button class="hint-btn" data-hint="Hidden size \u2014 \u0441\u043a\u043e\u043b\u044c\u043a\u043e \u2018\u044f\u0447\u0435\u0435\u043a\u2019 \u0432 \u0440\u0430\u0431\u043e\u0447\u0435\u0439 \u043f\u0430\u043c\u044f\u0442\u0438. \u0411\u043e\u043b\u044c\u0448\u0435 \u044f\u0447\u0435\u0435\u043a = \u0431\u043e\u043b\u044c\u0448\u0435 \u0434\u0435\u0442\u0430\u043b\u0435\u0439 \u043c\u043e\u0436\u0435\u0442 \u0443\u0434\u0435\u0440\u0436\u0430\u0442\u044c.">?</button>
        </span>
      </div>
      <div class="slider-wrap">
        <span class="slider-label">Hidden =</span>
        <input type="range" id="hidden-size" min="0" max="3" value="1" step="1">
        <span class="slider-value" id="hidden-size-val">8</span>
        <span style="font-size:13px;color:var(--text-muted);margin-left:8px;">\u043d\u0435\u0439\u0440\u043e\u043d\u043e\u0432 \u0432 \u0441\u043b\u043e\u0435</span>
      </div>
    </div>

    <!-- Generation -->
    <div class="card">
      <div class="card-title">\u0413\u0435\u043d\u0435\u0440\u0430\u0446\u0438\u044f</div>
      <div class="controls-row">
        <div style="display:flex;align-items:center;gap:8px;">
          <span style="font-size:14px;color:var(--text-muted);">\u041d\u0430\u0447\u0430\u0442\u044c \u0441\u043e \u0441\u043b\u043e\u0432\u0430:</span>
          <input type="text" id="start-word" class="word-input" value="\u0436\u0435\u043b\u0435\u0437\u043e" list="corpus-words">
          <datalist id="corpus-words">
            <option value="\u043a\u0443\u0437\u043d\u0435\u0446"><option value="\u043a\u043e\u0432\u0430\u043b"><option value="\u0436\u0435\u043b\u0435\u0437\u043e">
            <option value="\u0433\u043e\u0440\u044f\u0447\u0438\u043c"><option value="\u043c\u043e\u043b\u043e\u0442"><option value="\u0444\u043e\u0440\u043c\u0443">
          </datalist>
        </div>
        <div style="display:flex;align-items:center;gap:8px;">
          <span style="font-size:14px;color:var(--text-muted);">\u0421\u043b\u043e\u0432:</span>
          <input type="range" id="words-count" min="1" max="20" value="5" step="1" style="width:80px;">
          <span id="words-count-val" style="font-size:14px;font-weight:600;min-width:20px;">5</span>
        </div>
        <button class="btn btn-primary" id="generate-btn">\u25b6 \u0413\u0435\u043d\u0435\u0440\u0438\u0440\u043e\u0432\u0430\u0442\u044c</button>
      </div>
      <div class="result-box" id="generation-result" style="min-height:52px;line-height:2;">
        <span style="color:var(--text-muted);">\u041d\u0430\u0436\u043c\u0438\u0442\u0435 \u00ab\u0413\u0435\u043d\u0435\u0440\u0438\u0440\u043e\u0432\u0430\u0442\u044c\u00bb</span>
      </div>
    </div>

    <!-- Problem callout -->
    <div class="card">
      <div class="card-title">\u041f\u0440\u043e\u0431\u043b\u0435\u043c\u0430 RNN: \u0437\u0430\u0442\u0443\u0445\u0430\u044e\u0449\u0438\u0439 \u0433\u0440\u0430\u0434\u0438\u0435\u043d\u0442
        <span class="term" style="margin-left:8px;font-size:14px;font-weight:400;color:var(--text-muted);">vanishing gradient
          <button class="hint-btn" data-hint="\u0413\u0440\u0430\u0434\u0438\u0435\u043d\u0442 \u2014 \u0441\u0438\u0433\u043d\u0430\u043b \u043e\u0431\u0440\u0430\u0442\u043d\u043e\u0439 \u0441\u0432\u044f\u0437\u0438 \u043f\u0440\u0438 \u043e\u0431\u0443\u0447\u0435\u043d\u0438\u0438. \u041a\u043e\u0433\u0434\u0430 \u043e\u043d \u043f\u0440\u043e\u0445\u043e\u0434\u0438\u0442 \u0447\u0435\u0440\u0435\u0437 \u043c\u043d\u043e\u0433\u043e \u0448\u0430\u0433\u043e\u0432 \u2014 \u0437\u0430\u0442\u0443\u0445\u0430\u0435\u0442 \u043a\u0430\u043a \u044d\u0445\u043e \u0432 \u0434\u043b\u0438\u043d\u043d\u043e\u043c \u043a\u043e\u0440\u0438\u0434\u043e\u0440\u0435.">?</button>
        </span>
      </div>
      <div class="callout warn">
        <strong>\u0414\u043e\u043b\u0433\u0438\u0439 \u043f\u0443\u0442\u044c \u2192 \u0441\u043b\u0430\u0431\u044b\u0439 \u0441\u0438\u0433\u043d\u0430\u043b:</strong> hidden state \u043f\u0435\u0440\u0435\u0434\u0430\u0451\u0442\u0441\u044f \u0447\u0435\u0440\u0435\u0437 \u0443\u043c\u043d\u043e\u0436\u0435\u043d\u0438\u044f.
        \u041f\u0440\u0438 \u0434\u043b\u0438\u043d\u043d\u044b\u0445 \u043f\u043e\u0441\u043b\u0435\u0434\u043e\u0432\u0430\u0442\u0435\u043b\u044c\u043d\u043e\u0441\u0442\u044f\u0445 \u0441\u0438\u0433\u043d\u0430\u043b \u0437\u0430\u0442\u0443\u0445\u0430\u0435\u0442 \u2014 \u0440\u0430\u043d\u043d\u0438\u0435 \u0442\u043e\u043a\u0435\u043d\u044b \u043f\u043e\u0447\u0442\u0438 \u043d\u0435 \u0432\u043b\u0438\u044f\u044e\u0442 \u043d\u0430 \u043f\u0440\u0435\u0434\u0441\u043a\u0430\u0437\u0430\u043d\u0438\u0435.<br><br>
        <strong>\u042d\u0442\u043e \u043d\u0435 \u0436\u0451\u0441\u0442\u043a\u0438\u0439 \u043e\u0431\u0440\u044b\u0432</strong>, \u043a\u0430\u043a \u0443 N-gram, \u0430 \u043f\u043e\u0441\u0442\u0435\u043f\u0435\u043d\u043d\u0430\u044f \u0434\u0435\u0433\u0440\u0430\u0434\u0430\u0446\u0438\u044f.
        \u00ab\u041a\u0443\u0437\u043d\u0435\u0446\u00bb \u043a \u043a\u043e\u043d\u0446\u0443 \u043f\u0440\u0435\u0434\u043b\u043e\u0436\u0435\u043d\u0438\u044f \u043f\u043e\u043c\u043d\u0438\u0442\u0441\u044f \u0445\u0443\u0436\u0435, \u0447\u0435\u043c \u00ab\u0433\u043e\u0440\u044f\u0447\u0438\u043c\u00bb.
      </div>
    </div>

    <!-- RNN vs Transformer table -->
    <div class="card">
      <div class="card-title">RNN vs Transformer</div>
      <table class="cmp">
        <thead><tr>
          <th>\u0410\u0441\u043f\u0435\u043a\u0442</th>
          <th style="color:var(--epoch-rnn)">RNN / LSTM</th>
          <th style="color:#58a6ff">Transformer</th>
        </tr></thead>
        <tbody>
          <tr><td>\u0414\u043e\u0441\u0442\u0443\u043f \u043a \u043f\u0440\u043e\u0448\u043b\u043e\u043c\u0443</td><td class="tm">\u0417\u0430\u0442\u0443\u0445\u0430\u044e\u0449\u0438\u0439</td><td class="tg">\u0420\u0430\u0432\u043d\u044b\u0439 (Attention)</td></tr>
          <tr><td>\u041f\u0430\u0440\u0430\u043b\u043b\u0435\u043b\u0438\u0437\u043c</td><td class="tb">\u041d\u0435\u0442 (\u043f\u043e\u0441\u043b\u0435\u0434\u043e\u0432\u0430\u0442\u0435\u043b\u044c\u043d\u043e)</td><td class="tg">\u0414\u0430 (\u0432\u0441\u0435 \u0442\u043e\u043a\u0435\u043d\u044b \u0441\u0440\u0430\u0437\u0443)</td></tr>
          <tr><td>\u0414\u043b\u0438\u043d\u043d\u044b\u0435 \u0437\u0430\u0432\u0438\u0441\u0438\u043c\u043e\u0441\u0442\u0438</td><td class="tm">\u0427\u0430\u0441\u0442\u0438\u0447\u043d\u043e (LSTM)</td><td class="tg">\u0425\u043e\u0440\u043e\u0448\u043e</td></tr>
          <tr><td>\u0421\u043a\u043e\u0440\u043e\u0441\u0442\u044c \u043e\u0431\u0443\u0447\u0435\u043d\u0438\u044f</td><td class="tb">\u041c\u0435\u0434\u043b\u0435\u043d\u043d\u043e</td><td class="tg">\u0411\u044b\u0441\u0442\u0440\u043e (GPU)</td></tr>
          <tr><td>\u041f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u043e\u0432</td><td class="tg">\u041c\u0430\u043b\u043e</td><td class="tb">\u041c\u0438\u043b\u043b\u0438\u0430\u0440\u0434\u044b</td></tr>
        </tbody>
      </table>
    </div>

    <!-- LSTM gate explanation (only in LSTM mode) -->
    <div class="card" id="lstm-explain" style="display:none">
      <div class="card-title">LSTM: \u0432\u043e\u0440\u043e\u0442\u0430 \u0437\u0430\u0431\u044b\u0432\u0430\u043d\u0438\u044f
        <span class="term" style="margin-left:8px;font-size:14px;font-weight:400;color:var(--text-muted);">forget gate
          <button class="hint-btn" data-hint="Forget gate \u2014 \u2018\u0432\u0435\u043d\u0442\u0438\u043b\u044c\u2019 \u043a\u043e\u0442\u043e\u0440\u044b\u0439 \u0440\u0435\u0448\u0430\u0435\u0442, \u0447\u0442\u043e \u043e\u0441\u0442\u0430\u0432\u0438\u0442\u044c \u0432 \u043f\u0430\u043c\u044f\u0442\u0438, \u0430 \u0447\u0442\u043e \u0432\u044b\u0431\u0440\u043e\u0441\u0438\u0442\u044c. \u041a\u0430\u043a \u0447\u0435\u043b\u043e\u0432\u0435\u043a, \u043a\u043e\u0442\u043e\u0440\u044b\u0439 \u0440\u0435\u0448\u0430\u0435\u0442 \u0447\u0442\u043e \u0437\u0430\u043f\u0438\u0441\u0430\u0442\u044c \u0432 \u0431\u043b\u043e\u043a\u043d\u043e\u0442, \u0430 \u0447\u0442\u043e \u043d\u0435\u0442.">?</button>
        </span>
      </div>
      <div class="callout" style="border-color:var(--epoch-rnn)">
        LSTM \u0434\u043e\u0431\u0430\u0432\u043b\u044f\u0435\u0442 <strong>cell state (c)</strong> \u2014 \u0434\u043e\u043b\u0433\u043e\u0441\u0440\u043e\u0447\u043d\u0443\u044e \u043b\u0435\u043d\u0442\u0443 \u043f\u0430\u043c\u044f\u0442\u0438 \u043e\u0442\u0434\u0435\u043b\u044c\u043d\u043e \u043e\u0442 hidden state.
        \u0427\u0435\u0442\u044b\u0440\u0435 \u0432\u0435\u043d\u0442\u0438\u043b\u044f \u0443\u043f\u0440\u0430\u0432\u043b\u044f\u044e\u0442 \u043f\u043e\u0442\u043e\u043a\u043e\u043c:<br><br>
        <strong>f</strong> (forget) \u2014 \u0447\u0442\u043e \u0437\u0430\u0431\u044b\u0442\u044c \u0438\u0437 \u043f\u0440\u043e\u0448\u043b\u043e\u0433\u043e<br>
        <strong>i</strong> (input) \u2014 \u0447\u0442\u043e \u0437\u0430\u043f\u0438\u0441\u0430\u0442\u044c \u0438\u0437 \u043d\u043e\u0432\u043e\u0433\u043e \u0432\u0445\u043e\u0434\u0430<br>
        <strong>g</strong> (gate) \u2014 \u043a\u0430\u043d\u0434\u0438\u0434\u0430\u0442\u044b \u0434\u043b\u044f \u0437\u0430\u043f\u0438\u0441\u0438<br>
        <strong>o</strong> (output) \u2014 \u0447\u0442\u043e \u0432\u044b\u0434\u0430\u0442\u044c \u043d\u0430\u0440\u0443\u0436\u0443<br><br>
        \u0411\u043b\u0430\u0433\u043e\u0434\u0430\u0440\u044f \u043f\u0440\u044f\u043c\u043e\u043c\u0443 \u0433\u0440\u0430\u0434\u0438\u0435\u043d\u0442\u043d\u043e\u043c\u0443 \u043f\u0443\u0442\u0438 \u0447\u0435\u0440\u0435\u0437 c \u0441\u0435\u0442\u044c \u043b\u0443\u0447\u0448\u0435 \u043e\u0431\u0443\u0447\u0430\u0435\u0442\u0441\u044f \u043d\u0430 \u0434\u043b\u0438\u043d\u043d\u044b\u0445 \u043f\u043e\u0441\u043b\u0435\u0434\u043e\u0432\u0430\u0442\u0435\u043b\u044c\u043d\u043e\u0441\u0442\u044f\u0445.
      </div>
    </div>
  </main>

  <script src="/static/nav.js"></script>
  <script src="/static/utils.js"></script>
  <script src="/static/tooltip.js"></script>
  <script>
    const HIDDEN_SIZES = [4, 8, 16, 32];
    let currentMode = "rnn";

    document.getElementById("btn-rnn").addEventListener("click", () => setMode("rnn"));
    document.getElementById("btn-lstm").addEventListener("click", () => setMode("lstm"));

    function setMode(mode) {
      currentMode = mode;
      document.getElementById("btn-rnn").classList.toggle("active", mode === "rnn");
      document.getElementById("btn-lstm").classList.toggle("active", mode === "lstm");
      document.getElementById("svg-rnn").style.display = mode === "rnn" ? "" : "none";
      document.getElementById("svg-lstm").style.display = mode === "lstm" ? "" : "none";
      const titleEl = document.getElementById("diagram-title");
      titleEl.childNodes[0].textContent = mode === "rnn" ? "\u0410\u0440\u0445\u0438\u0442\u0435\u043a\u0442\u0443\u0440\u0430 RNN" : "\u0410\u0440\u0445\u0438\u0442\u0435\u043a\u0442\u0443\u0440\u0430 LSTM";
      document.getElementById("lstm-explain").style.display = mode === "lstm" ? "" : "none";
    }

    const hiddenSlider = document.getElementById("hidden-size");
    const hiddenVal = document.getElementById("hidden-size-val");
    hiddenSlider.addEventListener("input", () => {
      hiddenVal.textContent = HIDDEN_SIZES[parseInt(hiddenSlider.value)];
    });

    const wordsSlider = document.getElementById("words-count");
    const wordsVal = document.getElementById("words-count-val");
    wordsSlider.addEventListener("input", () => { wordsVal.textContent = wordsSlider.value; });

    async function generate() {
      const resultEl = document.getElementById("generation-result");
      const start = document.getElementById("start-word").value.trim() || "\u0436\u0435\u043b\u0435\u0437\u043e";
      const hiddenSize = HIDDEN_SIZES[parseInt(hiddenSlider.value)];
      const words = parseInt(wordsSlider.value);
      const seed = typeof getSeed === "function" ? getSeed() : 42;

      const endpoint = currentMode === "lstm"
        ? `/api/rnn/lstm/generate?hidden_size=${hiddenSize}&start=${encodeURIComponent(start)}&words=${words}&seed=${seed}`
        : `/api/rnn/generate?hidden_size=${hiddenSize}&start=${encodeURIComponent(start)}&words=${words}&seed=${seed}`;

      showLoading(resultEl);
      try {
        const data = await apiFetch(endpoint);
        const allTokens = [data.start, ...data.words];
        resultEl.innerHTML = allTokens.map((w, i) =>
          `<span class="gen-token ${i === 0 ? "seed" : ""}">${w}</span>`
        ).join(" ");
        const newEls = Array.from(resultEl.querySelectorAll(".gen-token:not(.seed)"));
        animateFlow(newEls, {
          delay: 400, cls: "active", keep: false,
          onStep: el => el.classList.add("active"),
          onDone: () => newEls.forEach(el => el.classList.add("done")),
        });
        if (data.fallback_used) {
          const note = document.createElement("div");
          note.style.cssText = "font-size:12px;color:var(--text-muted);margin-top:6px;";
          note.textContent = `\u0421\u043b\u043e\u0432\u043e "${start}" \u043d\u0435 \u043d\u0430\u0439\u0434\u0435\u043d\u043e \u0432 \u043a\u043e\u0440\u043f\u0443\u0441\u0435, \u0438\u0441\u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u043d: "${data.start}"`;
          resultEl.appendChild(note);
        }
      } catch(e) {
        showError(resultEl, e.message);
      }
    }

    document.getElementById("generate-btn").addEventListener("click", generate);
    document.addEventListener("animate", generate);

    document.addEventListener("DOMContentLoaded", () => {
      hiddenVal.textContent = HIDDEN_SIZES[parseInt(hiddenSlider.value)];
      initTooltips(document.getElementById("main-content"));
    });
  </script>
</body>
</html>"""

with open(r'c:/Projects/test/research/templates/module-2-rnn-lstm.html', 'w', encoding='utf-8') as f:
    f.write(content)
print('Done, size:', len(content))
