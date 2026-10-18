import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ------------------------------------------------------------------ konfigurasi
const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_LOCAL = "hand_landmarker.task";   // taruh di folder yang sama dengan index.html
const MODEL_REMOTE = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task";
const PINCH_ON = 0.42, PINCH_OFF = 0.58;       // rasio jarak jempol-telunjuk / ukuran tangan
const HOLD_MS = 700, AUTO_MS = 1400, MIN_FRAME = 110, SHATTER_MS = 950, SOLVED_MS = 5500;
const ACCENT = "#ffd60a", OK = "#4be37a";
const HAND_CONNECTIONS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];

// ------------------------------------------------------------------ elemen
const $ = s => document.querySelector(s);
const video = $("#webcam"), scene = $("#scene"), ctx = scene.getContext("2d");
const base = document.createElement("canvas"), bctx = base.getContext("2d");
const elStatus = $("#status"), elHint = $("#hint"), elLoader = $("#loader");
const elTime = $("#time"), elMoves = $("#moves"), gallery = $("#gallery"), elEmpty = $("#empty");
const selGrid = $("#grid"), rngBright = $("#bright"), chkMono = $("#mono");
const rngShadow = $("#shadow"), chkAuto = $("#auto"), chkSound = $("#sound"), rngDenoise = $("#denoise");
const elBest = $("#best"), elBestLbl = $("#bestLbl");
const gammaEls = ["#gR", "#gG", "#gB"].map(s => $(s));
const applyGamma = () => gammaEls.forEach(e => e.setAttribute("exponent", rngShadow.value));
rngShadow.addEventListener("input", applyGamma); applyGamma();

// ------------------------------------------------------------------ state
let W = 1280, H = 720;
let landmarker = null, lastVT = -1, lastTs = 0;
let hands = [], pinchState = [], smooth = [];
let state = "idle";                 // idle | shatter | puzzle | solved
let holdStart = 0, shatterStart = 0, solvedAt = 0, flash = 0;
let puzzle = null, wasPinch = false, lastPt = { x: 0, y: 0 };
let galleryCount = 0;
let anchor = null, steadyStart = 0, vig = null;

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const easeOut = t => 1 - Math.pow(1 - t, 3);

// ------------------------------------------------------------------ kamera & model
async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 30 }, facingMode: "user" }, audio: false });
  video.srcObject = stream;
  try {                                    // coba naikkan eksposur kamera (tidak semua webcam mendukung)
    const t = stream.getVideoTracks()[0], caps = (t.getCapabilities && t.getCapabilities()) || {}, adv = [];
    if (caps.exposureMode && caps.exposureMode.includes("continuous")) adv.push({ exposureMode: "continuous" });
    if (caps.exposureCompensation) adv.push({ exposureCompensation: Math.min(caps.exposureCompensation.max, 1.5) });
    if (adv.length) await t.applyConstraints({ advanced: adv });
  } catch (e) {}
  await video.play();
  const vw = video.videoWidth || 1280, vh = video.videoHeight || 720, sc = Math.min(1, 1280 / vw);   // proses maks. 1280px
  W = scene.width = base.width = Math.round(vw * sc);
  H = scene.height = base.height = Math.round(vh * sc);
  document.querySelector(".stage").style.aspectRatio = `${W} / ${H}`;
  try { const fps = stream.getVideoTracks()[0].getSettings().frameRate;
    $("#camInfo").textContent = `${vw}×${vh}` + (fps ? ` @${Math.round(fps)}fps` : ""); } catch (e) {}
}

async function pickModel() {
  try { const r = await fetch(MODEL_LOCAL, { method: "HEAD" }); if (r.ok) return MODEL_LOCAL; } catch (e) {}
  return MODEL_REMOTE;
}

async function loadModel() {
  const fileset = await FilesetResolver.forVisionTasks(WASM);
  const modelAssetPath = await pickModel();
  const opts = delegate => ({
    baseOptions: { modelAssetPath, delegate },
    runningMode: "VIDEO", numHands: 2,
    minHandDetectionConfidence: 0.4, minHandPresenceConfidence: 0.4, minTrackingConfidence: 0.4 });
  try { landmarker = await HandLandmarker.createFromOptions(fileset, opts("GPU")); }
  catch (e) { console.warn("GPU gagal, pakai CPU", e); landmarker = await HandLandmarker.createFromOptions(fileset, opts("CPU")); }
}

// ------------------------------------------------------------------ tangan
function parseHands(res) {
  const out = [];
  for (const lm of (res && res.landmarks) || []) {
    const pts = lm.map(p => ({ x: p.x * W, y: p.y * H }));
    const size = dist(pts[0], pts[9]) || 1;
    out.push({ pts, ratio: dist(pts[4], pts[8]) / size,
      pt: { x: (pts[4].x + pts[8].x) / 2, y: (pts[4].y + pts[8].y) / 2 } });
  }
  out.sort((a, b) => a.pt.x - b.pt.x);
  pinchState.length = out.length; smooth.length = out.length;
  out.forEach((h, i) => {
    h.pinch = h.ratio < (pinchState[i] ? PINCH_OFF : PINCH_ON);
    pinchState[i] = h.pinch;
    const s = smooth[i];
    if (s && dist(s, h.pt) < 250) h.pt = { x: s.x + (h.pt.x - s.x) * .55, y: s.y + (h.pt.y - s.y) * .55 };
    smooth[i] = { ...h.pt };
  });
  return out;
}

function drawHands() {
  const k = W / 1280;
  for (const h of hands) {
    ctx.shadowColor = h.pinch ? OK : "rgba(255,255,255,.8)"; ctx.shadowBlur = 8 * k;
    ctx.strokeStyle = h.pinch ? OK : "rgba(255,255,255,.75)";
    ctx.fillStyle = ctx.strokeStyle; ctx.lineWidth = 2 * k;
    ctx.beginPath();
    for (const [a, b] of HAND_CONNECTIONS) { ctx.moveTo(h.pts[a].x, h.pts[a].y); ctx.lineTo(h.pts[b].x, h.pts[b].y); }
    ctx.stroke();
    for (const p of h.pts) { ctx.beginPath(); ctx.arc(p.x, p.y, 3 * k, 0, 7); ctx.fill(); }
    ctx.strokeStyle = h.pinch ? OK : ACCENT; ctx.lineWidth = 3 * k;
    ctx.beginPath(); ctx.arc(h.pt.x, h.pt.y, (h.pinch ? 9 : 14) * k, 0, 7); ctx.stroke();
    ctx.font = `600 ${13 * k}px ui-monospace, Menlo, Consolas, monospace`;
    ctx.fillStyle = h.pinch ? OK : "#fff";
    ctx.shadowBlur = 0;
    ctx.fillText((h.pinch ? "CUBIT " : "buka ") + h.ratio.toFixed(2), h.pt.x + 18 * k, h.pt.y - 12 * k);
  }
}

// ------------------------------------------------------------------ frame kuning
function drawFrame(r, ok, prog) {
  const k = W / 1280;
  ctx.strokeStyle = ACCENT; ctx.lineWidth = 3 * k; ctx.globalAlpha = ok ? 1 : .45;
  ctx.strokeRect(r.x, r.y, r.w, r.h);
  const L = Math.min(34 * k, r.w / 3, r.h / 3);
  ctx.lineWidth = 7 * k;
  [[r.x, r.y, 1, 1], [r.x + r.w, r.y, -1, 1], [r.x, r.y + r.h, 1, -1], [r.x + r.w, r.y + r.h, -1, -1]].forEach(([x, y, sx, sy]) => {
    ctx.beginPath(); ctx.moveTo(x + sx * L, y); ctx.lineTo(x, y); ctx.lineTo(x, y + sy * L); ctx.stroke();
  });
  ctx.globalAlpha = 1;
  ctx.font = `600 ${14 * k}px ui-monospace, Menlo, Consolas, monospace`;
  ctx.fillStyle = ACCENT;
  ctx.fillText(`FRAME ${Math.round(r.w)}×${Math.round(r.h)}`, r.x, Math.max(16 * k, r.y - 8 * k));
  if (prog > 0) {
    ctx.fillStyle = "rgba(0,0,0,.6)"; ctx.fillRect(r.x, r.y + r.h + 10 * k, r.w, 8 * k);
    ctx.fillStyle = OK; ctx.fillRect(r.x, r.y + r.h + 10 * k, r.w * Math.min(prog, 1), 8 * k);
  }
}

// ------------------------------------------------------------------ foto & puzzle
function stylize(pc, w, h) {
  const img = pc.getImageData(0, 0, w, h), d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    let v = .299 * d[i] + .587 * d[i + 1] + .114 * d[i + 2];
    v = (v - 128) * 1.25 + 128 + (Math.random() - .5) * 30;
    d[i] = d[i + 1] = d[i + 2] = clamp(v, 0, 255);
  }
  pc.putImageData(img, 0, 0);
}

function addGallery(photo) {
  const url = photo.toDataURL("image/jpeg", .92);
  galleryCount++;
  const a = document.createElement("a");
  a.className = "thumb"; a.href = url; a.download = `puzzle-cam-${galleryCount}.jpg`; a.title = "Klik untuk unduh";
  const img = new Image(); img.src = url; a.appendChild(img);
  gallery.prepend(a); elEmpty.hidden = true;
  return a;
}

// ------------------------------------------------------------------ suara
let actx = null;
function unlockAudio() {
  try { actx ||= new (window.AudioContext || window.webkitAudioContext)(); if (actx.state === "suspended") actx.resume(); } catch (e) {}
}
addEventListener("pointerdown", unlockAudio); addEventListener("keydown", unlockAudio);
function tone(freq, dur = .12, type = "sine", vol = .15, when = 0) {
  if (!chkSound.checked || !actx || actx.state !== "running") return;
  try {
    const o = actx.createOscillator(), g = actx.createGain(), t = actx.currentTime + when;
    o.type = type; o.frequency.value = freq;
    g.gain.setValueAtTime(vol, t); g.gain.exponentialRampToValueAtTime(.0001, t + dur);
    o.connect(g).connect(actx.destination); o.start(t); o.stop(t + dur);
  } catch (e) {}
}
const sfx = {
  shutter() { tone(180, .08, "square", .1); tone(90, .16, "square", .09, .05); },
  pick() { tone(520, .06, "triangle"); },
  drop() { tone(300, .09, "triangle"); },
  swap() { tone(440, .08, "triangle"); tone(660, .12, "triangle", .15, .07); },
  win() { [523, 659, 784, 1047].forEach((f, i) => tone(f, .28, "sine", .18, i * .12)); },
};

// ------------------------------------------------------------------ konfeti
let confetti = [];
function burst(p) {
  const colors = [ACCENT, OK, "#ffffff", "#5ac8fa", "#ff6b9d"];
  for (let i = 0; i < 150; i++) confetti.push({
    x: W / 2 + (Math.random() - .5) * p.pw * .6, y: p.y0 + p.ph * .55,
    vx: (Math.random() - .5) * 14, vy: -Math.random() * 16 - 4, r: 4 + Math.random() * 5,
    c: colors[i % colors.length], rot: Math.random() * 6, vr: (Math.random() - .5) * .4, life: 1 });
}
function drawConfetti() {
  if (!confetti.length) return;
  const k = W / 1280;
  for (const c of confetti) {
    c.vy += .45; c.x += c.vx; c.y += c.vy; c.rot += c.vr; c.life -= .006;
    ctx.save(); ctx.globalAlpha = Math.max(0, c.life); ctx.translate(c.x, c.y); ctx.rotate(c.rot);
    ctx.fillStyle = c.c; ctx.fillRect(-c.r * k, -c.r * .5 * k, c.r * 2 * k, c.r * k); ctx.restore();
  }
  confetti = confetti.filter(c => c.life > 0 && c.y < H + 40);
}

// ------------------------------------------------------------------ rekor waktu
const bestKey = n => `puzzlecam-best-${n}`;
function getBest(n) { try { return +localStorage.getItem(bestKey(n)) || 0; } catch (e) { return 0; } }
function setBest(n, v) { try { localStorage.setItem(bestKey(n), String(v)); } catch (e) {} }
function refreshBest() {
  const n = +selGrid.value, b = getBest(n);
  elBestLbl.textContent = `(${n}×${n})`; elBest.textContent = b ? b.toFixed(1) + "s" : "–";
}
selGrid.addEventListener("change", refreshBest); refreshBest();

function buildPuzzle(photo, r) {
  const n = +selGrid.value, aspect = r.w / r.h;
  let ph = H * .66, pw = ph * aspect;
  if (pw > W * .62) { pw = W * .62; ph = pw / aspect; }
  const tw = Math.max(20, Math.floor(pw / n)), th = Math.max(20, Math.floor(ph / n));
  pw = tw * n; ph = th * n;
  const x0 = Math.round((W - pw) / 2), y0 = Math.round((H - ph) / 2);
  const tiles = [];
  for (let id = 0; id < n * n; id++) {
    const c = document.createElement("canvas"); c.width = tw; c.height = th;
    const sw = photo.width / n, sh = photo.height / n;
    c.getContext("2d").drawImage(photo, (id % n) * sw, Math.floor(id / n) * sh, sw, sh, 0, 0, tw, th);
    tiles.push(c);
  }
  const perm = [...Array(n * n).keys()];
  do { for (let i = perm.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; } }
  while (perm.every((v, i) => v === i));
  return { n, x0, y0, tw, th, pw, ph, tiles, perm, r, photo, newBest: false, held: -1, moves: 0, t0: 0, tEnd: 0, galleryEl: null };
}

const slotXY = (p, s) => ({ x: p.x0 + (s % p.n) * p.tw, y: p.y0 + Math.floor(s / p.n) * p.th });
function cellAt(p, pt) {
  if (pt.x < p.x0 || pt.y < p.y0 || pt.x >= p.x0 + p.pw || pt.y >= p.y0 + p.ph) return -1;
  return Math.floor((pt.y - p.y0) / p.th) * p.n + Math.floor((pt.x - p.x0) / p.tw);
}

function capture(r) {
  const x = Math.max(0, Math.round(r.x)), y = Math.max(0, Math.round(r.y));
  const w = Math.min(W - x, Math.round(r.w)), h = Math.min(H - y, Math.round(r.h));
  if (w < 40 || h < 40) return;
  const photo = document.createElement("canvas"); photo.width = w; photo.height = h;
  const pc = photo.getContext("2d");
  pc.drawImage(base, x, y, w, h, 0, 0, w, h);
  if (chkMono.checked) stylize(pc, w, h);
  puzzle = buildPuzzle(photo, { x, y, w, h });
  puzzle.galleryEl = addGallery(photo);
  state = "shatter"; shatterStart = performance.now(); flash = 1; holdStart = 0; wasPinch = false; anchor = null; steadyStart = 0;
  sfx.shutter();
}

function resetAll() { state = "idle"; puzzle = null; holdStart = 0; wasPinch = false; confetti = []; }

function reshuffle() {                       // acak ulang dengan foto yang sama (ukuran grid ikut pengaturan)
  if (!puzzle || state === "shatter") return;
  const old = puzzle, np = buildPuzzle(old.photo, old.r);
  np.galleryEl = old.galleryEl;
  puzzle = np; confetti = []; wasPinch = false;
  state = "shatter"; shatterStart = performance.now(); sfx.shutter();
}

function drawShatter(now) {
  const p = puzzle, n = p.n, t = clamp((now - shatterStart) / SHATTER_MS, 0, 1);
  for (let s = 0; s < n * n; s++) {
    const id = p.perm[s];
    const from = { x: p.r.x + (id % n) * p.r.w / n, y: p.r.y + Math.floor(id / n) * p.r.h / n, w: p.r.w / n, h: p.r.h / n };
    const to = slotXY(p, s);
    const e = easeOut(clamp(t * 1.3 - (s / (n * n)) * .3, 0, 1));
    const x = from.x + (to.x - from.x) * e, y = from.y + (to.y - from.y) * e;
    const w = from.w + (p.tw - from.w) * e, h = from.h + (p.th - from.h) * e;
    ctx.save();
    ctx.translate(x + w / 2, y + h / 2); ctx.rotate((1 - e) * Math.sin(s * 1.7) * .35);
    ctx.drawImage(p.tiles[id], -w / 2, -h / 2, w, h);
    ctx.strokeStyle = "rgba(0,0,0,.6)"; ctx.strokeRect(-w / 2, -h / 2, w, h);
    ctx.restore();
  }
  if (t >= 1) { state = "puzzle"; p.t0 = now; wasPinch = false; }
}

function drawPuzzle(now) {
  const p = puzzle, k = W / 1280;
  for (let s = 0; s < p.n * p.n; s++) {
    const { x, y } = slotXY(p, s);
    if (s === p.held) { ctx.fillStyle = "rgba(20,20,20,.9)"; ctx.fillRect(x, y, p.tw, p.th); }
    else ctx.drawImage(p.tiles[p.perm[s]], x, y, p.tw, p.th);
    ctx.strokeStyle = "rgba(0,0,0,.65)"; ctx.lineWidth = 1; ctx.strokeRect(x, y, p.tw, p.th);
    if (p.perm[s] === s && state === "puzzle") { ctx.fillStyle = OK; ctx.fillRect(x + 4, y + 4, 6 * k, 6 * k); }
  }
  ctx.strokeStyle = state === "solved" ? OK : ACCENT; ctx.lineWidth = 3 * k;
  ctx.strokeRect(p.x0, p.y0, p.pw, p.ph);

  const rw = 170 * k, rh = rw * p.photo.height / p.photo.width, rx = W - rw - 16 * k, ry = 16 * k;   // gambar contoh
  ctx.drawImage(p.photo, rx, ry, rw, rh);
  ctx.strokeStyle = "#fff"; ctx.lineWidth = 2 * k; ctx.strokeRect(rx, ry, rw, rh);
  ctx.font = `600 ${11 * k}px ui-monospace, Menlo, Consolas, monospace`; ctx.fillStyle = "#fff";
  ctx.fillText("CONTOH", rx, ry + rh + 14 * k);

  if (state === "puzzle") {
    for (const h of hands) {                      // sorot kotak di bawah jari
      if (h.pinch && p.held >= 0) continue;
      const c = cellAt(p, h.pt);
      if (c >= 0) { const { x, y } = slotXY(p, c); ctx.fillStyle = "rgba(255,214,10,.18)"; ctx.fillRect(x, y, p.tw, p.th);
        ctx.strokeStyle = ACCENT; ctx.lineWidth = 2 * k; ctx.strokeRect(x, y, p.tw, p.th); }
    }
    const active = hands.find(h => h.pinch);
    if (!p.tEnd) {
      if (active) {
        lastPt = active.pt;
        if (!wasPinch) { const c = cellAt(p, active.pt); if (c >= 0) { p.held = c; sfx.pick(); } }
      } else if (wasPinch && p.held >= 0) {
        const t = cellAt(p, lastPt);
        if (t >= 0 && t !== p.held) {
          [p.perm[p.held], p.perm[t]] = [p.perm[t], p.perm[p.held]]; p.moves++; sfx.swap();
          if (p.perm.every((v, i) => v === i)) {
            p.tEnd = now; state = "solved"; solvedAt = now;
            if (p.galleryEl) p.galleryEl.classList.add("done");
            const secs = (p.tEnd - p.t0) / 1000, best = getBest(p.n);
            if (!best || secs < best) { setBest(p.n, +secs.toFixed(1)); p.newBest = true; refreshBest(); }
            burst(p); sfx.win();
          }
        } else sfx.drop();
        p.held = -1;
      }
      wasPinch = !!active;
    }
    if (p.held >= 0 && active) {
      const x = clamp(active.pt.x - p.tw / 2, 0, W - p.tw), y = clamp(active.pt.y - p.th / 2, 0, H - p.th);
      ctx.save(); ctx.shadowColor = "rgba(0,0,0,.7)"; ctx.shadowBlur = 18 * k; ctx.shadowOffsetY = 6 * k;
      ctx.drawImage(p.tiles[p.perm[p.held]], x, y, p.tw, p.th); ctx.restore();
      ctx.strokeStyle = ACCENT; ctx.lineWidth = 3 * k; ctx.strokeRect(x, y, p.tw, p.th);
    }
  }

  if (state === "solved") {
    const secs = ((p.tEnd - p.t0) / 1000).toFixed(1);
    ctx.font = `700 ${26 * k}px ui-monospace, Menlo, Consolas, monospace`; ctx.textAlign = "center";
    const msg = `${p.newBest ? "REKOR BARU! " : "SELESAI · "}${secs}s · ${p.moves} gerakan`;
    ctx.fillStyle = "rgba(0,0,0,.65)"; ctx.fillRect(W / 2 - 280 * k, p.y0 + p.ph + 14 * k, 560 * k, 44 * k);
    ctx.fillStyle = OK; ctx.fillText(msg, W / 2, p.y0 + p.ph + 44 * k); ctx.textAlign = "start";
    if (now - solvedAt > SOLVED_MS) resetAll();
  }
}

// ------------------------------------------------------------------ UI teks
let lastHint = "", lastStatus = "";
function setText(el, v, cache) { if (cache.v !== v) { el.textContent = v; cache.v = v; } }
const hintC = {}, statC = {}, timeC = {}, moveC = {};

function updateUI(now, holding) {
  const n = hands.length;
  let hint;
  if (state === "idle") {
    hint = n === 0 ? "Tunjukkan kedua tangan ke kamera" : n === 1 ? "Munculkan tangan satunya lagi untuk membuat bingkai"
      : holding ? "Tahan… jangan bergerak" : "Tahan bingkai dengan diam → foto otomatis (atau cubit kedua tangan)";
    if (!landmarker) hint += "  (atau tekan SPACE)";
  } else if (state === "shatter") hint = "Memecah foto…";
  else if (state === "puzzle") hint = "Cubit potongan → geser → lepas di potongan lain · A = acak ulang";
  else hint = "Selesai! Foto tersimpan di galeri · tekan A untuk acak ulang";
  setText(elHint, hint, hintC);
  elStatus.classList.toggle("live", n > 0);
  setText(elStatus, !landmarker ? "● TANPA TRACKING" : n > 0 ? `● TRACKING · ${n} TANGAN` : "● MENCARI TANGAN", statC);
  if (!puzzle) setText(elTime, "0.0s", timeC);
  else if (puzzle.t0) setText(elTime, (((puzzle.tEnd || now) - puzzle.t0) / 1000).toFixed(1) + "s", timeC);
  setText(elMoves, String(puzzle ? puzzle.moves : 0), moveC);
}

// ------------------------------------------------------------------ loop
function loop() {
  requestAnimationFrame(loop);
  if (video.readyState < 2) return;
  const now = performance.now();

  // video cermin + kecerahan  ->  base
  bctx.save();
  bctx.filter = `url(#lift) brightness(${rngBright.value}) contrast(1.1) saturate(1.15)`;
  bctx.globalAlpha = 1 - (+rngDenoise.value);        // campur dengan frame sebelumnya = kurangi noise
  bctx.translate(W, 0); bctx.scale(-1, 1); bctx.drawImage(video, 0, 0, W, H);
  bctx.restore();

  if (landmarker && video.currentTime !== lastVT) {
    lastVT = video.currentTime;
    let ts = Math.floor(now); if (ts <= lastTs) ts = lastTs + 1; lastTs = ts;
    try { hands = parseHands(landmarker.detectForVideo(base, ts)); } catch (e) { console.warn(e); }
  }

  ctx.drawImage(base, 0, 0);
  if (!vig) { vig = ctx.createRadialGradient(W / 2, H / 2, H * .45, W / 2, H / 2, H * .98);
    vig.addColorStop(0, "rgba(0,0,0,0)"); vig.addColorStop(1, "rgba(0,0,0,.5)"); }
  ctx.fillStyle = vig; ctx.fillRect(0, 0, W, H);
  if (state !== "idle") { ctx.fillStyle = "rgba(0,0,0,.55)"; ctx.fillRect(0, 0, W, H); }

  let holding = false;
  if (state === "idle") {
    if (hands.length >= 2) {
      const a = hands[0].pt, b = hands[hands.length - 1].pt;
      const r = { x: Math.min(a.x, b.x), y: Math.min(a.y, b.y), w: Math.abs(a.x - b.x), h: Math.abs(a.y - b.y) };
      const ok = r.w > MIN_FRAME && r.h > MIN_FRAME;
      const pinchBoth = ok && hands[0].pinch && hands[hands.length - 1].pinch;
      if (pinchBoth) holdStart ||= now; else holdStart = 0;
      const same = anchor && Math.abs(r.x - anchor.x) < 45 && Math.abs(r.y - anchor.y) < 45 &&
                   Math.abs(r.w - anchor.w) < 60 && Math.abs(r.h - anchor.h) < 60;
      if (ok && chkAuto.checked) { if (!same) { anchor = { ...r }; steadyStart = now; } } else { anchor = null; steadyStart = 0; }
      const prog = Math.max(holdStart ? (now - holdStart) / HOLD_MS : 0, steadyStart ? (now - steadyStart) / AUTO_MS : 0);
      holding = prog > 0.08;
      drawFrame(r, ok, prog);
      if (prog >= 1) capture(r);
    } else { holdStart = 0; anchor = null; steadyStart = 0; }
  } else if (state === "shatter") drawShatter(now);
  else drawPuzzle(now);

  drawConfetti();
  drawHands();
  if (flash > 0) { ctx.fillStyle = `rgba(255,255,255,${flash})`; ctx.fillRect(0, 0, W, H); flash = Math.max(0, flash - .06); }
  updateUI(now, holding);
}

// ------------------------------------------------------------------ kontrol
function captureCenter() {
  if (state !== "idle" || video.readyState < 2) return;
  const s = Math.round(H * .6);
  capture({ x: (W - s) / 2, y: (H - s) / 2, w: s, h: s });
}
$("#btnShot").onclick = captureCenter;
$("#btnReset").onclick = resetAll;
$("#btnShuffle").onclick = reshuffle;
function toggleFull() {
  const st = document.querySelector(".stage");
  if (document.fullscreenElement) document.exitFullscreen(); else if (st.requestFullscreen) st.requestFullscreen();
}
$("#btnFull").onclick = toggleFull;
addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
  if (e.code === "Space") { e.preventDefault(); captureCenter(); }
  else if (e.key === "r" || e.key === "R") resetAll();
  else if (e.key === "a" || e.key === "A") reshuffle();
  else if (e.key === "f" || e.key === "F") toggleFull();
  else if (e.key === "+" || e.key === "=") selGrid.selectedIndex = Math.min(selGrid.options.length - 1, selGrid.selectedIndex + 1);
  else if (e.key === "-") selGrid.selectedIndex = Math.max(0, selGrid.selectedIndex - 1);
});

// ------------------------------------------------------------------ mulai
function showLoader(msg, err) { elLoader.hidden = !msg; elLoader.textContent = msg || ""; elLoader.classList.toggle("err", !!err); }
(async () => {
  try { await startCamera(); }
  catch (e) { showLoader("Kamera tidak bisa dibuka: " + e.message + ". Izinkan akses kamera lalu muat ulang halaman.", true); return; }
  requestAnimationFrame(loop);
  showLoader("Memuat model tangan…");
  try { await loadModel(); showLoader(""); }
  catch (e) { console.error(e); landmarker = null; showLoader("Model tangan gagal dimuat (" + e.message + "). Tombol SPACE tetap bisa dipakai.", true); }
})();
