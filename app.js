/* =========================================================================
   VoiceBot — voice-enabled chatbot
   - Speech-to-text : Web Speech API (SpeechRecognition)
   - Intent model   : feed-forward neural network in TensorFlow.js
   - Text-to-speech : Web Speech API (SpeechSynthesis)
   The model is trained in the browser from intents.json on first load and
   cached in IndexedDB, so later visits are instant.
   ========================================================================= */

'use strict';

const MODEL_VERSION = 'v2';                 // bump to force a retrain
const MODEL_KEY = `indexeddb://voicebot-${MODEL_VERSION}`;
const CONFIDENCE_THRESHOLD = 0.55;          // below this -> fallback reply

const FALLBACKS = [
  "Sorry, I didn't quite catch that. Could you rephrase?",
  "I'm not sure I understood. Try asking about the time, a joke, or say hello.",
  "Hmm, that's outside what I was trained on. Can you say it another way?",
];

// ---- DOM ----
const el = (id) => document.getElementById(id);
const transcriptEl = el('transcript');
const micBtn = el('micBtn');
const hintEl = el('hint');
const typeForm = el('typeForm');
const typeIn = el('typeIn');
const ttsToggle = el('ttsToggle');
const waveCanvas = el('wave');
const statusBox = el('modelStatus');
const statusText = el('statusText');

let intents = null;
let model = null;
let vocab = [];
let classes = [];
let vocabIndex = new Map();
let listening = false;

/* ------------------------------------------------------------------ NLU */
function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(Boolean);
}

function buildVocab(intentList) {
  const vocabSet = new Set();
  const cls = [];
  const trainTokens = [];
  const trainLabels = [];

  intentList.forEach((it) => {
    if (!cls.includes(it.tag)) cls.push(it.tag);
    const classIndex = cls.indexOf(it.tag);
    it.patterns.forEach((p) => {
      const toks = tokenize(p);
      toks.forEach((t) => vocabSet.add(t));
      trainTokens.push(toks);
      trainLabels.push(classIndex);
    });
  });

  const v = Array.from(vocabSet).sort();   // sorted => deterministic
  return { vocab: v, classes: cls, trainTokens, trainLabels };
}

function bagOfWords(tokens) {
  const vec = new Array(vocab.length).fill(0);
  tokens.forEach((t) => {
    if (vocabIndex.has(t)) vec[vocabIndex.get(t)] = 1;
  });
  return vec;
}

/* ---------------------------------------------------------------- model */
function buildModel(inputSize, numClasses) {
  const m = tf.sequential();
  m.add(tf.layers.dense({ inputShape: [inputSize], units: 24, activation: 'relu' }));
  m.add(tf.layers.dropout({ rate: 0.2 }));
  m.add(tf.layers.dense({ units: 24, activation: 'relu' }));
  m.add(tf.layers.dropout({ rate: 0.2 }));
  m.add(tf.layers.dense({ units: numClasses, activation: 'softmax' }));
  m.compile({
    optimizer: tf.train.adam(0.01),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });
  return m;
}

async function trainModel(trainTokens, trainLabels) {
  const xs = tf.tensor2d(trainTokens.map(bagOfWords));
  const ys = tf.oneHot(tf.tensor1d(trainLabels, 'int32'), classes.length);
  const m = buildModel(vocab.length, classes.length);

  // clean argmax accuracy on the training set (dropout is off during predict)
  const evalAcc = () => tf.tidy(() => {
    const preds = m.predict(xs).argMax(1).dataSync();
    let c = 0;
    for (let i = 0; i < preds.length; i++) if (preds[i] === trainLabels[i]) c++;
    return c / preds.length;
  });

  await m.fit(xs, ys, { epochs: 300, batchSize: 8, shuffle: true, verbose: 0 });
  let acc = evalAcc();

  // safeguard: keep training if an unlucky start left it under-fit
  let rounds = 0;
  while (acc < 0.95 && rounds < 3) {
    await m.fit(xs, ys, { epochs: 150, batchSize: 8, shuffle: true, verbose: 0 });
    acc = evalAcc();
    rounds++;
  }

  xs.dispose(); ys.dispose();
  return { model: m, acc };
}

async function classify(text) {
  const tokens = tokenize(text);
  const vec = bagOfWords(tokens);
  const knownWords = vec.reduce((a, b) => a + b, 0); // how many words the model knows
  const input = tf.tensor2d([vec]);
  const output = model.predict(input);
  const probs = await output.data();
  input.dispose(); output.dispose();

  let best = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
  return { tag: classes[best], confidence: probs[best], knownWords };
}

/* ---------------------------------------------------------- responses */
function pick(arr) { return arr[Math.floor(Math.random() * arr.length)]; }

function responseFor(tag) {
  const intent = intents.intents.find((i) => i.tag === tag);
  if (!intent) return pick(FALLBACKS);
  let text = pick(intent.responses);
  if (text === '__TIME__') {
    text = 'It is ' + new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) + '.';
  } else if (text === '__DATE__') {
    text = 'Today is ' + new Date().toLocaleDateString([], {
      weekday: 'long', year: 'numeric', month: 'long', day: 'numeric',
    }) + '.';
  }
  return text;
}

/* -------------------------------------------------------------- render */
function addTurn(role, text, meta) {
  const turn = document.createElement('div');
  turn.className = `turn ${role}`;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  turn.appendChild(bubble);
  if (meta) {
    const tag = document.createElement('div');
    tag.className = 'tag' + (meta.low ? ' low' : '');
    tag.innerHTML = meta.html;
    turn.appendChild(tag);
  }
  transcriptEl.appendChild(turn);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
  return turn;
}

// live interim (updated in place while the user is speaking)
let interimTurn = null;
function showInterim(text) {
  if (!interimTurn) {
    interimTurn = addTurn('user interim', text);
  } else {
    interimTurn.querySelector('.bubble').textContent = text;
    transcriptEl.scrollTop = transcriptEl.scrollHeight;
  }
}
function clearInterim() {
  if (interimTurn) { interimTurn.remove(); interimTurn = null; }
}

/* ----------------------------------------------------------- pipeline */
async function handleUserText(text) {
  const clean = text.trim();
  if (!clean) return;
  addTurn('user', clean);

  const { tag, confidence, knownWords } = await classify(clean);
  const pct = Math.round(confidence * 100);
  // uncertain if: no recognised words at all, or low confidence
  const uncertain = knownWords === 0 || confidence < CONFIDENCE_THRESHOLD;
  const reply = uncertain ? pick(FALLBACKS) : responseFor(tag);

  addTurn('bot', reply, {
    html: `intent <b>${uncertain ? 'uncertain' : tag}</b> · ${pct}%`,
    low: uncertain,
  });
  speak(reply);
}

/* ------------------------------------------------------ text-to-speech */
function speak(text) {
  if (!ttsToggle.checked) return;
  if (!('speechSynthesis' in window)) return;
  try {
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1.0;
    u.pitch = 1.0;
    window.speechSynthesis.speak(u);
  } catch (_) { /* non-fatal */ }
}

/* --------------------------------------------------- waveform (visual) */
let audioCtx = null, analyser = null, micStream = null, rafId = null;

function drawWave() {
  const ctx = waveCanvas.getContext('2d');
  const w = waveCanvas.width, h = waveCanvas.height;
  const style = getComputedStyle(document.documentElement);
  const amber = style.getPropertyValue('--amber').trim() || '#ffb020';

  const buf = analyser ? new Uint8Array(analyser.fftSize) : null;
  let t = 0;

  const render = () => {
    ctx.clearRect(0, 0, w, h);
    ctx.lineWidth = 2;
    ctx.strokeStyle = amber;
    ctx.beginPath();

    if (analyser) {
      analyser.getByteTimeDomainData(buf);
      const slice = w / buf.length;
      for (let i = 0; i < buf.length; i++) {
        const y = (buf[i] / 128.0) * (h / 2);
        const x = i * slice;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
    } else {
      // fallback synthetic waveform when mic capture is unavailable
      t += 0.15;
      for (let x = 0; x <= w; x += 4) {
        const y = h / 2 + Math.sin(x * 0.05 + t) * (h / 4) * (0.4 + 0.6 * Math.random());
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
    rafId = requestAnimationFrame(render);
  };
  render();
}

async function startWave() {
  waveCanvas.classList.add('active');
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioCtx.createMediaStreamSource(micStream);
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 512;
    source.connect(analyser);
  } catch (_) {
    analyser = null; // fall back to synthetic wave
  }
  drawWave();
}

function stopWave() {
  waveCanvas.classList.remove('active');
  if (rafId) cancelAnimationFrame(rafId);
  rafId = null;
  if (micStream) { micStream.getTracks().forEach((t) => t.stop()); micStream = null; }
  if (audioCtx) { audioCtx.close().catch(() => {}); audioCtx = null; }
  analyser = null;
  const ctx = waveCanvas.getContext('2d');
  ctx.clearRect(0, 0, waveCanvas.width, waveCanvas.height);
}

/* --------------------------------------------- speech recognition (STT) */
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;

function setupRecognition() {
  if (!SpeechRecognition) return null;
  const rec = new SpeechRecognition();
  rec.lang = 'en-US';
  rec.interimResults = true;
  rec.continuous = false;
  rec.maxAlternatives = 1;

  rec.onstart = () => {
    listening = true;
    micBtn.classList.add('listening');
    hintEl.textContent = 'Listening… speak now';
    hintEl.classList.remove('err');
    startWave();
  };

  rec.onresult = (event) => {
    let interim = '', final = '';
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const t = event.results[i][0].transcript;
      if (event.results[i].isFinal) final += t;
      else interim += t;
    }
    if (interim) showInterim(interim);
    if (final) {
      clearInterim();
      handleUserText(final);
    }
  };

  rec.onerror = (e) => {
    const map = {
      'no-speech': "I didn't hear anything. Tap the mic and try again.",
      'not-allowed': 'Microphone blocked. Allow mic access, or type below.',
      'service-not-allowed': 'Microphone blocked. Allow mic access, or type below.',
      'audio-capture': 'No microphone found. You can type instead.',
      'network': 'Network issue with speech recognition. Try again or type.',
    };
    hintEl.textContent = map[e.error] || ('Speech error: ' + e.error);
    hintEl.classList.add('err');
  };

  rec.onend = () => {
    listening = false;
    micBtn.classList.remove('listening');
    stopWave();
    clearInterim();
    if (!hintEl.classList.contains('err')) hintEl.textContent = 'Tap the mic and speak';
  };

  return rec;
}

function toggleListening() {
  if (!recognition) {
    hintEl.textContent = 'Speech recognition not supported here — use Chrome/Edge, or type below.';
    hintEl.classList.add('err');
    typeIn.focus();
    return;
  }
  if (listening) { recognition.stop(); return; }
  try {
    // some browsers need synthesis stopped before capturing audio
    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
    recognition.start();
  } catch (_) { /* start() throws if already started; ignore */ }
}

/* ------------------------------------------------------- status helper */
function setStatus(state, text) {
  statusBox.className = 'status-readout ' + state;
  statusText.textContent = text;
}

/* --------------------------------------------------------------- boot */
async function boot() {
  setStatus('busy', 'loading data…');
  if (window.__VOICEBOT_INTENTS__) {
    intents = window.__VOICEBOT_INTENTS__;        // single-file / embedded build
  } else {
    try {
      const res = await fetch('intents.json');    // normal multi-file build
      intents = await res.json();
    } catch (e) {
      setStatus('error', 'failed to load intents.json');
      hintEl.textContent = 'Could not load intents.json — is it in the same folder?';
      hintEl.classList.add('err');
      return;
    }
  }

  const built = buildVocab(intents.intents);
  vocab = built.vocab;
  classes = built.classes;
  vocabIndex = new Map(vocab.map((w, i) => [w, i]));

  el('mVocab').textContent = vocab.length;
  el('mClasses').textContent = classes.length;
  el('mSamples').textContent = built.trainTokens.length;

  // try cached model first
  let acc = null;
  try {
    setStatus('busy', 'loading model…');
    model = await tf.loadLayersModel(MODEL_KEY);
    el('mAcc').textContent = 'cached';
  } catch (_) {
    setStatus('busy', 'training model…');
    const out = await trainModel(built.trainTokens, built.trainLabels);
    model = out.model;
    acc = out.acc;
    el('mAcc').textContent = (acc * 100).toFixed(0) + '%';
    try { await model.save(MODEL_KEY); } catch (_) { /* private mode etc. */ }
  }

  recognition = setupRecognition();

  setStatus('ready', SpeechRecognition ? 'ready' : 'ready · type-only');
  addTurn('bot', pick(intents.intents.find((i) => i.tag === 'greeting').responses));

  if (!SpeechRecognition) {
    hintEl.textContent = 'Speech recognition needs Chrome/Edge — you can still type below.';
  }
}

/* ------------------------------------------------------------- events */
micBtn.addEventListener('click', toggleListening);
typeForm.addEventListener('submit', (e) => {
  e.preventDefault();
  const v = typeIn.value;
  typeIn.value = '';
  handleUserText(v);
});

boot();
