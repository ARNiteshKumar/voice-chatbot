# 🎙️ VoiceBot — Voice-Enabled Chatbot (Speech Recognition + Deep Learning)

A voice-enabled chatbot that listens to your voice, understands what you meant
using a **deep learning intent-classification model**, and replies — showing both
the recognized speech and the response, and speaking the answer back.

It runs **entirely in the browser**: no server, no API keys, no running costs.
That makes it a perfect fit for free, always-online hosting on **GitHub Pages**.

**🔗 Live demo:** https://arniteshkumar.github.io/voice-chatbot/
*(open in Google Chrome and allow microphone access)*

---

## What it does

1. **Speech → text** — captures your voice and transcribes it live (Web Speech API).
2. **Text → intent** — classifies the meaning with a neural network (TensorFlow.js).
3. **Intent → response** — replies from the matching intent, and can **speak it aloud**.
4. **Shows everything** — displays the recognized speech, the reply, and the
   detected **intent + confidence** so you can see the model working.

---

## Features

- 🎤 Real-time speech recognition with a live audio waveform while you talk
- 🧠 Deep-learning intent classifier trained in the browser (and cached, so later loads are instant)
- 🔊 Text-to-speech replies (toggleable)
- ⌨️ Text-input fallback — fully usable without a microphone
- 🛡️ Graceful handling of unknown / out-of-scope questions instead of confident wrong answers
- 📊 Live readout of vocabulary size, intent count, training samples, and accuracy

---

## Tech stack

| Part | Technology |
|---|---|
| Speech-to-text | Web Speech API (`SpeechRecognition`) |
| Deep learning model | Feed-forward neural network in **TensorFlow.js** |
| Text-to-speech | Web Speech API (`SpeechSynthesis`) |
| Audio waveform | Web Audio API (`AnalyserNode`) |
| Model caching | IndexedDB |
| Hosting | GitHub Pages (static, HTTPS, free) |
| Report figures | Python + Keras (`train_report.py`) |

---

## Dataset

The dataset (`intents.json`) is a set of **intents**. Each intent has a `tag`, a
list of example `patterns` (training sentences), and a list of `responses`.

- **Intents (classes):** 13
- **Training sentences (samples):** 126
- **Vocabulary size:** 156

Intents: `greeting`, `goodbye`, `thanks`, `about_bot`, `creator`, `help`, `time`,
`date`, `weather`, `joke`, `identity_project`, `mood`, and `unknown`
(an out-of-scope class that catches off-topic questions like recipes or sports scores).

---

## Model architecture

A bag-of-words feed-forward neural network:

```
Input (156 — bag-of-words vector)
  → Dense(24, ReLU) → Dropout(0.2)
  → Dense(24, ReLU) → Dropout(0.2)
  → Dense(13, Softmax)
```

- **Loss:** categorical cross-entropy   **Optimizer:** Adam (lr = 0.01)
- **Epochs:** 300, with a safeguard that keeps training if the model is under-fit
- **Batch size:** 8   **Trainable parameters:** ~4,000

---

## How it works

**1. Speech → text.** Tapping the mic starts `SpeechRecognition`. Interim words
stream into the transcript live; the final transcript is sent to the classifier.

**2. Text → intent.** The text is lowercased, tokenized, and turned into a
**bag-of-words** vector over the fixed vocabulary. That vector goes through the
network, which outputs a probability for each intent. The highest is the
prediction. The bot falls back to a safe reply when **either** the confidence is
below `0.55` **or** the sentence contains no words the model knows. Off-topic
questions are routed to the trained `unknown` intent.

**3. Intent → response.** A reply is chosen from that intent's `responses`. The
`time` and `date` intents are answered dynamically from the device clock.

**4. Response → speech.** If "Speak replies aloud" is on, the reply is spoken via
`SpeechSynthesis`.

The model trains in the browser on first load and is cached in IndexedDB, keyed by
`MODEL_VERSION` in `app.js` — **bump that value to force a retrain** after editing
the dataset. `train_report.py` trains the *same* architecture in Python/Keras to
produce accuracy/loss curves and a confusion matrix for the written report.

---

## Accuracy and limitations

The model reaches **100% accuracy on its training phrases** and classifies clear,
in-domain questions reliably (time, weather, jokes, greetings, help, etc.). Because
it is a compact bag-of-words classifier, it has expected limits:

- **No word order or context** — "how are you" and "you are how" look identical to it.
- **Small dataset** — casual filler ("okay", "I'm good") and topics outside the 13
  intents are matched to the nearest known intent, so they can be misrouted.
- **In-browser training variance** — each browser trains its own model with some
  randomness, so answers to unseen phrases can vary slightly between machines.
- **No live data** — `time`/`date` come from the local device clock (no time zones),
  and `weather` is a placeholder, not a live feed.

These are inherent to the model class and are documented in the project report.

---

## Run it locally

The app loads `intents.json` with `fetch()`, so serve it over a local web server
(don't just double-click the file):

```bash
python -m http.server 8000
# then open http://localhost:8000 in Chrome
```

---

## Deploy to GitHub Pages

**Option A — no command line (recommended):**

1. Create a new **public** repository on GitHub (e.g. `voice-chatbot`).
2. **Add file → Upload files**, drag in the files *inside* the project folder
   (not the folder itself), and **Commit changes**.
3. **Settings → Pages → Build and deployment → Source: Deploy from a branch →
   Branch: `main`, folder `/ (root)` → Save.**
4. Wait ~1 minute; your live link appears there:
   `https://<username>.github.io/<repo-name>/`.
5. Open it in Chrome, allow the microphone, and submit that link.

**Option B — git:**

```bash
git init
git add .
git commit -m "Voice-enabled chatbot"
git branch -M main
git remote add origin https://github.com/<username>/<repo-name>.git
git push -u origin main
```

Then enable Pages as in Option A, steps 3–5.

> The included `.nojekyll` file tells GitHub Pages to serve the files as-is.

**Updating a deployed site:** upload the changed files with the same names to
overwrite them, commit, then hard-refresh the page (`Ctrl+Shift+R`).

---

## Generate the report figures

```bash
pip install tensorflow scikit-learn matplotlib numpy
python train_report.py
```

Or paste `train_report.py` into a Google Colab cell (upload `intents.json` first).
It saves `report_assets/accuracy.png`, `loss.png`, and `confusion_matrix.png`.
The random seed is fixed, so the figures are reproducible.

---

## Project structure

```
.
├── index.html          # UI
├── style.css           # styling (dark "signal equipment" theme)
├── app.js              # speech recognition + TensorFlow.js model + text-to-speech
├── intents.json        # dataset: intents, patterns, responses
├── report.md           # report template (fill in your details + screenshots)
├── train_report.py     # Keras script → accuracy / loss / confusion-matrix figures
├── report_assets/      # generated report figures
├── .nojekyll           # GitHub Pages: serve files verbatim
└── README.md
```

---

## Extending it

- **Add intents:** edit `intents.json`, then bump `MODEL_VERSION` in `app.js`.
- **Real weather / time zones:** replace the `weather` (or `time`) response with a
  `fetch()` to a free API.
- **Another language:** change `recognition.lang` in `app.js` (e.g. `'hi-IN'`,
  `'ta-IN'`) and add patterns in that language.

---

## Submission checklist

- [ ] Live GitHub Pages link works in Chrome (speech + typing both respond)
- [ ] Screenshot showing **voice input** working (spoken words → text → reply)
- [ ] Source code pushed to a public GitHub repo
- [ ] Report completed (`report.md`) with dataset, architecture, methodology, results, and the figures

---

*Built as a deep learning lab project on speech recognition and chatbots.*
