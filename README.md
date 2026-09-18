# VoiceBot — Voice-Enabled Chatbot (Speech Recognition + Deep Learning)

A voice-enabled chatbot that:

1. Listens to your voice and converts **speech → text** (Web Speech API).
2. Classifies the **intent** of what you said using a **neural network** (TensorFlow.js).
3. Replies with an appropriate response and can **speak it back** (Text-to-Speech).
4. Displays **both** the recognized speech and the chatbot response, plus the
   detected intent and the model's confidence.

It runs **entirely in the browser** — no server, no API keys, no running costs —
which makes it a perfect fit for free static hosting on **GitHub Pages**.

**Live demo:** `https://<your-username>.github.io/<repo-name>/`
*(fill this in after you deploy — see below)*

---

## Why this architecture (and why GitHub Pages)

The assessment needs a **live, always-available link** for grading. Server-based
free tiers (Render, Railway, some HF Spaces) "sleep" after inactivity and show a
cold-start delay or an error when the grader clicks the link. This project avoids
that entirely: everything is client-side, so **GitHub Pages serves it as a static
site that is always instantly available and free forever**.

| Requirement | How it's met |
|---|---|
| Speech recognition for voice input | `SpeechRecognition` (Web Speech API), in-browser |
| Deep learning model for intent classification | Feed-forward neural network in TensorFlow.js |
| Show recognized speech **and** response | Chat transcript with user + bot turns |
| Deployed online, publicly accessible | GitHub Pages (static hosting, HTTPS, free) |
| Report: dataset, architecture, methodology, results | `report.md` + `train_report.py` figures |

> **Runner-up option:** if you specifically need server-side speech recognition
> (e.g. OpenAI Whisper) for robustness across browsers, deploy a Gradio/Streamlit
> app on **Hugging Face Spaces** instead. That's a valid alternative, but it adds
> latency and a backend to maintain. For this assessment, the client-side +
> GitHub Pages route is simpler and more reliable.

---

## Browser support

Speech recognition uses the Web Speech API, which is best supported in
**Google Chrome and Microsoft Edge** (desktop and Android). It also works in
Safari. Firefox has limited support. The page **always** includes a text input,
so it is fully usable even without a microphone or in an unsupported browser.
Microphone access requires **HTTPS** — GitHub Pages provides this automatically.

---

## Run it locally

Because the app uses `fetch()` to load `intents.json`, open it through a local
web server (not by double-clicking the file):

```bash
# Python 3
python -m http.server 8000
# then open http://localhost:8000
```

The first load trains the model in your browser (a few seconds) and caches it in
IndexedDB; subsequent loads are instant.

---

## Deploy to GitHub Pages (step by step)

1. Create a **new public repository** on GitHub, e.g. `voice-chatbot`.
2. Upload these files to the repository root (keep the structure below):
   - `index.html`, `style.css`, `app.js`, `intents.json`
   - `README.md`, `report.md`, `.nojekyll`
   ```bash
   git init
   git add .
   git commit -m "Voice-enabled chatbot"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```
3. In the repo, go to **Settings → Pages**.
4. Under **Build and deployment → Source**, choose **Deploy from a branch**.
5. Set **Branch** to `main` and folder to `/ (root)`, then **Save**.
6. Wait ~1 minute. Your live link appears at the top of the Pages settings:
   `https://<your-username>.github.io/<repo-name>/`
7. Open it in Chrome, allow microphone access, and submit that link.

> The `.nojekyll` file tells GitHub Pages to serve the files as-is (no Jekyll
> processing). It's included for you.

---

## Project structure

```
.
├── index.html          # UI
├── style.css           # styling
├── app.js              # speech recognition + TF.js model + TTS
├── intents.json        # dataset: intents, patterns, responses
├── report.md           # report template (fill in your results)
├── train_report.py     # Keras script -> accuracy/loss/confusion-matrix figures
├── .nojekyll           # GitHub Pages: serve files verbatim
└── README.md
```

---

## How it works

**1. Speech → text.** Tapping the mic starts `SpeechRecognition`. Interim results
stream live into the transcript; the final transcript is sent to the classifier.

**2. Text → intent.** The text is lowercased, tokenized, and turned into a
**bag-of-words** vector over a fixed vocabulary built from `intents.json`. That
vector is fed to a neural network:

```
Input (vocab size)
  → Dense(24, ReLU) → Dropout(0.2)
  → Dense(24, ReLU) → Dropout(0.2)
  → Dense(num_intents, Softmax)
```

The softmax output is a probability per intent. The highest one is the predicted
intent; if its confidence is below a threshold (0.55) — or the sentence contains
no words the model knows — the bot gives a graceful fallback instead of guessing.
An explicit `unknown` intent, trained on common out-of-scope questions (recipes,
sports scores, math, etc.), routes off-topic queries to a polite decline.

**3. Intent → response.** A response is chosen from that intent's `responses`
list. Two special tags, `time` and `date`, are answered dynamically.

**4. Response → speech.** If "Speak replies aloud" is on, the reply is spoken via
`SpeechSynthesis`.

The model is trained **in the browser** from `intents.json` on first load and
cached in IndexedDB (keyed by `MODEL_VERSION` in `app.js` — bump it to retrain
after editing the dataset). `train_report.py` trains the **same architecture** in
Python/Keras so you can generate clean accuracy/loss curves and a confusion matrix
for your written report.

---

## Extending it

- **Add intents:** edit `intents.json` (add `tag`, `patterns`, `responses`),
  then bump `MODEL_VERSION` in `app.js` so it retrains.
- **Live weather/time from an API:** replace the `weather` intent's response with
  a `fetch()` to a free weather API.
- **Different language:** change `recognition.lang` in `app.js` (e.g. `'hi-IN'`)
  and add patterns in that language.
