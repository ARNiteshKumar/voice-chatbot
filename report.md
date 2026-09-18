# Voice-Enabled Chatbot using Speech Recognition and Deep Learning
### Project Report

**Name:** _<your name>_  **Reg. No.:** _<your reg no>_  **Course:** _<course code>_
**Live link:** _<your GitHub Pages URL>_  **Source code:** _<your GitHub repo URL>_
**Date:** _<submission date>_

---

## 1. Abstract

This project implements and deploys an online voice-enabled chatbot. The system
captures the user's voice, transcribes it to text using browser-based speech
recognition, classifies the user's **intent** with a deep learning model
(a feed-forward neural network), and returns an appropriate response, which it can
also speak aloud. Both the recognized speech and the chatbot's reply are displayed,
along with the predicted intent and the model's confidence. The application runs
fully client-side and is deployed on GitHub Pages, making it free and always
available.

## 2. Objectives

- Integrate speech recognition to accept voice input.
- Build a deep learning model for intent classification.
- Generate and display appropriate responses to recognized speech.
- Deploy a publicly accessible web application and provide a live link.

## 3. System Architecture

```
 ┌──────────┐   voice    ┌────────────────────┐  text  ┌───────────────────────┐
 │  User /  │ ─────────▶ │ Speech Recognition │ ─────▶ │ Pre-processing         │
 │   Mic    │            │ (Web Speech API)   │        │ tokenize → bag-of-words│
 └──────────┘            └────────────────────┘        └───────────┬───────────┘
                                                                    │ vector
                                                                    ▼
 ┌──────────┐  speech  ┌──────────────────┐  text   ┌──────────────────────────┐
 │ Speaker  │ ◀─────── │ Text-to-Speech   │ ◀────── │ Neural network (TF.js)    │
 │          │          │ (SpeechSynthesis)│  reply  │ intent classifier + reply │
 └──────────┘          └──────────────────┘         └──────────────────────────┘
```

**Speech recognition:** The Web Speech API's `SpeechRecognition` transcribes audio
in real time, entirely in the browser (no audio leaves the device).

**Deep learning model:** A bag-of-words feed-forward neural network implemented in
TensorFlow.js:

| Layer | Units | Activation | Notes |
|------:|:-----:|:----------:|-------|
| Input | vocab size | – | binary bag-of-words vector |
| Dense | 16 | ReLU | |
| Dropout | – | – | rate 0.3 (regularization) |
| Dense | 16 | ReLU | |
| Dropout | – | – | rate 0.3 |
| Output | num intents | Softmax | probability per intent |

- **Loss:** categorical cross-entropy **Optimizer:** Adam (lr = 0.01)
- **Epochs:** 250 **Batch size:** 8

**Text-to-speech:** `SpeechSynthesis` reads the reply aloud (toggleable).

## 4. Dataset

The dataset (`intents.json`) is a set of **intents**. Each intent has a `tag`, a
list of example `patterns` (training sentences), and a list of `responses`.

- **Intents (classes):** _<fill: run the app, see the "intents" readout — e.g. 12>_
- **Training sentences (samples):** _<fill: the "samples" readout — e.g. 95>_
- **Vocabulary size:** _<fill: the "vocab" readout — e.g. 111>_

Example intents: greeting, goodbye, thanks, about_bot, creator, help, time, date,
weather, joke, identity_project, mood. _(Describe a couple of intents and give
example patterns/responses.)_

## 5. Methodology

1. **Pre-processing.** Each sentence is lowercased, stripped of punctuation, and
   tokenized. A vocabulary is built from all training tokens (sorted for
   determinism). Every sentence becomes a binary **bag-of-words** vector: position
   *i* is 1 if vocabulary word *i* is present.
2. **Labels.** Each intent tag is one-hot encoded.
3. **Training.** The network is trained with the settings in §3. Dropout reduces
   overfitting on the small dataset.
4. **Inference + confidence.** At runtime, the recognized text is vectorized the
   same way and passed through the network. The highest softmax probability gives
   the predicted intent and a confidence score. If confidence < 0.55, the bot
   returns a fallback message rather than guessing.
5. **Response generation.** A response is sampled from the predicted intent's
   `responses`. `time`/`date` intents are answered dynamically.

## 6. Deployment

The app is fully client-side (HTML + CSS + TensorFlow.js). It is hosted on
**GitHub Pages** (static hosting, HTTPS, free, always online). The model trains in
the browser on first visit and is cached in IndexedDB for instant subsequent
loads. _(See README for exact deployment steps. Insert your live URL here.)_

## 7. Results

_Generate the figures with `train_report.py` (see README / Colab) and insert them._

- **Training accuracy:** _<e.g. 100%>_ **Final loss:** _<e.g. 0.12>_
- **Test/validation accuracy:** _<from train_report.py>_

**Figures to include:**
1. Accuracy vs. epochs (`report_assets/accuracy.png`)
2. Loss vs. epochs (`report_assets/loss.png`)
3. Confusion matrix (`report_assets/confusion_matrix.png`)

**Screenshots to include:**
1. The app recognizing speech (user turn) and replying (bot turn).
2. The intent + confidence readout under a bot reply.
3. The GitHub Pages settings showing the live URL.

_Discuss the results: which intents were classified well, any confusions
(e.g. overlapping intents like `about_bot` vs `identity_project`), and how the
confidence threshold handles out-of-scope questions._

## 8. Limitations

- Web Speech API recognition quality varies by browser and accent; best in
  Chrome/Edge.
- The dataset is small; the model recognizes the trained intents, not open-domain
  conversation.
- Bag-of-words ignores word order (e.g. it can't distinguish "you help me" from
  "me help you").

## 9. Future Work

- Larger dataset and word embeddings / an LSTM or transformer for better
  generalization.
- Live weather/news via external APIs.
- Multilingual support (change `recognition.lang` and add patterns).
- Context/slot handling for multi-turn conversations.

## 10. References

- Web Speech API (MDN): SpeechRecognition, SpeechSynthesis.
- TensorFlow.js documentation.
- Classic "intents.json" chatbot formulation (bag-of-words + feed-forward network).
