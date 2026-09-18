"""
train_report.py
-----------------
Trains the SAME model architecture as the deployed browser app (bag-of-words
feed-forward network) on intents.json, and saves figures for the report:

    report_assets/accuracy.png
    report_assets/loss.png
    report_assets/confusion_matrix.png

It also prints a classification report and the final accuracy.

Run locally:
    pip install tensorflow scikit-learn matplotlib numpy
    python train_report.py

Run in Google Colab (recommended, zero setup):
    1. Upload intents.json to the Colab file panel.
    2. Paste this whole file into a cell and run it.
    3. Download the PNGs from the report_assets/ folder for your report.
"""

import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend so it works on servers/Colab
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

# reproducibility
np.random.seed(42)
tf.random.set_seed(42)

os.makedirs("report_assets", exist_ok=True)


# ---- same tokenisation as app.js ----
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t]


# ---- load dataset ----
with open("intents.json", "r", encoding="utf-8") as f:
    data = json.load(f)

vocab_set, classes = set(), []
docs = []  # (tokens, tag)
for intent in data["intents"]:
    if intent["tag"] not in classes:
        classes.append(intent["tag"])
    for pattern in intent["patterns"]:
        toks = tokenize(pattern)
        vocab_set.update(toks)
        docs.append((toks, intent["tag"]))

vocab = sorted(vocab_set)                      # sorted => deterministic (matches app.js)
vocab_index = {w: i for i, w in enumerate(vocab)}
class_index = {c: i for i, c in enumerate(classes)}

print(f"vocab size : {len(vocab)}")
print(f"intents    : {len(classes)}")
print(f"samples    : {len(docs)}")


def bag_of_words(tokens):
    vec = np.zeros(len(vocab), dtype="float32")
    for t in tokens:
        if t in vocab_index:
            vec[vocab_index[t]] = 1.0
    return vec


X = np.array([bag_of_words(toks) for toks, _ in docs])
y = np.array([class_index[tag] for _, tag in docs])
Y = tf.keras.utils.to_categorical(y, num_classes=len(classes))

# stratified split so every intent appears in train; small test set for a metric
X_tr, X_te, Y_tr, Y_te, y_tr, y_te = train_test_split(
    X, Y, y, test_size=0.2, random_state=42, stratify=y
)


# ---- same architecture as app.js ----
model = Sequential([
    Input(shape=(len(vocab),)),
    Dense(16, activation="relu"),
    Dropout(0.3),
    Dense(16, activation="relu"),
    Dropout(0.3),
    Dense(len(classes), activation="softmax"),
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

history = model.fit(
    X_tr, Y_tr,
    validation_data=(X_te, Y_te),
    epochs=250,
    batch_size=8,
    verbose=0,
)

train_acc = history.history["accuracy"][-1]
val_acc = history.history["val_accuracy"][-1]
print(f"\nfinal train accuracy: {train_acc:.4f}")
print(f"final val   accuracy: {val_acc:.4f}")


# ---- accuracy curve ----
plt.figure(figsize=(6, 4))
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="validation")
plt.title("Model accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("report_assets/accuracy.png", dpi=150)
plt.close()

# ---- loss curve ----
plt.figure(figsize=(6, 4))
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="validation")
plt.title("Model loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.tight_layout()
plt.savefig("report_assets/loss.png", dpi=150)
plt.close()

# ---- classification report + confusion matrix (on test set) ----
y_pred = model.predict(X_te, verbose=0).argmax(axis=1)
print("\nClassification report (test set):")
print(classification_report(y_te, y_pred, target_names=classes, zero_division=0))

cm = confusion_matrix(y_te, y_pred, labels=list(range(len(classes))))
plt.figure(figsize=(7, 6))
plt.imshow(cm, interpolation="nearest", cmap="viridis")
plt.title("Confusion matrix (test set)")
plt.colorbar()
ticks = np.arange(len(classes))
plt.xticks(ticks, classes, rotation=90)
plt.yticks(ticks, classes)
for i in range(len(classes)):
    for j in range(len(classes)):
        if cm[i, j] > 0:
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white", fontsize=8)
plt.ylabel("true intent")
plt.xlabel("predicted intent")
plt.tight_layout()
plt.savefig("report_assets/confusion_matrix.png", dpi=150)
plt.close()

print("\nSaved figures to report_assets/:")
print("  accuracy.png, loss.png, confusion_matrix.png")
