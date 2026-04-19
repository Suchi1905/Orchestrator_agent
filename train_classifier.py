"""
train_classifier.py
-------------------
Trains a semantic ML classifier to replace the Tier-2 keyword rule engine.

Anti-overfitting measures applied:
  1. FROZEN embeddings — all-MiniLM-L6-v2 pre-trained on 1B+ sentences (no fine-tuning).
  2. 60/40 train/test split — larger held-out set proves generalization, not memorization.
  3. Logistic Regression (C=0.3) — simpler, more regularized than SVC; linear boundary
     on semantic embeddings is hard to overfit with only 600 training rows.
  4. Gaussian noise injection — perturbs embeddings during training so the model learns
     robust, broad decision boundaries instead of exact embedding positions.
  5. 5-fold cross-validation on train split — confirms consistency, not lucky overfitting.

Run once:
    python train_classifier.py

Output:
    tier2_classifier.pkl   — the trained (calibrated) Logistic Regression classifier
"""

import os, csv
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sentence_transformers import SentenceTransformer

# ── Config ─────────────────────────────────────────────────────────────────
DATASET_PATH   = r"c:\Users\hp\Downloads\final_cleaned_dataset.csv"
MODEL_NAME     = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_PKL     = os.path.join(os.path.dirname(__file__), "tier2_classifier.pkl")

# Both datasets are combined for training.
# A 60/40 stratified split is applied to the COMBINED set, so the test rows
# are genuinely held-out and never seen during training.
DATASET_PATHS = [
    r"c:\Users\hp\Downloads\final_cleaned_dataset.csv",   # 1000 rows (original benchmark)
    r"c:\Users\hp\Desktop\orchestration_agent\dataset\new_dataset.csv",  # 550 rows (new patterns)
]

TEST_FRACTION  = 0.40    # 60% train / 40% test — larger holdout proves generalization
RANDOM_SEED    = 42
LR_C           = 0.3     # Strong regularization (lower = more regularized)
NOISE_STD      = 0.02    # Gaussian noise std for embedding augmentation

INTENT_CLASSES = ["greetings", "device_control", "service_request", "automations", "out_of_scope"]

ALIAS_MAP = {
    "greeting":        "greetings",   "greetings":       "greetings",
    "device_control":  "device_control",
    "service_request": "service_request", "service":     "service_request",
    "shopping":        "service_request", "queries":     "service_request",
    "automations":     "automations",  "automation":    "automations",
    "out_of_scope":    "out_of_scope", "guardrail":     "out_of_scope",
    "unsafe":          "out_of_scope",
}


def normalize_label(raw: str) -> str:
    n = (raw or "").strip().lower()
    if n in ALIAS_MAP:
        return ALIAS_MAP[n]
    for k, v in ALIAS_MAP.items():
        if n.startswith(k):
            return v
    return "out_of_scope"


# ── Load datasets (combined) ────────────────────────────────────────────────
print("Loading datasets...")
texts, labels = [], []
for path in DATASET_PATHS:
    if not os.path.exists(path):
        print(f"  WARNING: Not found, skipping: {path}")
        continue
    count_before = len(texts)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text  = str(row.get("input_text", "")).strip()
            label = normalize_label(str(row.get("actual_label", "")))
            if text:
                texts.append(text)
                labels.append(label)
    print(f"  {len(texts) - count_before} rows  <-  {os.path.basename(path)}")

print(f"\nCombined total: {len(texts)} rows")
labels_arr = np.array(labels)
unique, counts = np.unique(labels_arr, return_counts=True)
for cls, cnt in zip(unique, counts):
    print(f"  {cls:<20s}: {cnt}")


# ── Stratified 60/40 split ─────────────────────────────────────────────────
print(f"\nSplitting: {int((1-TEST_FRACTION)*100)}% train / {int(TEST_FRACTION*100)}% test  (seed={RANDOM_SEED})")
splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=RANDOM_SEED)
train_idx, test_idx = next(splitter.split(texts, labels_arr))

train_texts  = [texts[i] for i in train_idx]
train_labels = labels_arr[train_idx]
test_texts   = [texts[i] for i in test_idx]
test_labels  = labels_arr[test_idx]

print(f"  Train: {len(train_texts)} rows  |  Test: {len(test_texts)} rows")

# ── Encode with frozen sentence-transformer ────────────────────────────────
print(f"\nLoading embedding model: {MODEL_NAME} ...")
encoder = SentenceTransformer(MODEL_NAME)

print("Encoding train split...")
X_train_clean = encoder.encode(train_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

print("Encoding test split...")
X_test = encoder.encode(test_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

print(f"  Embedding dim: {X_train_clean.shape[1]}")

# ── Gaussian noise augmentation ────────────────────────────────────────────
# Adds small random perturbations to training embeddings so the model learns
# a wider decision boundary — reduces sensitivity to exact embedding positions.
print(f"\nApplying Gaussian noise augmentation (std={NOISE_STD}) to training embeddings...")
rng = np.random.default_rng(RANDOM_SEED)
noise = rng.normal(0, NOISE_STD, X_train_clean.shape).astype(np.float32)
X_train = X_train_clean + noise
# Re-normalize so embeddings stay on unit sphere
norms = np.linalg.norm(X_train, axis=1, keepdims=True)
X_train = X_train / np.maximum(norms, 1e-8)
print(f"  Augmented training embeddings shape: {X_train.shape}")

# ── Train Logistic Regression (regularized, calibrated) ───────────────────
print(f"\nTraining Logistic Regression (C={LR_C}, strong L2 regularization)...")
base_lr = LogisticRegression(
    C=LR_C,
    max_iter=2000,
    class_weight="balanced",
    solver="lbfgs",
    random_state=RANDOM_SEED,
)
# CalibratedClassifierCV gives reliable probability scores for confidence thresholding
clf = CalibratedClassifierCV(base_lr, cv=5, method="isotonic")
clf.fit(X_train, train_labels)
print("  Training complete.")

# ── 5-fold cross-validation on training data ───────────────────────────────
print("\nRunning 5-fold cross-validation on train split...")
base_lr_cv = LogisticRegression(
    C=LR_C, max_iter=2000, class_weight="balanced",
    solver="lbfgs", random_state=RANDOM_SEED,
)
cv_scores = cross_val_score(base_lr_cv, X_train, train_labels, cv=5, scoring="accuracy")
print(f"  CV accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
print(f"  Per fold:    {[round(s, 3) for s in cv_scores]}")

# ── Evaluate on HELD-OUT test set ──────────────────────────────────────────
print("\n" + "=" * 60)
print("HELD-OUT TEST SET RESULTS (40% of data, never seen during training)")
print("=" * 60)
test_preds = clf.predict(X_test)
test_acc   = accuracy_score(test_labels, test_preds)
print(f"\nOverall accuracy: {test_acc:.4f}  ({test_acc * 100:.2f}%)")
print()
print(classification_report(test_labels, test_preds, target_names=INTENT_CLASSES, zero_division=0))

gap = cv_scores.mean() - test_acc
print(f"CV vs test gap: {gap:.4f}")
if gap > 0.08:
    print("WARNING: Large gap detected — possible overfitting. Try lowering C further.")
elif test_acc >= 0.90:
    print("OK: Test accuracy >= 90% with strong regularization — good generalization signal.")
else:
    print("NOTE: Test accuracy < 90%. Consider more training data or tuning C.")

# ── Save classifier ────────────────────────────────────────────────────────
print(f"\nSaving classifier to: {OUTPUT_PKL}")
joblib.dump(clf, OUTPUT_PKL)
print("Done. The Streamlit app will load the new model automatically on next restart.")
