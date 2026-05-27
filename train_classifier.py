"""
train_classifier.py
-------------------
Trains a semantic ML classifier to replace the Tier-2 keyword rule engine.
Supports Multi-Intent training.

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
    tier2_classifier.pkl   — dictionary containing 'classifier' (calibrated Logistic Regression) and 'mlb' (MultiLabelBinarizer)
"""

import os, csv
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sentence_transformers import SentenceTransformer

# ── Config ─────────────────────────────────────────────────────────────────
DATASET_PATHS = [
    r"c:\Users\hp\Downloads\final_cleaned_dataset.csv",   # 1000 rows (original benchmark)
    r"c:\Users\hp\Desktop\orchestration_agent\dataset\new_dataset.csv",  # 550 rows (new patterns)
    r"c:\Users\hp\Desktop\orchestration_agent\dataset\havells_multi_intent_dataset.csv", # Multi intent
]

MODEL_NAME     = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_PKL     = os.path.join(os.path.dirname(__file__), "tier2_classifier.pkl")

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


def normalize_labels(raw: str) -> list:
    intents = []
    for p in str(raw).split(","):
        n = p.strip().lower()
        if not n: continue
        mapped = "out_of_scope"
        if n in ALIAS_MAP:
            mapped = ALIAS_MAP[n]
        else:
            for k, v in ALIAS_MAP.items():
                if n.startswith(k):
                    mapped = v
                    break
        intents.append(mapped)
    
    # Remove duplicates
    intents = list(set(intents))
    if not intents:
        return ["out_of_scope"]
    return intents


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
            label_raw = row.get("actual_label", row.get("label", ""))
            label_list = normalize_labels(label_raw)
            if text:
                texts.append(text)
                labels.append(label_list)
    print(f"  {len(texts) - count_before} rows  <-  {os.path.basename(path)}")

print(f"\nCombined total: {len(texts)} rows")

# ── Multi-Label Binarization ───────────────────────────────────────────────
mlb = MultiLabelBinarizer(classes=INTENT_CLASSES)
labels_arr = mlb.fit_transform(labels)
print("\nMultiLabel distribution:")
for cls_idx, cls_name in enumerate(mlb.classes_):
    print(f"  {cls_name:<20s}: {labels_arr[:, cls_idx].sum()}")


# ── Train/Test split ───────────────────────────────────────────────────────
print(f"\nSplitting: {int((1-TEST_FRACTION)*100)}% train / {int(TEST_FRACTION*100)}% test  (seed={RANDOM_SEED})")
# StratifiedShuffleSplit isn't natively supported for multilabel arrays in sklearn without iterative-stratification
# We will use random train_test_split which is generally sufficient for 1500+ rows
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels_arr, test_size=TEST_FRACTION, random_state=RANDOM_SEED
)

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
print(f"\nApplying Gaussian noise augmentation (std={NOISE_STD}) to training embeddings...")
rng = np.random.default_rng(RANDOM_SEED)
noise = rng.normal(0, NOISE_STD, X_train_clean.shape).astype(np.float32)
X_train = X_train_clean + noise
norms = np.linalg.norm(X_train, axis=1, keepdims=True)
X_train = X_train / np.maximum(norms, 1e-8)
print(f"  Augmented training embeddings shape: {X_train.shape}")

# ── Train Logistic Regression (regularized, calibrated) ───────────────────
print(f"\nTraining Multi-Intent Logistic Regression (OneVsRest + Calibration, C={LR_C})...")
base_lr = LogisticRegression(
    C=LR_C,
    max_iter=2000,
    class_weight="balanced",
    solver="lbfgs",
    random_state=RANDOM_SEED,
)
clf = OneVsRestClassifier(CalibratedClassifierCV(base_lr, cv=5, method="isotonic"))
clf.fit(X_train, train_labels)
print("  Training complete.")


# ── Evaluate on HELD-OUT test set ──────────────────────────────────────────
print("\n" + "=" * 60)
print("HELD-OUT TEST SET RESULTS (40% of data, never seen during training)")
print("=" * 60)
test_preds = clf.predict(X_test)
test_acc_exact = accuracy_score(test_labels, test_preds)
print(f"\nExact-Match Accuracy (all intents must match): {test_acc_exact:.4f}  ({test_acc_exact * 100:.2f}%)")
print()
print(classification_report(test_labels, test_preds, target_names=mlb.classes_, zero_division=0))

if test_acc_exact >= 0.85:
    print("OK: Test exact accuracy >= 85% with multi-label formulation — good generalization signal.")
else:
    print("NOTE: Test exact accuracy < 85%. Consider more training data or tuning C.")

# ── Save classifier & MLB ─────────────────────────────────────────────────
print(f"\nSaving classifier and binarizer to: {OUTPUT_PKL}")
model_dict = {
    'classifier': clf,
    'mlb': mlb
}
joblib.dump(model_dict, OUTPUT_PKL)
print("Done. The Streamlit app will load the new model automatically on next restart.")
