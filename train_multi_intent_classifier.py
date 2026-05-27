"""
train_multi_intent_classifier.py
---------------------------------
Trains the intent classifier on the new balanced multilingual dataset.

Dataset : final_dataset_improved_automation.json
          - 15,000 records | 5 intents × 3,000 each | 8 languages × 1,875 each
          - Single-label: every record carries exactly ONE intent

Embedding: paraphrase-multilingual-MiniLM-L12-v2
           - Multilingual sentence encoder (128 languages)
           - 384-dim embeddings, ~50ms CPU inference per query

Classifier: LogisticRegression (multinomial) + CalibratedClassifierCV
            - Reliable probability outputs for confidence-based routing
            - class_weight='balanced' (safe even on balanced data)
            - 5-fold sigmoid calibration

Data Split (stratified):
    Train      : 7,500  (50%)
    Validation : 3,750  (25%)
    Test       : 3,750  (25%)

Decision Logic (in agent.py):
    confidence >= 0.75  -> return ML result
    confidence <  0.75  -> escalate to Sarvam 30B LLM

Output:
    tier2_multi_intent_classifier.pkl
        keys: 'classifier'     — CalibratedClassifierCV
              'label_encoder'  — sklearn LabelEncoder
              'encoder_name'   — str (embedding model name)

Run:
    python train_multi_intent_classifier.py
"""

import json
import os
import time
import numpy as np
import joblib

from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sentence_transformers import SentenceTransformer

# ── Configuration ──────────────────────────────────────────────────────────────

DATASET_PATH = os.path.join(
    os.path.dirname(__file__),
    r"..\dataset\final_dataset_improved_automation.json",
)
DATASET_PATH = os.path.abspath(DATASET_PATH)

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
OUTPUT_PKL      = os.path.join(os.path.dirname(__file__), "tier2_multi_intent_classifier.pkl")

# Split sizes (must sum to 15,000)
N_TRAIN = 7_500
N_VAL   = 3_750
N_TEST  = 3_750

RANDOM_SEED = 42
LR_C        = 1.0      # L2 regularisation strength
NOISE_STD   = 0.03     # Gaussian noise for augmentation
BATCH_SIZE  = 128      # Embedding batch size
CONFIDENCE_THRESHOLD = 0.75  # Routing threshold (documented, not enforced here)

# Canonical label set used by agent.py
# The dataset uses 'automation' (no 's'); we normalise it here.
LABEL_ALIAS = {
    "automation":      "automations",
    "automations":     "automations",
    "device_control":  "device_control",
    "greetings":       "greetings",
    "greeting":        "greetings",
    "out_of_scope":    "out_of_scope",
    "service_request": "service_request",
    "service":         "service_request",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _banner(text: str) -> None:
    width = 68
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def _normalise_label(raw: str) -> str:
    """Map raw dataset label to canonical agent label."""
    key = str(raw).strip().lower()
    return LABEL_ALIAS.get(key, "out_of_scope")


# ── Load & Validate Dataset ────────────────────────────────────────────────────

_banner("STEP 1 — LOAD DATASET")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

with open(DATASET_PATH, encoding="utf-8") as f:
    raw_data = json.load(f)

print(f"  Raw records loaded : {len(raw_data):,}")

texts, labels, languages = [], [], []
skipped = 0

for record in raw_data:
    query = str(record.get("query", "")).strip()
    label_raw = record.get("expected_response_type", "")
    lang  = record.get("language", "unknown")

    if not query or not label_raw:
        skipped += 1
        continue

    canonical = _normalise_label(label_raw)
    texts.append(query)
    labels.append(canonical)
    languages.append(lang)

print(f"  Valid records      : {len(texts):,}")
print(f"  Skipped (empty)    : {skipped}")

# ── Distribution Report ────────────────────────────────────────────────────────

_banner("STEP 2 — DISTRIBUTION REPORT")

print("\n  Intent distribution:")
for intent, count in sorted(Counter(labels).items()):
    pct = count / len(labels) * 100
    print(f"    {intent:<20}: {count:>5,}  ({pct:.1f}%)")

print("\n  Language distribution:")
for lang, count in sorted(Counter(languages).items()):
    pct = count / len(languages) * 100
    print(f"    {lang:<15}: {count:>5,}  ({pct:.1f}%)")

# ── Stratified 3-Way Split ─────────────────────────────────────────────────────

_banner("STEP 3 — STRATIFIED SPLIT  (7,500 / 3,750 / 3,750)")

texts_arr  = np.array(texts)
labels_arr = np.array(labels)

# Step A: carve out 7,500 train vs 7,500 (val+test)
sss_a = StratifiedShuffleSplit(n_splits=1, test_size=N_VAL + N_TEST, random_state=RANDOM_SEED)
train_idx, temp_idx = next(sss_a.split(texts_arr, labels_arr))

train_texts  = texts_arr[train_idx].tolist()
train_labels = labels_arr[train_idx].tolist()

temp_texts  = texts_arr[temp_idx].tolist()
temp_labels = labels_arr[temp_idx].tolist()

# Step B: split the 7,500 remainder into val=3,750 and test=3,750
sss_b = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=RANDOM_SEED)
val_idx, test_idx = next(sss_b.split(temp_texts, temp_labels))

temp_texts_arr  = np.array(temp_texts)
temp_labels_arr = np.array(temp_labels)

val_texts   = temp_texts_arr[val_idx].tolist()
val_labels  = temp_labels_arr[val_idx].tolist()
test_texts  = temp_texts_arr[test_idx].tolist()
test_labels = temp_labels_arr[test_idx].tolist()

print(f"\n  Train : {len(train_texts):>5,} samples")
print(f"  Val   : {len(val_texts):>5,} samples")
print(f"  Test  : {len(test_texts):>5,} samples")

print("\n  Per-intent split:")
for intent in sorted(set(labels)):
    tr = train_labels.count(intent)
    va = val_labels.count(intent)
    te = test_labels.count(intent)
    print(f"    {intent:<20}: train={tr:>4}  val={va:>3}  test={te:>3}")

# ── Label Encoding ─────────────────────────────────────────────────────────────

le = LabelEncoder()
le.fit(sorted(set(labels)))   # deterministic ordering

y_train = le.transform(train_labels)
y_val   = le.transform(val_labels)
y_test  = le.transform(test_labels)

print(f"\n  Classes: {list(le.classes_)}")

# ── Embed with Sentence Transformer ───────────────────────────────────────────

_banner(f"STEP 4 — EMBED  [{EMBEDDING_MODEL}]")

print(f"\n  Loading model: {EMBEDDING_MODEL}  ...")
t0 = time.time()
encoder = SentenceTransformer(EMBEDDING_MODEL)
print(f"  Model loaded in {time.time() - t0:.1f}s")

print("\n  Encoding TRAIN split ...")
t0 = time.time()
X_train_clean = encoder.encode(
    train_texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
)
print(f"  Done in {time.time() - t0:.1f}s  |  shape: {X_train_clean.shape}")

print("\n  Encoding VAL split ...")
t0 = time.time()
X_val = encoder.encode(
    val_texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
)
print(f"  Done in {time.time() - t0:.1f}s  |  shape: {X_val.shape}")

print("\n  Encoding TEST split ...")
t0 = time.time()
X_test = encoder.encode(
    test_texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
)
print(f"  Done in {time.time() - t0:.1f}s  |  shape: {X_test.shape}")

print(f"\n  Embedding dimension: {X_train_clean.shape[1]}")

# ── Gaussian Noise Augmentation ────────────────────────────────────────────────

_banner("STEP 5 — GAUSSIAN NOISE AUGMENTATION")

print(f"\n  Applying noise std={NOISE_STD} to training embeddings ...")
rng = np.random.default_rng(RANDOM_SEED)
noise = rng.normal(0.0, NOISE_STD, X_train_clean.shape).astype(np.float32)
X_train = (X_train_clean + noise).astype(np.float32)

# Re-normalise after noise injection
norms = np.linalg.norm(X_train, axis=1, keepdims=True)
X_train = X_train / np.maximum(norms, 1e-8)
print(f"  Augmented train shape: {X_train.shape}")

# ── Train Classifier ───────────────────────────────────────────────────────────

_banner(f"STEP 6 — TRAIN  LogisticRegression (C={LR_C}, calibrated, cv=5)")

print("\n  Initialising classifier ...")
base_lr = LogisticRegression(
    C=LR_C,
    max_iter=2_000,
    class_weight="balanced",
    solver="lbfgs",
    random_state=RANDOM_SEED,
)
clf = CalibratedClassifierCV(base_lr, cv=5, method="sigmoid")

print("  Training ...")
t0 = time.time()
clf.fit(X_train, y_train)
print(f"  Training complete in {time.time() - t0:.1f}s")

# ── Evaluate on Validation Set ────────────────────────────────────────────────

_banner("STEP 7 — VALIDATION SET RESULTS")

val_preds   = clf.predict(X_val)
val_probas  = clf.predict_proba(X_val)
val_acc     = accuracy_score(y_val, val_preds)
val_f1_mac  = f1_score(y_val, val_preds, average="macro")
val_f1_wtd  = f1_score(y_val, val_preds, average="weighted")

print(f"\n  Accuracy (val)          : {val_acc:.4f}  ({val_acc * 100:.2f}%)")
print(f"  Macro F1 (val)          : {val_f1_mac:.4f}")
print(f"  Weighted F1 (val)       : {val_f1_wtd:.4f}")

# Confidence distribution
val_max_conf = val_probas.max(axis=1)
routed_ml   = (val_max_conf >= CONFIDENCE_THRESHOLD).sum()
routed_llm  = (val_max_conf <  CONFIDENCE_THRESHOLD).sum()
print(f"\n  Confidence >= {CONFIDENCE_THRESHOLD} (ML handles) : {routed_ml:>4} / {len(val_preds):>4}  ({routed_ml/len(val_preds)*100:.1f}%)")
print(f"  Confidence <  {CONFIDENCE_THRESHOLD} (-> Sarvam)  : {routed_llm:>4} / {len(val_preds):>4}  ({routed_llm/len(val_preds)*100:.1f}%)")

print(f"\n  Per-class report (validation):")
print(classification_report(
    y_val, val_preds,
    target_names=le.classes_,
    digits=4,
    zero_division=0,
))

print("  Confusion Matrix (val)  [rows=actual, cols=predicted]:")
cm_val = confusion_matrix(y_val, val_preds)
header = "  " + "".join(f"{c[:8]:>10}" for c in le.classes_)
print(header)
for i, row in enumerate(cm_val):
    row_str = "  " + f"{le.classes_[i][:10]:<10}" + "".join(f"{v:>10}" for v in row)
    print(row_str)

# ── Evaluate on Test Set ───────────────────────────────────────────────────────

_banner("STEP 8 — TEST SET RESULTS  (held-out, never seen during training)")

test_preds  = clf.predict(X_test)
test_probas = clf.predict_proba(X_test)
test_acc    = accuracy_score(y_test, test_preds)
test_f1_mac = f1_score(y_test, test_preds, average="macro")
test_f1_wtd = f1_score(y_test, test_preds, average="weighted")

print(f"\n  Accuracy (test)         : {test_acc:.4f}  ({test_acc * 100:.2f}%)")
print(f"  Macro F1 (test)         : {test_f1_mac:.4f}")
print(f"  Weighted F1 (test)      : {test_f1_wtd:.4f}")

# Confidence distribution on test
test_max_conf = test_probas.max(axis=1)
routed_ml_t  = (test_max_conf >= CONFIDENCE_THRESHOLD).sum()
routed_llm_t = (test_max_conf <  CONFIDENCE_THRESHOLD).sum()
print(f"\n  Confidence >= {CONFIDENCE_THRESHOLD} (ML handles) : {routed_ml_t:>4} / {len(test_preds):>4}  ({routed_ml_t/len(test_preds)*100:.1f}%)")
print(f"  Confidence <  {CONFIDENCE_THRESHOLD} (-> Sarvam)  : {routed_llm_t:>4} / {len(test_preds):>4}  ({routed_llm_t/len(test_preds)*100:.1f}%)")

print(f"\n  Per-class report (test):")
print(classification_report(
    y_test, test_preds,
    target_names=le.classes_,
    digits=4,
    zero_division=0,
))

print("  Confusion Matrix (test)  [rows=actual, cols=predicted]:")
cm_test = confusion_matrix(y_test, test_preds)
print(header)
for i, row in enumerate(cm_test):
    row_str = "  " + f"{le.classes_[i][:10]:<10}" + "".join(f"{v:>10}" for v in row)
    print(row_str)

# ── Latency Benchmark ──────────────────────────────────────────────────────────

_banner("STEP 9 — LATENCY BENCHMARK  (single-query, CPU)")

sample_queries = [
    "Turn on the fan",
    "Book a repair for my AC",
    "Set a timer to turn off the geyser after 2 hours",
    "Hello, good morning",
    "Tell me about history",
]

latencies = []
for q in sample_queries:
    t0 = time.perf_counter()
    emb = encoder.encode([q], normalize_embeddings=True, convert_to_numpy=True)
    _ = clf.predict_proba(emb)
    latencies.append((time.perf_counter() - t0) * 1000)

print(f"\n  Sample latencies (embed + predict):")
for q, lat in zip(sample_queries, latencies):
    status = "OK" if lat < 50 else "!!"
    print(f"    [{status}] {lat:5.1f}ms  |  {q[:55]}")

avg_lat = sum(latencies) / len(latencies)
print(f"\n  Average latency : {avg_lat:.1f}ms  (target: <50ms)")

# ── Final Verdict ──────────────────────────────────────────────────────────────

_banner("STEP 10 — VERDICT")

print()
if test_acc >= 0.93:
    print(f"  [EXCELLENT] Test accuracy {test_acc*100:.2f}%  ≥ 93%  — production-ready.")
elif test_acc >= 0.90:
    print(f"  [OK]        Test accuracy {test_acc*100:.2f}%  ≥ 90%  — meets target.")
else:
    print(f"  [WARNING]   Test accuracy {test_acc*100:.2f}%  < 90%  — needs investigation.")

if avg_lat < 50:
    print(f"  [OK]        Average latency {avg_lat:.1f}ms  < 50ms — latency target met.")
else:
    print(f"  [WARNING]   Average latency {avg_lat:.1f}ms  > 50ms — may need GPU.")

# ── Save Model ─────────────────────────────────────────────────────────────────

_banner("STEP 11 — SAVE MODEL")

payload = {
    "classifier":    clf,
    "label_encoder": le,
    "encoder_name":  EMBEDDING_MODEL,
    # Metadata for traceability
    "train_size":    len(train_texts),
    "val_size":      len(val_texts),
    "test_size":     len(test_texts),
    "val_accuracy":  round(val_acc, 6),
    "test_accuracy": round(test_acc, 6),
    "classes":       list(le.classes_),
}

print(f"\n  Saving to: {OUTPUT_PKL}")
joblib.dump(payload, OUTPUT_PKL)
size_kb = os.path.getsize(OUTPUT_PKL) / 1024
print(f"  File size : {size_kb:.0f} KB")
print(f"\n  Classes   : {list(le.classes_)}")
print(f"\n  Done! The agent will load this model automatically on next start.")
print(f"  -> tier2_multi_intent_classifier.pkl")
