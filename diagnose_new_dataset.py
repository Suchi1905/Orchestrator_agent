"""
Diagnose ML classifier failures on new_dataset.csv.
Shows: per-class accuracy, confidence of wrong predictions,
and what threshold would route them to Tier-3 LLM.
"""
import csv, sys, numpy as np
sys.path.insert(0, ".")
from agent import _ML_ENCODER, _ML_CLASSIFIER, _normalize_intent, ALLOWED_INTENTS

DATASET = r"c:\Users\hp\Desktop\orchestration_agent\dataset\havells_multi_intent_dataset.csv"

ALIAS_MAP = {
    "greeting":"greetings","greetings":"greetings",
    "device_control":"device_control",
    "service_request":"service_request","service":"service_request",
    "shopping":"service_request","queries":"service_request",
    "automations":"automations","automation":"automations",
    "out_of_scope":"out_of_scope","guardrail":"out_of_scope","unsafe":"out_of_scope",
}
def norm(label):
    intents = set()
    for p in str(label).split(","):
        n = p.strip().lower()
        if not n: continue
        mapped = "out_of_scope"
        if n in ALIAS_MAP:
            mapped = ALIAS_MAP[n]
        else:
            for k,v in ALIAS_MAP.items():
                if n.startswith(k):
                    mapped = v
                    break
        intents.add(mapped)
    if not intents:
        return "out_of_scope"
    return ",".join(sorted(list(intents)))

rows = list(csv.DictReader(open(DATASET, encoding="utf-8")))
texts  = [r["input_text"] for r in rows]
labels = [norm(r["actual_label"]) for r in rows]

print("Encoding with ML model...")
X = _ML_ENCODER.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
preds   = _ML_CLASSIFIER.predict(X)
probas  = _ML_CLASSIFIER.predict_proba(X)
confs   = probas.max(axis=1)
preds   = [norm(p) for p in preds]

correct = [p==a for p,a in zip(preds, labels)]
print(f"\nOverall ML accuracy: {sum(correct)}/{len(correct)} = {sum(correct)/len(correct)*100:.1f}%\n")

# Per-class
from collections import defaultdict
per = defaultdict(lambda: {"c":0,"t":0,"wrong_confs":[]})
for pred, actual, ok, conf in zip(preds, labels, correct, confs):
    per[actual]["t"] += 1
    if ok:
        per[actual]["c"] += 1
    else:
        per[actual]["wrong_confs"].append(conf)

print("Per-class accuracy:")
for cls in sorted(per.keys()):
    v = per[cls]
    t = v["t"] or 1
    pct = v["c"]/t*100
    wc = v["wrong_confs"]
    avg_wrong_conf = np.mean(wc) if wc else 0
    print(f"  {cls:<20s}: {v['c']:>3}/{t}  {pct:5.1f}%  | wrong predictions avg confidence: {avg_wrong_conf:.3f}")

# Threshold analysis — what fraction of WRONG predictions are above threshold?
print("\nThreshold analysis (what % of wrong predictions would reach Tier-3 LLM):")
for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
    wrong_above = sum(1 for ok, c in zip(correct, confs) if not ok and c >= thresh)
    wrong_total = sum(1 for ok in correct if not ok)
    correct_above = sum(1 for ok, c in zip(correct, confs) if ok and c >= thresh)
    correct_total = sum(1 for ok in correct if ok)
    print(f"  thresh={thresh:.2f}  |  wrong above (won't reach LLM): {wrong_above}/{wrong_total}  "
          f"|  correct above (Tier-2 handles OK): {correct_above}/{correct_total}")

print("\nSample wrong predictions:")
for i, (text, actual, pred, ok, conf) in enumerate(zip(texts, labels, preds, correct, confs)):
    if not ok:
        print(f"  [{actual:<20s}] -> [{pred:<20s}] conf={conf:.3f}  {text[:60]}")
        if i > 20: break
