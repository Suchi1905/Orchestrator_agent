import json, joblib
import numpy as np
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer

saved = joblib.load('tier2_multi_intent_classifier.pkl')
clf = saved['classifier']
le  = saved['label_encoder']
enc = SentenceTransformer(saved['encoder_name'])

ALIAS = {
    'automation': 'automations', 'automations': 'automations',
    'device_control': 'device_control', 'greetings': 'greetings',
    'greeting': 'greetings', 'out_of_scope': 'out_of_scope',
    'service_request': 'service_request', 'service': 'service_request',
}

with open(r'../dataset/final_test_dataset_15000_strong.json', encoding='utf-8') as f:
    data = json.load(f)

queries     = [d.get('query','') for d in data]
labels_raw  = [d.get('expected_response_type','') for d in data]
labels_true = [ALIAS.get(l.strip().lower(), l.strip().lower()) for l in labels_raw]
langs       = [d.get('language','unknown') for d in data]

print('Encoding 15,000 queries (this takes ~2 min on CPU)...')
embs   = enc.encode(queries, batch_size=256, normalize_embeddings=True, show_progress_bar=True)
probas = clf.predict_proba(embs)
preds_idx = probas.argmax(axis=1)
preds     = [le.classes_[i] for i in preds_idx]
confs     = probas.max(axis=1)

# ── Overall accuracy ──────────────────────────────────────────────────────
correct = sum(p == t for p, t in zip(preds, labels_true))
print(f'\nOverall ML Accuracy (pure ML, no Sarvam): {correct/len(preds)*100:.2f}%')
print(f'Total errors: {len(preds) - correct} / {len(preds)}')

# ── Confusion matrix ──────────────────────────────────────────────────────
classes = sorted(set(labels_true))
cm = defaultdict(lambda: defaultdict(int))
for t, p in zip(labels_true, preds):
    cm[t][p] += 1

pad = 26
print('\n--- CONFUSION MATRIX (rows=actual, cols=predicted) ---')
header = ' ' * pad + ''.join(c[:13].rjust(14) for c in classes)
print(header)
for actual in classes:
    row = actual.ljust(pad) + ''.join(str(cm[actual][pred]).rjust(14) for pred in classes)
    print(row)

# ── Per-intent accuracy ───────────────────────────────────────────────────
print('\n--- PER-INTENT ACCURACY ---')
for intent in classes:
    total     = sum(cm[intent].values())
    correct_i = cm[intent][intent]
    pct       = correct_i / total * 100 if total else 0
    print(f'  {intent:<25}: {correct_i:>5}/{total}  ({pct:.1f}%)  errors={total-correct_i}')

# ── Top misclassification pairs ───────────────────────────────────────────
print('\n--- TOP MISCLASSIFICATION PAIRS ---')
pairs = []
for actual in classes:
    for pred in classes:
        if actual != pred and cm[actual][pred] > 0:
            pairs.append((cm[actual][pred], actual, pred))
pairs.sort(reverse=True)
for count, actual, pred in pairs[:10]:
    print(f'  {actual:<25} -> {pred:<25}: {count:>5} errors')

# ── Sample wrong predictions for top 3 pairs ─────────────────────────────
print('\n--- SAMPLE WRONG PREDICTIONS (top 3 pairs) ---')
top3 = [(a, p) for _, a, p in pairs[:3]]
shown = defaultdict(int)
for i, (q, t, p, c) in enumerate(zip(queries, labels_true, preds, confs)):
    if (t, p) in top3 and shown[(t, p)] < 5:
        print(f'  [{t} -> {p}] conf={c:.3f} | {q[:80]}')
        shown[(t, p)] += 1

# ── Confidence of wrong predictions ──────────────────────────────────────
wrong_confs = [confs[i] for i in range(len(preds)) if preds[i] != labels_true[i]]
print(f'\n--- CONFIDENCE DISTRIBUTION OF {len(wrong_confs)} WRONG PREDICTIONS ---')
buckets = [
    (0.00, 0.50, '0.00-0.50 (-> Sarvam at 0.75 threshold)'),
    (0.50, 0.65, '0.50-0.65 (-> Sarvam at 0.75 threshold)'),
    (0.65, 0.75, '0.65-0.75 (-> Sarvam at 0.75 threshold)'),
    (0.75, 0.85, '0.75-0.85 (HIGH CONF WRONG - unfixable without retrain)'),
    (0.85, 0.95, '0.85-0.95 (HIGH CONF WRONG - unfixable without retrain)'),
    (0.95, 1.01, '0.95-1.00 (VERY HIGH CONF WRONG - critical errors)'),
]
for lo, hi, label in buckets:
    cnt = sum(1 for c in wrong_confs if lo <= c < hi)
    print(f'  {cnt:>4} ({cnt/len(wrong_confs)*100:.1f}%)  conf {label}')

# ── Per-language accuracy ─────────────────────────────────────────────────
print('\n--- PER-LANGUAGE ACCURACY ---')
lang_correct = defaultdict(int)
lang_total   = defaultdict(int)
for p, t, l in zip(preds, labels_true, langs):
    lang_total[l]  += 1
    if p == t:
        lang_correct[l] += 1
for lang in sorted(lang_total):
    tot = lang_total[lang]
    cor = lang_correct[lang]
    print(f'  {lang:<15}: {cor:>4}/{tot}  ({cor/tot*100:.1f}%)')

# ── Overlap vs novel query accuracy ──────────────────────────────────────
with open(r'../dataset/final_dataset_improved_automation.json', encoding='utf-8') as f:
    train_data = json.load(f)
train_queries = set(d['query'] for d in train_data)

overlap_ok  = sum(1 for i in range(len(queries)) if queries[i] in train_queries and preds[i] == labels_true[i])
overlap_tot = sum(1 for q in queries if q in train_queries)
novel_ok    = sum(1 for i in range(len(queries)) if queries[i] not in train_queries and preds[i] == labels_true[i])
novel_tot   = sum(1 for q in queries if q not in train_queries)

print('\n--- OVERLAP vs NOVEL QUERY ACCURACY ---')
if overlap_tot:
    print(f'  Queries seen in training : {overlap_ok}/{overlap_tot}  ({overlap_ok/overlap_tot*100:.1f}%)')
if novel_tot:
    print(f'  Queries NEVER seen       : {novel_ok}/{novel_tot}  ({novel_ok/novel_tot*100:.1f}%)')

print('\nDone.')
