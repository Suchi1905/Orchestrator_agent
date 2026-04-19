"""
Final end-to-end validation using the ML-powered agent pipeline.
Tests both known dataset rows AND paraphrased queries not in the training set
to verify generalization beyond vocabulary.
"""
import asyncio, csv, sys
sys.path.insert(0, ".")

from agent import orchestrate_request_with_meta, _normalize_intent

DATASET_PATH = r"c:\Users\hp\Downloads\final_cleaned_dataset.csv"

ALIAS_MAP = {
    "greeting":"greetings","greetings":"greetings",
    "device_control":"device_control",
    "service_request":"service_request","service":"service_request",
    "shopping":"service_request","queries":"service_request",
    "automations":"automations","automation":"automations",
    "out_of_scope":"out_of_scope","guardrail":"out_of_scope","unsafe":"out_of_scope",
}
def norm(label):
    n=(label or "").strip().lower()
    if n in ALIAS_MAP: return ALIAS_MAP[n]
    for k,v in ALIAS_MAP.items():
        if n.startswith(k): return v
    return "out_of_scope"

# ── Paraphrased queries NOT in the dataset (generalization test) ──────────
NOVEL_QUERIES = [
    ("Deactivate the illumination device in the hall",  "device_control"),
    ("Cut the power to the cooling unit",               "device_control"),
    ("Initiate the climate warming system",             "device_control"),
    ("Schedule the air purifier every weekday morning", "automations"),
    ("When I get home, start the geyser",               "automations"),
    ("My television is malfunctioning",                 "service_request"),
    ("I would like to obtain service coverage details", "service_request"),
    ("How are you doing today?",                        "greetings"),
    ("Greetings and salutations",                       "greetings"),
    ("Elaborate on the history of the Roman empire",    "out_of_scope"),
    ("What is the population of India?",                "out_of_scope"),
]

async def run():
    print("\n=== DATASET ROWS (sample: first 50) ===")
    rows = list(csv.DictReader(open(DATASET_PATH, encoding="utf-8")))[:50]
    correct = 0
    for r in rows:
        actual = norm(r["actual_label"])
        res    = await orchestrate_request_with_meta(r["input_text"])
        pred   = _normalize_intent(res["output"].get("intent",""))
        ok     = pred == actual
        correct += int(ok)
        if not ok:
            print(f"  FAIL [{actual}] -> [{pred}]: {r['input_text'][:65]}")
    print(f"  Accuracy on first 50 rows: {correct}/50 = {correct/50*100:.1f}%")

    print("\n=== NOVEL / PARAPHRASED QUERIES (generalization test) ===")
    novel_correct = 0
    for query, expected in NOVEL_QUERIES:
        res  = await orchestrate_request_with_meta(query)
        pred = _normalize_intent(res["output"].get("intent",""))
        ok   = pred == expected
        novel_correct += int(ok)
        status = "OK  " if ok else "FAIL"
        print(f"  {status} [{pred:<20s}] expected [{expected:<20s}] | {query[:60]}")
    print(f"\n  Novel query accuracy: {novel_correct}/{len(NOVEL_QUERIES)} = {novel_correct/len(NOVEL_QUERIES)*100:.1f}%")

asyncio.run(run())
