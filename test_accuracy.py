"""
test_accuracy.py
----------------
Offline accuracy check for BOTH the mixed-intent and multi-intent classifiers.

DEFINITIONS
-----------
Mixed-intent:  Each query has ONE canonical label.
               Classifier: tier2_classifier.pkl
               Dataset:    ALL three datasets combined.

Multi-intent:  Each query has TWO canonical labels (e.g. device_control+service_request).
               Classifier: tier2_multi_intent_classifier.pkl
               Dataset:    ONLY havells_multi_intent_dataset.csv
               Priority rules applied after prediction.

Run:
    python test_accuracy.py
"""
import csv
import re
import os
import sys

# ── Dataset paths ────────────────────────────────────────────────────────
MIXED_DATASET_PATHS = [
    r"c:\Users\hp\Downloads\final_cleaned_dataset.csv",
    r"c:\Users\hp\Desktop\orchestration_agent\dataset\new_dataset.csv",
]

MULTI_INTENT_DATASET_PATH = (
    r"c:\Users\hp\Desktop\orchestration_agent\dataset\havells_multi_intent_dataset.csv"
)

# ── Label normalisation ──────────────────────────────────────────────────
ALIAS_MAP = {
    "greeting":        "greetings",
    "greetings":       "greetings",
    "device_control":  "device_control",
    "service_request": "service_request",
    "service":         "service_request",
    "shopping":        "service_request",
    "queries":         "service_request",
    "automations":     "automations",
    "automation":      "automations",
    "out_of_scope":    "out_of_scope",
    "guardrail":       "out_of_scope",
    "unsafe":          "out_of_scope",
}


def normalize_labels(raw: str) -> frozenset:
    """Convert raw CSV label string to a frozenset of canonical intent names."""
    intents = set()
    for part in str(raw).split(","):
        token = part.strip().lower()
        if not token:
            continue
        mapped = "out_of_scope"
        if token in ALIAS_MAP:
            mapped = ALIAS_MAP[token]
        else:
            for k, v in ALIAS_MAP.items():
                if token.startswith(k):
                    mapped = v
                    break
        intents.add(mapped)
    return frozenset(intents) if intents else frozenset(["out_of_scope"])


# ── Signal sets (rule-based fallback) ────────────────────────────────────
GREETING_PHRASES = [
    "good morning", "good evening", "good afternoon", "good night",
    "warm greetings", "dear friend", "dear colleague",
    "welcome back", "welcome, ", "welcome folks", "welcome team",
    "welcome partner", "welcome everyone", "warm wishes", "warm regards",
    "delighted to see", "delighted to meet", "nice to see", "pleased to see",
]
GREETING_TOKENS = {
    "hi", "hello", "hey", "namaste", "greetings", "howdy",
    "welcome", "sup", "yo", "salutations",
}
GREETING_PATTERN = re.compile(
    r"^(warm (wishes|regards)|delighted to (see|meet)|"
    r"nice to (see|meet)|pleased to (see|meet)|"
    r"good (morning|evening|afternoon|night|day)|"
    r"hi (there|friends?|everyone|all|team|folks?|partner|neighbour)|"
    r"hello (there|everyone|all|team|folks?|friends?)|"
    r"hey (there|everyone|all|team|folks?|friends?))",
    re.IGNORECASE,
)

SERVICE_PHRASES = [
    "raise a service request", "raise a service",
    "book a repair", "book repair", "book service",
    "send a technician", "technician for", "need a technician",
    "log a complaint", "raise a complaint", "file a complaint",
    "register a complaint", "report an issue", "report issue",
    "create a repair ticket", "repair ticket", "service ticket",
    "customer support", "need support", "need help with",
    "schedule service",
    "buy a", "purchase a", "buy new", "new device", "new camera",
    "new plug", "new bulb", "new light", "new fan", "new heater",
    "new models", "new model", "best model", "best camera",
    "show new", "discover new", "help me discover", "looking to buy",
    "i want to buy", "recommend a", "recommend the latest",
    "recommendation for", "newest catalogue", "catalogue",
    "compare amc", "share amc", "amc options", "amc details",
    "amc coverage", "amc plan", "amc plans", "show amc",
    "maintenance plan", "maintenance plans", "maintenance coverage",
    "show maintenance",
    "loyalty plan", "loyalty points", "loyalty balance", "tell me the loyalty",
    "reward points", "show reward", "check reward",
    "warranty for", "warranty plan", "warranty registration",
    "complete warranty",
    "register my", "enroll my", "link my",
    "add to my account", "add to my profile",
    "update my profile", "manage my account", "my account",
    "my profile", "subscription",
    "not working", "broken", "making noise",
    "installation", "install the", "demo of",
    "show the points", "points available", "points for",
    "explore new", "i want to explore",
]

TIME_PATTERN = re.compile(
    r"\b("
    r"in \d+\s*(min|sec|hour|minute|second)s?|"
    r"after \d+\s*(min|sec|hour|minute|second)s?|"
    r"over the next \d+|"
    r"at \d{1,2}(:\d{2})?\s*([ap]m)?|"
    r"at midnight|at noon|at sunrise|at sunset|"
    r"tomorrow|every day|everyday|daily|"
    r"every (morning|evening|night|week|month)|"
    r"on weekdays|on weekends|"
    r"when i (leave|arrive|wake|sleep)|"
    r"schedule|automatically|routinely|recurring|"
    r"set a timer|set timer"
    r")\b",
    re.IGNORECASE,
)

CONDITION_PATTERN = re.compile(
    r"\bif (the |it )?(temperature|motion|room|it|light level|it gets)",
    re.IGNORECASE,
)

DEVICE_ACTION_PHRASES = [
    r"turn on", r"turn off", r"switch on", r"switch off",
    r"power on", r"power off", r"shut down", r"shutdown",
    r"set the", r"set temperature", r"set brightness", r"set volume",
    r"increase", r"decrease", r"raise the", r"lower", r"dim", r"brighten",
    r"open the", r"close the", r"lock the", r"unlock the",
    r"activate", r"deactivate", r"enable", r"disable",
    r"make.*inactive", r"make.*active",
    r"start the", r"start up", r"startup",
    r"stop the", r"pause the", r"resume the",
    r"plug in", r"unplug", r"reboot", r"restart", r"reset the",
    r"wake up the", r"put the.*to sleep", r"put.*to sleep",
    r"please turn", r"please switch",
    r"power up", r"boot up", r"bring.*online",
    r"take.*offline",
]

OOS_PHRASES = [
    "weather", "news today", "latest news",
    "joke", "tell me a joke",
    "history of ", "explain history", "discuss history",
    "can you discuss", "can you explain",
    "about science", "about history", "about politics",
    "about sports", "about music", "about movies", "about books",
    "about recipes", "about cooking",
    "sports update", "sports news",
    "politics", "political",
    "recipe for", "how to cook",
    "biography of", "who is the ceo",
    "school", "university", "college", "education system",
    "general knowledge", "internet speed",
    "write a poem", "write an essay", "write a story",
    "summarize this", "summarize the",
    "what is the capital", "how old is",
    "translate", "meaning of",
    "teach me about history", "teach me about science",
    "give a summary of books", "give a summary of movies",
    "i want to know about history", "i want to know about science",
    "tell me about history", "tell me about science",
    "tell me about sports", "tell me about movies",
    "tell me about books", "tell me about music",
    "tell me about politics", "tell me about weather",
]

DEVICE_TOKENS = {
    "lights", "light", "fan", "fans", "ac", "heater", "purifier",
    "tv", "camera", "bulb", "bulbs", "plug", "plugs", "washer",
    "washing machine", "fridge", "refrigerator", "oven", "geyser",
    "exhaust", "tube", "led", "cooler", "air fryer", "airfryer",
    "speaker", "router", "hub", "sensor", "microwave",
}

# ── Strong time/schedule signal (mirrors agent.py) ───────────────────────────
STRONG_TIME_PATTERN = re.compile(
    r"\b("
    r"at \d{1,2}(:\d{2})?\s*([ap]m)?"
    r"|at midnight|at noon|at sunrise|at sunset|at bedtime|at night"
    r"|in \d+\s*(min|sec|hour|minute|second)s?"
    r"|after \d+\s*(min|sec|hour|minute|second)s?"
    r"|every day|everyday|daily|nightly"
    r"|every\s+(morning|evening|night|afternoon|week|month|weekday)"
    r"|on weekdays|on weekends"
    r"|schedule[d]?"
    r"|automatically|routinely|recurring"
    r"|\btimer\b"
    r"|when i (leave|arrive|wake|sleep|reach)"
    r"|when (motion|humidity|temperature)"
    r"|before i (arrive|reach|wake|leave)"
    r"|30 mins before|20 mins before|\d+ minutes before"
    r")\b",
    re.IGNORECASE,
)

GENERIC_OOS_STARTERS = re.compile(
    r"^(what is |who is |where is |how old |what are |"
    r"explain |tell me about (history|science|sports|movies|books|music|politics|weather)|"
    r"teach me |give a summary of (books|movies)|"
    r"can you (discuss|explain)|i want to know about (history|science)|"
    r"summarize )",
    re.IGNORECASE,
)


def _rule_classify(user_input: str) -> frozenset:
    """Rule-based classifier — used as the fallback in both modes."""
    lower = re.sub(r"[^a-zA-Z0-9\s]", " ", user_input.lower()).strip()
    tokens = set(lower.split())

    has_greeting = (
        any(ph in lower for ph in GREETING_PHRASES)
        or bool(GREETING_PATTERN.match(lower))
        or bool(tokens & GREETING_TOKENS)
    )
    has_device_action = any(bool(re.search(ph, lower)) for ph in DEVICE_ACTION_PHRASES)
    has_service = any(
        (bool(re.search(ph, lower)) if any(c in ph for c in ".*()")else ph in lower)
        for ph in SERVICE_PHRASES
    )
    has_time      = bool(TIME_PATTERN.search(lower))
    has_condition = bool(CONDITION_PATTERN.search(lower))
    has_device_token = bool(tokens & DEVICE_TOKENS)
    has_oos = any(ph in lower for ph in OOS_PHRASES) or bool(GENERIC_OOS_STARTERS.match(lower))
    is_greeting_only = has_greeting and not has_device_action and not has_service

    intents = set()

    if (has_time or has_condition) and (has_device_action or has_device_token or has_service):
        intents.add("automations")
    if has_service:
        intents.add("service_request")
    if has_device_action and not has_time and not has_condition:
        intents.add("device_control")
    if is_greeting_only:
        intents.add("greetings")
    if has_oos:
        intents.add("out_of_scope")

    if not intents:
        return frozenset(["out_of_scope"])

    if len(intents) > 1 and "out_of_scope" in intents:
        intents.discard("out_of_scope")

    return frozenset(intents)


# ── ML classifier wrappers ────────────────────────────────────────────────
def _load_ml_models():
    """Load both ML models and the shared encoder. Returns (encoder, mixed_clf, mixed_mlb, multi_clf, multi_mlb)."""
    try:
        import joblib
        from sentence_transformers import SentenceTransformer

        encoder    = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        mixed_clf  = mixed_mlb  = None
        multi_clf  = multi_mlb  = None

        # Mixed-intent model
        mixed_pkl = os.path.join(os.path.dirname(__file__), "tier2_classifier.pkl")
        if os.path.exists(mixed_pkl):
            saved = joblib.load(mixed_pkl)
            if isinstance(saved, dict):
                mixed_clf = saved.get("classifier")
                mixed_mlb = saved.get("mlb")

        # Multi-intent model
        multi_pkl = os.path.join(os.path.dirname(__file__), "tier2_multi_intent_classifier.pkl")
        if os.path.exists(multi_pkl):
            saved = joblib.load(multi_pkl)
            if isinstance(saved, dict):
                multi_clf = saved.get("classifier")
                multi_mlb = saved.get("mlb")

        return encoder, mixed_clf, mixed_mlb, multi_clf, multi_mlb
    except Exception as e:
        print(f"  [WARN] Could not load ML models: {e}\n  Falling back to rule-based classifier.")
        return None, None, None, None, None


def _normalize_intent(intent: str) -> str:
    alias = {
        "greeting": "greetings", "greetings": "greetings",
        "service": "service_request", "service_request": "service_request",
        "shopping": "service_request", "queries": "service_request",
        "guardrail": "out_of_scope", "unsafe": "out_of_scope",
        "automation": "automations", "automations": "automations",
        "device_control": "device_control", "out_of_scope": "out_of_scope",
    }
    n = (intent or "").strip().lower()
    return alias.get(n, "out_of_scope")


def _apply_priority_rules(intents: list, user_input: str = "") -> list:
    """
    Multi-intent priority rules (mirrors agent.py _apply_multi_intent_priority_rules).
    Rules 8 and 9 use STRONG_TIME_PATTERN to distinguish genuine automations from
    false-positive automations triggered by irrelevant time-like preambles.
    """
    intent_set   = set(intents)
    if len(intent_set) <= 1:
        return list(intent_set) if intent_set else ["out_of_scope"]

    real_intents = intent_set - {"out_of_scope"}
    has_time     = bool(STRONG_TIME_PATTERN.search(user_input)) if user_input else False

    # Rule 1: device_control + out_of_scope -> device_control
    if "device_control" in real_intents and "out_of_scope" in intent_set and "service_request" not in real_intents:
        return ["device_control"]
    # Rule 2: service_request + out_of_scope -> service_request
    if ("service_request" in real_intents and "out_of_scope" in intent_set
            and "device_control" not in real_intents and "automations" not in real_intents):
        return ["service_request"]
    # Rule 3: greetings + out_of_scope -> out_of_scope
    if "greetings" in real_intents and "out_of_scope" in intent_set and len(real_intents) == 1:
        return ["out_of_scope"]
    # Rule 4: greetings + service_request -> service_request
    if "greetings" in real_intents and "service_request" in real_intents:
        return sorted(list(real_intents - {"greetings"}))
    # Rule 7: automations + out_of_scope -> automations
    if "automations" in real_intents and "out_of_scope" in intent_set and "service_request" not in real_intents:
        return ["automations"]
    # Rule 8: 3-way tie automations+device_control+service -> time signal decides
    if "automations" in real_intents and "device_control" in real_intents and "service_request" in real_intents:
        return ["automations", "service_request"] if has_time else ["device_control", "service_request"]
    # Rule 9: automations+service but no time signal -> device_control+service
    if ("automations" in real_intents and "service_request" in real_intents
            and "device_control" not in real_intents and not has_time):
        return ["device_control", "service_request"]
    # General: drop out_of_scope if real intents exist
    if real_intents and "out_of_scope" in intent_set:
        return sorted(list(real_intents))
    return sorted(list(intent_set))


def classify_mixed(user_input: str, encoder, clf, mlb) -> frozenset:
    """
    MIXED-INTENT classification: return ONE best label.
    Falls back to rule engine if model not available.
    """
    if encoder is None or clf is None or mlb is None:
        # Rule-based fallback: pick the single dominant intent
        raw = _rule_classify(user_input)
        if len(raw) == 1:
            return raw
        # For mixed mode: if multiple rules fire, pick by priority order
        priority = ["automations", "service_request", "device_control", "greetings", "out_of_scope"]
        for p in priority:
            if p in raw:
                return frozenset([p])
        return frozenset(["out_of_scope"])

    try:
        embedding = encoder.encode([user_input], normalize_embeddings=True)
        probas    = clf.predict_proba(embedding)[0]
        best_idx  = int(probas.argmax())
        if float(probas[best_idx]) < 0.45:
            return classify_mixed(user_input, None, None, None)
        intent = _normalize_intent(mlb.classes_[best_idx])
        return frozenset([intent])
    except Exception:
        return classify_mixed(user_input, None, None, None)


def classify_multi_intent(user_input: str, encoder, clf, mlb) -> frozenset:
    """
    MULTI-INTENT classification: return one or two labels with priority rules applied.
    Falls back to rule engine if model not available.
    """
    if encoder is None or clf is None or mlb is None:
        raw = _rule_classify(user_input)
        return frozenset(_apply_priority_rules(list(raw)))

    try:
        embedding = encoder.encode([user_input], normalize_embeddings=True)
        probas    = clf.predict_proba(embedding)[0]
        MULTI_THRESHOLD = 0.45  # matches agent.py — prevents false automations on weak signals
        passing = [i for i, p in enumerate(probas) if p >= MULTI_THRESHOLD]
        if not passing:
            raw = _rule_classify(user_input)
            return frozenset(_apply_priority_rules(list(raw)))

        raw_intents = []
        for idx in passing:
            canonical = _normalize_intent(mlb.classes_[idx])
            if canonical not in raw_intents:
                raw_intents.append(canonical)

        final = _apply_priority_rules(raw_intents, user_input=user_input)
        return frozenset(final)
    except Exception:
        raw = _rule_classify(user_input)
        return frozenset(_apply_priority_rules(list(raw), user_input=user_input))


# ── Load datasets ─────────────────────────────────────────────────────────
def load_rows(paths):
    rows = []
    for path in paths:
        if not os.path.exists(path):
            print(f"  [SKIP] Not found: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                label_raw = row.get("actual_label", row.get("label", ""))
                if row.get("input_text"):
                    rows.append({
                        "input_text": row["input_text"],
                        "actual_labels": normalize_labels(label_raw),
                    })
    return rows


# ── Evaluation helper ─────────────────────────────────────────────────────
def evaluate(rows, classify_fn, mode_name: str, show_mistakes: int = 20):
    correct_exact   = 0
    correct_partial = 0
    total           = len(rows)
    wrong_samples   = []

    for r in rows:
        actual    = r["actual_labels"]
        predicted = classify_fn(r["input_text"])

        if actual == predicted:
            correct_exact   += 1
            correct_partial += 1
        elif predicted & actual:          # at least one intent correct
            correct_partial += 1
            wrong_samples.append((r["input_text"], predicted, actual))
        else:
            wrong_samples.append((r["input_text"], predicted, actual))

    print(f"\n{'='*65}")
    print(f"  {mode_name.upper()} RESULTS  ({total} samples)")
    print(f"{'='*65}")
    print(f"  Exact-Match Accuracy : {correct_exact}/{total}  "
          f"= {correct_exact/total:.4f}  ({correct_exact/total*100:.2f}%)")
    print(f"  Partial Accuracy     : {correct_partial}/{total}  "
          f"= {correct_partial/total:.4f}  ({correct_partial/total*100:.2f}%)  "
          f"(at least one intent correct)")

    if wrong_samples:
        print(f"\n  --- Sample mistakes (first {show_mistakes}) ---")
        for text, pred, actual in wrong_samples[:show_mistakes]:
            print(f"  Input  : {text}")
            print(f"  Pred   : {sorted(pred)}")
            print(f"  Actual : {sorted(actual)}")
            print()


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("Loading ML models...")
    encoder, mixed_clf, mixed_mlb, multi_clf, multi_mlb = _load_ml_models()

    mixed_available = mixed_clf is not None and mixed_mlb is not None
    multi_available = multi_clf is not None and multi_mlb is not None

    print(f"  Mixed-intent model (tier2_classifier.pkl)           : "
          f"{'LOADED' if mixed_available else 'NOT FOUND — using rule engine'}")
    print(f"  Multi-intent model (tier2_multi_intent_classifier.pkl): "
          f"{'LOADED' if multi_available else 'NOT FOUND — run train_multi_intent_classifier.py'}")

    # ── 1. MIXED-INTENT evaluation (on single-label datasets only) ──────
    print("\n\nLoading MIXED-INTENT datasets...")
    mixed_rows = load_rows(MIXED_DATASET_PATHS)
    print(f"  Total rows: {len(mixed_rows)}")

    if mixed_rows:
        def f_mixed(text):
            return classify_mixed(
                text,
                encoder if mixed_available else None,
                mixed_clf,
                mixed_mlb,
            )
        evaluate(mixed_rows, f_mixed, "MIXED-INTENT (single label per query)")
    else:
        print("  [WARN] No mixed-intent dataset rows found.")

    # ── 2. MULTI-INTENT evaluation (on havells multi-intent dataset) ────
    print("\n\nLoading MULTI-INTENT dataset...")
    multi_rows = load_rows([MULTI_INTENT_DATASET_PATH])
    print(f"  Total rows: {len(multi_rows)}")

    if multi_rows:
        def f_multi(text):
            return classify_multi_intent(
                text,
                encoder if multi_available else None,
                multi_clf,
                multi_mlb,
            )
        evaluate(multi_rows, f_multi, "MULTI-INTENT (multiple labels per query)")
    else:
        print("  [WARN] Multi-intent dataset not found.")


if __name__ == "__main__":
    main()
