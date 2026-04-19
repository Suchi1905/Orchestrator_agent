"""
Quick offline accuracy check for the improved Tier-2 edge classifier.
Run:  python test_accuracy.py
"""
import csv
import re

DATASET_PATH = r"c:\Users\hp\Downloads\final_cleaned_dataset.csv"

# ── Label normalisation ────────────────────────────────────────────────────
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

def normalize_label(label: str) -> str:
    n = (label or "").strip().lower()
    if n in ALIAS_MAP:
        return ALIAS_MAP[n]
    for k, v in ALIAS_MAP.items():
        if n.startswith(k):
            return v
    return "out_of_scope"


# ── Signal sets ──────────────────────────────────────────────────────────
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
    # Repair / support tickets
    "raise a service request", "raise a service",
    "book a repair", "book repair", "book service",
    "send a technician", "technician for", "need a technician",
    "log a complaint", "raise a complaint", "file a complaint",
    "register a complaint", "report an issue", "report issue",
    "create a repair ticket", "repair ticket", "service ticket",
    "customer support", "need support", "need help with",
    "schedule service",  # 'schedule service for X' = service, not automation
    # Purchase / discovery
    "buy a", "purchase a", "buy new", "new device", "new camera",
    "new plug", "new bulb", "new light", "new fan", "new heater",
    "new models", "new model", "best model", "best camera",
    "show new", "discover new", "help me discover", "looking to buy",
    "i want to buy", "recommend a", "recommend the latest",
    "recommendation for", "latest.*options", "newest.*catalogue",
    "newest catalogue", "catalogue",
    # AMC / plans
    "compare amc", "share amc", "amc options", "amc details",
    "amc coverage", "amc plan", "amc plans", "show amc",
    "maintenance plan", "maintenance plans", "maintenance coverage",
    "show maintenance", "compare.*plan",
    # Loyalty / rewards / warranty
    "loyalty plan", "loyalty points", "loyalty balance", "tell me the loyalty",
    "reward points", "show reward", "check reward",
    "warranty for", "warranty plan", "warranty registration",
    "complete warranty",
    # Account / registration
    "register my", "enroll my", "link my",
    "add to my account", "add to my profile",
    "update my profile", "manage my account", "my account",
    "my profile", "subscription",
    # Device issues
    "not working", "broken", "making noise",
    "installation", "install the", "demo of",
    # Points / loyalty
    "show the points", "points available", "points for",
    # Product exploration
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

GENERIC_OOS_STARTERS = re.compile(
    r"^(what is |who is |where is |how old |what are |"
    r"explain |tell me about (history|science|sports|movies|books|music|politics|weather)|"
    r"teach me |give a summary of (books|movies)|"
    r"can you (discuss|explain)|i want to know about (history|science)|"
    r"summarize )",
    re.IGNORECASE,
)


def classify(user_input: str) -> str:
    lower = re.sub(r"[^a-zA-Z0-9\s]", " ", user_input.lower()).strip()
    tokens = set(lower.split())

    has_greeting = (
        any(ph in lower for ph in GREETING_PHRASES)
        or bool(GREETING_PATTERN.match(lower))
        or bool(tokens & GREETING_TOKENS)
    )
    has_device_action = any(bool(re.search(ph, lower)) for ph in DEVICE_ACTION_PHRASES)
    has_service = any(
        (bool(re.search(ph, lower)) if any(c in ph for c in ".*()") else ph in lower)
        for ph in SERVICE_PHRASES
    )
    has_time      = bool(TIME_PATTERN.search(lower))
    has_condition = bool(CONDITION_PATTERN.search(lower))
    has_device_token = bool(tokens & DEVICE_TOKENS)
    has_oos = any(ph in lower for ph in OOS_PHRASES) or bool(GENERIC_OOS_STARTERS.match(lower))

    is_greeting_only = has_greeting and not has_device_action and not has_service

    # ── Priority order ───────────────────────────────────────────────────
    # 0. Service (override — explicit service phrases outrank device actions)
    if has_service and not has_device_action:
        return "service_request"

    # 1. Automations: time/conditional trigger + device context
    if (has_time or has_condition) and (has_device_action or has_device_token):
        return "automations"

    # 2. Device control: immediate action, no scheduler, no service
    if has_device_action and not has_time and not has_condition and not has_service:
        return "device_control"

    # 3. Pure greeting
    if is_greeting_only:
        return "greetings"

    # 4. Out of scope
    if has_oos:
        return "out_of_scope"

    return "out_of_scope"


# ── Evaluation ─────────────────────────────────────────────────────────────
def main():
    with open(DATASET_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    CLASSES = ["greetings", "device_control", "service_request", "automations", "out_of_scope"]
    per_class = {c: {"correct": 0, "total": 0, "wrong": []} for c in CLASSES}
    correct_total = 0

    for r in rows:
        actual    = normalize_label(r["actual_label"])
        predicted = classify(r["input_text"])

        if actual not in per_class:
            per_class[actual] = {"correct": 0, "total": 0, "wrong": []}
        per_class[actual]["total"] += 1

        if predicted == actual:
            correct_total += 1
            per_class[actual]["correct"] += 1
        else:
            per_class[actual]["wrong"].append((r["input_text"], predicted))

    total = len(rows)
    print(f"\nOverall accuracy: {correct_total}/{total} = {correct_total/total:.4f}  ({correct_total/total*100:.2f}%)\n")
    print("Per-class accuracy:")
    for cls in CLASSES:
        v = per_class[cls]
        t = v["total"] or 1
        pct = v["correct"] / t * 100
        bar = "#" * int(pct / 5)
        print(f"  {cls:<20s}: {v['correct']:>3}/{t}  {pct:5.1f}%  {bar}")

    print("\n--- Remaining mistakes (first 15 per class) ---")
    for cls in CLASSES:
        wrongs = per_class[cls]["wrong"]
        if wrongs:
            print(f"\n  [{cls}] — {len(wrongs)} errors:")
            for text, pred in wrongs[:15]:
                print(f"    -> [{pred:<20s}]  {text[:72]}")


if __name__ == "__main__":
    main()
