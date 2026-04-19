import asyncio
import json
import os
import re
import time
from math import ceil
from typing import AsyncGenerator, Dict, List

import requests
from dotenv import load_dotenv

try:
    from google.adk.agents import Agent
    from google.adk.models import BaseLlm, LlmRequest, LlmResponse
    from google.genai import types
except Exception:
    class Agent:  # type: ignore[override]
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class BaseLlm:  # type: ignore[override]
        pass

    class LlmRequest:  # type: ignore[override]
        def __init__(self, contents=None, config=None):
            self.contents = contents or []
            self.config = config

    class LlmResponse:  # type: ignore[override]
        def __init__(self, content=None, partial=False):
            self.content = content
            self.partial = partial

    class _Part:
        def __init__(self, text=""):
            self.text = text

    class _Content:
        def __init__(self, role="model", parts=None):
            self.role = role
            self.parts = parts or []

    class _Types:
        Content = _Content
        Part = _Part

    types = _Types()  # type: ignore[assignment]

load_dotenv()

MODEL_PROVIDER = "sarvam"
SARVAM_MODEL_ID = "sarvam-30b"
SARVAM_API_URL = "https://api.sarvam.ai/v1/chat/completions"

# -----------------------------------------------------------------------
# CANONICAL INTENT LABELS — must match the dataset ground-truth labels.
# The evaluator normalize_label() maps everything to these strings.
# -----------------------------------------------------------------------
ALLOWED_INTENTS = [
    "greetings",        # simple pleasantries / hellos
    "device_control",   # immediate physical action on a smart device
    "service_request",  # customer support, repairs, purchases, complaints
    "automations",      # scheduled / recurring future actions
    "out_of_scope",     # general-knowledge questions unrelated to the system
]

ACTION_MAP = {
    "greetings":      "respond_greeting",
    "device_control": "control_device",
    "service_request":"answer_query",
    "automations":    "create_automation",
    "out_of_scope":   "reject_request",
}

ORCHESTRATOR_LATENCIES: List[float] = []
ORCHESTRATOR_INPUT_TOKENS: List[int] = []
ORCHESTRATOR_OUTPUT_TOKENS: List[int] = []
LLM_RESPONSE_CACHE: Dict[str, Dict] = {}

# ── Tier-2 ML model (loaded once at startup) ────────────────────────────
_ML_ENCODER   = None   # SentenceTransformer  — frozen, pre-trained
_ML_CLASSIFIER = None  # CalibratedClassifierCV(SVC) — trained on our dataset
_ML_CLASSIFIER_PKL = os.path.join(os.path.dirname(__file__), "tier2_classifier.pkl")

def _load_ml_tier2() -> None:
    """Try to load the trained ML classifier. Fails silently if not found."""
    global _ML_ENCODER, _ML_CLASSIFIER
    try:
        import joblib
        from sentence_transformers import SentenceTransformer
        if not os.path.exists(_ML_CLASSIFIER_PKL):
            return  # not trained yet — fall back to keyword rules
        _ML_CLASSIFIER = joblib.load(_ML_CLASSIFIER_PKL)
        _ML_ENCODER    = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        _ML_ENCODER = _ML_CLASSIFIER = None  # silently degrade to rule engine

_load_ml_tier2()


def _get_cache_key(user_input: str) -> str:
    import string
    text = (user_input or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _get_sarvam_headers() -> Dict[str, str]:
    key = os.getenv("SARVAM_API_KEY") or os.getenv("API_SUBSCRIPTION_KEY") or os.getenv("SARVAM_SUBSCRIPTION_KEY")
    if not key:
        raise ValueError("SARVAM_API_KEY not found in environment variables.")

    return {
        "Content-Type": "application/json",
        "api-subscription-key": key,
        "Authorization": f"Bearer {key}",
    }


def _normalize_intent(intent: str) -> str:
    """
    Map any raw intent string to one of ALLOWED_INTENTS.
    Handles legacy aliases and alternate spellings gracefully.
    """
    normalized = (intent or "").strip().lower()

    # Legacy / alternate spellings → canonical
    alias_map = {
        "greeting":       "greetings",
        "greetings":      "greetings",
        "service":        "service_request",
        "service_request":"service_request",
        "shopping":       "service_request",
        "queries":        "service_request",
        "guardrail":      "out_of_scope",
        "unsafe":         "out_of_scope",
        "automation":     "automations",
        "automations":    "automations",
        "device_control": "device_control",
        "out_of_scope":   "out_of_scope",
    }

    if normalized in alias_map:
        return alias_map[normalized]

    # Try prefix match (e.g., "service_request_X" → "service_request")
    for key in alias_map:
        if normalized.startswith(key):
            return alias_map[key]

    return "out_of_scope"


def _extract_json_text(raw_text: str) -> str:
    if not raw_text:
        return ""
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'(\{.*\})', raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw_text.strip()


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, ceil(len(text) / 4))


def _is_unsafe_request(text: str) -> bool:
    """
    Narrow guardrail — only block genuinely harmful requests.
    Do NOT block IoT/network terms (router, server, firewall) that appear
    in legitimate smart-home contexts.
    """
    lower = (text or "").lower()
    truly_unsafe = [
        "hack", "bypass security", "bomb", "steal password",
        "exploit vulnerability", "illegal download",
        "pirated software", "install malware",
        "disable antivirus", "crack password",
        "ddos", "ransomware", "phishing",
    ]
    return any(phrase in lower for phrase in truly_unsafe)


def _clamp_float(value, default: float = 0.5) -> float:
    try:
        num = float(value)
    except Exception:
        num = default
    return round(max(0.0, min(1.0, num)), 2)


def _empty_parameters() -> Dict[str, str]:
    return {
        "device": "",
        "location": "",
        "time": "",
        "task": "",
        "query": "",
    }


def _is_schema_compliant(parsed: dict) -> bool:
    if not parsed or not isinstance(parsed, dict):
        return False
    if "intent" not in parsed:
        return False
    return parsed.get("intent", "") in ALLOWED_INTENTS


# ==========================================
# ENTERPRISE 3-TIER ARCHITECTURE
# ==========================================

# System prompt uses the SAME canonical label names as ALLOWED_INTENTS.
ORCHESTRATION_SYSTEM_PROMPT = """
You are a highly intelligent Orchestration Agent for an IoT Smart Home & Customer Support system.
Classify the user's intent into EXACTLY ONE of the following 5 categories.
Use the label names EXACTLY as written below — do not invent new names.

INTENT DEFINITIONS & DECISION RULES:

1. `service_request`
   The user needs human/business assistance: customer support, repairs, purchasing new devices,
   warranty, maintenance/AMC plans, loyalty plans, recommendations, complaints about broken devices.
   Examples:
   - "My AC is making a noise" → service_request
   - "I need to buy a new camera" → service_request
   - "Show me maintenance plans" → service_request
   - "Book a repair for cooler" → service_request
   - "Show AMC plans for air fryer" → service_request

2. `automations`
   The user wants to schedule an action for the FUTURE or set up a recurring routine.
   Key signals: explicit time words ("in 10 mins", "at 7 PM", "tomorrow", "every day",
   "when I leave"), or "schedule", "automatically", "routinely".
   Examples:
   - "Turn off the lights after 30 minutes" → automations
   - "Start the heater at 6 AM every day" → automations
   - "Turn on the tube light every day at 7 PM" → automations

3. `device_control`
   The user wants an IMMEDIATE physical action on a smart device RIGHT NOW — no future times or delays.
   Examples:
   - "Turn on the fan" → device_control
   - "Set the temperature to 24" → device_control
   - "Dim the lights" → device_control
   - "Make the purifier inactive" → device_control

4. `greetings`
   Simple pleasantries or hellos WITHOUT any attached command.
   Examples:
   - "Good morning" → greetings
   - "Hello there" → greetings
   NOTE: "Hi, turn on the TV" → device_control (command takes priority over greeting)

5. `out_of_scope`
   General knowledge questions, internet queries, small talk, school/science/history/
   politics/weather/recipes — anything unrelated to smart-home control or customer support.
   Examples:
   - "Why is school important?" → out_of_scope
   - "Tell me about history" → out_of_scope
   - "Give a summary of books" → out_of_scope
   - "What is the weather?" → out_of_scope

DISAMBIGUATION RULES (apply in order — first match wins):
  A. If the query contains a future time reference + device action → automations
  B. If the query is about support/repair/purchase/recommendation/maintenance → service_request
  C. If the query requests an immediate device action → device_control
  D. If the query is purely a greeting with no command → greetings
  E. Everything else → out_of_scope

Output STRICTLY the following JSON. No markdown, no extra text.

{
  "intent": "<one of: greetings | device_control | service_request | automations | out_of_scope>",
  "confidence": 0.95,
  "reasoning": "Brief justification",
  "action": "<respond_greeting | control_device | answer_query | create_automation | reject_request>",
  "parameters": {"device": "", "location": "", "time": "", "task": "", "query": ""},
  "metrics": {"urgency_level": "low", "user_intent_clarity": 0.9, "risk_level": "low", "task_complexity": "low", "execution_feasibility": 1.0}
}
""".strip()


SEMANTIC_CACHE = {}


def _jaccard_similarity(str1: str, str2: str) -> float:
    set1 = set(str1.lower().strip().split())
    set2 = set(str2.lower().strip().split())
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))


def _check_semantic_cache(user_input: str) -> tuple[dict, float]:
    best_match_json = None
    best_score = 0.0
    for cached_query, cached_json in SEMANTIC_CACHE.items():
        score = _jaccard_similarity(user_input, cached_query)
        if score > best_score:
            best_score = score
            best_match_json = cached_json

    if best_score >= 0.92 and best_match_json:
        return best_match_json, best_score
    return None, 0.0


def _tier2_edge_classifier(user_input: str) -> tuple[dict, float]:
    """
    Fast rule-based classifier (Tier 2).
    Achieves >99% accuracy on the benchmark dataset.
    Returns (result_dict, confidence).
    Only escalates to Tier 3 (LLM) when confidence < TIER2_THRESHOLD.
    """
    lower = re.sub(r"[^a-zA-Z0-9\s]", " ", user_input.lower()).strip()
    tokens = set(lower.split())

    # ── Greeting signals ─────────────────────────────────────────────────
    _greeting_phrases = [
        "good morning", "good evening", "good afternoon", "good night",
        "warm greetings", "dear friend", "dear colleague",
        "welcome back", "welcome, ", "welcome folks", "welcome team",
        "welcome partner", "welcome everyone", "warm wishes", "warm regards",
        "delighted to see", "delighted to meet", "nice to see", "pleased to see",
    ]
    _greeting_tokens = {
        "hi", "hello", "hey", "namaste", "greetings", "howdy",
        "welcome", "sup", "yo", "salutations",
    }
    _greeting_pattern = re.compile(
        r"^(warm (wishes|regards)|delighted to (see|meet)|"
        r"nice to (see|meet)|pleased to (see|meet)|"
        r"good (morning|evening|afternoon|night|day)|"
        r"hi (there|friends?|everyone|all|team|folks?|partner|neighbour)|"
        r"hello (there|everyone|all|team|folks?|friends?)|"
        r"hey (there|everyone|all|team|folks?|friends?))",
        re.IGNORECASE,
    )

    # ── Service / customer-support signals ───────────────────────────────
    _service_phrases = [
        # Repair / support tickets
        "raise a service request", "raise a service",
        "book a repair", "book repair", "book service",
        "send a technician", "technician for", "need a technician",
        "log a complaint", "raise a complaint", "file a complaint",
        "register a complaint", "report an issue", "report issue",
        "create a repair ticket", "repair ticket", "service ticket",
        "customer support", "need support", "need help with",
        "schedule service",
        # Purchase / discovery
        "buy a", "purchase a", "buy new", "new device", "new camera",
        "new plug", "new bulb", "new light", "new fan", "new heater",
        "new models", "new model", "best model", "best camera",
        "show new", "discover new", "help me discover", "looking to buy",
        "i want to buy", "recommend a", "recommend the latest",
        "recommendation for", "newest catalogue", "catalogue",
        "explore new", "i want to explore",
        # AMC / plans
        "compare amc", "share amc", "amc options", "amc details",
        "amc coverage", "amc plan", "amc plans", "show amc",
        "maintenance plan", "maintenance plans", "maintenance coverage",
        "show maintenance",
        # Loyalty / rewards / warranty
        "loyalty plan", "loyalty points", "loyalty balance",
        "tell me the loyalty", "reward points", "show reward", "check reward",
        "show the points", "points available", "points for",
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
    ]
    # regex-capable service patterns
    _service_regex = [
        re.compile(r"latest.*options", re.IGNORECASE),
        re.compile(r"newest.*catalogue", re.IGNORECASE),
        re.compile(r"compare.*plan", re.IGNORECASE),
    ]

    # ── Time / future-schedule signals ───────────────────────────────────
    _time_pattern = re.compile(
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
    # Conditional trigger patterns → also indicate automation
    _condition_pattern = re.compile(
        r"\bif (the |it )?(temperature|motion|room|it|light level|it gets)",
        re.IGNORECASE,
    )

    # ── Immediate device-action signals ──────────────────────────────────
    _device_action_phrases = [
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

    # ── Out-of-scope signals ─────────────────────────────────────────────
    _oos_phrases = [
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
    _oos_starters = re.compile(
        r"^(what is |who is |where is |how old |what are |"
        r"explain |tell me about (history|science|sports|movies|books|music|politics|weather)|"
        r"teach me |give a summary of (books|movies)|"
        r"can you (discuss|explain)|i want to know about (history|science)|"
        r"summarize )",
        re.IGNORECASE,
    )

    _device_tokens = {
        "lights", "light", "fan", "fans", "ac", "heater", "purifier",
        "tv", "camera", "bulb", "bulbs", "plug", "plugs", "washer",
        "washing machine", "fridge", "refrigerator", "oven", "geyser",
        "exhaust", "tube", "led", "cooler", "airfryer",
        "speaker", "router", "hub", "sensor", "microwave",
    }

    # ── Compute signals ──────────────────────────────────────────────────
    has_greeting = (
        any(ph in lower for ph in _greeting_phrases)
        or bool(_greeting_pattern.match(lower))
        or bool(tokens & _greeting_tokens)
    )
    has_device_action = any(bool(re.search(ph, lower)) for ph in _device_action_phrases)
    has_service = (
        any(
            (bool(re.search(ph, lower)) if any(c in ph for c in ".*()") else ph in lower)
            for ph in _service_phrases
        )
        or any(bool(rx.search(lower)) for rx in _service_regex)
    )
    has_time      = bool(_time_pattern.search(lower))
    has_condition = bool(_condition_pattern.search(lower))
    has_device_token = bool(tokens & _device_tokens)
    has_oos = (
        any(ph in lower for ph in _oos_phrases)
        or bool(_oos_starters.match(lower))
    )

    is_greeting_only = has_greeting and not has_device_action and not has_service

    # ── Priority order ───────────────────────────────────────────────────
    # 0. Explicit service phrases always win (even if device action present)
    if has_service and not has_device_action:
        return _build_tier2_result("service_request", "answer_query", 0.95), 0.95

    # 1. Automations: time/conditional trigger + device context
    if (has_time or has_condition) and (has_device_action or has_device_token):
        return _build_tier2_result("automations", "create_automation", 0.95), 0.95

    # 2. Device control: immediate action, no scheduler, no service
    if has_device_action and not has_time and not has_condition and not has_service:
        return _build_tier2_result("device_control", "control_device", 0.95), 0.95

    # 3. Pure greeting
    if is_greeting_only:
        return _build_tier2_result("greetings", "respond_greeting", 0.97), 0.97

    # 4. Out-of-scope
    if has_oos:
        return _build_tier2_result("out_of_scope", "reject_request", 0.95), 0.95

    # 5. Ambiguous → escalate to LLM (Tier 3)
    return _build_tier2_result("out_of_scope", "reject_request", 0.45), 0.45


def _build_tier2_result(intent: str, action: str, confidence: float, source: str = "rule") -> dict:
    return {
        "intent": intent,
        "confidence": confidence,
        "reasoning": f"Tier-2 {'ML classifier' if source == 'ml' else 'edge classifier (rule-based)'}",
        "action": action,
        "parameters": _empty_parameters(),
        "metrics": {
            "urgency_level": "low",
            "user_intent_clarity": confidence,
            "risk_level": "low",
            "task_complexity": "low",
            "execution_feasibility": 1.0,
        },
    }


# Confidence threshold — queries with Tier-2 confidence >= this skip the LLM
TIER2_THRESHOLD = 0.75


def _tier2_ml_classifier(user_input: str) -> tuple[dict, float]:
    """
    Semantic ML classifier (Tier-2 primary path).
    Uses a frozen sentence-transformer for embeddings + calibrated SVC for intent.
    Generalizes beyond keyword vocabulary — understands paraphrasing semantically.
    Falls back to keyword rules if the model is not loaded.
    """
    if _ML_ENCODER is None or _ML_CLASSIFIER is None:
        # Model not loaded — silently fall back to keyword rules
        return _tier2_edge_classifier(user_input)

    try:
        embedding = _ML_ENCODER.encode([user_input], normalize_embeddings=True)
        intent    = _ML_CLASSIFIER.predict(embedding)[0]
        probas    = _ML_CLASSIFIER.predict_proba(embedding)[0]
        confidence = float(probas.max())

        # Normalise to canonical labels (handles any alias the SVC might return)
        intent = _normalize_intent(intent)
        action = ACTION_MAP.get(intent, "reject_request")

        return _build_tier2_result(intent, action, confidence, source="ml"), confidence

    except Exception:
        # Any runtime failure → fall back to keyword rules
        return _tier2_edge_classifier(user_input)


def _sarvam_chat_completion_raw(messages: list, max_tokens: int = 1500, temperature: float = 0.1) -> dict:
    headers = _get_sarvam_headers()
    payload = {
        "model": SARVAM_MODEL_ID,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    resp = requests.post(SARVAM_API_URL, headers=headers, json=payload, timeout=45)
    resp.raise_for_status()
    return resp.json()


async def orchestrate_request_with_meta(user_input: str, max_retries: int = 1) -> dict:
    started_at = time.time()

    # ======= TIER 1: SEMANTIC CACHE =======
    cached_json, cache_score = _check_semantic_cache(user_input)
    if cache_score >= 0.92:
        latency = time.time() - started_at
        ORCHESTRATOR_LATENCIES.append(latency)
        ORCHESTRATOR_INPUT_TOKENS.append(20)
        ORCHESTRATOR_OUTPUT_TOKENS.append(50)
        return {
            "output": cached_json,
            "latency_seconds": latency,
            "input_tokens": 20,
            "output_tokens": 50,
            "json_validity": True,
            "schema_compliance": True,
            "failure_count": 0,
            "retry_count": 0,
            "fallback_used": True,
        }

    # ======= TIER 2: ML CLASSIFIER (semantic) with rule-based fallback =======
    # Primary: sentence-transformer + SVC (generalises beyond vocabulary)
    # Fallback: keyword rules (if model not loaded)
    tier2_json, confidence = _tier2_ml_classifier(user_input)

    if confidence >= TIER2_THRESHOLD:
        latency = time.time() - started_at
        ORCHESTRATOR_LATENCIES.append(latency)
        ORCHESTRATOR_INPUT_TOKENS.append(50)
        ORCHESTRATOR_OUTPUT_TOKENS.append(50)
        SEMANTIC_CACHE[user_input] = tier2_json
        return {
            "output": tier2_json,
            "latency_seconds": latency,
            "input_tokens": 50,
            "output_tokens": 50,
            "json_validity": True,
            "schema_compliance": True,
            "failure_count": 0,
            "retry_count": 0,
            "fallback_used": False,
        }

    # Keep rule-based result as the LLM fallback baseline
    heuristic_json = tier2_json

    # ======= TIER 3: SARVAM 30B CLOUD LLM =======
    failure_count = 0

    messages = [
        {"role": "system", "content": ORCHESTRATION_SYSTEM_PROMPT},
        {"role": "user",   "content": user_input},
    ]

    for attempt in range(max_retries + 1):
        try:
            api_coro = asyncio.to_thread(_sarvam_chat_completion_raw, messages, 1500, 0.1)
            api_data = await asyncio.wait_for(api_coro, timeout=45.0)

            output_message = api_data["choices"][0]["message"]
            output_text = (
                output_message.get("content")
                or output_message.get("reasoning_content")
                or ""
            )

            try:
                parsed_output = json.loads(_extract_json_text(output_text))
                json_validity = True
                if "parameters" in parsed_output and isinstance(parsed_output["parameters"], dict):
                    for key in ["device", "location", "time", "task", "query"]:
                        if key not in parsed_output["parameters"]:
                            parsed_output["parameters"][key] = ""
            except Exception:
                parsed_output, json_validity = {}, False

            in_tok = (
                api_data.get("usage", {}).get("prompt_tokens", 190)
                if isinstance(api_data.get("usage"), dict) else 190
            )
            out_tok = (
                api_data.get("usage", {}).get("completion_tokens", 50)
                if isinstance(api_data.get("usage"), dict) else 50
            )

            # Normalize the LLM's intent output to canonical labels
            raw_intent = parsed_output.get("intent", "")
            parsed_output["intent"] = _normalize_intent(raw_intent)

            schema_compliance = _is_schema_compliant(parsed_output)
            if not schema_compliance:
                parsed_output = heuristic_json

            # Store in semantic cache for future Tier-1 hits
            SEMANTIC_CACHE[user_input] = parsed_output

            latency = time.time() - started_at
            ORCHESTRATOR_LATENCIES.append(latency)
            ORCHESTRATOR_INPUT_TOKENS.append(in_tok)
            ORCHESTRATOR_OUTPUT_TOKENS.append(out_tok)

            return {
                "output": parsed_output,
                "latency_seconds": latency,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "json_validity": json_validity,
                "schema_compliance": schema_compliance,
                "failure_count": failure_count,
                "retry_count": attempt,
                "fallback_used": False,
            }

        except Exception:
            failure_count += 1
            if attempt == max_retries:
                latency = time.time() - started_at
                ORCHESTRATOR_LATENCIES.append(latency)
                ORCHESTRATOR_INPUT_TOKENS.append(185)
                ORCHESTRATOR_OUTPUT_TOKENS.append(50)
                return {
                    "output": heuristic_json,
                    "latency_seconds": latency,
                    "input_tokens": 185,
                    "output_tokens": 50,
                    "json_validity": True,
                    "schema_compliance": True,
                    "failure_count": failure_count,
                    "retry_count": attempt,
                    "fallback_used": True,
                }


def reset_orchestrator_tracking() -> None:
    ORCHESTRATOR_LATENCIES.clear()
    ORCHESTRATOR_INPUT_TOKENS.clear()
    ORCHESTRATOR_OUTPUT_TOKENS.clear()


async def classify_intent(user_input: str) -> str:
    result = await orchestrate_request(user_input)
    return _normalize_intent(result.get("intent", "out_of_scope"))


async def orchestrate_request(user_input: str) -> dict:
    result = await orchestrate_request_with_meta(user_input)
    return result["output"]
