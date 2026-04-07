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

ALLOWED_INTENTS = ["greeting", "device_control", "service", "automations", "out_of_scope"]
ACTION_MAP = {
    "greeting": "respond_greeting",
    "device_control": "control_device",
    "service": "answer_query",
    "automations": "create_automation",
    "out_of_scope": "reject_request",
}

ORCHESTRATOR_LATENCIES: List[float] = []
ORCHESTRATOR_INPUT_TOKENS: List[int] = []
ORCHESTRATOR_OUTPUT_TOKENS: List[int] = []


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
    normalized = (intent or "").strip().lower()
    if normalized == "shopping":
        return "service"
    if normalized == "queries":
        return "service"
    if normalized == "guardrail":
        return "out_of_scope"
    if normalized in ALLOWED_INTENTS:
        return normalized
    return "out_of_scope"


def _extract_json_text(raw_text: str) -> str:
    if not raw_text:
        return ""
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # Lightweight approximation when provider usage is absent.
    return max(1, ceil(len(text) / 4))


def _is_unsafe_request(text: str) -> bool:
    lower = (text or "").lower()
    unsafe_keywords = [
        "hack",
        "bypass",
        "bomb",
        "steal",
        "password",
        "exploit",
        "illegal",
        "pirated",
        "spy",
        "malware",
        "firewall",
        "router",
        "server",
        "disable antivirus",
        "disable security",
        "crack",
    ]
    return any(keyword in lower for keyword in unsafe_keywords)


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


def _heuristic_fallback(user_input: str) -> Dict:
    text = user_input.strip()
    lower = text.lower()

    unsafe_keywords = [
        "hack", "bypass", "bomb", "steal", "password", "exploit", "illegal", "pirated", "spy", "malware"
    ]
    greeting_keywords = ["hi", "hello", "hey", "good morning", "good evening", "yo"]
    automation_keywords = ["remind", "schedule", "every day", "daily", "tomorrow", "at ", "workflow", "automate"]
    device_keywords = ["turn on", "turn off", "switch", "set", "ac", "fan", "lights", "cooler", "washing machine"]

    has_automation_signal = any(k in lower for k in automation_keywords)
    has_device_signal = any(k in lower for k in device_keywords)
    has_greeting_signal = any(k in lower for k in greeting_keywords)

    if any(k in lower for k in unsafe_keywords):
        intent = "out_of_scope"
    elif has_automation_signal:
        intent = "automations"
    elif has_device_signal:
        intent = "device_control"
    elif has_greeting_signal and len(lower.split()) <= 6:
        intent = "greeting"
    else:
        intent = "service"

    params = _empty_parameters()
    params["query"] = text if intent == "service" else ""

    if intent == "device_control":
        for device in ["ac", "fan", "lights", "cooler", "washing machine"]:
            if device in lower:
                params["device"] = device
                break
        room_match = re.search(r"(bedroom|living room|kitchen|hall|office)", lower)
        if room_match:
            params["location"] = room_match.group(1)

    if intent == "automations":
        params["task"] = text
        time_match = re.search(r"\b(\d{1,2}(:\d{2})?\s?(am|pm)?)\b", lower)
        if time_match:
            params["time"] = time_match.group(1)

    urgency = "low"
    if any(k in lower for k in ["now", "immediately", "urgent", "asap"]):
        urgency = "high"
    elif intent in {"device_control", "automations"}:
        urgency = "medium"

    risk = "high" if intent == "out_of_scope" and any(k in lower for k in unsafe_keywords) else "low"
    complexity = "medium" if " and " in lower else "low"
    feasibility = 0.2 if risk == "high" else (0.95 if intent in {"service", "greeting"} else 0.9)
    clarity = 0.95 if len(lower.split()) >= 3 else 0.8
    confidence = 0.95 if intent in {"greeting", "device_control", "out_of_scope"} else 0.85

    return {
        "intent": intent,
        "confidence": confidence,
        "action": ACTION_MAP[intent],
        "parameters": params,
        "metrics": {
            "urgency_level": urgency,
            "user_intent_clarity": clarity,
            "risk_level": risk,
            "task_complexity": complexity,
            "execution_feasibility": feasibility,
        },
    }


def _sanitize_output(raw_obj: Dict, user_input: str) -> Dict:
    obj = raw_obj if isinstance(raw_obj, dict) else {}

    intent = _normalize_intent(obj.get("intent", "out_of_scope"))
    confidence = _clamp_float(obj.get("confidence", 0.6), 0.6)
    action = ACTION_MAP[intent]

    params = _empty_parameters()
    incoming_params = obj.get("parameters", {}) if isinstance(obj.get("parameters"), dict) else {}
    for key in params:
        value = incoming_params.get(key, "")
        params[key] = "" if value is None else str(value)

    metrics_in = obj.get("metrics", {}) if isinstance(obj.get("metrics"), dict) else {}
    urgency = str(metrics_in.get("urgency_level", "low")).lower()
    if urgency not in {"low", "medium", "high"}:
        urgency = "low"

    risk = str(metrics_in.get("risk_level", "low")).lower()
    if risk not in {"low", "medium", "high"}:
        risk = "low"

    complexity = str(metrics_in.get("task_complexity", "low")).lower()
    if complexity not in {"low", "medium", "high"}:
        complexity = "low"

    clarity = _clamp_float(metrics_in.get("user_intent_clarity", 0.6), 0.6)
    feasibility = _clamp_float(metrics_in.get("execution_feasibility", 0.8), 0.8)

    safe_obj = {
        "intent": intent,
        "confidence": confidence,
        "action": action,
        "parameters": params,
        "metrics": {
            "urgency_level": urgency,
            "user_intent_clarity": clarity,
            "risk_level": risk,
            "task_complexity": complexity,
            "execution_feasibility": feasibility,
        },
    }

    # Ensure required query/task fields are not lost for generic requests.
    if safe_obj["intent"] == "service" and not safe_obj["parameters"]["query"]:
        safe_obj["parameters"]["query"] = user_input.strip()
    if safe_obj["intent"] == "automations" and not safe_obj["parameters"]["task"]:
        safe_obj["parameters"]["task"] = user_input.strip()

    # Deterministic safety override to prevent unsafe prompts from slipping through.
    if _is_unsafe_request(user_input):
        safe_obj["intent"] = "out_of_scope"
        safe_obj["action"] = ACTION_MAP["out_of_scope"]
        safe_obj["confidence"] = max(safe_obj["confidence"], 0.9)
        safe_obj["parameters"] = _empty_parameters()
        safe_obj["metrics"]["risk_level"] = "high"
        safe_obj["metrics"]["execution_feasibility"] = min(safe_obj["metrics"]["execution_feasibility"], 0.3)

        if safe_obj["metrics"]["urgency_level"] == "low":
            safe_obj["metrics"]["urgency_level"] = "medium"

    return safe_obj


def _is_schema_compliant(payload: Dict) -> bool:
    if not isinstance(payload, dict):
        return False

    required_top = {"intent", "confidence", "action", "parameters", "metrics"}
    if not required_top.issubset(set(payload.keys())):
        return False

    if payload.get("intent") not in ALLOWED_INTENTS:
        return False

    expected_action = ACTION_MAP.get(payload.get("intent"))
    if payload.get("action") != expected_action:
        return False

    params = payload.get("parameters")
    if not isinstance(params, dict):
        return False
    for key in ["device", "location", "time", "task", "query"]:
        if key not in params:
            return False

    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return False

    if metrics.get("urgency_level") not in {"low", "medium", "high"}:
        return False
    if metrics.get("risk_level") not in {"low", "medium", "high"}:
        return False
    if metrics.get("task_complexity") not in {"low", "medium", "high"}:
        return False

    try:
        clarity = float(metrics.get("user_intent_clarity", -1))
        feasibility = float(metrics.get("execution_feasibility", -1))
        confidence = float(payload.get("confidence", -1))
    except Exception:
        return False

    return (
        0.0 <= clarity <= 1.0
        and 0.0 <= feasibility <= 1.0
        and 0.0 <= confidence <= 1.0
    )


def _sarvam_chat_completion_raw(messages: List[Dict[str, str]], max_tokens: int = 350, temperature: float = 0.1) -> Dict:
    payload = {
        "model": SARVAM_MODEL_ID,
        "messages": messages,
        "temperature": temperature,
        "top_p": 1,
        "max_tokens": max_tokens,
        "stream": False,
    }

    response = requests.post(
        SARVAM_API_URL,
        headers=_get_sarvam_headers(),
        json=payload,
        timeout=45,
    )
    response.raise_for_status()
    return response.json()


def _sarvam_chat_completion(messages: List[Dict[str, str]], max_tokens: int = 350, temperature: float = 0.1) -> str:
    data = _sarvam_chat_completion_raw(messages, max_tokens=max_tokens, temperature=temperature)
    return data["choices"][0]["message"]["content"]


def _extract_token_usage(api_data: Dict, user_input: str, output_text: str) -> Dict[str, int]:
    usage = api_data.get("usage", {}) if isinstance(api_data, dict) else {}
    prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None

    if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
        return {
            "input_tokens": max(0, prompt_tokens),
            "output_tokens": max(0, completion_tokens),
        }

    return {
        "input_tokens": _estimate_tokens(user_input),
        "output_tokens": _estimate_tokens(output_text),
    }


class SarvamModel(BaseLlm):
    model: str = SARVAM_MODEL_ID

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        messages = []
        if llm_request.config and llm_request.config.system_instruction:
            messages.append({"role": "system", "content": llm_request.config.system_instruction})

        for content in llm_request.contents:
            role = "assistant" if content.role == "model" else content.role
            text = "".join(part.text for part in content.parts if part.text)
            messages.append({"role": role, "content": text})

        try:
            output_text = await asyncio.to_thread(_sarvam_chat_completion, messages)
        except Exception as e:
            output_text = f"Model error: {str(e)}"

        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text=output_text)]),
            partial=False,
        )


model = SarvamModel()

device_control_agent = Agent(
    name="device_control",
    model=model,
    description="Handles direct control commands for supported home devices.",
    instruction="Return concise appliance-control acknowledgement text only.",
)

service_agent = Agent(
    name="service",
    model=model,
    description="Handles general assistant queries and service-related requests.",
    instruction="Return concise informational/helpful text only.",
)

automations_agent = Agent(
    name="automations",
    model=model,
    description="Handles reminders, schedules, workflows, and repeat automation requests.",
    instruction="Return concise automation acknowledgement text only.",
)

orchestrator_agent = Agent(
    name="orchestrator_agent",
    model=model,
    description="Structural placeholder only. Routing is done manually via orchestrate_request().",
    instruction="Manual orchestrator placeholder",
    sub_agents=[device_control_agent, service_agent, automations_agent],
)


ORCHESTRATION_SYSTEM_PROMPT = """
You are a production-grade AI Orchestrator Agent powered by Sarvam-30B.

Use ONLY these intents:
- greeting
- device_control
- service
- automations
- out_of_scope

Migration rules:
- shopping -> service
- queries -> service
- guardrail -> out_of_scope

Action mapping:
- greeting -> respond_greeting
- device_control -> control_device
- service -> answer_query
- automations -> create_automation
- out_of_scope -> reject_request

Always return STRICT JSON with this schema and no extra keys:
{
  "intent": "<intent>",
  "confidence": <0..1>,
  "action": "<action_name>",
  "parameters": {
    "device": "",
    "location": "",
    "time": "",
    "task": "",
    "query": ""
  },
  "metrics": {
    "urgency_level": "low|medium|high",
    "user_intent_clarity": <0..1>,
    "risk_level": "low|medium|high",
    "task_complexity": "low|medium|high",
    "execution_feasibility": <0..1>
  }
}

Decision rules:
- Choose exactly one intent
- If unclear, lower confidence
- Unsafe/illegal/harmful/security-sensitive => out_of_scope
- Keep numeric scores conservative and consistent

Mixed-intent handling (important):
- If input contains both greeting + actionable request, prioritize actionable intent.
- Example: "good morning, turn on fan" => device_control (not greeting)
- Example: "hi, remind me at 10pm" => automations (not greeting)
- Use greeting only when the user message is purely social/casual with no task.
""".strip()


async def orchestrate_request(user_input: str) -> Dict:
    result = await orchestrate_request_with_meta(user_input)
    return result["output"]


async def orchestrate_request_with_meta(user_input: str, max_retries: int = 1) -> Dict:
    user_input = (user_input or "").strip()
    started_at = time.time()

    if not user_input:
        output = _heuristic_fallback(user_input)
        latency = time.time() - started_at
        input_tokens = 0
        output_tokens = _estimate_tokens(json.dumps(output, ensure_ascii=False))

        ORCHESTRATOR_LATENCIES.append(latency)
        ORCHESTRATOR_INPUT_TOKENS.append(input_tokens)
        ORCHESTRATOR_OUTPUT_TOKENS.append(output_tokens)

        return {
            "output": output,
            "latency_seconds": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "json_validity": True,
            "schema_compliance": _is_schema_compliant(output),
            "failure_count": 0,
            "retry_count": 0,
            "fallback_used": True,
        }

    messages = [
        {"role": "system", "content": ORCHESTRATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    parsed_output = None
    json_validity = False
    failure_count = 0
    retry_count = 0
    input_tokens = _estimate_tokens(user_input)
    output_tokens = 0
    fallback_used = False

    for attempt in range(max_retries + 1):
        try:
            api_data = await asyncio.to_thread(_sarvam_chat_completion_raw, messages, 400, 0.1)
            raw_text = api_data["choices"][0]["message"]["content"]
            cleaned = _extract_json_text(raw_text)
            parsed = json.loads(cleaned)
            parsed_output = _sanitize_output(parsed, user_input)
            json_validity = True

            usage = _extract_token_usage(api_data, user_input, raw_text)
            input_tokens = usage["input_tokens"]
            output_tokens = usage["output_tokens"]
            break
        except Exception:
            failure_count += 1
            if attempt < max_retries:
                retry_count += 1
                continue

    if parsed_output is None:
        parsed_output = _heuristic_fallback(user_input)
        json_validity = False
        fallback_used = True
        if output_tokens == 0:
            output_tokens = _estimate_tokens(json.dumps(parsed_output, ensure_ascii=False))

    schema_compliance = _is_schema_compliant(parsed_output)
    latency = time.time() - started_at

    ORCHESTRATOR_LATENCIES.append(latency)
    ORCHESTRATOR_INPUT_TOKENS.append(input_tokens)
    ORCHESTRATOR_OUTPUT_TOKENS.append(output_tokens)

    return {
        "output": parsed_output,
        "latency_seconds": latency,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "json_validity": json_validity,
        "schema_compliance": schema_compliance,
        "failure_count": failure_count,
        "retry_count": retry_count,
        "fallback_used": fallback_used,
    }


def get_orchestrator_tracking_snapshot() -> Dict:
    return {
        "latencies": list(ORCHESTRATOR_LATENCIES),
        "input_tokens": list(ORCHESTRATOR_INPUT_TOKENS),
        "output_tokens": list(ORCHESTRATOR_OUTPUT_TOKENS),
    }


def reset_orchestrator_tracking() -> None:
    ORCHESTRATOR_LATENCIES.clear()
    ORCHESTRATOR_INPUT_TOKENS.clear()
    ORCHESTRATOR_OUTPUT_TOKENS.clear()


async def classify_intent(user_input: str) -> str:
    result = await orchestrate_request(user_input)
    return _normalize_intent(result.get("intent", "out_of_scope"))
