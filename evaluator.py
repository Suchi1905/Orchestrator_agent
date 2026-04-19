import argparse
import asyncio
import csv
import json
import os
import re
from typing import Dict, List

from agent import (
    ALLOWED_INTENTS,
    MODEL_PROVIDER,
    SARVAM_MODEL_ID,
    orchestrate_request_with_meta,
    reset_orchestrator_tracking,
)
from metrics_engine import (
    compute_accuracy,
    compute_false_positive_rate,
    compute_latency_percentiles,
    compute_precision_recall_f1,
)


def normalize_label(label: str) -> str:
    """
    Map any raw label string to one of the canonical ALLOWED_INTENTS.
    Handles legacy/alternate spellings so older datasets still work.
    """
    normalized = (label or "").strip().lower()

    _alias_map = {
        # greetings
        "greeting":        "greetings",
        "greetings":       "greetings",
        # device control
        "device_control":  "device_control",
        # service / customer support
        "service_request": "service_request",
        "service":         "service_request",
        "shopping":        "service_request",
        "queries":         "service_request",
        # automations
        "automations":     "automations",
        "automation":      "automations",
        # out of scope / guardrail
        "out_of_scope":    "out_of_scope",
        "guardrail":       "out_of_scope",
        "unsafe":          "out_of_scope",
    }

    if normalized in _alias_map:
        return _alias_map[normalized]

    # Partial / prefix match fallback
    for key, canonical in _alias_map.items():
        if normalized.startswith(key):
            return canonical

    return "out_of_scope"


def detect_language_tag(text: str) -> str:
    if not text:
        return "English"

    hindi_script = re.search(r"[\u0900-\u097F]", text)
    if hindi_script:
        return "Hindi"

    malyalam_script = re.search(r"[\u0D00-\u0D7F]", text)
    if malyalam_script:
        return "Malayalam"

    lower = text.lower()
    hinglish_markers = ["hai", "karo", "karna", "krdo", "mera", "mujhe", "please", "yaar"]
    translit_markers = ["namaste", "kaise", "aap", "dhanyavaad", "shukriya"]

    if any(word in lower for word in translit_markers):
        return "Transliteration"
    if any(word in lower for word in hinglish_markers):
        return "Hinglish"
    return "English"


def load_dataset(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    required = {"input_text", "actual_label"}
    if not required.issubset(set(reader.fieldnames or [])):
        missing = required - set(reader.fieldnames or [])
        raise ValueError(f"Dataset missing required columns: {', '.join(sorted(missing))}")

    return rows


def build_confusion_matrix(labels: List[str]) -> Dict[str, Dict[str, int]]:
    matrix_labels = list(ALLOWED_INTENTS)
    matrix = {actual: {pred: 0 for pred in matrix_labels} for actual in matrix_labels}
    for actual, pred in labels:
        a = actual if actual in matrix else "out_of_scope"
        p = pred if pred in matrix[a] else "out_of_scope"
        matrix[a][p] += 1
    return matrix


def save_confusion_matrix_csv(path: str, matrix: Dict[str, Dict[str, int]]) -> None:
    labels = list(ALLOWED_INTENTS)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["actual/predicted"] + labels + ["total"])
        for actual in labels:
            total = sum(matrix[actual][pred] for pred in labels)
            writer.writerow([actual] + [matrix[actual][pred] for pred in labels] + [total])


def save_detailed_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_csv(path: str, report: Dict) -> None:
    entries = [
        ("model_provider", report["model_config"]["model_provider"]),
        ("model_id", report["model_config"]["model_id"]),
        ("dataset", report["model_config"]["dataset"]),
        ("total_requests", report["model_config"]["total_requests"]),
        ("latency_p50", report["latency_seconds"]["p50"]),
        ("latency_p90", report["latency_seconds"]["p90"]),
        ("latency_p99", report["latency_seconds"]["p99"]),
        ("overall_accuracy", report["classification"]["overall_accuracy"]),
        ("guardrail_false_positive_rate", report["guardrail"]["false_positive_rate"]),
        ("avg_input_tokens", report["token_usage"]["avg_input_tokens"]),
        ("avg_output_tokens", report["token_usage"]["avg_output_tokens"]),
        ("avg_cost_per_request", report["cost"]["avg_cost_per_request"]),
        ("cost_per_1k_queries", report["cost"]["cost_per_1k_queries"]),
        ("json_validity_rate", report["quality"]["json_validity_rate"]),
        ("schema_compliance_rate", report["quality"]["schema_compliance_rate"]),
        ("failure_count", report["quality"]["failure_count"]),
        ("retry_count", report["quality"]["retry_count"]),
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for row in entries:
            writer.writerow(row)


async def run_benchmark(dataset_path: str, output_dir: str) -> Dict:
    rows = load_dataset(dataset_path)
    os.makedirs(output_dir, exist_ok=True)

    input_cost_per_1k_tokens = float(os.getenv("INPUT_COST_PER_1K_TOKENS", "0.0"))
    output_cost_per_1k_tokens = float(os.getenv("OUTPUT_COST_PER_1K_TOKENS", "0.0"))

    predictions: List[str] = []
    actuals: List[str] = []
    latencies: List[float] = []
    input_tokens_all: List[int] = []
    output_tokens_all: List[int] = []

    detailed_rows: List[Dict] = []
    language_tracker: Dict[str, Dict[str, int]] = {}

    json_valid_true = 0
    schema_compliant_true = 0
    failure_count = 0
    retry_count = 0

    reset_orchestrator_tracking()

    sem = asyncio.Semaphore(20)

    async def _process_row(idx, row):
        query = str(row.get("input_text", "")).strip()
        actual_label = normalize_label(str(row.get("actual_label", "")))
        language = (row.get("language") or "").strip() or detect_language_tag(query)

        async with sem:
            result = await orchestrate_request_with_meta(query, max_retries=2)
            
        return idx, query, actual_label, language, result

    tasks = [_process_row(idx, row) for idx, row in enumerate(rows, start=1)]
    results = await asyncio.gather(*tasks)

    for idx, query, actual_label, language, result in results:

        output = result["output"]

        predicted_label = normalize_label(output.get("intent", "out_of_scope"))
        predictions.append(predicted_label)
        actuals.append(actual_label)

        latency = float(result.get("latency_seconds", 0.0))
        in_tok = int(result.get("input_tokens", 0))
        out_tok = int(result.get("output_tokens", 0))
        latencies.append(latency)
        input_tokens_all.append(in_tok)
        output_tokens_all.append(out_tok)

        json_valid = bool(result.get("json_validity", False))
        schema_ok = bool(result.get("schema_compliance", False))
        if json_valid:
            json_valid_true += 1
        if schema_ok:
            schema_compliant_true += 1

        failure_count += int(result.get("failure_count", 0))
        retry_count += int(result.get("retry_count", 0))

        if language not in language_tracker:
            language_tracker[language] = {"total": 0, "correct": 0}
        language_tracker[language]["total"] += 1
        if predicted_label == actual_label:
            language_tracker[language]["correct"] += 1

        request_cost = ((in_tok / 1000.0) * input_cost_per_1k_tokens) + ((out_tok / 1000.0) * output_cost_per_1k_tokens)

        detailed_rows.append(
            {
                "index": idx,
                "input_text": query,
                "language": language,
                "actual_label": actual_label,
                "predicted_label": predicted_label,
                "correct": predicted_label == actual_label,
                "latency_seconds": f"{latency:.6f}",
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "json_validity": json_valid,
                "schema_compliance": schema_ok,
                "failure_count": int(result.get("failure_count", 0)),
                "retry_count": int(result.get("retry_count", 0)),
                "request_cost": round(request_cost, 8),
            }
        )

    total_requests = len(actuals)
    confusion_matrix = build_confusion_matrix(list(zip(actuals, predictions)))
    latency_metrics = {k: f"{v:.6f}" for k, v in compute_latency_percentiles(latencies).items()}
    overall_accuracy = compute_accuracy(predictions, actuals)
    cls_metrics = compute_precision_recall_f1(confusion_matrix)
    guardrail_fpr = compute_false_positive_rate(actuals, predictions, out_of_scope_label="out_of_scope")

    total_input_tokens = sum(input_tokens_all)
    total_output_tokens = sum(output_tokens_all)
    avg_input_tokens = (total_input_tokens / total_requests) if total_requests else 0.0
    avg_output_tokens = (total_output_tokens / total_requests) if total_requests else 0.0

    total_cost = ((total_input_tokens / 1000.0) * input_cost_per_1k_tokens) + ((total_output_tokens / 1000.0) * output_cost_per_1k_tokens)
    avg_cost_per_request = (total_cost / total_requests) if total_requests else 0.0
    cost_per_1k_queries = avg_cost_per_request * 1000.0

    multilingual_accuracy = {}
    for language, counts in language_tracker.items():
        total = counts["total"]
        acc = (counts["correct"] / total) if total else 0.0
        multilingual_accuracy[language] = {
            "total": total,
            "correct": counts["correct"],
            "accuracy": round(acc, 6),
        }

    report = {
        "model_config": {
            "model_provider": MODEL_PROVIDER,
            "model_id": SARVAM_MODEL_ID,
            "dataset": dataset_path,
            "total_requests": total_requests,
            "allowed_intents": list(ALLOWED_INTENTS),
            "pricing": {
                "input_cost_per_1k_tokens": input_cost_per_1k_tokens,
                "output_cost_per_1k_tokens": output_cost_per_1k_tokens,
            },
        },
        "latency_seconds": latency_metrics,
        "token_usage": {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "avg_input_tokens": round(avg_input_tokens, 6),
            "avg_output_tokens": round(avg_output_tokens, 6),
        },
        "classification": {
            "overall_accuracy": overall_accuracy,
            "per_intent": cls_metrics,
        },
        "guardrail": {
            "false_positive_rate": guardrail_fpr,
        },
        "multilingual": multilingual_accuracy,
        "cost": {
            "total_cost": round(total_cost, 8),
            "avg_cost_per_request": round(avg_cost_per_request, 8),
            "cost_per_1k_queries": round(cost_per_1k_queries, 8),
        },
        "quality": {
            "json_validity_rate": round((json_valid_true / total_requests) if total_requests else 0.0, 6),
            "schema_compliance_rate": round((schema_compliant_true / total_requests) if total_requests else 0.0, 6),
            "failure_count": failure_count,
            "retry_count": retry_count,
        },
        "misclassification_matrix": confusion_matrix,
    }

    misclassification_csv = os.path.join(output_dir, "misclassification_matrix.csv")
    detailed_csv = os.path.join(output_dir, "benchmark_detailed_results.csv")
    summary_csv = os.path.join(output_dir, "benchmark_summary.csv")
    json_report = os.path.join(output_dir, "benchmark_report.json")

    save_confusion_matrix_csv(misclassification_csv, confusion_matrix)
    save_detailed_csv(detailed_csv, detailed_rows)
    save_summary_csv(summary_csv, report)

    with open(json_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return {
        "report": report,
        "paths": {
            "misclassification_matrix_csv": misclassification_csv,
            "detailed_csv": detailed_csv,
            "summary_csv": summary_csv,
            "json_report": json_report,
        },
    }


def print_report(report: Dict, paths: Dict[str, str]) -> None:
    print("\n=== ORCHESTRATOR BENCHMARK REPORT ===")
    print(f"Model: {report['model_config']['model_provider']} / {report['model_config']['model_id']}")
    print(f"Dataset: {report['model_config']['dataset']}")
    print(f"Total requests: {report['model_config']['total_requests']}")

    print("\n-- Latency (seconds) --")
    print(f"P50: {report['latency_seconds']['p50']}")
    print(f"P90: {report['latency_seconds']['p90']}")
    print(f"P99: {report['latency_seconds']['p99']}")

    print("\n-- Accuracy --")
    print(f"Overall accuracy: {report['classification']['overall_accuracy']}")
    print(f"Guardrail false positive rate: {report['guardrail']['false_positive_rate']}")

    print("\nPer-intent precision/recall/F1:")
    per_intent = report["classification"]["per_intent"]
    for label, values in per_intent.items():
        if not isinstance(values, dict):
            continue
        print(
            f"{label}: precision={values.get('precision', 0)}, "
            f"recall={values.get('recall', 0)}, f1={values.get('f1', 0)}"
        )

    print("\n-- Token usage --")
    print(f"Average input tokens: {report['token_usage']['avg_input_tokens']}")
    print(f"Average output tokens: {report['token_usage']['avg_output_tokens']}")

    print("\n-- Cost --")
    print(f"Total cost: {report['cost']['total_cost']}")
    print(f"Average cost/request: {report['cost']['avg_cost_per_request']}")
    print(f"Cost per 1K queries: {report['cost']['cost_per_1k_queries']}")

    print("\n-- Quality --")
    print(f"JSON validity rate: {report['quality']['json_validity_rate']}")
    print(f"Schema compliance rate: {report['quality']['schema_compliance_rate']}")
    print(f"Failure count: {report['quality']['failure_count']}")
    print(f"Retry count: {report['quality']['retry_count']}")

    print("\n-- Multilingual accuracy --")
    for language, values in report["multilingual"].items():
        print(f"{language}: accuracy={values['accuracy']} (correct={values['correct']}, total={values['total']})")

    print("\nOutput files:")
    for key, value in paths.items():
        print(f"- {key}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run orchestrator benchmarking pipeline")
    parser.add_argument("--dataset", default="dataset.csv", help="Path to dataset CSV")
    parser.add_argument("--output-dir", default="benchmark_outputs", help="Directory for benchmark outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = asyncio.run(run_benchmark(args.dataset, args.output_dir))
    print_report(result["report"], result["paths"])


if __name__ == "__main__":
    main()
