from typing import Dict, List


def _percentile(values: List[float], percentile_value: float) -> float:
    if not values:
        return 0.0

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return round(sorted_values[0], 6)

    rank = (len(sorted_values) - 1) * percentile_value
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    result = sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
    return round(result, 6)


def compute_latency_percentiles(latencies: List[float]) -> Dict[str, float]:
    return {
        "p50": _percentile(latencies, 0.50),
        "p90": _percentile(latencies, 0.90),
        "p99": _percentile(latencies, 0.99),
    }


def compute_accuracy(predictions: List[str], labels: List[str]) -> float:
    if not labels:
        return 0.0
    correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
    return round(correct / len(labels), 6)


def compute_precision_recall_f1(confusion_matrix: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
    labels = sorted(confusion_matrix.keys())
    metrics: Dict[str, Dict[str, float]] = {}

    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0

    for label in labels:
        tp = confusion_matrix[label].get("tp", 0)
        fp = confusion_matrix[label].get("fp", 0)
        fn = confusion_matrix[label].get("fn", 0)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        metrics[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }

        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

    count = len(labels) if labels else 1
    metrics["macro_avg"] = {
        "precision": round(macro_precision / count, 6),
        "recall": round(macro_recall / count, 6),
        "f1": round(macro_f1 / count, 6),
    }

    return metrics


def compute_false_positive_rate(
    actual_labels: List[str],
    predicted_labels: List[str],
    out_of_scope_label: str = "out_of_scope",
) -> float:
    valid_indices = [i for i, label in enumerate(actual_labels) if out_of_scope_label not in set(label.split(","))]
    total_valid = len(valid_indices)
    if total_valid == 0:
        return 0.0

    false_positives = sum(
        1
        for i in valid_indices
        if out_of_scope_label in set(predicted_labels[i].split(","))
    )
    return round(false_positives / total_valid, 6)
