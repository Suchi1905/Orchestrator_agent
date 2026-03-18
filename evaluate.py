import csv
import asyncio
import json
import logging
import os
from google.adk.runners import InMemoryRunner
from google.genai import types
from agent import orchestrator_agent

logging.getLogger("google_genai.types").setLevel(logging.ERROR)


# ---------------- CLEAN RESPONSE ----------------
def extract_json_text(response):
    if not response:
        return response
    cleaned = response.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


# ---------------- GET PREDICTION ----------------
async def get_prediction(runner, user_id, session_id, user_input):
    for attempt in range(1):
        try:
            response = None

            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=types.Content(
                    role="user",
                    parts=[types.Part(text=user_input)],
                ),
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    response = "\n".join(
                        part.text for part in event.content.parts if part.text
                    )

            response = extract_json_text(response)

            # ✅ CASE 1: JSON response
            try:
                parsed = json.loads(response)
                intent = parsed.get("intent", "").strip().lower()
                if intent:
                    return intent
            except:
                pass

            # ✅ CASE 2: plain text fallback
            cleaned = response.strip().lower()

            valid_labels = [
                "greeting",
                "device_control",
                "guardrail",
                "out_of_scope",
                "shopping",
                "queries"
            ]

            for label in valid_labels:
                if label in cleaned:
                    return label

            return "out_of_scope"

        except Exception as e:
            print(f"[Retry {attempt+1}] error:", e)
            await asyncio.sleep(15)

    return "error"


# ---------------- MAIN ----------------
async def main():
    runner = InMemoryRunner(agent=orchestrator_agent)

    user_id = "eval_user"
    session_id = "eval_session"

    await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id,
    )

    # -------- LOAD DATASET --------
    with open("dataset.csv", "r", encoding="utf-8") as f:
        dataset = list(csv.DictReader(f))

    # -------- LOAD PROGRESS --------
    start_index = 0
    if os.path.exists("progress.json"):
        with open("progress.json", "r") as f:
            progress = json.load(f)
            start_index = progress.get("last_index", 0)

    print(f"Resuming from index: {start_index}\n")

    actual = []
    predicted = []

    # Load previous results (important for confusion matrix)
    if os.path.exists("evaluation_results.csv"):
        with open("evaluation_results.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                actual.append(row["Actual Label"])
                predicted.append(row["Predicted Label"])

    print("Evaluating dataset...\n")

    # -------- MAIN LOOP --------
    for i in range(start_index, len(dataset)):
        item = dataset[i]
        text = item["input_text"]
        label = item["actual_label"]

        pred = await get_prediction(runner, user_id, session_id, text)

        actual.append(label)
        predicted.append(pred)

        print(f"[{text}] -> Actual: {label}, Predicted: {pred}")

        # -------- SAVE RESULT --------
        file_exists = os.path.exists("evaluation_results.csv")

        with open("evaluation_results.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Input Query", "Actual Label", "Predicted Label", "Correct"])
            writer.writerow([text, label, pred, label == pred])

        # -------- SAVE PROGRESS --------
        with open("progress.json", "w") as f:
            json.dump({"last_index": i + 1}, f)

        await asyncio.sleep(15)

    # -------- CONFUSION MATRIX --------
    print("\n================== Confusion Matrix ==================")

    labels = ["device_control", "greeting", "guardrail", "out_of_scope", "shopping", "queries"]
    labels = [l for l in labels if l in set(actual + predicted)]

    matrix = {a: {p: 0 for p in labels} for a in labels}

    for a, p in zip(actual, predicted):
        matrix[a][p] += 1

    # -------- PRINT MATRIX --------
    header = f"{'Actual \\ Pred':<20} | " + " | ".join([f"{l:<12}" for l in labels])
    print(header)
    print("-" * len(header))

    for a in labels:
        row = f"{a:<20} | "
        for p in labels:
            row += f"{matrix[a][p]:<12} | "
        print(row)

    # -------- ACCURACY --------
    correct = sum(1 for a, p in zip(actual, predicted) if a == p)
    total = len(actual)
    print(f"\nAccuracy: {correct}/{total} ({(correct/total)*100:.2f}%)")

    # -------- SAVE EXCEL-FRIENDLY CSV --------
    with open("confusion_matrix.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        # header
        writer.writerow(["Actual \\ Predicted"] + labels + ["Total"])

        # rows
        for a in labels:
            row = [a]
            total_row = 0

            for p in labels:
                val = matrix[a][p]
                row.append(val)
                total_row += val

            row.append(total_row)
            writer.writerow(row)

        # column totals
        totals = ["Total"]
        for p in labels:
            col_sum = sum(matrix[a][p] for a in labels)
            totals.append(col_sum)

        totals.append(sum(totals[1:]))
        writer.writerow(totals)

    print("\n✅ confusion_matrix.csv saved (Excel ready)")
    print("✅ progress.json updated")


if __name__ == "__main__":
    asyncio.run(main())