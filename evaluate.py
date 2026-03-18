import csv
import asyncio
import json
import logging
import os
from google.adk.runners import InMemoryRunner
from google.genai import types
from agent import orchestrator_agent
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment

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
def save_pretty_excel(matrix, labels):
    wb = Workbook()
    ws = wb.active
    ws.title = "Metrics"

    header_fill = PatternFill(start_color="4F81BD", end_color="4F81BD", fill_type="solid")
    subheader_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    green = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    red = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")

    bold = Font(bold=True, color="FFFFFF")
    center = Alignment(horizontal="center", vertical="center")

    # Title
    ws.merge_cells("A1:F1")
    ws["A1"] = "MISCLASSIFICATION MATRIX"
    ws["A1"].fill = header_fill
    ws["A1"].font = bold
    ws["A1"].alignment = center

    # Description
    ws.merge_cells("A2:F2")
    ws["A2"] = "Rows = Actual | Columns = Predicted | Green = Correct | Red = Errors"
    ws["A2"].alignment = center

    # Subheader
    ws.merge_cells("A4:F4")
    ws["A4"] = "Orchestrator Evaluation"
    ws["A4"].fill = subheader_fill
    ws["A4"].alignment = center

    # Headers
    ws.cell(row=5, column=1, value="Actual ↓ Predicted →").alignment = center

    for j, label in enumerate(labels, start=2):
        ws.cell(row=5, column=j, value=label).alignment = center

    ws.cell(row=5, column=len(labels)+2, value="Total")

    # Matrix
    for i, actual in enumerate(labels, start=6):
        ws.cell(row=i, column=1, value=actual)

        row_total = 0
        for j, pred in enumerate(labels, start=2):
            value = matrix[actual].get(pred, 0)
            row_total += value

            cell = ws.cell(row=i, column=j, value=value)
            cell.alignment = center

            if actual == pred:
                cell.fill = green
            elif value > 0:
                cell.fill = red

        ws.cell(row=i, column=len(labels)+2, value=row_total)

    for col in ws.columns:
        ws.column_dimensions[col[0].column_letter].width = 22

    wb.save("orchestration_metrics.xlsx")
    print("✅ Excel file created")


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

    # Load previous results
    if os.path.exists("evaluation_results.csv"):
        with open("evaluation_results.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                actual.append(row["Actual Label"])
                predicted.append(row["Predicted Label"])

    print("Evaluating dataset...\n")

    try:
        # -------- MAIN LOOP --------
        for i in range(start_index, len(dataset)):
            item = dataset[i]
            text = item["input_text"]
            label = item["actual_label"]

            try:
                pred = await get_prediction(runner, user_id, session_id, text)
            except Exception as e:
                print(f"❌ Error at index {i}: {e}")
                pred = "error"   # important fallback

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

    except Exception as e:
        print("⚠️ Evaluation stopped due to error:", e)

    finally:
        print("\n📊 Generating reports...")

        # -------- CONFUSION MATRIX --------
        labels = ["device_control", "greeting", "guardrail", "out_of_scope", "shopping", "queries", "error"]
        labels = [l for l in labels if l in set(actual + predicted)]

        matrix = {a: {p: 0 for p in labels} for a in labels}

        for a, p in zip(actual, predicted):
            matrix[a][p] += 1

        # -------- PRINT MATRIX --------
        print("\n================== Confusion Matrix ==================")

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

        # -------- SAVE CSV --------
        with open("confusion_matrix.csv", "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)

            writer.writerow(["Actual \\ Predicted"] + labels + ["Total"])

            for a in labels:
                row = [a]
                total_row = 0

                for p in labels:
                    val = matrix[a][p]
                    row.append(val)
                    total_row += val

                row.append(total_row)
                writer.writerow(row)

            totals = ["Total"]
            for p in labels:
                col_sum = sum(matrix[a][p] for a in labels)
                totals.append(col_sum)

            totals.append(sum(totals[1:]))
            writer.writerow(totals)

        print("\n✅ confusion_matrix.csv saved")

        # -------- SAVE EXCEL --------
        save_pretty_excel(matrix, labels)
        print("✅ Excel file created")

        print("✅ progress.json updated")


if __name__ == "__main__":
    asyncio.run(main())