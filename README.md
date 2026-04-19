# IoT Orchestration Agent

A production-ready, three-tier intent classification pipeline for IoT / smart-home voice assistants.

## Architecture

```
User Query
    │
    ▼
Tier 1 — Semantic Cache (Jaccard similarity ≥ 0.92)
    │   Instant return for repeated/similar queries (~0ms)
    │
    ▼  cache miss
Tier 2 — ML Classifier  (sentence-transformers + Logistic Regression)
    │   Fast semantic intent classification (~30ms)
    │   Returns result if confidence ≥ 0.75
    │
    ▼  confidence < 0.75
Tier 3 — Sarvam 30B LLM  (cloud fallback for ambiguous queries)
         Handles edge cases and novel phrasing (~2–12s)
```

### Supported Intents
| Intent | Example |
|---|---|
| `greetings` | "Hi there", "Good morning" |
| `device_control` | "Turn on the fan", "Set AC to 22°C" |
| `service_request` | "My TV is not working", "Show AMC plans" |
| `automations` | "Turn on lights every day at 7 PM" |
| `out_of_scope` | "Tell me a joke", "Who is the PM of India?" |

---

## Setup

### 1. Clone the repo
```bash
git clone <your-repo-url>
cd Orchestrator_Agent
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure your API key
```bash
cp .env.example .env
```
Open `.env` and replace `your_sarvam_api_key_here` with your actual Sarvam API key.

### 4. Run the dashboard
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## ML Classifier (Tier 2)

The trained classifier (`tier2_classifier.pkl`) is **already included** — no retraining needed.

If you want to retrain on your own dataset:
```bash
python train_classifier.py
```

Update the `DATASET_PATHS` list inside `train_classifier.py` to point to your CSV files.  
Each CSV must have columns: `input_text`, `actual_label`.

---

## Benchmarking

Upload a CSV with `input_text` and `actual_label` columns via the **Benchmark** tab in the dashboard.

---

## Project Structure

```
Orchestrator_Agent/
├── agent.py              # Core 3-tier orchestration logic
├── app.py                # Streamlit dashboard
├── evaluator.py          # Benchmark evaluation engine
├── metrics_engine.py     # Accuracy / latency metrics
├── train_classifier.py   # Retrain the Tier-2 ML classifier
├── tier2_classifier.pkl  # Pre-trained classifier (ready to use)
├── requirements.txt      # Python dependencies
├── .env.example          # API key template
└── dataset.csv           # Sample dataset for quick testing
```

---

## Performance (on benchmark datasets)

| Dataset | Accuracy |
|---|---|
| `final_cleaned_dataset` (1000 rows) | ~98% |
| `new_dataset` (550 rows, unseen) | ~95% |
| Combined held-out test (620 rows) | **95.32%** |

Latency:
- P50: ~30ms (ML classifier)
- P90: ~45ms (ML classifier)  
- P99: ~2–12s (rare LLM calls, near-zero after cache warm-up)
