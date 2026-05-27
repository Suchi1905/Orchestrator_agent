# Orchestrator Agent Project

This project focuses on building an intent-classification Orchestrator Agent using the Google ADK and Gemini 2.5 Flash model. The system intelligently routes user queries (such as home appliance control, electronics shopping, or post-purchase support) to specialized sub-agents while also providing guardrails for out-of-scope or unsafe requests.

Below is a detailed explanation of each file in this repository:

## 1. Core Source Code Files

- **`agent.py`**
  This is the core definition file for the agentic structure. It initializes the Gemini API model and defines three specialized sub-agents: `device_control_agent`, `shopping_agent`, and `queries_agent` with their system instructions and scope. Finally, it encapsulates them inside the `orchestrator_agent` which acts as the main router classifying user input into six possible intents (`greeting`, `device_control`, `shopping`, `queries`, `guardrail`, `out_of_scope`) and either responding directly or transferring to a sub-agent.

- **`running_agent.py`**
  This is an interactive command-line interface script for testing the orchestrator agent manually. It uses an `InMemoryRunner` to launch a local session. Users can continuously type chat queries into the console, and the script tracks the intent inference, the routing transfer path (e.g., `orchestrator_agent -> device_control`), and the final parsed JSON response returned by the targeted sub-agent.

- **`evaluate.py`**
  A high-level testing, benchmarking, and metrics generation script for the orchestrator setup. The script runs automated evaluations mapping queries from `dataset.csv` against the `orchestrator_agent`. It logs iteration progress (to allow resuming interrupted runs), evaluates if the prediction matches the ground truth label, generates accuracy percentages, and plots a complete Confusion Matrix in real-time. It exports these final evaluation metrics to `.csv` and graphically to an `.xlsx` (Excel) file.

## 2. Dataset and Metrics Files

- **`dataset.csv`**
  The main input dataset file providing the testing and benchmarking samples. It features two columns: `input_text` and `actual_label`, providing the raw query string and its intended ground truth intent.

- **`evaluation_results.csv`**
  An autosaved, continuously appending output file generated uniquely by `evaluate.py`. It provides row-by-row debugging visibility over the evaluation run: showing the user's base query, the actual intent label, what the agent predicted, and a boolean correctly predicting it.

- **`confusion_matrix.csv`**
  The raw CSV output file describing the classification confusion matrix. It quantifies how accurately the orchestrator routed intents correctly versus misclassifying them (e.g. mapping `guardrail` inputs incorrectly to `device_control`), ensuring performance safety can be audited.

- **`README.md`** 
  Historically contained a preview/draft block of the text classification dataset.

## 3. Configuration & Dependency Files

- **`.gitignore`**
  Standard Git exclusion file indicating which hidden or local project files (e.g., Python cache, local environments like `.env`, Excel output files, progress JSON files) should not be committed to source-control.

- **`requirements.txt`**
  Specifies the Python package dependencies and library versions needed to run the project. Users run `pip install -r requirements.txt` to align the environment, likely including libraries like `google-genai`, `google-adk`, `python-dotenv`, and `openpyxl`.
