import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from models import generate_public, generate_private

DATA_FILE = "data/pupa_tnb_train.csv"
RESULTS_FILE = "results/responses.csv"
REPEATS = 1         # each question sent once per model
WARMUP = 10         # warm-up requests before measuring (Irugalbandara et al., 2024)
LIMIT = None        # None = all 237 questions; a number = test run on the first N
PAUSE_PUBLIC = 4    # seconds between Gemini calls (free-tier rate limit)

# Fixed column order: every row is saved with these columns, matched by name
COLUMNS = ["q_index", "question_id", "category", "repeat", "deployment", "timestamp",
           "model", "response", "input_tokens", "output_tokens", "thinking_tokens",
           "latency_s", "latency_per_token_s", "attempts", "error"]

questions = pd.read_csv(DATA_FILE)
if LIMIT:
    questions = questions.head(LIMIT)

# Resume: skip (question number, repeat, deployment) already answered without error
done = set()
if Path(RESULTS_FILE).exists():
    old = pd.read_csv(RESULTS_FILE)
    if "q_index" in old.columns:
        ok = old[old["error"].isna()]
        done = set(zip(ok["q_index"], ok["repeat"], ok["deployment"]))
    print(f"Resuming: {len(done)} answers already saved")


def save(row):
    """Append one row, with columns matched by name (never by position)."""
    Path(RESULTS_FILE).parent.mkdir(exist_ok=True)
    pd.DataFrame([row]).reindex(columns=COLUMNS).to_csv(
        RESULTS_FILE, mode="a", index=False, header=not Path(RESULTS_FILE).exists())


# Warm-up so the local model is loaded before anything is measured
print(f"Warming up the local model with {WARMUP} requests...")
for _ in range(WARMUP):
    generate_private(questions.iloc[0]["user_query"])

for q_index, q in questions.iterrows():
    for repeat in range(1, REPEATS + 1):
        for deployment, generate in [("public", generate_public), ("private", generate_private)]:
            if (q_index, repeat, deployment) in done:
                continue
            print(f"Question {q_index + 1}/{len(questions)} | {deployment}")
            row = {"q_index": q_index, "question_id": q["conversation_hash"],
                   "category": q["predicted_category"], "repeat": repeat,
                   "deployment": deployment,
                   "timestamp": datetime.now().isoformat(timespec="seconds")}
            try:
                row.update(generate(q["user_query"]))
                row["error"] = None
            except Exception as error:
                row["error"] = str(error)[:200]
                print("  failed:", row["error"])
            save(row)
            if deployment == "public":
                time.sleep(PAUSE_PUBLIC)

print("Done. Results saved to", RESULTS_FILE)