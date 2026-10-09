from datetime import datetime
from pathlib import Path
import time

import pandas as pd
from models import generate_public, generate_private

DATA_FILE = "data/pupa_tnb_train.csv"
RESULTS_FILE = "results/speed_test.csv"
WARMUP = 10     # warm-up requests per model (Irugalbandara et al., 2024)
MEASURED = 10   # measured requests per model (Irugalbandara et al., 2024)
PAUSE_PUBLIC = 4

COLUMNS = ["run", "deployment", "timestamp", "model", "input_tokens", "output_tokens",
           "thinking_tokens", "latency_s", "latency_per_token_s", "attempts", "error"]

# Same request every time: the first PUPA-TNB question
question = pd.read_csv(DATA_FILE).iloc[0]["user_query"]
rows = []

for deployment, generate in [("public", generate_public), ("private", generate_private)]:
    print(f"Warming up {deployment} with {WARMUP} requests...")
    for _ in range(WARMUP):
        generate(question)
        if deployment == "public":
            time.sleep(PAUSE_PUBLIC)

    print(f"Measuring {deployment} with {MEASURED} requests...")
    for run in range(1, MEASURED + 1):
        row = {"run": run, "deployment": deployment,
               "timestamp": datetime.now().isoformat(timespec="seconds")}
        try:
            row.update(generate(question))
            row["error"] = None
        except Exception as error:
            row["error"] = str(error)[:200]
        rows.append(row)
        print(f"  {deployment} run {run}: {row.get('latency_s', float('nan')):.2f} s, "
              f"{row.get('output_tokens')} tokens")
        if deployment == "public":
            time.sleep(PAUSE_PUBLIC)

# Answer text is not needed for speed; columns matched by name
results = pd.DataFrame(rows).reindex(columns=COLUMNS)
Path(RESULTS_FILE).parent.mkdir(exist_ok=True)
results.to_csv(RESULTS_FILE, mode="a", index=False, header=not Path(RESULTS_FILE).exists())

print(results.groupby("deployment")[["latency_s", "latency_per_token_s", "output_tokens"]]
      .agg(["mean", "std"]).round(4))