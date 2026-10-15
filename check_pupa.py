import pandas as pd

local = pd.read_csv("data/pupa_tnb_train.csv").reset_index(drop=True)
local["q_index"] = local.index
res = pd.read_csv("results/responses_final.csv")
res = res[res["deployment"] == "private"][["q_index", "question_id", "category"]]

m = res.merge(local[["q_index", "conversation_hash", "predicted_category"]], on="q_index")
same = (m["question_id"] == m["conversation_hash"]) & (m["category"] == m["predicted_category"])
print(f"{same.sum()} of {len(m)} rows match")