"""
Paired significance tests: private vs public on the same queries.

  Quality (pass/fail per answer)        -> McNemar exact test (McNemar, 1947)
  Leakage (% per query) and SEM-BLEU    -> Wilcoxon signed-rank test (Wilcoxon, 1945)

Run from the project folder:            python stats_tests.py
For the Arabic run, inside arabic_run:   python ../stats_tests.py
Saves results/stats_tests.csv
"""
import os
import pandas as pd
from scipy.stats import binomtest, wilcoxon

R = "results"
JUDGES = {"gpt-oss:20b": "", "qwen3:32b": "_qwen"}
rows = []

def quality_scores(path):
    q = pd.read_csv(path).pivot_table(index=["q_index", "deployment"], columns="order", values="verdict").reset_index()
    q["lenient"] = ((q[1] == q[2]) | (q[1] == 1)).astype(int)     # Siyan et al.'s rule (Q3)
    q["strict"] = (q[1] == 1).astype(int)                          # order 1 = yes (Q4)
    return q

def mcnemar(df, col):
    w = df.pivot(index="q_index", columns="deployment", values=col).dropna()
    b = int(((w.private == 1) & (w.public == 0)).sum())            # only private passes
    c = int(((w.private == 0) & (w.public == 1)).sum())            # only public passes
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0          # exact McNemar
    return len(w), b, c, p

def paired_wilcoxon(df, col):
    w = df.pivot(index="q_index", columns="deployment", values=col).dropna()
    d = w.private - w.public
    p = wilcoxon(w.private, w.public).pvalue if (d != 0).any() else 1.0
    return len(w), w.private.mean(), w.public.mean(), p

for judge, sfx in JUDGES.items():
    qf, lf = f"{R}/quality_judgments{sfx}.csv", f"{R}/leakage_judgments{sfx}.csv"
    if os.path.exists(qf):
        q = quality_scores(qf)
        for col in ["lenient", "strict"]:
            n, b, c, p = mcnemar(q, col)
            rows.append({"measure": f"quality_{col}", "judge": judge, "test": "McNemar (exact)", "n_pairs": n,
                         "private_only_pass": b, "public_only_pass": c, "p_value": p})
    if os.path.exists(lf):
        l = pd.read_csv(lf).groupby(["q_index", "deployment"])["verdict"].mean().rename("leak").reset_index()
        n, mp, mq, p = paired_wilcoxon(l, "leak")
        rows.append({"measure": "leakage", "judge": judge, "test": "Wilcoxon signed-rank", "n_pairs": n,
                     "private_mean": mp * 100, "public_mean": mq * 100, "p_value": p})

sf = f"{R}/sem_bleu_scores.csv"
if os.path.exists(sf):
    s = pd.read_csv(sf)
    n, mp, mq, p = paired_wilcoxon(s, "sem_bleu")
    rows.append({"measure": "sem_bleu", "judge": "-", "test": "Wilcoxon signed-rank", "n_pairs": n,
                 "private_mean": mp, "public_mean": mq, "p_value": p})

res = pd.DataFrame(rows)
res["significant_5pct"] = res.p_value < 0.05
pd.set_option("display.width", 200)
print(res.round(4).to_string(index=False))
res.to_csv(f"{R}/stats_tests.csv", index=False)
print(f"\nsaved {R}/stats_tests.csv")