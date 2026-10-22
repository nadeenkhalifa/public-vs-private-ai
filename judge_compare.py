"""
Compares the two AI judges (gpt-oss:20b vs qwen3:32b) on the same answers, same Siyan et al. (2025) prompts.
Agreement % follows Siyan et al. (judge vs human agreement, Sec. 5); Cohen's kappa = agreement corrected for chance.

Run from the project folder:  python judge_compare.py
Needs in results/: quality_judgments.csv, leakage_judgments.csv, quality_judgments_qwen.csv, leakage_judgments_qwen.csv
"""
import pandas as pd

JUDGES = {"gpt-oss:20b": "", "qwen3:32b": "_qwen"}

def kappa(a, b):
    a, b = pd.Series(a).astype(int).values, pd.Series(b).astype(int).values
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")

def outcomes(path):
    q = pd.read_csv(path).pivot_table(index=["q_index", "deployment"], columns="order", values="verdict").reset_index()
    q["outcome"] = q.apply(lambda r: {(1, 0): "better", (1, 1): "tie", (0, 0): "undecided", (0, 1): "worse"}[(int(r[1]), int(r[2]))], axis=1)
    q["lenient"] = ((q[1] == q[2]) | (q[1] == 1)).astype(int)   # Siyan's rule: if s1 == s2 -> 1, else s1
    q["strict"] = q["outcome"].isin(["better", "tie"]).astype(int)
    return q

def leak_answers(path):
    return pd.read_csv(path).groupby(["q_index", "deployment"])["verdict"].mean().rename("leak").reset_index()

Q = {j: outcomes(f"results/quality_judgments{s}.csv") for j, s in JUDGES.items()}
L = {j: leak_answers(f"results/leakage_judgments{s}.csv") for j, s in JUDGES.items()}
RAWQ = {j: pd.read_csv(f"results/quality_judgments{s}.csv") for j, s in JUDGES.items()}
RAWL = {j: pd.read_csv(f"results/leakage_judgments{s}.csv") for j, s in JUDGES.items()}

# ---------- 1. Results side by side ----------
rows = []
for j in JUDGES:
    for dep in ["private", "public"]:
        q, l = Q[j][Q[j].deployment == dep], L[j][L[j].deployment == dep]
        oc = q["outcome"].value_counts()
        rows.append({"judge": j, "deployment": dep, "answers": len(q),
                     "quality_lenient_%": q["lenient"].mean() * 100, "quality_strict_%": q["strict"].mean() * 100,
                     "better": oc.get("better", 0), "tie": oc.get("tie", 0),
                     "undecided": oc.get("undecided", 0), "worse": oc.get("worse", 0),
                     "leak_mean_%": l["leak"].mean() * 100, "leak_any_%": (l["leak"] > 0).mean() * 100,
                     "leak_full_%": (l["leak"] == 1).mean() * 100})
res = pd.DataFrame(rows).round(1)
print("=== 1. RESULTS BY JUDGE ===\n", res.to_string(index=False))

# ---------- 2. Does the ranking hold? (public minus private) ----------
print("\n=== 2. GAP: PUBLIC minus PRIVATE (percentage points) ===")
for j in JUDGES:
    r = res[res.judge == j].set_index("deployment")
    print(f"{j:12s}  lenient {r.loc['public','quality_lenient_%'] - r.loc['private','quality_lenient_%']:+.1f}"
          f" | strict {r.loc['public','quality_strict_%'] - r.loc['private','quality_strict_%']:+.1f}"
          f" | leakage mean {r.loc['public','leak_mean_%'] - r.loc['private','leak_mean_%']:+.1f}")

# ---------- 3. How often do the judges say "yes"? ----------
print("\n=== 3. YES RATE PER JUDGE (%) ===")
for j in JUDGES:
    print(f"{j:12s}  quality verdicts {RAWQ[j]['verdict'].mean()*100:.1f} | leakage verdicts {RAWL[j]['verdict'].mean()*100:.1f}")

# ---------- 4. Agreement between the judges ----------
a, b = list(JUDGES)
print("\n=== 4. AGREEMENT BETWEEN JUDGES ===")
mq = RAWQ[a].merge(RAWQ[b], on=["q_index", "deployment", "order"], suffixes=("_a", "_b"))
ml = RAWL[a].merge(RAWL[b], on=["q_index", "deployment", "unit"], suffixes=("_a", "_b"))
mo = Q[a].merge(Q[b], on=["q_index", "deployment"], suffixes=("_a", "_b"))
for name, d, x, y in [("Quality, single verdicts", mq, "verdict_a", "verdict_b"),
                      ("Quality, lenient pass/fail", mo, "lenient_a", "lenient_b"),
                      ("Quality, strict pass/fail", mo, "strict_a", "strict_b"),
                      ("Leakage, single verdicts", ml, "verdict_a", "verdict_b")]:
    print(f"{name:28s} n={len(d):5d}  agreement {(d[x] == d[y]).mean()*100:5.1f}%  kappa {kappa(d[x], d[y]):.2f}")
print(f"{'Quality, same outcome':28s} n={len(mo):5d}  agreement {(mo.outcome_a == mo.outcome_b).mean()*100:5.1f}%")
print("\nOutcome cross-table (rows = gpt-oss, columns = qwen):\n", pd.crosstab(mo.outcome_a, mo.outcome_b))

res.to_csv("results/judge_comparison.csv", index=False, encoding="utf-8")