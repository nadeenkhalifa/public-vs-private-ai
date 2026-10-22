import pandas as pd

q = pd.read_csv("results/judge/quality_judgments.csv")
s = pd.read_csv("results/judge/judge_scores.csv")

v = q.pivot_table(index=["q_index", "deployment"], columns="order", values="verdict").reset_index()
v["outcome"] = "undecided (no/no)"
v.loc[(v[1] == 1) & (v[2] == 0), "outcome"] = "model better (yes/no)"
v.loc[(v[1] == 1) & (v[2] == 1), "outcome"] = "tie (yes/yes)"
v.loc[(v[1] == 0) & (v[2] == 1), "outcome"] = "reference better (no/yes)"

print("OUTCOME COUNTS")
print(pd.crosstab(v["outcome"], v["deployment"], margins=True))

v["lenient"] = ((v[1] == v[2]) | (v[1] == 1)).astype(int)   # Siyan's rule
v["strict"]  = (v[1] == 1).astype(int)                       # no/no counted as fail
print("\nQUALITY % (lenient = Siyan's rule, strict = undecided counted as fail)")
print((v.groupby("deployment")[["lenient", "strict"]].mean() * 100).round(1))

print("\nLEAKAGE % per answer (mean), answers with any leak, answers with full leak")
s2 = s.dropna(subset=["leakage"])
print(s2.groupby("deployment")["leakage"].agg(
    mean_pct=lambda x: round(x.mean() * 100, 1),
    any_leak_pct=lambda x: round((x > 0).mean() * 100, 1),
    full_leak_pct=lambda x: round((x == 1).mean() * 100, 1),
    answers="count"))