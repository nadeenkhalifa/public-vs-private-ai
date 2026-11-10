"""
Compares the original run with the Arabic run (non-English queries translated into Arabic).
Shows each measure for: all queries, the translated queries (split into full and mixed), and the unchanged (English) queries.
Comparing the translated subset isolates the effect of the translation; the unchanged subset shows
normal run-to-run variation (new answers at temperature 0.7).

Run from the project folder:  python compare_runs.py
Needs: results/ and data/ (original run), arabic_run/results/ and arabic_run/data/ (Arabic run)
"""
import os
import pandas as pd

RUNS = {"Original": ".", "Arabic": "arabic_run"}
JUDGES = {"gpt-oss:20b": "", "qwen3:32b": "_qwen"}

ar = pd.read_csv("arabic_run/data/pupa_tnb_train.csv").reset_index(drop=True)
translated = set(ar.index[ar["translated"] == True])
ttype = ar["translation_type"] if "translation_type" in ar.columns else pd.Series("", index=ar.index)
full = set(ar.index[ttype == "full"])
mixed = set(ar.index[ttype == "mixed"])
print(f"Translated queries: {len(translated)} -> {sorted(translated)}")
print(f"  full (whole query translated): {len(full)} -> {sorted(full)}")
print(f"  mixed (only non-English parts translated): {len(mixed)} -> {sorted(mixed)}\n")

SUBSETS = ["all", "translated", "full", "mixed", "unchanged"]

def subset(df, which):
    if which == "translated":
        return df[df.q_index.isin(translated)]
    if which == "full":
        return df[df.q_index.isin(full)]
    if which == "mixed":
        return df[df.q_index.isin(mixed)]
    if which == "unchanged":
        return df[~df.q_index.isin(translated)]
    return df

def quality(path):
    q = pd.read_csv(path).pivot_table(index=["q_index", "deployment"], columns="order", values="verdict").reset_index()
    q["lenient"] = ((q[1] == q[2]) | (q[1] == 1)).astype(int)        # Siyan et al.'s rule
    q["strict"] = (q[1] == 1).astype(int)                             # better or tie (order 1 = yes)
    return q

rows = []
for run, folder in RUNS.items():
    R = os.path.join(folder, "results")
    for judge, sfx in JUDGES.items():
        qf, lf = f"{R}/quality_judgments{sfx}.csv", f"{R}/leakage_judgments{sfx}.csv"
        if not (os.path.exists(qf) and os.path.exists(lf)):
            print(f"[{run} / {judge}] judge files not found - skipped"); continue
        q = quality(qf)
        l = pd.read_csv(lf).groupby(["q_index", "deployment"])["verdict"].mean().rename("leak").reset_index()
        for which in SUBSETS:
            qs, ls = subset(q, which), subset(l, which)
            for dep in ["private", "public"]:
                a, b = qs[qs.deployment == dep], ls[ls.deployment == dep]
                rows.append({"subset": which, "judge": judge, "deployment": dep, "run": run, "n": len(a),
                             "quality_lenient_%": a.lenient.mean() * 100, "quality_strict_%": a.strict.mean() * 100,
                             "leakage_mean_%": b.leak.mean() * 100})
res = pd.DataFrame(rows).round(1)

pd.set_option("display.width", 200)
for which in SUBSETS:
    t = res[res.subset == which].pivot_table(index=["judge", "deployment"], columns="run",
                                             values=["quality_lenient_%", "quality_strict_%", "leakage_mean_%"])
    n = res[(res.subset == which)].groupby("run")["n"].max().to_dict()
    print(f"=== {which.upper()} QUERIES (answers per model: {n}) ===\n{t.round(1).to_string()}\n")

# ---------- answers: speed, length, loops ----------
ans = []
for run, folder in RUNS.items():
    f = os.path.join(folder, "results", "responses_final.csv")
    if not os.path.exists(f):
        continue
    r = pd.read_csv(f)
    for which in SUBSETS:
        s = subset(r, which)
        for dep in ["private", "public"]:
            d = s[s.deployment == dep]
            ans.append({"subset": which, "deployment": dep, "run": run, "answers": len(d),
                        "median_latency_s": d.latency_s.median(), "mean_output_tokens": d.output_tokens.mean(),
                        "loops_40960": int((d.output_tokens >= 40960).sum()),
                        "loop_q_index": sorted(d[d.output_tokens >= 40960].q_index.tolist())})
ans = pd.DataFrame(ans)
print("=== ANSWERS: speed, length, repetition loops ===")
print(ans.round(2).to_string(index=False), "\n")

# ---------- SEM-BLEU (if sem_bleu.py was run in both folders) ----------
sb = []
for run, folder in RUNS.items():
    f = os.path.join(folder, "results", "sem_bleu_scores.csv")
    if os.path.exists(f):
        s = pd.read_csv(f)
        for which in SUBSETS:
            m = subset(s, which).groupby("deployment")["sem_bleu"].mean()
            sb += [{"subset": which, "deployment": d, "run": run, "sem_bleu": v} for d, v in m.items()]
if sb:
    print("=== SEM-BLEU ===")
    print(pd.DataFrame(sb).pivot_table(index=["subset", "deployment"], columns="run", values="sem_bleu").round(3).to_string())

# ---------- speed test + cost (equations 1-10, Irugalbandara et al., 2024, Sec. VI-E) ----------
IN_PRICE, OUT_PRICE, GPU_HOUR, HOURS, UTIL = 0.30, 2.50, 0.526, 730, 0.8
sc = []
for run, folder in RUNS.items():
    R = os.path.join(folder, "results")
    row = {"run": run}
    if os.path.exists(f"{R}/speed_test.csv"):
        st = pd.read_csv(f"{R}/speed_test.csv")
        st = st[st["error"].isna() | (st["error"].astype(str).str.strip() == "")]
        for dep in ["private", "public"]:
            d = st[st.deployment == dep]
            row[f"speedtest_{dep}_s"] = f"{d.latency_s.mean():.2f} ± {d.latency_s.std():.2f}"
            row[f"speedtest_{dep}_s_per_token"] = f"{d.latency_per_token_s.mean():.4f} ± {d.latency_per_token_s.std():.4f}"
    if os.path.exists(f"{R}/responses_final.csv"):
        r = pd.read_csv(f"{R}/responses_final.csv")
        r = r[~r.get("note", pd.Series("", index=r.index)).fillna("").astype(str).str.startswith("failed")]
        pub, pri = r[r.deployment == "public"], r[r.deployment == "private"]
        api_cost = (pub.input_tokens.sum() * IN_PRICE + (pub.output_tokens.sum() + pub.thinking_tokens.fillna(0).sum()) * OUT_PRICE) / 1e6
        tok_h = pri.output_tokens.sum() / pri.latency_s.sum() * 3600 * UTIL
        row.update({"median_latency_private_s": round(pri.latency_s.median(), 2), "median_latency_public_s": round(pub.latency_s.median(), 2),
                    "median_s_per_token_private": round(pri.latency_per_token_s.median(), 4),
                    "median_s_per_token_public": round(pub.latency_per_token_s.median(), 4),
                    "api_cost_total_$": round(api_cost, 3), "api_per_query_$": round(api_cost / len(pub), 5),
                    "api_per_1K_$": round(api_cost / pub.output_tokens.sum() * 1000, 5),
                    "private_tokens_per_s": round(pri.output_tokens.sum() / pri.latency_s.sum(), 2),
                    "private_per_1K_$": round(GPU_HOUR / tok_h * 1000, 5),
                    "private_vs_api": round((GPU_HOUR / tok_h * 1000) / (api_cost / pub.output_tokens.sum() * 1000), 2),
                    "capacity_queries_month": round(tok_h * HOURS / pri.output_tokens.mean()),
                    "break_even_queries_month": round(GPU_HOUR * HOURS / (api_cost / len(pub)))})
    sc.append(row)
sc = pd.DataFrame(sc).set_index("run").T
print("\n=== SPEED AND COST (exposure: private 0%, public 100% in both runs) ===")
print(sc.to_string())
sc.to_csv("run_comparison_speed_cost.csv", encoding="utf-8")

res.to_csv("run_comparison.csv", index=False, encoding="utf-8")
ans.to_csv("run_comparison_answers.csv", index=False, encoding="utf-8")
print("\nsaved run_comparison.csv, run_comparison_answers.csv and run_comparison_speed_cost.csv")