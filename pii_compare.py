"""
Compares PUPA's PII units (Siyan et al., 2025; extracted with gpt-4-turbo) with a second extraction
by Qwen3 32B using Siyan's own prompt and code (thesis_pii_extraction.ipynb), then checks whether the
leakage results change when only the units both extractions agree on are used.

Recall / precision follow Singhal et al. (2024), cited by Siyan et al. for the redaction method.

Run from the project folder:  python pii_compare.py
Needs: data/pupa_tnb_train.csv, results/pii_extraction_qwen.csv,
       results/leakage_judgments.csv (+ results/leakage_judgments_qwen.csv if present)
"""
import os, re
import pandas as pd

def norm(s):
    s = re.sub(r"\s+", " ", str(s).lower()).strip()
    s = re.sub(r"^(\d+[.)]|[-*•])\s+", "", s)          # drop list numbering added by the LLM ("3. pidilite")
    return s.strip(" .,;:!?'\"()[]<>-")

def split_units(s):
    return [] if pd.isna(s) else sorted(set(u.strip() for u in str(s).split("||") if u.strip()))

def same(a, b):  # same item if equal, or one contains the other (at least 3 characters)
    a, b = norm(a), norm(b)
    return a == b or (min(len(a), len(b)) >= 3 and (a in b or b in a))

# ---------- load ----------
pupa = pd.read_csv("data/pupa_tnb_train.csv").reset_index(drop=True)
pupa["q_index"] = pupa.index
pupa = pupa[pupa["pii_units"].notna()]
qw = pd.read_csv("results/pii_extraction_qwen.csv")
df = pupa[["q_index", "user_query", "pii_units", "predicted_category"]].merge(qw, on="q_index", how="left")
print(f"Queries: {len(df)} in PUPA with PII units, {qw['q_index'].nunique()} processed by Qwen")

# ---------- unit-level matching ----------
rows, per_q = [], []
for _, r in df.iterrows():
    P, Q = split_units(r["pii_units"]), split_units(r["units_qwen"])
    p_hit = [any(same(p, q) for q in Q) for p in P]
    q_hit = [any(same(q, p) for p in P) for q in Q]
    rows += [{"q_index": r.q_index, "source": "pupa", "unit": u, "found_by_other": h} for u, h in zip(P, p_hit)]
    rows += [{"q_index": r.q_index, "source": "qwen", "unit": u, "found_by_other": h} for u, h in zip(Q, q_hit)]
    per_q.append({"q_index": r.q_index, "category": r["predicted_category"], "n_pupa": len(P), "n_qwen": len(Q),
                  "all_pupa_found": all(p_hit) if P else True,
                  "suspect": isinstance(r["redacted"], str) and len(r["redacted"]) > 1.5 * len(str(r["user_query"]))})
units = pd.DataFrame(rows)
pq = pd.DataFrame(per_q)
units.to_csv("results/pii_compare_units.csv", index=False, encoding="utf-8")

pu, qu = units[units.source == "pupa"], units[units.source == "qwen"]
recall, precision = pu["found_by_other"].mean(), qu["found_by_other"].mean()
f1 = 2 * recall * precision / (recall + precision)

print("\n=== 1. EXTRACTION OVERVIEW ===")
print("How Qwen's spans were recovered:\n", qw["method"].value_counts().to_string())
print(f"Average PII units per query: PUPA {pq.n_pupa.mean():.2f} (paper, Table 3: 3.01) | Qwen {pq.n_qwen.mean():.2f}")
print(f"Queries where Qwen found no PII: {(pq.n_qwen == 0).sum()}")
print(f"Queries where Qwen's output looks cut off / looped: {pq.suspect.sum()} -> q_index {pq[pq.suspect].q_index.tolist()}")

print("\n=== 2. AGREEMENT ===")
print(f"PUPA units also found by Qwen (recall):   {recall*100:.1f}%  ({pu.found_by_other.sum()} of {len(pu)})")
print(f"Qwen units also in PUPA (precision):      {precision*100:.1f}%  ({qu.found_by_other.sum()} of {len(qu)})")
print(f"F1:                                       {f1*100:.1f}%")
print(f"Queries where Qwen found ALL PUPA units:  {pq.all_pupa_found.mean()*100:.1f}%")
cat = units.merge(pq[["q_index", "category"]], on="q_index").groupby(["category", "source"])["found_by_other"].mean().unstack() * 100
print("\nBy category (pupa = recall, qwen = precision):\n", cat.round(1).to_string())
print("\nExamples of PUPA units Qwen did NOT find:\n", pu[~pu.found_by_other].head(10)[["q_index", "unit"]].to_string(index=False))
print("\nExamples of Qwen units NOT in PUPA (possible missed PII or over-redaction):\n",
      qu[~qu.found_by_other].head(10)[["q_index", "unit"]].to_string(index=False))

# ---------- sensitivity of leakage ----------
def leak_table(l):
    a = l.groupby(["q_index", "deployment"])["verdict"].mean().rename("leak").reset_index()
    return a.groupby("deployment")["leak"].agg(
        answers="count", mean_pct=lambda x: x.mean() * 100,
        any_pct=lambda x: (x > 0).mean() * 100, full_pct=lambda x: (x == 1).mean() * 100).round(1)

confirmed = pu.rename(columns={"found_by_other": "confirmed"})[["q_index", "unit", "confirmed"]]
print("\n=== 3. LEAKAGE: ALL PUPA UNITS vs UNITS BOTH EXTRACTIONS AGREE ON ===")
for judge, path in {"gpt-oss:20b": "results/leakage_judgments.csv",
                    "qwen3:32b": "results/leakage_judgments_qwen.csv"}.items():
    if not os.path.exists(path):
        print(f"\n[{judge}] {path} not found - skipped"); continue
    l = pd.read_csv(path).merge(confirmed, on=["q_index", "unit"], how="left")
    print(f"\n[{judge}]  verdicts matched to a unit: {l.confirmed.notna().mean()*100:.1f}%")
    print("A) All PUPA units (reported result)\n", leak_table(l).to_string())
    print("C) Agreed units only\n", leak_table(l[l.confirmed == True]).to_string())