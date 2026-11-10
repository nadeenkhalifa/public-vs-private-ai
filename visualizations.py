"""
Makes the 6 thesis figures (PNG, 300 dpi) in figures/ and a table comparing each figure with the articles.

  Fig 1  Quality (Siyan's rule) for both judges next to Siyan et al.'s values   -> Siyan et al. (2025), Tables 4 and 6
  Fig 2  Outcome breakdown (better / tie / undecided / worse)                    -> Siyan et al. (2025), Alg. 1
  Fig 3  SEM-BLEU, SBERT, BLEU                                                   -> Irugalbandara et al. (2024), IV-B3
  Fig 4  Speed: latency per request and per token, 10 measured runs (box plots)  -> Irugalbandara et al. (2024), Figs 11-12
  Fig 5  Privacy: exposure and leakage (dumbbell: private vs public)             -> Siyan et al. (2025), Table 4
  Fig 6  Cost: per 1K tokens + monthly cost vs volume                            -> Irugalbandara et al. (2024), Figs 16-17, Sec. VI-E
  Fig 7  This study vs the articles: who is better on each dimension (scorecard)  -> Siyan et al. (2025); Irugalbandara et al. (2024)

Run from the project folder:  python visualizations.py
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, OUT = "results", "figures"
os.makedirs(OUT, exist_ok=True)

# ---------- style: no grid, light axes, direct labels ----------
# ONE palette for all figures, each colour with a fixed role:
# Theme "Muted academic":
#   blue = private model | orange = public model | dark grey lines = values reported in the articles
#   green / light green / yellow / red = good -> bad levels (Fig 2 outcomes, Fig 5 leakage)
PRIV, PUB = "#4e79a7", "#f28e2b"                  # private = muted blue, public = muted orange (always)
ARTICLE = "#3a3a3a"                               # dark grey: benchmark values from Siyan / Irugalbandara
PRIV_L, PUB_L = "#b8cbe0", "#f9cfa5"              # lighter tints (box fills in Fig 4)
INK, INK2, MUTED, AXIS = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
OUTCOME = {"better": "#59a14f", "tie": "#a6d396", "undecided": "#edc948", "worse": "#e15759"}   # green = pass ... red = fail
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white"})
SOURCE = "Data: 236 judged PUPA-TNB queries per model."

def save(fig, name, note=SOURCE, handles=None, labels=None, ncol=2, bottom=0.13, top=0.92):
    fig.tight_layout(rect=[0, bottom, 1, top])
    if handles:
        fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 0.035), ncol=ncol, fontsize=7.5)
    fig.text(0.0, 0.0, note, fontsize=6.5, color=MUTED, ha="left", va="bottom")
    fig.savefig(f"{OUT}/{name}.png")
    plt.close(fig)
    print("saved", f"{OUT}/{name}.png")

def title(fig, text):
    fig.suptitle(text, x=0.0, y=0.99, ha="left", fontsize=10, fontweight="bold")

def label_bars(ax, bars, fmt="{:.1f}", dy=1.2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy, fmt.format(b.get_height()),
                ha="center", va="bottom", fontsize=7.5, color=INK2)

def swatch(c):
    return plt.Rectangle((0, 0), 1, 1, color=c)

# ---------- load judge results ----------
JUDGES = {"gpt-oss:20b": "", "qwen3:32b": "_qwen"}

def outcomes(path):
    q = pd.read_csv(path).pivot_table(index=["q_index", "deployment"], columns="order", values="verdict").reset_index()
    q["outcome"] = [{(1, 0): "better", (1, 1): "tie", (0, 0): "undecided", (0, 1): "worse"}[(int(a), int(b))]
                    for a, b in zip(q[1], q[2])]
    q["lenient"] = ((q[1] == q[2]) | (q[1] == 1)).astype(int)     # Siyan et al.'s rule
    q["strict"] = q["outcome"].isin(["better", "tie"]).astype(int)
    return q

Q = {j: outcomes(f"{R}/quality_judgments{s}.csv") for j, s in JUDGES.items()}
L = {j: pd.read_csv(f"{R}/leakage_judgments{s}.csv").groupby(["q_index", "deployment"])["verdict"].mean()
        .rename("leak").reset_index() for j, s in JUDGES.items()}
qual = {j: Q[j].groupby("deployment")[["lenient", "strict"]].mean() * 100 for j in JUDGES}

# ---------- Fig 1: quality, with Siyan et al.'s values as labelled benchmark lines ----------
SIYAN_LOCAL, SIYAN_API = 81.5, 88.2      # Siyan et al. (2025): local-only Llama-3.1-8B (Table 6); API-only GPT-4o-mini (Table 4)
fig, ax = plt.subplots(figsize=(7.2, 3.5))
x, w = np.arange(2), 0.34
priv = [qual[j].loc["private", "lenient"] for j in JUDGES]
pub = [qual[j].loc["public", "lenient"] for j in JUDGES]
b1 = ax.bar(x - w / 2 - 0.01, priv, w, color=PRIV)
b2 = ax.bar(x + w / 2 + 0.01, pub, w, color=PUB)
label_bars(ax, list(b1) + list(b2))
for y, c, txt, va, dy, ls in [(SIYAN_API, ARTICLE, f"Siyan et al.: API model (GPT-4o-mini) {SIYAN_API}", "bottom", 1, "--"),
                              (SIYAN_LOCAL, ARTICLE, f"Siyan et al.: local model (Llama-3.1-8B) {SIYAN_LOCAL}", "top", -1, ":")]:
    ax.axhline(y, color=c, lw=1.6, ls=ls, zorder=3)
    ax.text(1.6, y + dy, txt, fontsize=7, color=INK2, va=va, ha="left")
    ax.text(-0.52, y + dy * 1.6, f"{y}", fontsize=7, color=c, va="center", ha="center", fontweight="bold", zorder=4,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=c, lw=0.8))
ax.set_xticks(x, [f"Judge: {j}" for j in JUDGES]); ax.set_xlim(-0.65, 1.55)
ax.set_ylim(0, 108); ax.set_ylabel("Quality (%)")
title(fig, "Figure 1. Quality: share of answers at least as good as the reference (Siyan et al.'s rule)")
save(fig, "fig1_quality", SOURCE + " Grey lines: Siyan et al. (2025), Tables 4 (API model, dashed) and 6 (local model, dotted). Strict quality: see Figure 2.",
     [swatch(PRIV), swatch(PUB), plt.Line2D([], [], color=ARTICLE, lw=1.6, ls="--")],
     ["Private (Mistral 7B)", "Public (Gemini 3.5 Flash-Lite)", "Siyan et al. (2025) benchmark"], ncol=3)

# ---------- Fig 2: outcome breakdown (every segment labelled) ----------
rows = [(j, d) for j in JUDGES for d in ["private", "public"]]
fig, ax = plt.subplots(figsize=(7.2, 2.9))
for i, (j, d) in enumerate(rows):
    share = Q[j][Q[j].deployment == d]["outcome"].value_counts(normalize=True) * 100
    left, small = 0, []
    for o, c in OUTCOME.items():
        v = share.get(o, 0)
        ax.barh(i, v, left=left, color=c, edgecolor="white", linewidth=1.5, height=0.6)
        if v >= 6:     # wide enough: label inside
            ax.text(left + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=7.5,
                    color=INK if o in ("tie", "undecided") else "white")
        elif v > 0:    # too narrow: listed at the end of the bar
            small.append(f"{o} {v:.0f}%")
        left += v
    if small:
        ax.text(101.5, i, " · ".join(small), va="center", fontsize=7, color=INK2, clip_on=False)
ax.set_yticks(range(len(rows)), [f"{d.capitalize()} · judge {j}" for j, d in rows])
ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("Share of answers (%)")
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
title(fig, "Figure 2. Outcome of asking the judge in both orders (Siyan et al., 2025, Alg. 1)")
save(fig, "fig2_outcomes", SOURCE + " Lenient quality = better + tie + undecided; strict = better + tie.",
     [swatch(c) for c in OUTCOME.values()],
     ["Better than reference (yes / no)", "Tie (yes / yes)", "Undecided (no / no)", "Worse (no / yes)"], ncol=4, bottom=0.16)

# ---------- Fig 3: SEM-BLEU ----------
sb = pd.read_csv(f"{R}/sem_bleu_scores.csv").groupby("deployment")[["sbert", "bleu", "sem_bleu"]].mean()
fig, ax = plt.subplots(figsize=(7.2, 3.0))
x = np.arange(3)
b1 = ax.bar(x - w / 2 - 0.01, sb.loc["private"], w, color=PRIV)
b2 = ax.bar(x + w / 2 + 0.01, sb.loc["public"], w, color=PUB)
label_bars(ax, list(b1) + list(b2), "{:.3f}", 0.012)
ax.set_xticks(x, ["SBERT (meaning)", "BLEU (word overlap)", "SEM-BLEU (mean of both)"])
ax.set_ylim(0, 1); ax.set_ylabel("Similarity to reference (0-1)")
title(fig, "Figure 3. SEM-BLEU: similarity to the reference answer (Irugalbandara et al., 2024)")
save(fig, "fig3_sem_bleu", "Data: 236 answers per model with a PUPA reference answer; SBERT all-MiniLM-L6-v2, BLEU with smoothing.",
     [swatch(PRIV), swatch(PUB)], ["Private (Mistral 7B)", "Public (Gemini 3.5 Flash-Lite)"], bottom=0.16)

# ---------- Fig 4: speed (box plots of 10 runs, as Irugalbandara Figs 11-12) ----------
sp_file = f"{R}/speed_test.csv"
sp = pd.read_csv(sp_file) if os.path.exists(sp_file) else pd.concat(
    [pd.read_csv(f) for f in glob.glob(f"{R}/speed_p*.csv")], ignore_index=True)
sp = sp[sp["error"].isna() | (sp["error"].astype(str).str.strip() == "")]
rng = np.random.default_rng(0)
fig, axes = plt.subplots(2, 1, figsize=(7.2, 3.9))
for ax, col, lab, fmt in [(axes[0], "latency_s", "Seconds per request", "{:.1f} s"),
                          (axes[1], "latency_per_token_s", "Seconds per output token", "{:.4f} s")]:
    data = [sp[sp.deployment == d][col].values for d in ["private", "public"]]
    if col == "latency_s":
        data_req = data
    bp = ax.boxplot(data, vert=False, widths=0.6, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1.8), whiskerprops=dict(color=INK2, lw=1),
                    capprops=dict(color=INK2, lw=1))
    for patch, c in zip(bp["boxes"], [PRIV_L, PUB_L]):
        patch.set_facecolor(c); patch.set_edgecolor("none")
    hi = max(np.max(d) for d in data)
    for k, (d, c) in enumerate(zip(data, [PRIV, PUB]), 1):
        ax.scatter(d, k + rng.uniform(-0.1, 0.1, len(d)), s=18, color=c, edgecolor="white", linewidth=0.6, zorder=3)
        ax.text(np.max(d) + hi * 0.03, k, "median " + fmt.format(np.median(d)), va="center", fontsize=7.5, color=INK2)
    ax.set_yticks([1, 2], ["Private\n(Mistral 7B)", "Public\n(Gemini Flash-Lite)"])
    ax.invert_yaxis(); ax.set_xlim(0, hi * 1.3)
    ax.set_xlabel(lab); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
title(fig, "Figure 4. Speed over 10 measured requests (Irugalbandara et al., 2024, Figs 11-12)")
sp_ratio = np.median(data_req[0]) / np.median(data_req[1])
fig.text(0.0, 0.935, f"Article: Mistral-7B was within 1 s of GPT-4 per request and faster per token "
                     f"(Irugalbandara et al., 2024, Sec. VI-C; no exact values reported).\n"
                     f"This study: private is {sp_ratio:.1f}x slower per request (median).",
         ha="left", va="top", fontsize=7.5, color=ARTICLE, style="italic")
save(fig, "fig4_speed", "Same query each time, after 10 warm-up requests. Dots = the 10 runs; "
                        "shaded box = middle 50%; black line = median; whiskers = range.", bottom=0.06, top=0.87)

# ---------- Fig 5: privacy (exposure bars + leakage as 100% stacked bars of answers) ----------
LEAK_LVL = {"none": "#59a14f", "some": "#edc948", "all": "#e15759"}   # green = no PII repeated ... red = all repeated
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1, 2.3]})
ax = axes[0]
eb = ax.bar([0, 1], [0, 100], 0.6, color=[PRIV, PUB])
ax.text(0, 2, "0%", ha="center", va="bottom", fontsize=8, color=INK2)
ax.text(1, 94, "100%", ha="center", va="top", fontsize=8, color="white", fontweight="bold")
ax.axhline(100, color=ARTICLE, lw=1.6, ls="--", zorder=3)
ax.text(-0.45, 103, "Siyan et al.: API-only\nleakage 100 (Table 4)", fontsize=6.8, color=INK2, va="bottom")
ax.set_xticks([0, 1], ["Private", "Public"]); ax.set_ylim(0, 122); ax.set_ylabel("Queries sent outside (%)")
ax.set_title("Exposure", fontsize=9)

ax = axes[1]
rows = [(j, d) for j in JUDGES for d in ["private", "public"]]
for i, (j, d) in enumerate(rows):
    s = L[j][L[j].deployment == d]["leak"]
    parts = {"none": (s == 0).mean() * 100, "some": ((s > 0) & (s < 1)).mean() * 100, "all": (s == 1).mean() * 100}
    left = 0
    for k, v in parts.items():
        ax.barh(i, v, left=left, color=LEAK_LVL[k], edgecolor="white", linewidth=1.5, height=0.6)
        if v >= 6:
            ax.text(left + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=7.5,
                    color="white" if k == "all" else INK)
        left += v
    ax.text(101.5, i, f"mean {s.mean() * 100:.0f}%", va="center", fontsize=7, color=INK2, clip_on=False)
ax.set_yticks(range(len(rows)), [f"{d.capitalize()} · judge {j.split(':')[0]}" for j, d in rows])
ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("Share of answers (%)")
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
ax.set_title("Leakage: how much of the query's PII each answer repeats", fontsize=9)
title(fig, "Figure 5. Privacy: only the public model sends data out; both repeat PII in answers about equally")
save(fig, "fig5_privacy", "Data: 236 judged PUPA-TNB queries per model. Mean = average share of a query's PII units repeated in the answer.",
     [swatch(c) for c in LEAK_LVL.values()],
     ["Repeats none of the PII", "Repeats some of the PII", "Repeats all of the PII"], ncol=3)

# ---------- Fig 6: cost (Irugalbandara Fig 16 + Sec. VI-E break-even) ----------
# cost equations 1-10, computed from this run's results/responses_final.csv (first run: $0.00157, $0.00263, $0.00696, 103,547)
IN_PRICE, OUT_PRICE, GPU_HOUR, HOURS, UTIL = 0.30, 2.50, 0.526, 730, 0.8   # Gemini $/1M tokens; AWS g4dn.xlarge $/h
rf = pd.read_csv(f"{R}/responses_final.csv")
rf = rf[~rf.get("note", pd.Series("", index=rf.index)).fillna("").astype(str).str.startswith("failed")]
pubr, prir = rf[rf.deployment == "public"], rf[rf.deployment == "private"]
api_cost = (pubr.input_tokens.sum() * IN_PRICE + (pubr.output_tokens.sum() + pubr.thinking_tokens.fillna(0).sum()) * OUT_PRICE) / 1e6  # eq. 1
API_PER_QUERY = api_cost / len(pubr)                                       # eq. 2
API_PER_1K = api_cost / pubr.output_tokens.sum() * 1000                     # eq. 3
tok_per_h = prir.output_tokens.sum() / prir.latency_s.sum() * 3600 * UTIL   # eq. 4-5 (throughput x 80%)
PRIV_PER_1K = GPU_HOUR / tok_per_h * 1000                                   # eq. 6
SERVER_MONTH = GPU_HOUR * HOURS                                             # eq. 8 ($383.98)
CAPACITY = tok_per_h * HOURS / prir.output_tokens.mean()                    # eq. 10
BREAK_EVEN = SERVER_MONTH / API_PER_QUERY                                   # eq. 9
print(f"Cost: API ${API_PER_QUERY:.5f}/query, ${API_PER_1K:.5f}/1K; private ${PRIV_PER_1K:.5f}/1K; "
      f"capacity {CAPACITY:,.0f}; break-even {BREAK_EVEN:,.0f}")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={"width_ratios": [1, 2.3]})
ax = axes[0]
bb = ax.bar([0, 1], [PRIV_PER_1K, API_PER_1K], 0.6, color=[PRIV, PUB])
for b, t in zip(bb, [f"${PRIV_PER_1K:.4f}", f"${API_PER_1K:.4f}"]):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.00015, t, ha="center", fontsize=8, color=INK2)
ax.text(0.5, PRIV_PER_1K * 1.2, f"private {PRIV_PER_1K / API_PER_1K:.1f}x\nmore expensive", ha="center", fontsize=7.5, color=INK)
ax.set_xticks([0, 1], ["Private", "Public"]); ax.set_ylim(0, PRIV_PER_1K * 1.5)
ax.set_ylabel("US$ per 1,000 tokens"); ax.set_title("Cost per 1K tokens", fontsize=9)

ax = axes[1]
v = np.linspace(0, 300_000, 301)
ax.plot(v / 1000, v * API_PER_QUERY, color=PUB, lw=2.2)
ax.plot([0, CAPACITY / 1000], [SERVER_MONTH] * 2, color=PRIV, lw=2.2)
ax.plot([CAPACITY / 1000, 300], [SERVER_MONTH] * 2, color=PRIV, lw=1.2, ls="--", alpha=0.7)
ax.scatter([CAPACITY / 1000], [SERVER_MONTH], s=30, color=PRIV, zorder=4)
ax.scatter([BREAK_EVEN / 1000], [SERVER_MONTH], s=40, facecolor="white", edgecolor=INK, zorder=4)
ax.text(CAPACITY / 1000, SERVER_MONTH - 22, f"one T4 is full\n({CAPACITY:,.0f} queries)", ha="center", va="top", fontsize=7, color=INK2)
ax.text(BREAK_EVEN / 1000 + 6, SERVER_MONTH - 40, f"break-even\n{BREAK_EVEN:,.0f}", ha="left", va="top", fontsize=7, color=INK2)
ax.set_xlim(0, 300); ax.set_ylim(0, SERVER_MONTH * 1.5)
ax.set_xlabel("Queries per month (thousands)"); ax.set_ylabel("US$ per month")
ax.set_title("Monthly cost vs volume", fontsize=9)
title(fig, "Figure 6. Cost: public is cheaper at every volume one T4 can serve (Irugalbandara et al., 2024, Sec. VI-E)")
save(fig, "fig6_cost", "Prices: Gemini $0.30/$2.50 per 1M tokens; AWS g4dn.xlarge $0.526 per hour x 730 h; "
                       "80% utilization (Irugalbandara et al., 2024).",
     [plt.Line2D([], [], color=PRIV, lw=2.2), plt.Line2D([], [], color=PRIV, lw=1.2, ls="--", alpha=0.7),
      plt.Line2D([], [], color=PUB, lw=2.2),
      plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK)],
     ["Private: $384/month for one T4", "Private: more queries than one T4 can answer",
      f"Public API: ${API_PER_QUERY:.5f} per query", "Break-even (API cost = one T4)"], ncol=2, bottom=0.16)

# ---------- Fig 7: this study vs the articles (scorecard: who is better on each dimension) ----------
SIYAN_Q = SIYAN_LOCAL / SIYAN_API                          # Siyan et al. (2025): 81.5 / 88.2 (Tables 6 and 4)
IRUG_COST = (5, 29)                                        # Irugalbandara et al. (2024), Sec. VI-E: self-hosting 5x-29x cheaper
lat_ratio = pubr.latency_s.median() / prir.latency_s.median()   # < 1 = private slower
cost_ratio = API_PER_1K / PRIV_PER_1K                           # < 1 = private more expensive
q_g = qual["gpt-oss:20b"].loc["private", "lenient"] / qual["gpt-oss:20b"].loc["public", "lenient"]
q_q = qual["qwen3:32b"].loc["private", "lenient"] / qual["qwen3:32b"].loc["public", "lenient"]

def verdict(r, tol=0.03):          # r = private / public, turned so that > 1 = private better
    return "equal" if abs(r - 1) <= tol else ("private" if r > 1 else "public")

def times(r, good, bad):
    return f"private {r:.1f}x {good}" if r >= 1 else f"private {1 / r:.1f}x {bad}"

pq, gq = qual["gpt-oss:20b"], qual["qwen3:32b"]
lat_pri, lat_pub = prir.latency_s.median(), pubr.latency_s.median()
# (dimension, source, article verdict, article values, study verdict, study values, note if same, note if different)
card = [
    ("Quality\n(judge gpt-oss)", "Siyan et al. (2025)", verdict(SIYAN_Q), f"{SIYAN_LOCAL} vs {SIYAN_API}  ({SIYAN_Q:.0%})",
     verdict(q_g), f"{pq.loc['private', 'lenient']:.1f} vs {pq.loc['public', 'lenient']:.1f}  ({q_g:.0%})",
     "Same direction; the gap is\n" + ("larger" if q_g < SIYAN_Q else "smaller") + " with this judge", "This judge finds no gap"),
    ("Quality\n(judge Qwen)", "Siyan et al. (2025)", verdict(SIYAN_Q), f"{SIYAN_LOCAL} vs {SIYAN_API}  ({SIYAN_Q:.0%})",
     verdict(q_q), f"{gq.loc['private', 'lenient']:.1f} vs {gq.loc['public', 'lenient']:.1f}  ({q_q:.0%})",
     "Same direction as Siyan", "Lenient judge: accepts almost\nall answers from both models"),
    ("Privacy\n(data sent out)", "Siyan et al. (2025)", "private", "0% vs 100% of PII sent",
     "private", "0% vs 100% of PII sent",
     "Local model keeps all data\ninside the organisation", ""),
    ("Speed\n(per request)", "Irugalbandara et al. (2024)", "equal", "Mistral-7B within 1 s of GPT-4",
     verdict(lat_ratio, 0.1), f"{lat_pri:.1f} s vs {lat_pub:.1f} s  ({times(lat_ratio, 'faster', 'slower').replace('private ', '')})",
     "Same as Irugalbandara", "Flash-Lite is far faster than\nGPT-4 (2023); Mistral on one T4"),
    ("Cost\n(per 1K tokens)", "Irugalbandara et al. (2024)", "private", f"private {IRUG_COST[0]}-{IRUG_COST[1]}x cheaper than GPT-4",
     verdict(cost_ratio, 0.1), f"\\${PRIV_PER_1K:.4f} vs \\${API_PER_1K:.4f}  ({times(cost_ratio, 'cheaper', 'higher').replace('private ', '')})",
     "Same as Irugalbandara", "API prices fell sharply since\nGPT-4 (2023)"),
]
# light tile + coloured stripe and heading (easier to read and print than solid blocks)
WIN = {"private": ("#e3ebf4", PRIV, "#2f5d8a", "Private better"),
       "public": ("#fdebdb", PUB, "#b35c12", "Public better"),
       "equal": ("#eeecea", "#9c948f", "#5f5a56", "About equal")}
AGREE = {True: ("#e2f1df", "#3d7a35", "\u2713  Same"), False: ("#fbe1e1", "#b8333a", "\u2717  Different")}
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(8.0, 4.8))
cols = {"art": (1.6, 2.55), "me": (4.2, 2.55), "agree": (6.8, 1.45), "why": (8.35, 2.3)}   # x start, width
ax.set_xlim(0, 10.65); ax.set_ylim(len(card) - 0.48, -1.12); ax.axis("off")
heads = [("art", "Articles", "older models: GPT-4o-mini / GPT-4\nvs Llama-3.1-8B / Mistral-7B"),
         ("me", "This study", "Gemini 3.5 Flash-Lite\nvs Mistral 7B (one T4)"),
         ("agree", "Same result?", ""), ("why", "Why", "")]
for key, txt, sub in heads:
    x0, wd = cols[key]
    xc, ha = (x0 + 0.05, "left") if key == "why" else (x0 + wd / 2, "center")
    ax.text(xc, -0.98, txt, ha=ha, va="center", fontsize=9, fontweight="bold", color=INK)
    if sub:
        ax.text(xc, -0.67, sub, ha=ha, va="center", fontsize=6.3, color=MUTED, linespacing=1.1)
ax.plot([0.1, 10.6], [-0.5, -0.5], color=AXIS, lw=0.8)

def tile(x0, wd, y, bg, stripe, fg, head, sub):
    ax.add_patch(FancyBboxPatch((x0 + 0.05, y - 0.4), wd - 0.1, 0.8, boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor=bg, edgecolor="none"))
    ax.add_patch(plt.Rectangle((x0 + 0.05, y - 0.4), 0.06, 0.8, facecolor=stripe, edgecolor="none"))
    ax.text(x0 + wd / 2 + 0.03, y - 0.13, head, ha="center", va="center", fontsize=8.5, fontweight="bold", color=fg)
    ax.text(x0 + wd / 2 + 0.03, y + 0.17, sub, ha="center", va="center", fontsize=6.5, color=INK2)

n_same = 0
for i, (dim, src, av, ad, mv, md, why_same, why_diff) in enumerate(card):
    ax.text(1.5, i - 0.08, dim, ha="right", va="center", fontsize=8.5, color=INK, linespacing=1.1)
    ax.text(1.5, i + 0.3, src, ha="right", va="center", fontsize=6, color=MUTED, style="italic")
    for key, v, d in [("art", av, ad), ("me", mv, md)]:
        bg, stripe, fg, head = WIN[v]
        tile(*cols[key], i, bg, stripe, fg, head, d)
    same = av == mv; n_same += same
    bg, fg, txt = AGREE[same]
    x0, wd = cols["agree"]
    ax.add_patch(FancyBboxPatch((x0 + 0.06, i - 0.2), wd - 0.12, 0.4, boxstyle="round,pad=0,rounding_size=0.2",
                                facecolor=bg, edgecolor="none"))
    ax.text(x0 + wd / 2, i, txt, ha="center", va="center", fontsize=8, fontweight="bold", color=fg)
    ax.text(cols["why"][0] + 0.05, i, why_same if same else why_diff, ha="left", va="center",
            fontsize=6.8, color=INK2, linespacing=1.2)
    if i < len(card) - 1:
        ax.plot([0.1, 10.6], [i + 0.5, i + 0.5], color="#ecebe6", lw=0.6)
title(fig, f"Figure 7. This study vs the articles: {n_same} of {len(card)} findings match")
save(fig, "fig7_vs_articles", "Values = private vs public (private as % or x of public). About equal = within 3% (quality) "
                              "or 10% (speed, cost). Quality: lenient score (Siyan's rule).\nSpeed: median latency of all answers. "
                              "Cost: equations 1-10. Articles: Siyan et al. (2025), Tables 4 and 6; Irugalbandara et al. (2024), "
                              "Sec. III-B, VI-C and VI-E.",
     [swatch(PRIV), swatch(PUB), swatch("#9c948f")], ["Private better", "Public better", "About equal"],
     ncol=3, bottom=0.13, top=0.97)

# ---------- comparison with the articles ----------
g, qn = qual["gpt-oss:20b"], qual["qwen3:32b"]
lat = sp.groupby("deployment")["latency_s"].agg(["mean", "std"])
comp = pd.DataFrame([
    ["Fig 1 Quality", "Siyan et al. (2025), Tables 4 and 6",
     "API-only GPT-4o-mini 88.2; local-only Llama-3.1-8B 81.5",
     f"gpt-oss: public {g.loc['public','lenient']:.1f}, private {g.loc['private','lenient']:.1f}; "
     f"qwen: public {qn.loc['public','lenient']:.1f}, private {qn.loc['private','lenient']:.1f}",
     "Same direction with gpt-oss (API higher); qwen finds no difference (lenient judge)"],
    ["Fig 2 Outcomes", "Siyan et al. (2025), Alg. 1", "No figure (scoring rule only)",
     "gpt-oss mostly better/worse; qwen mostly tie", "No equivalent"],
    ["Fig 3 SEM-BLEU", "Irugalbandara et al. (2024), Sec. IV-B3", "SEM-BLEU matched human rankings best",
     f"Private {sb.loc['private','sem_bleu']:.3f} vs public {sb.loc['public','sem_bleu']:.3f}",
     "Different: SEM-BLEU ranks the models opposite to the judge"],
    ["Fig 4 Speed", "Irugalbandara et al. (2024), Figs 11-12",
     "Mistral-7B within 1 s of GPT-4 per request; faster per token",
     f"Private {lat.loc['private','mean']:.2f} ± {lat.loc['private','std']:.2f} s vs "
     f"public {lat.loc['public','mean']:.2f} ± {lat.loc['public','std']:.2f} s",
     "Different: private about 2.5x slower in the speed test (5x in the full run); T4 + quantised model vs Flash-Lite"],
    ["Fig 5 Privacy", "Siyan et al. (2025), Table 4", "API-only leakage 100",
     "Exposure: public 100%, private 0%; answer leakage about equal for both models",
     "Same for exposure; answer leakage is an own measure"],
    ["Fig 6 Cost", "Irugalbandara et al. (2024), Figs 16-17, Sec. VI-E", "Self-hosting 5-29x cheaper than GPT-4",
     f"Private {PRIV_PER_1K / API_PER_1K:.1f}x more expensive per token; break-even {BREAK_EVEN:,.0f} queries > capacity {CAPACITY:,.0f}",
     "Opposite: current API prices are far lower than GPT-4's in 2023"],
    ["Fig 7 Summary", "Siyan et al. (2025); Irugalbandara et al. (2024)",
     f"Quality private/public {SIYAN_Q:.2f}; speed about equal; private {IRUG_COST[0]}-{IRUG_COST[1]}x cheaper",
     f"Quality {q_g:.2f} (gpt-oss), {q_q:.2f} (Qwen); private {1 / lat_ratio:.1f}x slower; "
     f"private {1 / cost_ratio:.1f}x more expensive",
     "Quality: same direction (gpt-oss); speed and cost: opposite (newer, cheaper, faster API)"],
], columns=["Figure", "Article", "Their result", "Your result", "Same or different, and why"])
comp.to_csv(f"{OUT}/comparison_with_articles.csv", index=False, encoding="utf-8")
print(f"saved {OUT}/comparison_with_articles.csv")