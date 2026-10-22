"""
SEM-BLEU: second quality measure, following Irugalbandara et al. (2024), Section IV-B3.
SEM-BLEU = mean of (1) SBERT semantic similarity (Reimers & Gurevych, 2019) and
(2) BLEU word overlap (Papineni et al., 2002), each comparing a model answer
with PUPA's reference answer (target_response). No AI judge is involved.

Run from the project folder:  python sem_bleu.py
Install once:  pip install sentence-transformers nltk langdetect
"""
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0                        # makes language detection repeatable
SBERT_MODEL = "all-MiniLM-L6-v2"                # standard SBERT model (Reimers & Gurevych, 2019); light enough for a laptop CPU

# ---------- load answers + reference answers (matched by q_index) ----------
resp = pd.read_csv("results/responses_final.csv", encoding="utf-8")
pupa = pd.read_csv("data/pupa_tnb_train.csv").reset_index(drop=True)
pupa["q_index"] = pupa.index
df = resp.merge(pupa[["q_index", "target_response"]], on="q_index", how="left")
df = df[df["target_response"].notna() & df["response"].notna()].copy()   # drops q217 (no reference)
print(len(df), "answers scored (expected 472 = 236 per model)")

def lang(text):
    try:
        return detect(str(text)[:1000])
    except Exception:
        return "unknown"
df["ref_lang"] = df["target_response"].apply(lang)

# ---------- SBERT: cosine similarity of sentence embeddings ----------
model = SentenceTransformer(SBERT_MODEL)
emb_a = model.encode(df["response"].astype(str).tolist(), convert_to_tensor=True, show_progress_bar=True)
emb_r = model.encode(df["target_response"].astype(str).tolist(), convert_to_tensor=True, show_progress_bar=True)
df["sbert"] = util.cos_sim(emb_a, emb_r).diagonal().cpu().numpy().clip(0, 1)

# ---------- BLEU: n-gram word overlap with the reference ----------
smooth = SmoothingFunction().method1           # avoids zero scores for short texts
df["bleu"] = [sentence_bleu([str(r).split()], str(a).split(), smoothing_function=smooth)
              for a, r in zip(df["response"], df["target_response"])]

# ---------- SEM-BLEU = mean of the two ----------
df["sem_bleu"] = (df["sbert"] + df["bleu"]) / 2

df[["q_index", "deployment", "ref_lang", "sbert", "bleu", "sem_bleu"]].to_csv(
    "results/sem_bleu_scores.csv", index=False, encoding="utf-8")

def table(d):
    return d.groupby("deployment")[["sbert", "bleu", "sem_bleu"]].agg(["mean", "std"]).round(3)

print("\nALL ANSWERS (mean, SD)\n", table(df))
print("\nENGLISH REFERENCES ONLY (mean, SD)\n", table(df[df["ref_lang"] == "en"]))
print("\nAnswers per language of the reference:\n", df[df.deployment == "private"]["ref_lang"].value_counts())