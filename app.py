# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
from typing import List
import os

# Optional transformers (keberadaan tidak penting)
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

app = FastAPI(title="CarbonSwap Recommendation API")

# WEIGHTS: gunakan nama fitur yang sesuai CSV Anda
WEIGHTS = {
    "carbon_absorbed": 0.25,
    "annual_survival_rate": 0.20,
    "area_ha": 0.15,
    "trees_planted": 0.10,
    "people_involved": 0.08,
    "review_rating": 0.07,
    "review_count": 0.05,
    "nlp_sentiment": 0.10
}

CSV_PATH = os.getenv("LOCATIONS_CSV", "./data/location.csv")

# ---------- helpers ----------
def parse_int_like(s):
    if pd.isna(s):
        return 0
    if isinstance(s, (int, float)) and not np.isnan(s):
        return int(s)
    s = str(s).strip()
    if s == "":
        return 0
    s = s.replace(",", "")
    m = re.match(r"^([0-9]*\.?[0-9]+)\s*[kK].*$", s)
    if m:
        return int(float(m.group(1)) * 1000)
    m3 = re.search(r"([0-9]*\.?[0-9]+)", s)
    if m3:
        try:
            return int(float(m3.group(1)))
        except:
            return 0
    return 0

def parse_numeric_allow_k(s):
    if pd.isna(s):
        return 0.0
    if isinstance(s, (int, float)) and not np.isnan(s):
        return float(s)
    st = str(s).strip()
    if st == "":
        return 0.0
    for u in ["Kg", "kg", "CO2eq", "CO2", "mm"]:
        st = st.replace(u, "")
    if "%" in st:
        try:
            return float(st.replace("%", "").strip()) / 100.0
        except:
            pass
    m = re.match(r"^([0-9]*\.?[0-9]+)\s*[kK].*$", st)
    if m:
        return float(m.group(1)) * 1000.0
    st = st.replace(",", "")
    m2 = re.search(r"([0-9]*\.?[0-9]+)", st)
    if m2:
        try:
            return float(m2.group(1))
        except:
            return 0.0
    return 0.0

def safe_to_float(x):
    try:
        return float(x)
    except:
        return 0.0

def pick_column_value(row: pd.Series, candidates: List[str], default=""):
    """
    Return first non-empty value from candidates present in row (stringify non-strings).
    """
    for c in candidates:
        if c in row.index and pd.notna(row.get(c)):
            v = row.get(c)
            if isinstance(v, str):
                if v.strip() == "":
                    continue
                return v
            if v is not None:
                return str(v)
    return default

# ---------- sentiment (optional HF, fallback rule-based) ----------
SENT_MODEL_NAME = os.getenv("HF_SENT_MODEL", "indolem/indobert-base-uncased-sentiment")
sentiment_pipe = None
if HF_AVAILABLE:
    try:
        sentiment_pipe = pipeline("sentiment-analysis", model=SENT_MODEL_NAME)
    except Exception:
        sentiment_pipe = None

def avg_sentiment_for_texts(texts: List[str]) -> float:
    vals = []
    if sentiment_pipe is None:
        # simple rule-based fallback using common positive/negative tokens (EN + ID)
        pos = ["good","great","active","easy","suitable","success","maintained","clean","support","baik","bagus"]
        neg = ["dead","bad","difficult","erosion","damaged","abandoned","trash","buruk","rusak"]
        for t in texts:
            if not isinstance(t, str) or t.strip() == "":
                continue
            low = t.lower()
            score = 0.5
            for p in pos:
                if p in low:
                    score += 0.18
            for n in neg:
                if n in low:
                    score -= 0.18
            vals.append(max(0.0, min(1.0, score)))
    else:
        for t in texts:
            if not isinstance(t, str) or t.strip() == "":
                continue
            try:
                out = sentiment_pipe(t[:512])
                label = out[0]['label'].lower()
                sc = float(out[0]['score'])
                if 'pos' in label or 'positive' in label:
                    vals.append(sc)
                else:
                    vals.append(1.0 - sc)
            except:
                vals.append(0.5)
    if len(vals) == 0:
        return 0.5
    return float(sum(vals) / len(vals))

# ---------- core scoring ----------
def compute_scores(df: pd.DataFrame, include_nlp: bool = True) -> pd.DataFrame:
    dfc = df.copy()

    # canonical numeric columns using your CSV header names (english primary, id fallback)
    dfc["area_ha"] = dfc.get("area_ha", dfc.get("luas_ha", 0)).apply(parse_numeric_allow_k)
    dfc["people_involved"] = dfc.get("people_involved", dfc.get("orang_terlibat", 0)).apply(parse_int_like)
    dfc["trees_planted"] = dfc.get("trees_planted", dfc.get("pohon_tertanam", 0)).apply(parse_int_like)
    # CSV has 'carbon_absorbed' per your header
    dfc["carbon_absorbed"] = dfc.get("carbon_absorbed", dfc.get("carbon_sequestered", dfc.get("karbon_terserap", 0))).apply(parse_numeric_allow_k)
    dfc["annual_survival_rate"] = dfc.get("annual_survival_rate", 0).apply(parse_numeric_allow_k)
    dfc["annual_survival_rate"] = dfc["annual_survival_rate"].apply(lambda v: v/100.0 if v>1 else v)

    # rating and counts (header: review_rating, review_count)
    dfc["review_rating"] = dfc.get("review_rating", dfc.get("rating", dfc.get("rating_ulasan", 0))).apply(lambda x: safe_to_float(parse_numeric_allow_k(x)))
    dfc["review_count"] = dfc.get("review_count", dfc.get("jumlah_ulasan", 0)).apply(parse_int_like)

    # NLP sentiment: detect 'ulasan_' (Indonesian) or 'review_' (English)
    review_cols = [c for c in dfc.columns if c.lower().startswith("ulasan") or c.lower().startswith("review")]
    nlp_scores = []
    for _, row in dfc.iterrows():
        texts = []
        for c in review_cols:
            v = row.get(c)
            if isinstance(v, str) and v.strip():
                texts.append(v.strip())
        nlp_scores.append(avg_sentiment_for_texts(texts))
    dfc["nlp_sentiment"] = nlp_scores

    # pick features that exist in dfc according to WEIGHTS keys
    features = [k for k in WEIGHTS.keys() if k in dfc.columns]

    if not features:
        raise ValueError(f"No numeric features found for scoring. Expected keys: {list(WEIGHTS.keys())}. CSV columns: {list(dfc.columns)}")

    # ensure numeric
    for f in features:
        dfc[f] = pd.to_numeric(dfc[f], errors='coerce').fillna(0.0)

    X = dfc[features].astype(float).values
    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X)
    df_scaled = pd.DataFrame(Xs, columns=features, index=dfc.index)

    score = np.zeros(len(dfc))
    for key, w in WEIGHTS.items():
        if key in df_scaled.columns:
            score += df_scaled[key].values * w

    dfc["score"] = score
    dfc_sorted = dfc.sort_values("score", ascending=False).reset_index(drop=True)
    return dfc_sorted

# ---------- Pydantic model ----------
class RecommendationItem(BaseModel):
    locName: str = ""
    locDesc: str = ""
    locImage: str = ""
    province: str = ""
    treeType: str = ""
    score: float = 0.0

# ---------- endpoints ----------
@app.get("/recommendations", response_model=List[RecommendationItem])
def get_recommendations(top_k: int = 10, include_nlp: bool = True):
    if not os.path.exists(CSV_PATH):
        return []
    df = pd.read_csv(CSV_PATH)
    df_sorted = compute_scores(df, include_nlp=include_nlp)

    # candidates tuned to your CSV header
    name_cols = ["location_name", "name", "locName", "title", "site_name", "nama_lokasi", "site", "id"]
    desc_cols = ["description", "locDesc", "deskripsi", "desc", "details"]
    image_cols = ["image", "image_url", "locImage", "photo"]
    province_cols = ["province", "provinsi", "prov"]
    tree_cols = ["seed_type", "treeType", "jenis_bibit", "jenis", "seed", "species"]

    # create friendly columns so selection won't KeyError
    df_sorted["locName"] = df_sorted.apply(lambda r: pick_column_value(r, name_cols, default=""), axis=1)
    df_sorted["locDesc"] = df_sorted.apply(lambda r: pick_column_value(r, desc_cols, default=""), axis=1)
    df_sorted["locImage"] = df_sorted.apply(lambda r: pick_column_value(r, image_cols, default=""), axis=1)
    df_sorted["province"] = df_sorted.apply(lambda r: pick_column_value(r, province_cols, default=""), axis=1)
    df_sorted["treeType"] = df_sorted.apply(lambda r: pick_column_value(r, tree_cols, default=""), axis=1)

    # fallback fill for empty names (so Postman never shows empty locName)
    for i, _ in df_sorted.iterrows():
        if not df_sorted.at[i, "locName"]:
            # prefer id if exists
            idx_val = df_sorted.at[i, "id"] if "id" in df_sorted.columns else None
            if pd.notna(idx_val) and str(idx_val).strip() != "":
                df_sorted.at[i, "locName"] = f"Site {int(idx_val)}"
            else:
                df_sorted.at[i, "locName"] = f"Site {i+1}"

    top = df_sorted.head(top_k)[["locName", "locDesc", "locImage", "province", "treeType", "score"]].copy()
    top["score"] = top["score"].round(4)
    return top.to_dict(orient="records")

@app.get("/ranked_full")
def get_full_ranked(include_nlp: bool = True):
    if not os.path.exists(CSV_PATH):
        return {"error": "no csv"}
    df = pd.read_csv(CSV_PATH)
    df_sorted = compute_scores(df, include_nlp=include_nlp)

    # same friendly mapping
    name_cols = ["location_name", "name", "locName", "title", "site_name", "nama_lokasi", "site", "id"]
    desc_cols = ["description", "locDesc", "deskripsi", "desc", "details"]
    image_cols = ["image", "image_url", "locImage", "photo"]
    province_cols = ["province", "provinsi", "prov"]
    tree_cols = ["seed_type", "treeType", "jenis_bibit", "jenis", "seed", "species"]

    df_sorted["locName"] = df_sorted.apply(lambda r: pick_column_value(r, name_cols, default=""), axis=1)
    df_sorted["locDesc"] = df_sorted.apply(lambda r: pick_column_value(r, desc_cols, default=""), axis=1)
    df_sorted["locImage"] = df_sorted.apply(lambda r: pick_column_value(r, image_cols, default=""), axis=1)
    df_sorted["province"] = df_sorted.apply(lambda r: pick_column_value(r, province_cols, default=""), axis=1)
    df_sorted["treeType"] = df_sorted.apply(lambda r: pick_column_value(r, tree_cols, default=""), axis=1)

    for i, _ in df_sorted.iterrows():
        if not df_sorted.at[i, "locName"]:
            idx_val = df_sorted.at[i, "id"] if "id" in df_sorted.columns else None
            if pd.notna(idx_val) and str(idx_val).strip() != "":
                df_sorted.at[i, "locName"] = f"Site {int(idx_val)}"
            else:
                df_sorted.at[i, "locName"] = f"Site {i+1}"

    df_sorted["score"] = df_sorted["score"].round(4)
    return df_sorted.to_dict(orient="records")