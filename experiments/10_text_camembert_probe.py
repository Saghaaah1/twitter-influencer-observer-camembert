"""CHEAP PROBE: does frozen CamemBERT text embedding add honest signal on top of metadata?

The public leader is "AngryBERT" -> their edge is likely a transformer on the French text.
Our shallow TF-IDF text barely helped (+0.05-0.12%); this checks whether a real language
model's embeddings do better. Frozen (no fine-tuning) is a lower bound: if even frozen
embeddings lift the honest GroupKFold OOF, a fine-tuned CamemBERT will lift it more.

Pipeline: mean-pooled camembert-base embedding of each tweet (cached) -> global PCA(128)
-> concat to the metadata features -> LightGBM under StratifiedGroupKFold -> compare
metadata-only vs metadata+text, raw and per-user-smoothed.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import os
import numpy as np
import pandas as pd
from pandas import json_normalize
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import config
import features as F

HERE = os.path.dirname(os.path.abspath(__file__))
EMB_CACHE = os.path.join(config.OUTPUT_DIR, "text_emb_camembert.npy")
NROWS = None  # full

print("Loading train...")
df = pd.read_json(config.TRAIN_PATH, lines=True, nrows=NROWS)
df = json_normalize(df.to_dict(orient="records"))
y = df["label"].values.astype(int)
groups = df.get("user.created_at", pd.Series("", index=df.index)).fillna("").astype(str).values
texts = df.apply(F.get_text, axis=1).fillna("").astype(str).tolist()
n = len(y)
print(f"rows {n}")

# ---- frozen CamemBERT embeddings (cached) ----
if os.path.exists(EMB_CACHE):
    emb = np.load(EMB_CACHE)
    print(f"loaded cached embeddings {emb.shape}")
else:
    import torch
    from transformers import AutoTokenizer, AutoModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name = "almanach/camembert-base"  # canonical repo (has model.safetensors)
    print(f"embedding with {name} on {dev} ...")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(dev).half().eval()
    bs, maxlen = 256, 96
    out = np.zeros((n, model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, bs):
            batch = texts[i:i + bs]
            enc = tok(batch, padding=True, truncation=True, max_length=maxlen,
                      return_tensors="pt").to(dev)
            hs = model(**enc).last_hidden_state              # [b, t, h]
            mask = enc["attention_mask"].unsqueeze(-1).half()  # [b, t, 1]
            pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)
            out[i:i + bs] = pooled.float().cpu().numpy()
            if (i // bs) % 50 == 0:
                print(f"  {i}/{n}")
    emb = out
    np.save(EMB_CACHE, emb)
    print(f"saved embeddings {emb.shape}")

# global unsupervised PCA (no labels -> not leakage)
emb_pca = PCA(n_components=128, random_state=config.SEED).fit_transform(
    (emb - emb.mean(0)) / (emb.std(0) + 1e-6))
print(f"PCA emb {emb_pca.shape}")

# ---- metadata features ----
Xb = F.build_features(df).replace([np.inf, -np.inf], np.nan)
ps = F.parse_source_series(df)
Xb = F.add_static_source_features(ps, df, Xb)

sgkf = StratifiedGroupKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
params = dict(objective="binary", num_leaves=63, learning_rate=0.05, n_estimators=400,
              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
              random_state=config.SEED, n_jobs=-1, verbose=-1)
emb_cols = [f"emb_{i}" for i in range(emb_pca.shape[1])]


def smooth(p):
    return pd.Series(p).groupby(groups).transform("mean").values


def best_acc(p):
    return max(accuracy_score(y, (p > t).astype(int)) for t in np.arange(0.40, 0.601, 0.005))


def cv(with_text):
    oof = np.zeros(n)
    for tr, va in sgkf.split(Xb, y, groups):
        smap, gm = F.compute_source_te(ps.iloc[tr].values, y[tr])
        X = Xb.copy()
        X["source_te"] = F.apply_source_te(ps, smap, gm)
        if with_text:
            X[emb_cols] = emb_pca
        X = X.fillna(-1)
        m = lgb.LGBMClassifier(**params)
        m.fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof


print("\n=== GroupKFold LightGBM ===")
for label, wt in [("metadata only", False), ("metadata + CamemBERT", True)]:
    oof = cv(wt)
    print(f"{label:24s}: raw {best_acc(oof)*100:.2f}%  | smoothed {best_acc(smooth(oof))*100:.2f}%")
