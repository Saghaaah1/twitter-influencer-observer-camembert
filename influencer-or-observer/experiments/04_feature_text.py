"""Does TEXT content (user bio / tweet text) add HONEST lift under GroupKFold(user)?
Bio words like 'journaliste'/'officiel'/'media' identify social role and generalize
across users (unlike a per-user identity proxy). Measured model-agnostically.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import re
import numpy as np
import pandas as pd
from pandas import json_normalize
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack, csr_matrix
import features as F

NROWS = 120000
print(f"Loading {NROWS} rows...")
df = pd.read_json(config.TRAIN_PATH, lines=True, nrows=NROWS)
df = json_normalize(df.to_dict(orient="records"))
y = df["label"].values.astype(int)
groups = df.get("user.created_at", pd.Series("", index=df.index)).fillna("").astype(str).values

bio = df.get("user.description", pd.Series("", index=df.index)).fillna("").astype(str)
txt = df.apply(F.get_text, axis=1).fillna("").astype(str)

# base honest metadata + source_te (parsed once; TE done fold-safe inside cv)
Xb = F.build_features(df).replace([np.inf, -np.inf], np.nan)
parsed_src = F.parse_source_series(df)
Xb = F.add_static_source_features(parsed_src, df, Xb)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
lgb_params = dict(objective="binary", num_leaves=63, learning_rate=0.05, n_estimators=300,
                  subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)


def text_only_acc(corpus, label):
    """LogReg on TF-IDF of `corpus` alone, GroupKFold. Vectorizer fit per-fold (fold-safe)."""
    oof = np.zeros(len(y))
    for tr, va in sgkf.split(corpus, y, groups):
        vec = TfidfVectorizer(min_df=5, max_features=20000, ngram_range=(1, 2),
                              sublinear_tf=True)
        Xtr = vec.fit_transform(corpus.iloc[tr]); Xva = vec.transform(corpus.iloc[va])
        clf = LogisticRegression(C=1.0, max_iter=200, n_jobs=-1)
        clf.fit(Xtr, y[tr]); oof[va] = clf.predict(Xva)
    print(f"  text-only [{label:8s}] GroupKFold acc: {accuracy_score(y, oof)*100:.2f}%")


def combined_acc(corpus, label, svd_dim=40):
    """Metadata + source_te + SVD(TF-IDF(corpus)) -> LightGBM, GroupKFold (all fold-safe)."""
    from sklearn.decomposition import TruncatedSVD
    oof = np.zeros(len(y))
    for tr, va in sgkf.split(Xb, y, groups):
        smap, sgm = F.compute_source_te(parsed_src.iloc[tr].values, y[tr])
        X = Xb.copy()
        X["source_te"] = F.apply_source_te(parsed_src, smap, sgm)
        X = X.fillna(-1)
        vec = TfidfVectorizer(min_df=5, max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
        Ttr = vec.fit_transform(corpus.iloc[tr]); Tall = vec.transform(corpus)
        svd = TruncatedSVD(n_components=svd_dim, random_state=42)
        Str = svd.fit_transform(Ttr)
        Sall = svd.transform(Tall)
        scols = [f"svd_{i}" for i in range(svd_dim)]
        Xall = X.copy()
        Xall[scols] = Sall
        m = lgb.LGBMClassifier(**lgb_params)
        m.fit(Xall.iloc[tr], y[tr]); oof[va] = m.predict(Xall.iloc[va])
    print(f"  metadata + {label:8s} SVD: {accuracy_score(y, oof)*100:.2f}%")


print("base pos rate %.3f" % y.mean())
print("=== text-only (is there ANY generalizing signal?) ===")
text_only_acc(bio, "bio")
text_only_acc(txt, "tweet")
print("=== metadata baseline vs +text SVD ===")
# baseline (no text) for reference
oof = np.zeros(len(y))
for tr, va in sgkf.split(Xb, y, groups):
    smap, sgm = F.compute_source_te(parsed_src.iloc[tr].values, y[tr])
    X = Xb.copy(); X["source_te"] = F.apply_source_te(parsed_src, smap, sgm); X = X.fillna(-1)
    m = lgb.LGBMClassifier(**lgb_params); m.fit(X.iloc[tr], y[tr]); oof[va] = m.predict(X.iloc[va])
print(f"  metadata baseline       : {accuracy_score(y, oof)*100:.2f}%")
combined_acc(bio, "bio")
combined_acc(txt, "tweet")
