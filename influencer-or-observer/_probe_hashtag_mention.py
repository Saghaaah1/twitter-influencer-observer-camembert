"""Cheap probe: do hashtag / mentioned-account TARGET ENCODINGS add honest signal?

We used hashtag/mention COUNTS but never WHICH ones. News/campaign hashtags and mentioning
media/brand accounts may indicate role and generalize across users. Fold-safe TE: per-tag
smoothed mean-label computed on the fold's TRAIN rows only, aggregated per tweet (mean & max
over its tags), applied to val/test. Compared under StratifiedGroupKFold (disjoint users).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from pandas import json_normalize
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score
import config, features as F

df = json_normalize(pd.read_json(config.TRAIN_PATH, lines=True).to_dict(orient="records"))
y = df["label"].values.astype(int)
groups = df.get("user.created_at", pd.Series("", index=df.index)).fillna("").astype(str).values


def list_field(col, key):
    s = df.get(col, pd.Series([None]*len(df), index=df.index))
    out = []
    for v in s:
        if isinstance(v, list):
            out.append([str(d.get(key, "")).lower() for d in v if isinstance(d, dict) and d.get(key)])
        else:
            out.append([])
    return out


tags = list_field("entities.hashtags", "text")          # list[str] per row
ments = list_field("entities.user_mentions", "screen_name")
print(f"rows {len(y)} | tweets w/ hashtag {np.mean([len(t)>0 for t in tags])*100:.1f}% | "
      f"w/ mention {np.mean([len(m)>0 for m in ments])*100:.1f}%")

Xb = F.build_features(df).replace([np.inf, -np.inf], np.nan)
ps = F.parse_source_series(df)
Xb = F.add_static_source_features(ps, df, Xb)

sgkf = StratifiedGroupKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
params = dict(objective="binary", num_leaves=63, n_estimators=400, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=42,
              n_jobs=-1, verbose=-1)


def te_map(items_train, y_train, k=20.0):
    """Smoothed mean-label per token over exploded (token,label) train pairs."""
    gm = float(np.mean(y_train))
    cnt, pos = {}, {}
    for toks, lab in zip(items_train, y_train):
        for t in set(toks):
            cnt[t] = cnt.get(t, 0) + 1
            pos[t] = pos.get(t, 0) + lab
    return {t: (pos[t] + k*gm) / (cnt[t] + k) for t in cnt}, gm


def agg(items, m, gm):
    """Per-tweet mean & max TE over its tokens (gm if none)."""
    mean_, max_ = np.full(len(items), gm), np.full(len(items), gm)
    for i, toks in enumerate(items):
        vals = [m.get(t) for t in set(toks) if t in m]
        if vals:
            mean_[i], max_[i] = np.mean(vals), np.max(vals)
    return mean_, max_


def smooth(p):
    return pd.Series(p).groupby(groups).transform("mean").values


def best(p):
    return max(accuracy_score(y, (p > t).astype(int)) for t in np.arange(0.40, 0.601, 0.005))


def cv(with_te):
    oof = np.zeros(len(y))
    for tr, va in sgkf.split(Xb, y, groups):
        smap, sgm = F.compute_source_te(ps.iloc[tr].values, y[tr])
        X = Xb.copy(); X["source_te"] = F.apply_source_te(ps, smap, sgm)
        if with_te:
            hm, hgm = te_map([tags[i] for i in tr], y[tr])
            mm, mgm = te_map([ments[i] for i in tr], y[tr])
            X["htag_te_mean"], X["htag_te_max"] = agg(tags, hm, hgm)
            X["ment_te_mean"], X["ment_te_max"] = agg(ments, mm, mgm)
        X = X.fillna(-1)
        m = lgb.LGBMClassifier(**params).fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof


for label, wt in [("metadata only", False), ("metadata + htag/mention TE", True)]:
    o = cv(wt)
    print(f"{label:30s}: raw {best(o)*100:.2f}%  | smoothed {best(smooth(o))*100:.2f}%")
