"""Throwaway validation: does a fold-safe URL-domain TE (+ media/place/shortener flags)
add HONEST lift under GroupKFold(user)? Measured with LightGBM (model-agnostic).

Baseline   = build_features + fold-safe source_te
Candidate  = baseline + url_domain_te + has_media + has_shortener + n_domains + has_place
If the candidate beats baseline under GroupKFold (disjoint users ~ public LB), the URL
domain carries genuine, generalizing signal — not a per-user identity proxy.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import config
import re
import numpy as np
import pandas as pd
from pandas import json_normalize
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score
import features as F

NROWS = 120000
print(f"Loading {NROWS} rows...")
df = pd.read_json(config.TRAIN_PATH, lines=True, nrows=NROWS)
df = json_normalize(df.to_dict(orient="records"))
y = df["label"].values.astype(int)

# Proxy user group key (created_at is per-account constant; name absent here).
created = df.get("user.created_at", pd.Series("", index=df.index)).fillna("").astype(str)
groups = created.values

# --- base honest features + parsed source ---
Xb = F.build_features(df).replace([np.inf, -np.inf], np.nan)
parsed_src = F.parse_source_series(df)
Xb = F.add_static_source_features(parsed_src, df, Xb)

# --- candidate raw signals ---
SHORTENERS = ("bit.ly", "dlvr.it", "ift.tt", "buff.ly", "ow.ly", "paper.li",
              "trib.al", "tinyurl", "is.gd", "shar.es", "ln.is")

def primary_domain(urlist):
    """First non-twitter expanded-url domain (the content link), else 'none'."""
    if not isinstance(urlist, list):
        return "none"
    doms = []
    for u in urlist:
        ex = (u.get("expanded_url") or u.get("display_url") or "") if isinstance(u, dict) else ""
        m = re.search(r"https?://([^/]+)", str(ex))
        d = m.group(1).lower() if m else str(ex).split("/")[0].lower()
        if d.startswith("www."):
            d = d[4:]
        if d:
            doms.append(d)
    for d in doms:
        if "twitter.com" not in d:
            return d
    return doms[0] if doms else "none"

urls = df.get("entities.urls", pd.Series([[]] * len(df), index=df.index))
dom = urls.apply(primary_domain)
n_dom = urls.apply(lambda x: len(x) if isinstance(x, list) else 0)
has_short = dom.apply(lambda d: int(any(s in d for s in SHORTENERS)))

def mlen(x):
    return len(x) if isinstance(x, list) else 0
media = (df.get("extended_entities.media", pd.Series(np.nan, index=df.index)).apply(mlen)
         + df.get("entities.media", pd.Series(np.nan, index=df.index)).apply(mlen))
has_media = (media > 0).astype(int)
has_place = df.get("place.country_code", pd.Series(np.nan, index=df.index)).notna().astype(int)

params = dict(objective="binary", num_leaves=63, learning_rate=0.05, n_estimators=300,
              subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)


def te_fit(cat_tr, y_tr, smoothing=20.0):
    s = pd.Series(np.asarray(cat_tr)); yt = pd.Series(np.asarray(y_tr), index=s.index)
    gm = float(yt.mean()); g = yt.groupby(s.values)
    sm = (g.count() * g.mean() + smoothing * gm) / (g.count() + smoothing)
    return sm.to_dict(), gm


def cv(add_candidate):
    oof = np.zeros(len(y))
    for tr, va in sgkf.split(Xb, y, groups):
        # fold-safe source TE
        smap, sgm = F.compute_source_te(parsed_src.iloc[tr].values, y[tr])
        Xtr = Xb.copy()
        Xtr["source_te"] = F.apply_source_te(parsed_src, smap, sgm)
        if add_candidate:
            dmap, dgm = te_fit(dom.iloc[tr].values, y[tr])
            Xtr["url_domain_te"] = pd.Series(np.asarray(dom)).map(dmap).fillna(dgm).values
            Xtr["n_url_domains"] = n_dom.values
            Xtr["has_shortener"] = has_short.values
            Xtr["has_media"] = has_media.values
            Xtr["has_place"] = has_place.values
        Xtr = Xtr.fillna(-1)
        m = lgb.LGBMClassifier(**params)
        m.fit(Xtr.iloc[tr], y[tr])
        oof[va] = m.predict(Xtr.iloc[va])
    return accuracy_score(y, oof)


print("Running GroupKFold(user) LightGBM...")
a_base = cv(False)
a_cand = cv(True)
print(f"\n=== GroupKFold OOF (disjoint users ~ public LB) ===")
print(f"baseline (source_te only)        : {a_base*100:.2f}%")
print(f"+ url_domain_te/media/place/short: {a_cand*100:.2f}%")
print(f"HONEST lift                      : {(a_cand-a_base)*100:+.2f}%")
