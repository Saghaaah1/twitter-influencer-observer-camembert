"""Final honest-feature sweep: do curated bio role-words + text-style ratios + emoji/digit
ratios + quoted completeness collectively break the ~83.8% GroupKFold ceiling?
Add features one GROUP at a time (additive) so we see which group, if any, actually lifts.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
groups = df.get("user.created_at", pd.Series("", index=df.index)).fillna("").astype(str).values

Xb = F.build_features(df).replace([np.inf, -np.inf], np.nan)
parsed_src = F.parse_source_series(df)
Xb = F.add_static_source_features(parsed_src, df, Xb)

bio = df.get("user.description", pd.Series("", index=df.index)).fillna("").astype(str)
txt = df.apply(F.get_text, axis=1).fillna("").astype(str)

# --- group 1: curated bio role-word flags (FR+EN) ---
ROLE_WORDS = ("journ", "reporter", "media", "média", "presse", "press", "news", "actu",
              "officiel", "official", "président", "ministre", "député", "sénateur",
              "maire", "politiq", "parti ", "syndicat", "communicat", "community manager",
              "rédac", "radio", "télé", " tv", "chaîne", "magazine", "blog", "influence",
              "marketing", "digital", "entrepreneur", "ceo", "fondateur", "founder",
              "directeur", "auteur", "écrivain", "artist", "musicien", "chanteur",
              "comédien", "acteur", "sport", "football", "podcast", "youtube", "twitch")
bio_l = bio.str.lower()
g1 = pd.DataFrame(index=df.index)
g1["bio_role_words"] = bio_l.apply(lambda s: sum(1 for w in ROLE_WORDS if w in s))
g1["bio_word_count"] = bio.str.split().str.len().fillna(0)
g1["bio_emoji_ratio"] = bio.apply(lambda s: sum(1 for c in s if ord(c) > 0x2100) / max(len(s), 1))
g1["bio_digit_ratio"] = bio.apply(lambda s: sum(c.isdigit() for c in s) / max(len(s), 1))
g1["bio_upper_ratio"] = bio.apply(lambda s: sum(c.isupper() for c in s) / max(len(s), 1))

# --- group 2: tweet text-style ratios ---
g2 = pd.DataFrame(index=df.index)
wc = txt.str.split().str.len().replace(0, np.nan)
g2["digit_ratio"] = txt.apply(lambda s: sum(c.isdigit() for c in s) / max(len(s), 1))
g2["emoji_ratio"] = txt.apply(lambda s: sum(1 for c in s if ord(c) > 0x2100) / max(len(s), 1))
g2["mention_per_word"] = (txt.str.count(r"@\w+") / wc).fillna(0)
g2["hashtag_per_word"] = (txt.str.count(r"#\w+") / wc).fillna(0)
g2["url_per_word"] = (txt.str.count(r"http") / wc).fillna(0)
g2["nonascii_ratio"] = txt.apply(lambda s: sum(1 for c in s if ord(c) > 127) / max(len(s), 1))

# --- group 3: quoted completeness / richer ---
g3 = pd.DataFrame(index=df.index)
qv = df.get("quoted_status.user.verified", pd.Series(np.nan, index=df.index))
g3["quoted_user_verified"] = qv.map({True: 1.0, False: 0.0}).fillna(-1)
g3["quoted_has_url"] = df.get("quoted_status.user.url", pd.Series(np.nan, index=df.index)).notna().astype(int)
g3["quoted_is_reply"] = df.get("quoted_status.in_reply_to_status_id", pd.Series(np.nan, index=df.index)).notna().astype(int)
g3["quoted_lang_fr"] = (df.get("quoted_status.lang", pd.Series("", index=df.index)) == "fr").astype(int)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
params = dict(objective="binary", num_leaves=63, learning_rate=0.05, n_estimators=300,
              subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)


def cv(extra):
    oof = np.zeros(len(y))
    for tr, va in sgkf.split(Xb, y, groups):
        smap, sgm = F.compute_source_te(parsed_src.iloc[tr].values, y[tr])
        X = Xb.copy()
        X["source_te"] = F.apply_source_te(parsed_src, smap, sgm)
        if extra is not None:
            X = pd.concat([X, extra], axis=1)
        X = X.fillna(-1)
        m = lgb.LGBMClassifier(**params)
        m.fit(X.iloc[tr], y[tr]); oof[va] = m.predict(X.iloc[va])
    return accuracy_score(y, oof)


base = cv(None)
print(f"\nbaseline                         : {base*100:.2f}%")
print(f"+ g1 bio role/style              : {cv(g1)*100:.2f}%  ({(cv(g1)-base)*100:+.2f})")
print(f"+ g2 tweet style ratios          : {cv(g2)*100:.2f}%  ({(cv(g2)-base)*100:+.2f})")
print(f"+ g3 quoted richer               : {cv(g3)*100:.2f}%  ({(cv(g3)-base)*100:+.2f})")
allg = pd.concat([g1, g2, g3], axis=1)
print(f"+ ALL groups                     : {cv(allg)*100:.2f}%  ({(cv(allg)-base)*100:+.2f})")
