"""
Feature engineering for the deep-MLP pipeline.

This module is self-contained so the repo does not depend on any parent-directory
scripts: build_features (and its helpers parse_date / safe_len / get_text) and the
parse_source / pro_tools logic are vendored in here directly.

LEAKAGE POLICY
--------------
Only "honest" features are produced — features legitimately available per tweet.
The main user object has user.id / id_str / screen_name STRIPPED, so any cross-row
"user aggregation" or per-user target encoding would actually aggregate across the
whole dataset = leakage (this is what inflated sub10 to 94.4% and sub21 to 97.48%).

The single target-dependent feature here is a *fold-safe* source target encoding:
compute_source_te is only ever called on a fold's training rows (see train.py),
and the resulting map is applied to that fold's val rows and the test set.

Identity-proxy columns (profile colors / background image) listed in
config.BANNED_COLUMNS are never read.
"""
import re
from datetime import datetime

import numpy as np
import pandas as pd

import config

REF_DATE = datetime(*config.REF_DATE_YMD)


# ---------------------------------------------------------------------------
# Module constants for the feature upgrade
# ---------------------------------------------------------------------------
# Constant (std=0) columns measured on the real data — pure dead weight for an MLP
# (a zero-variance column wastes input weight + a BatchNorm channel). Dropped at the
# END of build_features (some are intermediate inputs to other features).
#   - lang is 100% 'fr' across the whole file -> lang_enc constant
#   - all tweet-level engagement counts are 0 (freshly-streamed corpus)
#   - the listed booleans are uniform
DEAD_FEATURES = [
    "lang_enc", "user_default_profile_image", "user_protected",
    "user_contributors_enabled", "retweet_count", "favorite_count",
    "quote_count", "reply_count", "engagement",
    "log_rt", "log_fav_tweet", "log_engagement",
]

# Quoted-tweet columns that are ABSENT in ~65% of rows. Instead of the -1 sentinel
# (which pollutes StandardScaler since it mixes with magnitudes up to millions), these
# are left as NaN by build_features and median-imputed FOLD-SAFE in train.py.
QUOTED_IMPUTE_COLS = [
    "quoted_followers", "quoted_friends", "quoted_listed", "quoted_statuses",
    "quoted_rt", "quoted_fav", "quoted_reply", "quoted_engagement",
    "quoted_user_age_days", "quoted_fol_ratio", "quoted_tweets_per_day",
    "quoted_listed_per_tweet", "log_q_fol", "log_q_fri", "log_q_rt",
    "log_q_fav", "log_q_engagement", "author_vs_quoted_fol",
    "quoted_verified", "quoted_default_profile",
]

# Source-app category keywords (substring match, lower-cased). Handles French locale
# variants ("Twitter pour iPhone") and unseen test-only apps deterministically.
OFFICIAL_KW = ("twitter for ", "twitter web", "twitter pour ", "twitter para ",
               "twitter per ", "twitter media studio", "twitter for advertisers")
SCHED_KW = ("tweetdeck", "hootsuite", "buffer", "sprout social", "agorapulse",
            "echobox", "swello", "sociallymap", "limber", "social")
AUTO_KW = ("dlvr.it", "ifttt", "zapier", "wordpress", "paper.li", "revive",
           "blog2social", "scoop.it", "rss", "feed", "automat", "bot",
           "tweeted times", "nonli", "overblog", "mashup")
SOURCE_CATS = ("official", "scheduler", "automation", "other", "unknown")


def source_category(app):
    """Map a parsed source-app name to one of SOURCE_CATS (substring keyword match)."""
    a = str(app).lower()
    if app == "unknown" or a.strip() == "":
        return "unknown"
    if any(k in a for k in OFFICIAL_KW):
        return "official"
    if any(k in a for k in SCHED_KW):
        return "scheduler"
    if any(k in a for k in AUTO_KW):
        return "automation"
    return "other"


# ---------------------------------------------------------------------------
# Helpers copied verbatim from sub16_honest_features.py
# ---------------------------------------------------------------------------
def parse_date(s):
    if pd.isna(s) or s is None:
        return None
    try:
        return datetime.strptime(str(s), "%a %b %d %H:%M:%S +0000 %Y")
    except Exception:
        return None


def safe_len(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0
    if isinstance(x, list):
        return len(x)
    return 0


def get_text(row):
    text = row.get("text", "") or ""
    ext = row.get("extended_tweet.full_text")
    if ext and not (isinstance(ext, float) and np.isnan(ext)):
        text = ext
    return str(text)


def build_features(df):
    """Extract ~74 honest per-tweet features. Copied from sub16_honest_features."""
    f = pd.DataFrame(index=df.index)

    # === USER PROFILE (same across all tweets from same user) ===
    f["user_statuses_count"] = df.get("user.statuses_count", pd.Series(0, index=df.index)).fillna(0)
    f["user_listed_count"] = df.get("user.listed_count", pd.Series(0, index=df.index)).fillna(0)
    f["user_favourites_count"] = df.get("user.favourites_count", pd.Series(0, index=df.index)).fillna(0)
    f["user_geo_enabled"] = df.get("user.geo_enabled", pd.Series(False, index=df.index)).fillna(False).astype(int)
    f["user_default_profile"] = df.get("user.default_profile", pd.Series(True, index=df.index)).fillna(True).astype(int)
    f["user_default_profile_image"] = df.get("user.default_profile_image", pd.Series(True, index=df.index)).fillna(True).astype(int)
    f["user_has_url"] = df.get("user.url", pd.Series(np.nan, index=df.index)).notna().astype(int)
    f["user_has_banner"] = df.get("user.profile_banner_url", pd.Series(np.nan, index=df.index)).notna().astype(int)
    f["user_has_desc"] = df.get("user.description", pd.Series("", index=df.index)).apply(
        lambda x: 0 if (x is None or (isinstance(x, float) and np.isnan(x)) or str(x).strip() == "") else 1)
    f["user_protected"] = df.get("user.protected", pd.Series(False, index=df.index)).fillna(False).astype(int)
    f["user_contributors_enabled"] = df.get("user.contributors_enabled", pd.Series(False, index=df.index)).fillna(False).astype(int)
    f["user_geo_enabled"] = df.get("user.geo_enabled", pd.Series(False, index=df.index)).fillna(False).astype(int)
    # Replaces near-constant user_is_translator with a sharper behavioral flag.
    f["user_translator_is_regular"] = (
        df.get("user.translator_type", pd.Series("none", index=df.index)) == "regular"
    ).astype(int)
    f["user_profile_use_bg_image"] = df.get("user.profile_use_background_image", pd.Series(True, index=df.index)).fillna(True).astype(int)
    # Behavioral styling toggle (NOT an identity-proxy color/image field).
    f["user_bg_tile"] = df.get("user.profile_background_tile", pd.Series(False, index=df.index)).fillna(False).astype(int)
    f["user_has_location"] = df.get("user.location", pd.Series(np.nan, index=df.index)).apply(
        lambda x: 0 if (x is None or (isinstance(x, float) and np.isnan(x)) or str(x).strip() == "") else 1)

    # Profile-completeness composite (honest behavioral signal).
    f["profile_completeness"] = (
        f["user_has_url"] + f["user_has_banner"] + f["user_has_desc"]
        + f["user_has_location"] + f["user_geo_enabled"]
    ).astype(float)

    # Description / location behavioral features — LENGTH & flags only, never raw text.
    # Coerce absent to "" (behavioral 0, consistent with user_has_desc; NOT the -1 sentinel).
    _desc = df.get("user.description", pd.Series("", index=df.index)).fillna("").astype(str)
    f["desc_len"] = _desc.str.len()
    f["log_desc_len"] = np.log1p(f["desc_len"])
    f["desc_has_url"] = _desc.str.contains("http", case=False, regex=False).astype(int)
    f["desc_has_mention"] = _desc.str.contains("@", regex=False).astype(int)
    _loc = df.get("user.location", pd.Series("", index=df.index)).fillna("").astype(str)
    f["loc_len"] = _loc.str.len()

    # Account age
    user_created = df.get("user.created_at", pd.Series(np.nan, index=df.index))
    f["user_account_age_days"] = user_created.apply(
        lambda s: max(0, (REF_DATE - parse_date(s)).days) if parse_date(s) else 0)

    # Log transforms
    f["log_age"] = np.log1p(f["user_account_age_days"])
    f["log_statuses"] = np.log1p(f["user_statuses_count"])
    f["log_listed"] = np.log1p(f["user_listed_count"])
    f["log_favs"] = np.log1p(f["user_favourites_count"])

    # Behavioral ratios
    f["tweets_per_day"] = f["user_statuses_count"] / (f["user_account_age_days"] + 1)
    f["listed_per_tweet"] = f["user_listed_count"] / (f["user_statuses_count"] + 1)
    f["fav_per_tweet"] = f["user_favourites_count"] / (f["user_statuses_count"] + 1)
    f["listed_per_day"] = f["user_listed_count"] / (f["user_account_age_days"] + 1)
    f["log_tweets_per_day"] = np.log1p(f["tweets_per_day"])
    f["log_listed_per_tweet"] = np.log1p(f["listed_per_tweet"])
    f["log_fav_per_tweet"] = np.log1p(f["fav_per_tweet"])

    # === TWEET ENGAGEMENT ===
    f["retweet_count"] = df.get("retweet_count", pd.Series(0, index=df.index)).fillna(0)
    f["favorite_count"] = df.get("favorite_count", pd.Series(0, index=df.index)).fillna(0)
    f["quote_count"] = df.get("quote_count", pd.Series(0, index=df.index)).fillna(0)
    f["reply_count"] = df.get("reply_count", pd.Series(0, index=df.index)).fillna(0)
    f["engagement"] = f["retweet_count"] + f["favorite_count"] + f["quote_count"] + f["reply_count"]
    f["log_rt"] = np.log1p(f["retweet_count"])
    f["log_fav_tweet"] = np.log1p(f["favorite_count"])
    f["log_engagement"] = np.log1p(f["engagement"])

    # === TWEET METADATA ===
    f["is_quote"] = df.get("is_quote_status", pd.Series(False, index=df.index)).fillna(False).astype(int)
    f["is_reply"] = df.get("in_reply_to_status_id", pd.Series(np.nan, index=df.index)).notna().astype(int)
    f["truncated"] = df.get("truncated", pd.Series(False, index=df.index)).fillna(False).astype(int)

    lang_map = {"fr": 0, "en": 1, "es": 2, "ar": 3, "de": 4, "pt": 5, "it": 6, "und": 7, "ja": 8, "tr": 9}
    f["lang_enc"] = df.get("lang", pd.Series("fr", index=df.index)).fillna("fr").map(lambda x: lang_map.get(x, 99))

    # === TEXT SURFACE FEATURES ===
    texts = df.apply(get_text, axis=1)
    f["text_len"] = texts.str.len().fillna(0)
    f["word_count"] = texts.str.split().str.len().fillna(0)
    f["hashtag_count"] = texts.str.count(r"#\w+")
    f["mention_count"] = texts.str.count(r"@\w+")
    f["url_count"] = texts.str.count(r"http")
    f["caps_ratio"] = texts.apply(lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1))
    f["is_rt"] = texts.str.startswith("RT ").astype(int)
    f["exclaim_count"] = texts.str.count("!")
    f["question_count"] = texts.str.count(r"\?")
    f["avg_word_len"] = texts.apply(lambda t: np.mean([len(w) for w in t.split()]) if t.split() else 0)
    f["unique_words"] = texts.apply(lambda t: len(set(t.lower().split())) if t.split() else 0)
    f["type_token_ratio"] = f["unique_words"] / (f["word_count"] + 1)

    # === ENTITIES ===
    if "entities.hashtags" in df.columns:
        f["ent_hashtags"] = df["entities.hashtags"].apply(safe_len)
        f["ent_mentions"] = df["entities.user_mentions"].apply(safe_len) if "entities.user_mentions" in df.columns else pd.Series(0, index=df.index)
        f["ent_urls"] = df["entities.urls"].apply(safe_len) if "entities.urls" in df.columns else pd.Series(0, index=df.index)
        f["ent_media"] = df["entities.media"].apply(safe_len) if "entities.media" in df.columns else pd.Series(0, index=df.index)

    # === QUOTED TWEET FEATURES — NaN on absent (median-imputed FOLD-SAFE in train.py) ===
    # ~65% of rows have no quoted_status. Leaving these as NaN (not -1) keeps the -1
    # sentinel out of StandardScaler; has_quoted is the single channel that says "absent".
    if "quoted_status.user.followers_count" in df.columns:
        q_fol = df["quoted_status.user.followers_count"]          # NaN on absent
        q_fri = df["quoted_status.user.friends_count"]
        q_statuses = df.get("quoted_status.user.statuses_count", pd.Series(np.nan, index=df.index))
        f["has_quoted"] = q_fol.notna().astype(int)
        f["quoted_followers"] = q_fol
        f["quoted_friends"] = q_fri
        f["quoted_listed"] = df.get("quoted_status.user.listed_count", pd.Series(np.nan, index=df.index))
        f["quoted_statuses"] = q_statuses
        f["quoted_verified"] = df.get("quoted_status.user.verified",
            pd.Series(np.nan, index=df.index)).map({True: 1.0, False: 0.0})
        f["quoted_default_profile"] = df.get("quoted_status.user.default_profile",
            pd.Series(np.nan, index=df.index)).map({True: 1.0, False: 0.0})
        f["quoted_fol_ratio"] = q_fol / (q_fri + 1)              # NaN/(NaN+1)=NaN on absent
        f["log_q_fol"] = np.log1p(q_fol)                         # present values are >= 0
        f["log_q_fri"] = np.log1p(q_fri)
        f["quoted_rt"] = df.get("quoted_status.retweet_count", pd.Series(np.nan, index=df.index))
        f["quoted_fav"] = df.get("quoted_status.favorite_count", pd.Series(np.nan, index=df.index))
        f["quoted_reply"] = df.get("quoted_status.reply_count", pd.Series(np.nan, index=df.index))
        f["log_q_rt"] = np.log1p(f["quoted_rt"])
        f["log_q_fav"] = np.log1p(f["quoted_fav"])
        f["quoted_engagement"] = f["quoted_rt"] + f["quoted_fav"] + f["quoted_reply"]
        f["log_q_engagement"] = np.log1p(f["quoted_engagement"])

        # Quoted user account age (NaN on absent).
        q_created = df.get("quoted_status.user.created_at", pd.Series(np.nan, index=df.index))
        f["quoted_user_age_days"] = q_created.apply(
            lambda s: max(0, (REF_DATE - parse_date(s)).days) if parse_date(s) else np.nan)
        f["quoted_tweets_per_day"] = q_statuses / (f["quoted_user_age_days"] + 1)
        f["quoted_listed_per_tweet"] = f["quoted_listed"] / (q_statuses + 1)

        # Ratio: author stats vs quoted user stats.
        f["author_vs_quoted_fol"] = f["log_statuses"] / (f["log_q_fol"] + 1)
    else:
        f["has_quoted"] = 0
        for _c in QUOTED_IMPUTE_COLS:
            f[_c] = np.nan

    # Drop dead (constant) features. Fill ONLY non-quoted columns with -1; leave
    # QUOTED_IMPUTE_COLS as NaN for the fold-safe median imputer in train.py.
    f = f.drop(columns=DEAD_FEATURES, errors="ignore")
    other = [c for c in f.columns if c not in QUOTED_IMPUTE_COLS]
    f[other] = f[other].replace([np.inf, -np.inf], np.nan).fillna(-1)
    f[QUOTED_IMPUTE_COLS] = f[QUOTED_IMPUTE_COLS].replace([np.inf, -np.inf], np.nan)
    return f


# ---------------------------------------------------------------------------
# Source app handling (parse_source + pro_tools from sub19_with_source.py)
# ---------------------------------------------------------------------------
def parse_source(s):
    if not s or (isinstance(s, float) and np.isnan(s)):
        return "unknown"
    m = re.search(r">([^<]+)<", str(s))
    return m.group(1).strip() if m else str(s)[:50]


PRO_TOOLS = {
    "TweetDeck", "Hootsuite Inc.", "dlvr.it", "Buffer", "Echobox",
    "AgoraPulse Manager", "Twitter Media Studio", "Sprout Social",
    "Zapier.com", "IFTTT", "WordPress.com",
}


def parse_source_series(df):
    """Parse the `source` HTML column once, returning a Series of app names."""
    sources = df.get("source", pd.Series("", index=df.index)).fillna("")
    return sources.apply(parse_source)


def add_static_source_features(parsed_src, df, X):
    """Target-INDEPENDENT source/time features (identical on train & test → no leakage)."""
    X = X.copy()
    # Source category one-hot (5-way), replaces the single source_is_pro binary.
    # Substring keyword match -> strong honest separation across 270 apps.
    cat = parsed_src.apply(source_category)
    for c in SOURCE_CATS:
        X[f"src_cat_{c}"] = (cat == c).astype(int)

    # Timestamp-derived features. Hour is encoded cyclically (sin/cos); a raw 0..23
    # integer through StandardScaler misleads the MLP. NaN hour -> 0/0 (neutral),
    # NOT -1 (which would land on the cycle and corrupt it).
    if "timestamp_ms" in df.columns:
        ts = pd.to_numeric(df["timestamp_ms"], errors="coerce") / 1000
        dt = pd.to_datetime(ts, unit="s", utc=True, errors="coerce")
        h = dt.dt.hour
        rad = 2 * np.pi * h / 24.0
        X["hour_sin"] = np.where(h.notna(), np.sin(rad), 0.0)
        X["hour_cos"] = np.where(h.notna(), np.cos(rad), 0.0)
        X["day_of_week"] = dt.dt.dayofweek.fillna(-1)
        X["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    else:
        X["hour_sin"] = 0.0
        X["hour_cos"] = 0.0
        X["day_of_week"] = -1
        X["is_weekend"] = 0
    return X


# ---------------------------------------------------------------------------
# Fold-safe source target encoding (the ONLY target-dependent feature)
# ---------------------------------------------------------------------------
def compute_source_te(train_src, y_train, smoothing=config.SOURCE_TE_SMOOTHING):
    """Smoothed mean-label per source, computed on TRAINING ROWS ONLY.

    Returns (te_map: dict[str -> float], global_mean: float).
    """
    train_src = pd.Series(np.asarray(train_src))
    y_train = pd.Series(np.asarray(y_train), index=train_src.index)
    global_mean = float(y_train.mean())

    grp = y_train.groupby(train_src.values)
    counts = grp.count()
    means = grp.mean()
    smoothed = (counts * means + smoothing * global_mean) / (counts + smoothing)
    return smoothed.to_dict(), global_mean


def apply_source_te(src_series, te_map, global_mean):
    """Map each source to its smoothed rate; unseen sources -> global_mean."""
    return pd.Series(np.asarray(src_series)).map(te_map).fillna(global_mean).values
