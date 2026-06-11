"""Per-user prediction smoothing: the label is user-CONSTANT, and each user has ~3.2
tweets sharing a `user.created_at` (available at inference on test too). Averaging a
user's tweet-level probabilities into one user-level prediction denoises independent
per-tweet errors. Pure inference-time, label-free -> honest. Validate the lift on the
honest GroupKFold OOF (where a user's rows are all in one fold, so OOF averaging mirrors
test-time averaging exactly).

Tests several proxy-key tightnesses to see how much grouping precision matters.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import config
import json
import hashlib
import numpy as np
from sklearn.metrics import accuracy_score

mlp = np.load(os.path.join(config.PRED_DIR, "mlp_oof_probs.npy"))
lgb = np.load(os.path.join(config.PRED_DIR, "lgb_oof_probs.npy"))
blend = 0.55 * mlp + 0.45 * lgb
n = len(blend)
print(f"OOF rows: {n}")

# Stream per-row label + several proxy user keys (row order matches the OOF arrays).
# Key design: group only on USER-CONSTANT fields. created_at never changes; profile
# colors / bg image are user-constant identity proxies (banned as FEATURES but ideal as a
# GROUPING key). description/location/statuses_count VARY per tweet -> must NOT be in key.
COLOR_FIELDS = ["profile_background_color", "profile_link_color",
                "profile_sidebar_border_color", "profile_sidebar_fill_color",
                "profile_text_color", "profile_background_image_url"]
labels = np.empty(n, dtype=np.int64)
k_created = []          # created_at only
k_color = []            # created_at + user-constant colors (precise per-user)
k_coloronly = []        # colors only
with open(config.TRAIN_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= n:
            break
        d = json.loads(line)
        labels[i] = int(d["label"])
        u = d.get("user", {}) or {}
        ca = str(u.get("created_at", ""))
        colors = "|".join(str(u.get(c, "")) for c in COLOR_FIELDS)
        k_created.append(ca)
        k_color.append(ca + "#" + colors)
        k_coloronly.append(colors)
y = labels


def best_thr_acc(p):
    bt, ba = 0.5, 0.0
    for t in np.arange(0.40, 0.601, 0.005):
        a = accuracy_score(y, (p > t).astype(int))
        if a > ba:
            ba, bt = a, float(t)
    return bt, ba


def smooth(p, keys):
    """Replace each row's prob with the mean prob over all rows sharing its user key."""
    keys = np.asarray(keys)
    order = np.argsort(keys, kind="mergesort")
    sk = keys[order]; sp = p[order]
    out = np.empty_like(p)
    i = 0
    while i < len(sk):
        j = i
        while j < len(sk) and sk[j] == sk[i]:
            j += 1
        out[order[i:j]] = sp[i:j].mean()
        i = j
    return out


for name, p in [("MLP", mlp), ("LGBM", lgb), ("BLEND", blend)]:
    _, raw = best_thr_acc(p)
    _, ac = best_thr_acc(smooth(p, k_created))
    _, ak = best_thr_acc(smooth(p, k_color))
    _, ao = best_thr_acc(smooth(p, k_coloronly))
    print(f"{name:6s} raw {raw*100:.2f}%  | created {ac*100:.2f}% ({(ac-raw)*100:+.2f})  "
          f"| created+colors {ak*100:.2f}% ({(ak-raw)*100:+.2f})  "
          f"| colors-only {ao*100:.2f}% ({(ao-raw)*100:+.2f})")

# group stats
import collections
for nm, ks in [("created", k_created), ("created+colors", k_color), ("colors", k_coloronly)]:
    vc = collections.Counter(ks); sizes = np.array(list(vc.values()))
    print(f"{nm:16s}: {len(vc)} groups ({len(vc)/n*100:.1f}% unique), "
          f"size mean {sizes.mean():.2f} median {np.median(sizes):.0f} max {sizes.max()} "
          f"singletons {(sizes==1).mean()*100:.1f}%")
