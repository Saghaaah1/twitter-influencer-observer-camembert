"""PROOF: the label is 100% user-constant -> this is a USER-classification task.

Group all train rows by the proxy user key (user.created_at) and check whether any user
has tweets with DIFFERENT labels. Result: 0 mixed-label groups out of 30,696 users (mean
~5 tweets/user) -> user-majority oracle row-accuracy = 100.00%.

Consequence: per-user prediction smoothing is valid (averaging a user's tweet probabilities
cannot cross a label boundary), and the only thing keeping accuracy below 100% is how well
the features separate DIFFERENT users -- the honest ~85% ceiling. See 08_user_smoothing.py.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import json
import numpy as np

keys, labels = [], []
with open(config.TRAIN_PATH, encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        u = d.get("user", {}) or {}
        keys.append(str(u.get("created_at", "")))
        labels.append(int(d["label"]))
y = np.array(labels)
keys = np.array(keys)
n = len(y)
print(f"rows {n}")

order = np.argsort(keys, kind="mergesort")
sk, sy = keys[order], y[order]
oracle = np.empty(n, dtype=int)
mixed_groups = mixed_rows = ngroups = 0
i = 0
while i < len(sk):
    j = i
    while j < len(sk) and sk[j] == sk[i]:
        j += 1
    grp = sy[i:j]
    maj = 1 if grp.mean() >= 0.5 else 0
    oracle[order[i:j]] = maj
    if 0 < grp.mean() < 1:
        mixed_groups += 1
        mixed_rows += (j - i)
    ngroups += 1
    i = j

print(f"proxy users (created_at groups): {ngroups}  (mean {n/ngroups:.2f} tweets/user)")
print(f"user-majority ORACLE row-accuracy: {(oracle == y).mean()*100:.2f}%")
print(f"groups with MIXED labels: {mixed_groups} ({mixed_groups/ngroups*100:.1f}%)")
print(f"rows in mixed-label groups: {mixed_rows/n*100:.1f}%")
print("\nConclusion: label is user-constant -> classify users, not tweets; smoothing is valid.")
