# Experiments — evidence behind the design decisions

These are the validation scripts that justify every non-obvious choice in the pipeline.
They are kept as a reproducible audit trail: each one is model-agnostic (mostly LightGBM
under honest `StratifiedGroupKFold`) and answers a single question. Run any of them with the
project's Python environment from the repo root, e.g. `python experiments/02_train_test_overlap.py`.

The thread running through all of them: **the public test set is disjoint-user, so only
signal that generalizes across *different* users counts.** Row-wise CV that lets the same
user span folds inflates the score by memorizing identity.

| # | Script | Question | Conclusion |
|---|--------|----------|------------|
| 01 | `01_leakage_diagnosis.py` | Do user-constant "profile" features leak identity under row-wise CV? | Yes — they help `StratifiedKFold` but not `GroupKFold`. Confirms row-wise OOF is inflated by identity memorization. |
| 02 | `02_train_test_overlap.py` | How many test users appear in train? | **0%** (0 of 32,117 test proxy-users). The public LB is 100% disjoint-user → identity features are dead weight. |
| 03 | `03_feature_url_domain.py` | Does a fold-safe TE of the tweet's URL domain add signal? | **−0.07%** — automation shorteners (bit.ly/dlvr.it) are already captured by `source_te`. No lift. |
| 04 | `04_feature_text.py` | Does bio / tweet text (TF-IDF→SVD) add signal? | Bio alone 66.9%, tweet 64.2%; on top of metadata only **+0.12% / +0.05%**. Text is near-useless here. |
| 05 | `05_feature_extra.py` | Bio role-words, text-style ratios, emoji/digit ratios, richer quoted features? | All groups **≤ +0.08%** (noise). |
| 06 | `06_feature_user_aggregation.py` | Label-free within-dataset user aggregation (counts, RT fraction, source diversity)? | **+0.04%** (noise). |
| 07 | `07_label_user_consistency.py` | Is the label consistent within a user? | **100% user-constant** (0 mixed-label groups / 30,696 users). → This is a *user*-classification task. |
| 08 | `08_user_smoothing.py` | Does averaging each user's tweet predictions help? | **+0.47%** (blend 84.31% → 84.78%). The single biggest post-feature gain. `created_at` is the right key; per-tweet-varying fields over-split it; colors alone collide. |
| 09 | `09_user_level_model.py` | Does an explicit user-level aggregated model beat smoothing? | 84.48% — slightly *below* the smoothed diverse blend; aggregation ≈ smoothing, and model diversity matters more. |

### Headline numbers (honest `GroupKFold` OOF)

```
metadata baseline (LightGBM) ......... 83.83%
  + URL domain / text / extra / agg .. +0.04% to +0.12%  (all noise)
honest feature ceiling ............... ~83.9%
MLP + LightGBM blend ................. 84.31%
  + per-user smoothing ............... 84.78%   -> public 0.848
4-model stack (+ HistGBM/ExtraTrees) . 84.84%   (marginal)
```

**Takeaway:** the metadata signal is saturated at the ~84–85% honest ceiling; feature
engineering does not move it. The gains came from (a) honest CV to stop fooling ourselves,
(b) recognizing the task is user-level, and (c) smoothing + model-diversity blending.
