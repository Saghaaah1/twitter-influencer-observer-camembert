# Influencer or Observer — honest social-role prediction

A self-contained PyTorch + LightGBM pipeline for the Kaggle task *"Influencer or Observer:
Predicting Social Roles"* — a binary classification of Twitter accounts from a single
tweet's metadata (French-language corpus). The entire design is organized around **one
principle: do not fool yourself with identity leakage.**

> Cautionary history: earlier submissions scored 94–97% in cross-validation but only 0.839
> on the public leaderboard, because some features act as per-user *identity proxies* under
> row-wise CV. The public test set is **100% disjoint-user** (see
> [`experiments/02`](experiments/02_train_test_overlap.py)), so any signal that does not
> generalize across *different* users is worthless. Every choice here keeps the OOF estimate
> honest and aligned with the public LB.

## Result

| Stage | honest OOF (GroupKFold) | public LB |
|-------|-------------------------|-----------|
| MLP only | 83.94% | — |
| + LightGBM blend (α=0.55) | 84.31% | — |
| **+ per-user smoothing** | **84.78%** | **0.848** |
| 4-model stack (+ HistGBM/ExtraTrees) | 84.84% | (marginal) |

The honest information ceiling of this feature set is ~85%; see [`experiments/`](experiments/).

## Layout

```
.
├── config.py          # single source of truth: paths, seed, hyperparams, banned columns
├── features.py        # "honest" feature engineering + fold-safe source target encoding
├── model.py           # TabularMLP (256→128→64→1, BatchNorm+GELU+Dropout)
├── train.py           # orchestration: load → CV → train → blend → smooth → submission
├── stack_models.py    # optional multi-model stack (LGBM/HistGBM/ExtraTrees[/CatBoost/XGBoost])
├── experiments/       # the evidence behind every design decision (see its README)
└── PIPELINE.zh.md     # detailed pipeline walkthrough (Chinese)
```

## Setup

```bash
conda activate sara        # or: pip install pandas scikit-learn lightgbm torch
# place train.jsonl and kaggle_test.jsonl in the PARENT directory of this repo
```

## Run

```bash
python train.py --group --blend --smooth   # recommended: honest CV + ensemble + smoothing
python train.py --group                     # honest StratifiedGroupKFold, MLP only
python train.py                             # default row-wise CV (OPTIMISTIC OOF — for contrast)
python train.py --smoke                     # tiny 5k-row sanity run
python stack_models.py                      # multi-model stack -> submission_stack.csv
```

## Key ideas

1. **Honest cross-validation.** `--group` uses `StratifiedGroupKFold` keyed by a proxy user
   id (`user.created_at`, which is constant per account). This keeps every user inside one
   fold so user-constant features cannot be memorized as identity. The gap between default
   and `--group` OOF measures the row-wise inflation.
2. **Only honest features.** Per-tweet metadata, behavioral ratios, text surface stats,
   quoted-user stats, cyclic time, and a *fold-safe* source-app target encoding. Identity-
   proxy columns (profile colors / background image) are hard-banned via an assertion.
3. **The task is user-level.** The label is 100% user-constant
   ([`experiments/07`](experiments/07_label_user_consistency.py)), so `--smooth` averages
   each user's tweet probabilities before thresholding — the largest post-feature gain.
4. **Diversity blend.** `--blend` trains a LightGBM on the same folds and ensembles it with
   the MLP; weight and threshold are tuned on aligned OOF only.

See [`experiments/README.md`](experiments/README.md) for the full evidence trail.
