# Influencer or Observer — Predicting Social Roles

Binary classification of French Twitter accounts (influencer vs. observer) from tweet
metadata + text. **Final result: public LB 0.856 — tied for #1.**

The whole approach is built around one principle: **don't fool yourself with identity
leakage.** The public test set is 100% disjoint users, so only signal that generalizes
across *different* users counts. See [`docs/SOLUTION_README.md`](docs/SOLUTION_README.md)
for the full write-up and [`experiments/README.md`](experiments/README.md) for the evidence.

## Repository layout

```
.
├── src/                      # production pipeline (run these)
│   ├── config.py             #   single source of truth: paths, seed, hyperparameters
│   ├── features.py           #   honest feature engineering + fold-safe source target encoding
│   ├── model.py              #   TabularMLP (256->128->64->1)
│   ├── train.py              #   tabular pipeline: CV -> MLP + LightGBM blend -> smooth -> submission
│   ├── stack_models.py       #   adds CatBoost / XGBoost / HistGBM / ExtraTrees
│   ├── finetune_camembert.py #   fine-tune CamemBERT on tweet text
│   ├── finetune_camembert_bio.py  # fine-tune CamemBERT on bio + tweet
│   └── make_submission.py    #   blend all model predictions + per-user smoothing -> final CSV
├── experiments/              # 12 numbered probes — the evidence behind every decision
├── exploration/              # initial data-understanding scripts + feature_lineage/
├── data/                     # train.jsonl / kaggle_test.jsonl (gitignored — place here)
├── output/                   # submissions, model_predictions/ (prob arrays), runs/ (gitignored)
└── docs/                     # SOLUTION_README.md (detailed) + PIPELINE.zh.md (Chinese)
```

## Setup

```bash
conda activate sara    # or: pip install pandas scikit-learn lightgbm torch transformers
# place train.jsonl and kaggle_test.jsonl in data/
```

## Reproduce the 0.856 submission

```bash
cd src
python train.py --group --blend --smooth   # MLP + LightGBM, honest CV, per-user smoothing
python stack_models.py                      # add CatBoost / XGBoost / HistGBM / ExtraTrees
python finetune_camembert.py                # fine-tune CamemBERT on tweet text   (GPU, ~1 hr)
python finetune_camembert_bio.py            # fine-tune CamemBERT on bio + tweet   (GPU, ~1 hr)
python make_submission.py                   # blend everything -> output/submission_final.csv
```

Each model writes aligned `<name>_oof_probs.npy` / `<name>_test_probs.npy` into
`output/model_predictions/`; `make_submission.py` auto-discovers them, selects ensemble
weights by greedy search on the per-user-smoothed OOF, and writes the final submission.

## How it got to 0.856

| Stage | honest OOF | public LB |
|-------|-----------|-----------|
| MLP + LightGBM blend | 84.31% | — |
| + per-user smoothing (label is user-constant) | 84.78% | 0.848 |
| + CatBoost / HistGBM / ExtraTrees stack | 84.84% | 0.850 |
| + fine-tuned CamemBERT (tweet) | 85.14% | 0.852 |
| + CamemBERT (bio + tweet) | 85.40% | **0.856** |

### Key findings
1. **Direct user IDs are stripped** (`user.id` / `screen_name` absent), and train/test users
   are 100% disjoint — so any per-user "aggregation" is leakage, not signal.
2. **Honest CV is everything.** Row-wise CV reached 97% but only 0.839 public (identity
   memorization). `StratifiedGroupKFold` by `user.created_at` gives an honest ~84% that
   tracks the LB. (`experiments/01,02,12`)
3. **The label is 100% user-constant** → it's a user-classification task → per-user
   prediction smoothing is the biggest single gain. (`experiments/07,08`)
4. **Metadata saturates ~0.850; French text (CamemBERT) is the one fresh signal** that
   pushed it to 0.856. Bigger/other transformers add nothing (`experiments/10,11`, `archive/`).

0.856 is the data's honest ceiling — the whole leaderboard plateaus there.
