"""
Central configuration for the PyTorch deep-MLP pipeline.
Single source of truth for paths, seed, and hyperparameters.
"""
import os

# === Paths ===
# Data lives one directory up from the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.dirname(_HERE)

TRAIN_PATH = os.path.join(_DATA_DIR, "train.jsonl")
TEST_PATH = os.path.join(_DATA_DIR, "kaggle_test.jsonl")

SUBMISSION_PATH = os.path.join(_HERE, "submission_mlp.csv")
OOF_PATH = os.path.join(_HERE, "mlp_oof_probs.npy")
TEST_PROBS_PATH = os.path.join(_HERE, "mlp_test_probs.npy")

# TensorBoard log directory (one run per launch -> subfolders created in train.py).
LOG_DIR = os.path.join(_HERE, "runs")

# === Reproducibility ===
SEED = 42

# === Cross-validation ===
N_FOLDS = 5

# === Feature engineering ===
SOURCE_TE_SMOOTHING = 20.0  # smoothing for fold-safe source target encoding
REF_DATE_YMD = (2021, 6, 1)  # reference date for account-age computation

# === MLP architecture ===
HIDDEN = [256, 128, 64]
DROPOUT = [0.30, 0.30, 0.20]

# === Training ===
BATCH_SIZE = 1024
EPOCHS = 40
LR = 1e-3
MAX_LR = 3e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 6  # early-stopping patience on validation accuracy

# === Blend companion model (--blend): honest GBDT to ensemble with the MLP ===
# Trees are scale-invariant and capture axis-aligned interactions the MLP misses, so a
# diversity blend typically beats either model alone. Trained per-fold on the SAME folds
# as the MLP (aligned OOF) so the blend weight is chosen honestly on OOF.
LGB_PARAMS = dict(
    objective="binary", num_leaves=63, learning_rate=0.05, n_estimators=400,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=SEED, n_jobs=-1, verbose=-1,
)

# === Smoke test (--smoke) ===
SMOKE_NROWS = 5000
SMOKE_EPOCHS = 2
SMOKE_FOLDS = 2

# Columns that act as identity proxies under row-wise CV. These MUST NEVER
# enter the feature matrix (they inflated sub21 to 97.48% CV / 0.839 public).
BANNED_COLUMNS = [
    "user.profile_background_color",
    "user.profile_link_color",
    "user.profile_sidebar_border_color",
    "user.profile_sidebar_fill_color",
    "user.profile_text_color",
    "user.profile_background_image_url",
    "user.profile_background_image_url_https",
]
