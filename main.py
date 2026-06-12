"""main.py — one-command pipeline runner for Influencer or Observer.

Runs the production pipeline end to end, in the canonical order, by invoking each
src/ script as a subprocess (most of them execute at import time, so subprocess is
the clean way to sequence them). Each stage's stdout/stderr is streamed live and
also captured to output/src_logs/<stage>.log.

PIPELINE (default order):
    1. train.py --group --blend --smooth   tabular MLP + LightGBM, honest CV, smoothing
    2. stack_models.py                      CatBoost / XGBoost / HistGBM / ExtraTrees
    3. finetune_camembert.py                CamemBERT on tweet text          (GPU, ~1 h)
    4. finetune_camembert_bio.py            CamemBERT on bio + tweet         (GPU, ~1 h)
    5. finetune_camembert_user.py           user-level CamemBERT             (GPU, ~1 h)
    6. make_submission.py                   blend everything -> submission_final.csv

USAGE
    python main.py                  # full pipeline (needs a GPU for stages 3-5)
    python main.py --no-text        # skip the 3 CamemBERT stages (tabular only, fast)
    python main.py --smoke          # quick sanity check: train.py --smoke only
    python main.py --only make_submission        # run one stage
    python main.py --only train,stack_models     # run a subset, in this order
    python main.py --list           # show stage names and exit

Place train.jsonl and kaggle_test.jsonl in data/ first (see README).
"""
import os
import sys
import argparse
import subprocess
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
DATA = os.path.join(ROOT, "data")
LOG_DIR = os.path.join(ROOT, "output", "src_logs")

# (stage name, [script + args], needs_gpu)
STAGES = [
    ("train",                ["train.py", "--group", "--blend", "--smooth"], False),
    ("stack_models",         ["stack_models.py"],                            False),
    ("finetune_camembert",     ["finetune_camembert.py"],                    True),
    ("finetune_camembert_bio", ["finetune_camembert_bio.py"],                True),
    ("finetune_camembert_user",["finetune_camembert_user.py"],               True),
    ("make_submission",      ["make_submission.py"],                         False),
]
TEXT_STAGES = {"finetune_camembert", "finetune_camembert_bio", "finetune_camembert_user"}


def check_data():
    """Fail early with a clear message if the competition files are missing."""
    missing = [f for f in ("train.jsonl", "kaggle_test.jsonl")
               if not os.path.exists(os.path.join(DATA, f))]
    if missing:
        print(f"ERROR: missing data file(s) in data/: {', '.join(missing)}")
        print("Place train.jsonl and kaggle_test.jsonl in data/ first (see README).")
        sys.exit(1)


def run_stage(name, argv):
    """Run one src/ script as a subprocess; stream output and tee to a log file."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{name}.log")
    cmd = [sys.executable] + argv
    print("\n" + "=" * 78)
    print(f">>> STAGE: {name}   ({' '.join(argv)})")
    print(f">>> started {datetime.now():%Y-%m-%d %H:%M:%S}  | log -> {os.path.relpath(log_path, ROOT)}")
    print("=" * 78, flush=True)

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    t0 = datetime.now()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"### {name}: {' '.join(argv)}  started {t0:%Y-%m-%d %H:%M:%S}\n")
        log.flush()
        # cwd=SRC so scripts that import config/features and read relative paths work
        proc = subprocess.Popen(cmd, cwd=SRC, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        proc.wait()
        dt = (datetime.now() - t0).total_seconds()
        log.write(f"[exit code: {proc.returncode}] {dt:.0f}s\n")

    if proc.returncode != 0:
        print(f"\n!!! STAGE '{name}' FAILED (exit {proc.returncode}). See {log_path}")
        print("    Pipeline stopped. Fix the error and re-run (use --only to resume).")
        sys.exit(proc.returncode)
    print(f">>> {name} done in {dt:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Run the Influencer-or-Observer pipeline.")
    ap.add_argument("--no-text", action="store_true",
                    help="skip the 3 CamemBERT GPU stages (tabular models only)")
    ap.add_argument("--smoke", action="store_true",
                    help="quick sanity check: only train.py --smoke")
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated stage name(s) to run, in given order")
    ap.add_argument("--list", action="store_true", help="list stage names and exit")
    args = ap.parse_args()

    all_names = [s[0] for s in STAGES]
    if args.list:
        print("Pipeline stages (in order):")
        for nm, argv, gpu in STAGES:
            print(f"  {nm:26s} {'[GPU]' if gpu else '     '}  {' '.join(argv)}")
        return

    if args.smoke:
        check_data()
        run_stage("train_smoke", ["train.py", "--smoke", "--group", "--blend", "--smooth"])
        print("\nSmoke test complete. The pipeline runs; use 'python main.py' for the real run.")
        return

    # build the stage list to run
    if args.only:
        want = [s.strip() for s in args.only.split(",") if s.strip()]
        unknown = [w for w in want if w not in all_names]
        if unknown:
            print(f"ERROR: unknown stage(s): {unknown}\nValid: {all_names}")
            sys.exit(1)
        plan = [(nm, argv) for (nm, argv, _gpu) in STAGES if nm in want]
        # honor the order the user gave
        plan.sort(key=lambda p: want.index(p[0]))
    else:
        plan = [(nm, argv) for (nm, argv, gpu) in STAGES
                if not (args.no_text and nm in TEXT_STAGES)]

    check_data()
    print(f"Running {len(plan)} stage(s): {[p[0] for p in plan]}")
    t0 = datetime.now()
    for nm, argv in plan:
        run_stage(nm, argv)
    print("\n" + "=" * 78)
    print(f"PIPELINE COMPLETE in {(datetime.now()-t0).total_seconds():.0f}s")
    sub = os.path.join(ROOT, "output", "submission_final.csv")
    if os.path.exists(sub):
        print(f"Submission: {os.path.relpath(sub, ROOT)}")
    print("=" * 78)


if __name__ == "__main__":
    main()

