"""Step 3b: cache the stage-1 feature matrix of the train/hold-out rows (input of s05_stage2_train.py)."""
from s03_train import load_truth
from x01_learning_curve import CACHE, build_cache

if __name__ == "__main__":
    if not (CACHE / "meta.parquet").exists():
        build_cache(load_truth()[0])
    print("cache ready:", CACHE)
