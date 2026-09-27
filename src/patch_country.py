"""Re-run one test country after changing its country-specific normalization (no retraining; the models stay as they are).

Usage: python patch_country.py France
Uses ER_WORK_DIR / ER_OUT_DIR / ER_OUT_DIR2 like the other steps; other countries' predictions are reused.
"""
import subprocess
import sys

from common import WORK_DIR

if __name__ == "__main__":
    country = sys.argv[1]
    steps = [["s07_country_maps.py"], ["s02_block.py", "test", country]]
    for cmd in steps:
        print(">>>", " ".join(cmd), flush=True)
        if subprocess.call([sys.executable, "-u", *cmd]) != 0:
            sys.exit(f"failed: {cmd}")
    for f in ("pred.parquet", "pred2.parquet"):
        p = WORK_DIR / "test" / country / f
        if p.exists():
            p.unlink()
    for cmd in (["s04_predict.py", "--reuse"], ["s06_predict_stage2.py", "--reuse"]):
        print(">>>", " ".join(cmd), flush=True)
        if subprocess.call([sys.executable, "-u", *cmd]) != 0:
            sys.exit(f"failed: {cmd}")
    print("done")
