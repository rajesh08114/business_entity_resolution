"""Run the whole pipeline end to end (steps 1-6), each step in its own process, stopping at the first failure.

Usage: python run_pipeline.py [first_step]      e.g. `python run_pipeline.py s03` resumes from training.
Paths come from ER_DATA_DIR / ER_WORK_DIR / ER_OUT_DIR / ER_OUT_DIR2 (see common.py). Logs go to ER_WORK_DIR/logs/.
"""
import subprocess
import sys
import time

from common import WORK_DIR

STEPS = [
    ("s01", ["s01_lexicon.py"]),
    ("s02_test", ["s02_block.py", "test"]),
    ("s02_train", ["s02_block.py", "train"]),
    ("gate", None),
    ("s03", ["s03_train.py", "0.15", "0.45"]),
    ("s05", ["s05_stage2_train.py"]),
    ("s04", ["s04_predict.py"]),
    ("s06", ["s06_predict_stage2.py"]),
]
MIN_BLOCKING_RECALL = 0.9824  # full-scale train pair recall of the previous version (v4)


def blocking_gate():
    import numpy as np
    import pandas as pd
    from common import pack_pair
    sys.argv = sys.argv[:1]  # s03_train parses sys.argv at import time
    from s03_train import load_truth
    truth, n_true = load_truth()
    hit = tot = 0
    for c in sorted((WORK_DIR / "train").iterdir()):
        C = pd.read_parquet(c / "cand.parquet", columns=["s1_num", "b_num", "src"])
        hit += int(np.isin(pack_pair(C.s1_num.values, C.b_num.values, C.src.values), truth).sum())
        tot += len(C)
    rec = hit / len(truth)
    print(f"GATE full-scale train blocking: pair recall {rec:.4f}, candidates per S1 {tot / len(n_true):.1f} (previous v4: 0.9824 at 36.1)", flush=True)
    return rec > MIN_BLOCKING_RECALL


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else STEPS[0][0]
    names = [s for s, _ in STEPS]
    logs = WORK_DIR / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for name, cmd in STEPS[names.index(start):]:
        t = time.time()
        print(f"[{t - t0:7.0f}s] >>> {name}", flush=True)
        if cmd is None:
            if not blocking_gate():
                print("GATE FAILED: blocking recall did not improve; stopping before training.", flush=True)
                sys.exit(2)
            continue
        with open(logs / f"{name}.log", "w", encoding="utf-8") as f:
            rc = subprocess.call([sys.executable, "-u", *cmd], stdout=f, stderr=subprocess.STDOUT)
        print(f"[{time.time() - t0:7.0f}s] <<< {name} exit={rc} ({(time.time() - t) / 60:.1f} min)", flush=True)
        if rc != 0:
            print(f"FAILED at {name}; see {logs / (name + '.log')}", flush=True)
            sys.exit(rc)
    print(f"ALL DONE in {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
