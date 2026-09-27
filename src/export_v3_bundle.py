"""Build the transfer bundle for training neural matcher v3 on a second machine (see ../v3_remote/README.md).

Writes <out>/model_v2, pairs_pool.parquet (text pairs + labels from TRAINING entities only) and val_hard.parquet
(disjoint entities). Usage: python export_v3_bundle.py D:/Dataset_ML_C/transfer_v3
"""
import shutil
import sys
from pathlib import Path

import lightgbm  # noqa: F401  keep before pandas/numpy on this Windows setup
import numpy as np
import pandas as pd

import nn_train_v2 as m2
from nn_rescorer import NN_DIR, pair_text, texts

if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    m2.N_SCAN = 20_000_000
    m2.QUOTA.update({"hard_pos": 800_000, "pos": 1_500_000, "hard_neg": 2_500_000, "similar_neg": 1_200_000, "rand_neg": 1_000_000})
    s = m2.mine()
    ta = np.empty(len(s), object)
    tb = np.empty(len(s), object)
    for c, g in s.groupby("country"):
        T = texts("train", c)
        a, b = pair_text(T, g.s1_num.to_numpy(), g.b_num.to_numpy(), g.src.to_numpy())
        ta[g.index.to_numpy()], tb[g.index.to_numpy()] = a, b
    df = pd.DataFrame({"ta": ta, "tb": tb, "y": s.y.to_numpy(), "group": s.group.to_numpy(), "country": s.country.to_numpy()})
    is_val = (pd.util.hash_array(s.s1_num.to_numpy().astype(np.uint64)) % 100) == 0  # 1% of entities, disjoint from training
    df[is_val].sample(min(60_000, int(is_val.sum())), random_state=0)[["ta", "tb", "y"]].to_parquet(out / "val_hard.parquet", index=False)
    df[~is_val].to_parquet(out / "pairs_pool.parquet", index=False)
    if not (out / "model_v2").exists():
        shutil.copytree(NN_DIR / "model_v2", out / "model_v2")
    print(f"bundle written to {out}: pool {int((~is_val).sum()):,} pairs, validation {min(60_000, int(is_val.sum())):,} pairs")
