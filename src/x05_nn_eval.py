"""Does the neural re-scorer help? Stage-2 cross-validation on the held-out entities, with and without the NN score,
same folds and parameters as s05_stage2_train.py. Needs cache/X2.npy + meta2.parquet (s05) and nn/nn_hold.parquet."""
import os
import sys

import lightgbm as lgb  # keep before pandas/numpy on this Windows setup
import numpy as np
import pandas as pd

from common import WORK_DIR
from s03_train import load_truth, macro_f05
from stage2 import expected_f_select

CACHE = WORK_DIR / "cache"
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=12)


def main():
    truth, n_true = load_truth()
    X2 = np.load(CACHE / "X2.npy", mmap_mode="r")
    m2 = pd.read_parquet(CACHE / "meta2.parquet")
    nn = pd.read_parquet(WORK_DIR / "nn" / os.environ.get("NN_HOLD_FILE", "nn_hold.parquet"))
    m2 = m2.merge(nn, on=["country", "s1_num", "b_num", "src"], how="left")
    nnp = m2.nn.to_numpy(np.float32)
    logit = np.log(np.clip(nnp, 1e-6, 1 - 1e-6) / (1 - np.clip(nnp, 1e-6, 1 - 1e-6)))
    extra = np.stack([nnp, logit, np.isnan(nnp).astype(np.float32)], axis=1)
    print(f"rows {len(m2):,}; with an NN score {np.isfinite(nnp).mean():.1%}")
    y = m2.y.to_numpy().astype(np.float32)
    fold = (pd.util.hash_array(m2.s1_num.to_numpy().astype(np.uint64)) % 5).astype(int)
    ents = {c: g.s1_num.unique() for c, g in m2.groupby("country")}
    cidx = {c: i for i, c in enumerate(sorted(m2.country.unique()))}
    ck = (m2.b_num.to_numpy().astype(np.int64) * 2 + (m2.src.to_numpy() == 3)) * 4 + m2.country.map(cidx).to_numpy()
    s1, yy, country = m2.s1_num.to_numpy(), m2.y.to_numpy(), m2.country.to_numpy()

    def evaluate(p):
        idx = np.flatnonzero(p >= 0.05)
        o = idx[np.lexsort((-p[idx], ck[idx]))]
        own = o[np.r_[True, ck[o][1:] != ck[o][:-1]]]
        res = {}
        for t in (0.6, 0.65, 0.7, 0.75, 0.8):
            r = own[p[own] >= t]
            res[f"thr {t}"] = np.mean(np.concatenate([macro_f05(ids, n_true, s1[r[country[r] == c]], yy[r[country[r] == c]]) for c, ids in ents.items()]))
        keep = own[expected_f_select(s1[own], p[own])]
        res["expected-F"] = np.mean(np.concatenate([macro_f05(ids, n_true, s1[keep[country[keep] == c]], yy[keep[country[keep] == c]]) for c, ids in ents.items()]))
        return res

    out = {}
    for tag, use_nn in (("stage 2 + neural re-scorer", True),):
        oof = np.zeros(len(y))
        for k in range(5):
            tr, va = fold != k, fold == k
            Xtr = np.asarray(X2[np.flatnonzero(tr)])
            if use_nn:
                Xtr = np.concatenate([Xtr, extra[tr]], axis=1)
            b = lgb.train(PARAMS, lgb.Dataset(Xtr, y[tr]), num_boost_round=300)
            del Xtr
            Xva = np.asarray(X2[np.flatnonzero(va)])
            if use_nn:
                Xva = np.concatenate([Xva, extra[va]], axis=1)
            oof[va] = b.predict(Xva)
        out[tag] = evaluate(oof)
        print(tag, {k: round(v, 5) for k, v in out[tag].items()}, flush=True)
    df = pd.DataFrame(out)
    print(df.round(5).to_string())
    df.to_csv(WORK_DIR / "nn" / ("eval_" + os.environ.get("NN_HOLD_FILE", "nn_hold.parquet") + ".csv"))


if __name__ == "__main__":
    main()
