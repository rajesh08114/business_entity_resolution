"""Experiment: does more training data help? Trains on 25% / 50% / 100% of the training entities and scores the same held-out entities.

Caches the feature matrix (X.npy, y.npy, meta.parquet) under WORK_DIR/cache so later experiments can reuse it.
"""
import json
import sys
import time

import lightgbm as lgb  # keep before pandas/numpy on this Windows setup
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss

from common import WORK_DIR, pack_pair
from features import ALL_COLS, COMP_COLS, PairFeaturizer, competition_features
from s03_train import load_truth, macro_f05, u01

HFRAC = 0.15
CACHE = WORK_DIR / "cache"
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.0f}s] {m}", flush=True)


def build_cache(truth):
    CACHE.mkdir(exist_ok=True)
    metas, n_rows = [], 0
    xs = []
    for cdir in sorted((WORK_DIR / "train").iterdir()):
        country = cdir.name
        cand = pd.read_parquet(cdir / "cand.parquet")
        ra = pd.read_parquet(cdir / "records.parquet")
        recs = {s: ra[ra.src == s].reset_index(drop=True) for s in (1, 2, 3)}
        dup = recs[1].groupby("name_n").num.transform("size")
        comp = competition_features(cand, pd.Series(dup.to_numpy(), index=recs[1].num.to_numpy()))[COMP_COLS].to_numpy(np.float32)
        u = u01(cand.s1_num.to_numpy())
        bk = cand.b_num.to_numpy().astype(np.int64) * 2 + (cand.src.to_numpy() == 3)
        in_hold = (u >= HFRAC * 0.7) & (u < HFRAC)
        hold_rows = np.isin(bk, np.unique(bk[in_hold]))
        train_rows = (u < HFRAC * 0.7) & ~hold_rows
        sel = np.flatnonzero(hold_rows | train_rows)
        c = cand.iloc[sel].reset_index(drop=True)
        pf = PairFeaturizer(str(cdir / "idf.pkl"), recs)
        base = pf.compute(c)
        pf.close()
        X = np.concatenate([base, comp[sel]], axis=1)
        y = np.isin(pack_pair(c.s1_num.to_numpy(), c.b_num.to_numpy(), c.src.to_numpy()), truth)
        np.save(CACHE / f"X_{country}.npy", X)
        metas.append(pd.DataFrame({"country": country, "s1_num": c.s1_num.to_numpy(), "b_num": c.b_num.to_numpy(), "src": c.src.to_numpy(),
                                   "hold": hold_rows[sel], "hold_s1": in_hold[sel], "y": y}))
        ents = recs[1].num.to_numpy()
        u1 = u01(ents)
        pd.DataFrame({"country": country, "s1_num": ents[(u1 >= HFRAC * 0.7) & (u1 < HFRAC)]}).to_parquet(CACHE / f"hold_entities_{country}.parquet")
        log(f"{country}: cached {len(c):,} rows")
        del cand, comp, c, base, X
    pd.concat(metas, ignore_index=True).to_parquet(CACHE / "meta.parquet")


def load_cache():
    meta = pd.read_parquet(CACHE / "meta.parquet")
    X = np.empty((len(meta), len(ALL_COLS)), np.float32)
    pos = 0
    for c in sorted(meta.country.unique()):
        a = np.load(CACHE / f"X_{c}.npy", mmap_mode="r")
        X[pos:pos + len(a)] = a
        pos += len(a)
        del a
    ents = pd.concat([pd.read_parquet(CACHE / f"hold_entities_{c}.parquet") for c in sorted(meta.country.unique())])
    return X, meta, ents


def predict_chunks(booster, X, idx, it, chunk=2_000_000):
    out = np.empty(len(idx), np.float32)
    for lo in range(0, len(idx), chunk):
        out[lo:lo + chunk] = booster.predict(X[idx[lo:lo + chunk]], num_iteration=it)
    return out


def main():
    truth, n_true = load_truth()
    if not (CACHE / "meta.parquet").exists():
        build_cache(truth)
    X, meta, ents = load_cache()
    y = meta.y.to_numpy().astype(np.float32)
    hold = meta.hold.to_numpy()
    u = u01(meta.s1_num.to_numpy())
    hold_idx = np.flatnonzero(hold)
    log(f"loaded X {X.shape}, train-eligible rows {int((~hold).sum()):,}, hold rows {len(hold_idx):,}")
    params = dict(objective="binary", learning_rate=0.06, num_leaves=127, min_child_samples=50, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=12)
    h = meta.iloc[hold_idx].reset_index(drop=True)
    h["bk"] = h.b_num.astype(np.int64) * 2 + (h.src == 3)
    h["ck"] = h.bk * 4 + h.country.map({c: i for i, c in enumerate(sorted(meta.country.unique()))}).astype(np.int64)
    ent_by_c = {c: g.s1_num.to_numpy() for c, g in ents.groupby("country")}
    rows = []
    for frac in (0.25, 0.5, 1.0):
        tr_idx = np.flatnonzero(~hold & (u < HFRAC * 0.7 * frac))
        dtr = lgb.Dataset(X[tr_idx], y[tr_idx], feature_name=ALL_COLS)
        dho = lgb.Dataset(X[hold_idx[::10]], y[hold_idx[::10]], reference=dtr)
        b = lgb.train(params, dtr, num_boost_round=800, valid_sets=[dho], callbacks=[lgb.early_stopping(30, verbose=False)])
        p = predict_chunks(b, X, hold_idx, b.best_iteration)
        h["p"] = p
        yy = h.y.to_numpy()
        owned = h.loc[h[h.p >= 0.05].groupby("ck").p.idxmax()]
        best = (-1, None)
        for t in np.arange(0.5, 0.91, 0.05):
            sel = owned[(owned.p >= t) & owned.hold_s1]
            tot = n = 0
            for c, ids in ent_by_c.items():
                s = sel[sel.country == c]
                f = macro_f05(ids, n_true, s.s1_num.to_numpy(), s.y.to_numpy())
                tot += f.sum()
                n += len(f)
            if tot / n > best[0]:
                best = (tot / n, round(float(t), 2))
        hs = h.hold_s1.to_numpy()
        r = {"train fraction of entities": frac, "train rows": len(tr_idx), "trees": b.best_iteration,
             "held-out macro F0.5": best[0], "best threshold": best[1],
             "held-out AP (entity rows)": average_precision_score(yy[hs], p[hs]),
             "held-out logloss (entity rows)": log_loss(yy[hs], np.clip(p[hs], 1e-6, 1 - 1e-6))}
        rows.append(r)
        log(json.dumps(r))
    out = pd.DataFrame(rows)
    out.to_csv(WORK_DIR / "learning_curve.csv", index=False)
    print(out.round(5).to_string())


if __name__ == "__main__":
    main()
