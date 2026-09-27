"""How much does a model lose on a country it never saw (a labelled stand-in for France), and what recovers it?

For each direction SOURCE -> TARGET (India -> US, US -> India), using the stage-1 feature cache of a finished run:
  in-country  : trained on TARGET training rows          (what we have for India/US today)
  unseen      : trained on SOURCE training rows only     (what we have for France today)
  + percentile: features mapped to within-country percentiles before training / scoring (country-neutral scale)
  + self-train: unseen model, plus TARGET rows it is very sure about (p >= 0.97 -> match, p <= 0.03 -> non-match)
Evaluation: macro F0.5 on the TARGET held-out entities, one-owner rule, threshold 0.70 (no tuning on the target).
"""
import json
import time

import lightgbm as lgb  # keep before pandas/numpy on this Windows setup
import numpy as np
import pandas as pd

from common import WORK_DIR
from features import ALL_COLS
from s03_train import load_truth, macro_f05

CACHE = WORK_DIR / "cache"
N_TRAIN = 4_000_000
N_PSEUDO = 2_000_000
PARAMS = dict(objective="binary", learning_rate=0.06, num_leaves=127, min_child_samples=50, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=12)
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.0f}s] {m}", flush=True)


def load():
    meta = pd.read_parquet(CACHE / "meta.parquet", columns=["country", "s1_num", "b_num", "src", "hold", "hold_s1", "y"])
    out = {}
    for c in sorted(meta.country.unique()):
        mc = meta[meta.country == c].reset_index(drop=True)
        out[c] = {"meta": mc, "X": np.load(CACHE / f"X_{c}.npy", mmap_mode="r"),
                  "hold_ents": pd.read_parquet(CACHE / f"hold_entities_{c}.parquet").s1_num.to_numpy()}
    return out


def rows(X, idx):
    return np.asarray(X[np.sort(idx)])


def percentile_maps(X, rng, n=400_000):
    """Per-feature quantile grid of one country (on a random sample of its candidate rows)."""
    s = rows(X, rng.choice(X.shape[0], size=min(n, X.shape[0]), replace=False))
    return [np.unique(np.nanquantile(s[:, j], np.linspace(0, 1, 201))) for j in range(s.shape[1])]


def to_pct(A, grids):
    B = np.empty_like(A)
    for j, g in enumerate(grids):
        col = A[:, j]
        B[:, j] = np.where(np.isnan(col), np.nan, np.searchsorted(g, col, side="right") / max(len(g), 1))
    return B


def evaluate(p, mh, ents, n_true, thr=0.70):
    ck = mh.b_num.to_numpy().astype(np.int64) * 2 + (mh.src.to_numpy() == 3)
    idx = np.flatnonzero(p >= 0.05)
    o = idx[np.lexsort((-p[idx], ck[idx]))]
    own = o[np.r_[True, ck[o][1:] != ck[o][:-1]]]
    r = own[(p[own] >= thr) & mh.hold_s1.to_numpy()[own]]
    return macro_f05(ents, n_true, mh.s1_num.to_numpy()[r], mh.y.to_numpy()[r]).mean()


def predict(b, X, idx, transform=None, chunk=2_000_000):
    out = np.empty(len(idx), np.float32)
    for lo in range(0, len(idx), chunk):
        A = np.asarray(X[idx[lo:lo + chunk]])
        out[lo:lo + chunk] = b.predict(transform(A) if transform else A)
    return out


def main():
    truth, n_true = load_truth()
    D = load()
    rng = np.random.default_rng(0)
    grids = {c: percentile_maps(d["X"], rng) for c, d in D.items()}
    log("data loaded, percentile grids built")
    results = []
    for src, tgt in (("India", "US"), ("US", "India")):
        S, T = D[src], D[tgt]
        s_tr = rng.choice(np.flatnonzero(~S["meta"].hold.to_numpy()), size=N_TRAIN, replace=False)
        t_tr = rng.choice(np.flatnonzero(~T["meta"].hold.to_numpy()), size=N_TRAIN, replace=False)
        t_hold = np.flatnonzero(T["meta"].hold.to_numpy())
        mh = T["meta"].iloc[t_hold].reset_index(drop=True)
        ys, yt = S["meta"].y.to_numpy(), T["meta"].y.to_numpy()

        def fit(Xtr, ytr):
            return lgb.train(PARAMS, lgb.Dataset(Xtr, ytr.astype(np.float32)), num_boost_round=400)

        def record(tag, p):
            f = evaluate(p, mh, T["hold_ents"], n_true)
            results.append({"direction": f"{src} -> {tgt}", "setup": tag, "macro F0.5": f})
            log(f"{src} -> {tgt} | {tag:<40} {f:.5f}")

        b = fit(rows(T["X"], t_tr), yt[np.sort(t_tr)])
        record("in-country (trained on target)", predict(b, T["X"], t_hold))
        b_un = fit(rows(S["X"], s_tr), ys[np.sort(s_tr)])
        p_un = predict(b_un, T["X"], t_hold)
        record("unseen country (trained on source only)", p_un)
        b = fit(to_pct(rows(S["X"], s_tr), grids[src]), ys[np.sort(s_tr)])
        record("unseen + percentile features", predict(b, T["X"], t_hold, lambda A: to_pct(A, grids[tgt])))
        # self-training: pseudo-label target training rows (no labels used)
        pool = rng.choice(np.flatnonzero(~T["meta"].hold.to_numpy()), size=N_TRAIN, replace=False)
        pool = np.sort(pool)
        pp = predict(b_un, T["X"], pool)
        conf = (pp >= 0.97) | (pp <= 0.03)
        sel = pool[conf]
        lab = (pp[conf] >= 0.97)
        if len(sel) > N_PSEUDO:
            k = rng.choice(len(sel), N_PSEUDO, replace=False)
            sel, lab = sel[k], lab[k]
        o = np.argsort(sel)
        sel, lab = sel[o], lab[o]
        acc = (lab == yt[sel]).mean()
        log(f"pseudo-labels: {len(sel):,} rows, {lab.mean():.3f} positive, agreement with the (hidden) truth {acc:.4f}")
        Xst = np.concatenate([rows(S["X"], s_tr), rows(T["X"], sel)])
        yst = np.concatenate([ys[np.sort(s_tr)], lab])
        b = fit(Xst, yst)
        record("unseen + self-training on target", predict(b, T["X"], t_hold))
        del Xst
    df = pd.DataFrame(results).pivot(index="setup", columns="direction", values="macro F0.5")
    print(df.round(5).to_string())
    df.to_csv(WORK_DIR / "unseen_country.csv")


if __name__ == "__main__":
    main()
