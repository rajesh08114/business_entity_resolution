"""Step 3: stage-1 matcher. Builds features for train/hold-out rows, trains LightGBM, tunes the threshold, writes the
feature cache used by stage 2 (cache/X_<country>.npy + cache/meta.parquet) and hold-out predictions (hold_pred.parquet).

Entity split (hash of the S1 id, u in [0,1)):
  hold-out  : 0.105 <= u < 0.15   -> evaluation + stage-2 training (out-of-sample stage-1 probabilities)
  training  : u < 0.105  or  0.15 <= u < TRAIN_UP_TO
Rows:
  hold rows  = every candidate pair whose record is a candidate of some hold-out entity (the one-owner rule can then be
               evaluated exactly)
  train rows = candidate pairs of training entities whose record is not among the hold rows' records
Features are written to disk (memory-mapped) and LightGBM reads them in batches, so the training set can exceed RAM.
"""
import gc
import json
import sys
import time

import lightgbm as lgb  # keep before pandas/numpy on this Windows setup
import numpy as np
import pandas as pd

from common import DATA_DIR, WORK_DIR, id_num, pack_pair
from features import ALL_COLS, COMP_COLS, PairFeaturizer, competition_features

HFRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.15
TRAIN_UP_TO = float(sys.argv[2]) if len(sys.argv) > 2 else 0.45
SEEDS = (1, 2)
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_child_samples=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, verbose=-1, num_threads=12)
CACHE = WORK_DIR / "cache"
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.0f}s] {m}", flush=True)


def u01(s1_num):
    return ((s1_num.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(2 ** 32)).astype(np.float64) / 2 ** 32


def load_truth():
    gt = pd.read_csv(DATA_DIR / "train" / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    n_true = pd.Series(np.where(gt.matched_entity_ids == "", 0, gt.matched_entity_ids.str.count(",") + 1), index=id_num(gt.source1_entity_id))
    gt = gt[gt.matched_entity_ids != ""]
    p = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
    keys = pack_pair(id_num(p.source1_entity_id), id_num(p.m), p.m.str.get(1).astype(int).to_numpy())
    return np.sort(keys), n_true


def macro_f05(s1_nums, n_true, sel_s1, sel_true):
    """s1_nums: evaluated entities; sel_*: accepted pairs (s1 and whether correct)."""
    tot = pd.Series(1, index=sel_s1).groupby(level=0).sum().reindex(s1_nums).fillna(0).to_numpy()
    tp = pd.Series(sel_true.astype(int), index=sel_s1).groupby(level=0).sum().reindex(s1_nums).fillna(0).to_numpy()
    nt = n_true.reindex(s1_nums).to_numpy()
    prec = np.where(tot > 0, tp / np.maximum(tot, 1), 0.0)
    rec = np.where(nt > 0, tp / np.maximum(nt, 1), 0.0)
    f = np.where(prec + rec > 0, 1.25 * prec * rec / (0.25 * prec + rec + 1e-12), 0.0)
    f = np.where(nt == 0, (tot == 0).astype(float), f)
    return f


class Rows(lgb.Sequence):
    """Selected rows of a memory-mapped feature matrix, read by LightGBM in batches."""

    def __init__(self, arr, idx, batch_size=200_000):
        self.arr, self.idx, self.batch_size = arr, np.sort(idx), batch_size

    def __getitem__(self, i):
        return np.asarray(self.arr[self.idx[i]], dtype=np.float64)

    def __len__(self):
        return len(self.idx)


def build_features(truth):
    CACHE.mkdir(parents=True, exist_ok=True)
    metas, ents = [], {}
    for cdir in sorted((WORK_DIR / "train").iterdir()):
        country = cdir.name
        cand = pd.read_parquet(cdir / "cand.parquet")
        ra = pd.read_parquet(cdir / "records.parquet")
        recs = {s: ra[ra.src == s].reset_index(drop=True) for s in (1, 2, 3)}
        del ra
        u1 = u01(recs[1].num.to_numpy())
        ents[country] = recs[1].num.to_numpy()[(u1 >= HFRAC * 0.7) & (u1 < HFRAC)]
        pd.DataFrame({"country": country, "s1_num": ents[country]}).to_parquet(CACHE / f"hold_entities_{country}.parquet")
        dup = recs[1].groupby("name_n").num.transform("size")
        comp = competition_features(cand, pd.Series(dup.to_numpy(), index=recs[1].num.to_numpy()))[COMP_COLS].to_numpy(np.float32)
        u = u01(cand.s1_num.to_numpy())
        bk = cand.b_num.to_numpy().astype(np.int64) * 2 + (cand.src.to_numpy() == 3)
        in_hold = (u >= HFRAC * 0.7) & (u < HFRAC)
        in_train = (u < HFRAC * 0.7) | ((u >= HFRAC) & (u < TRAIN_UP_TO))
        hold_rows = np.isin(bk, np.unique(bk[in_hold]))
        train_rows = in_train & ~hold_rows
        sel = np.flatnonzero(hold_rows | train_rows)
        c = cand.iloc[sel].reset_index(drop=True)
        comp = comp[sel]
        del cand, bk, u
        log(f"{country}: selected {len(c):,} rows (hold {int(hold_rows.sum()):,}, train {int(train_rows.sum()):,})")
        X = np.lib.format.open_memmap(CACHE / f"X_{country}.npy", mode="w+", dtype=np.float32, shape=(len(c), len(ALL_COLS)))
        pf = PairFeaturizer(str(cdir / "idf.pkl"), recs)
        nb = len(ALL_COLS) - len(COMP_COLS)
        for lo, hi, base in pf.iter_chunks(c):
            X[lo:hi, :nb] = base
            X[lo:hi, nb:] = comp[lo:hi]
        pf.close()
        X.flush()
        del X, comp
        y = np.isin(pack_pair(c.s1_num.to_numpy(), c.b_num.to_numpy(), c.src.to_numpy()), truth)
        metas.append(pd.DataFrame({"country": country, "s1_num": c.s1_num.to_numpy(), "b_num": c.b_num.to_numpy(), "src": c.src.to_numpy(),
                                   "hold": hold_rows[sel], "hold_s1": in_hold[sel], "y": y}))
        log(f"{country}: features written, positives {int(y.sum()):,}")
        del c, recs, hold_rows, train_rows, in_hold, sel
        gc.collect()
    meta = pd.concat(metas, ignore_index=True)
    meta.to_parquet(CACHE / "meta.parquet", index=False)
    return meta, ents


def main():
    truth, n_true = load_truth()
    log(f"truth pairs {len(truth):,}")
    if (CACHE / "meta.parquet").exists():
        meta = pd.read_parquet(CACHE / "meta.parquet")
        ents = {c: pd.read_parquet(CACHE / f"hold_entities_{c}.parquet").s1_num.to_numpy() for c in sorted(meta.country.unique())}
    else:
        meta, ents = build_features(truth)
    countries = sorted(meta.country.unique())
    X = {c: np.load(CACHE / f"X_{c}.npy", mmap_mode="r") for c in countries}
    local = {}
    for c in countries:
        mc = meta.country.to_numpy() == c
        local[c] = (np.flatnonzero(mc), meta.hold.to_numpy()[mc], meta.y.to_numpy()[mc])
    tr_seq, tr_y, va_seq, va_y = [], [], [], []
    for c in countries:
        _, hold, y = local[c]
        ti = np.flatnonzero(~hold)
        vi = np.flatnonzero(hold)[::8]
        tr_seq.append(Rows(X[c], ti))
        tr_y.append(y[ti])
        va_seq.append(Rows(X[c], vi))
        va_y.append(y[vi])
    tr_y = np.concatenate(tr_y).astype(np.float32)
    va_y = np.concatenate(va_y).astype(np.float32)
    log(f"train rows {len(tr_y):,} (positives {int(tr_y.sum()):,}), early-stopping rows {len(va_y):,}")

    dtr = lgb.Dataset(tr_seq, tr_y, feature_name=ALL_COLS, free_raw_data=True)
    dva = lgb.Dataset(va_seq, va_y, reference=dtr)
    models = []
    for seed in SEEDS:
        params = dict(PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
        b = lgb.train(params, dtr, num_boost_round=2500, valid_sets=[dva], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
        name = f"model_seed{seed}.txt"
        b.save_model(str(WORK_DIR / name))
        models.append(name)
        log(f"seed {seed}: best iteration {b.best_iteration}")
    del dtr, dva
    gc.collect()

    boosters = [lgb.Booster(model_file=str(WORK_DIR / n)) for n in models]
    h = meta[meta.hold].reset_index(drop=True)
    p = np.empty(len(h), np.float32)
    pos = 0
    for c in countries:
        _, hold, _ = local[c]
        hi = np.flatnonzero(hold)
        for lo in range(0, len(hi), 1_000_000):
            rows = np.asarray(X[c][hi[lo:lo + 1_000_000]])
            p[pos:pos + len(rows)] = np.mean([b.predict(rows) for b in boosters], axis=0)
            pos += len(rows)
        log(f"{c}: hold-out rows scored")
    h["p"] = p
    h[["country", "s1_num", "b_num", "src", "hold", "hold_s1", "y", "p"]].to_parquet(WORK_DIR / "hold_pred.parquet", index=False)
    imp = sum(pd.Series(b.feature_importance("gain"), index=ALL_COLS) for b in boosters)
    print((imp / imp.sum()).sort_values(ascending=False).head(15).round(4).to_string())

    cidx = {c: i for i, c in enumerate(countries)}
    ck = (h.b_num.to_numpy().astype(np.int64) * 2 + (h.src.to_numpy() == 3)) * 4 + h.country.map(cidx).to_numpy()
    pp = h.p.to_numpy()
    idx = np.flatnonzero(pp >= 0.05)
    o = idx[np.lexsort((-pp[idx], ck[idx]))]
    own = o[np.r_[True, ck[o][1:] != ck[o][:-1]]]
    own = own[h.hold_s1.to_numpy()[own]]
    best, rows = (-1, None), []
    for t in np.arange(0.3, 0.96, 0.05):
        r = own[pp[own] >= t]
        tot_f = tot_n = 0
        per_c = {}
        for c in countries:
            rc = r[h.country.to_numpy()[r] == c]
            f = macro_f05(ents[c], n_true, h.s1_num.to_numpy()[rc], h.y.to_numpy()[rc])
            per_c[c] = f.mean()
            tot_f += f.sum()
            tot_n += len(f)
        rows.append((round(float(t), 2), tot_f / tot_n, per_c))
        if tot_f / tot_n > best[0]:
            best = (tot_f / tot_n, round(float(t), 2))
    for t, f, pc in rows:
        print(f"t={t}: macro F0.5={f:.4f}  " + "  ".join(f"{k}={v:.4f}" for k, v in pc.items()))
    log(f"best threshold {best[1]} with macro F0.5 {best[0]:.4f} on {sum(len(v) for v in ents.values()):,} held-out entities")
    with open(WORK_DIR / "decision.json", "w") as f:
        json.dump({"threshold": best[1], "holdout_macro_f05": best[0], "features": ALL_COLS, "models": models}, f)


if __name__ == "__main__":
    main()
