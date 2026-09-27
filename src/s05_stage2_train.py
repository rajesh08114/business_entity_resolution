"""Step 5: train the second-stage model on held-out entities (whose stage-1 probabilities are out-of-sample).

Uses the feature cache written by x01_learning_curve.py and hold_pred.parquet written by s03_train.py.
Stage-2 rows = candidate pairs of the held-out entities (~3M). Context features use stage-1 probabilities of ALL hold rows
(including competitor rows), so B-level competition is exact. Evaluation is 5-fold cross-validation grouped by S1 entity.
"""
import json
import sys
import time

import lightgbm as lgb  # keep before pandas/numpy on this Windows setup
import numpy as np
import pandas as pd

from common import WORK_DIR
from features import ALL_COLS, BASE_COLS, PairFeaturizer
from s03_train import load_truth, macro_f05, u01
from stage2 import ANCHOR_COLS, CTX_COLS, NN_COLS, STAGE2_EXTRA, context_features, expected_f_select, nn_features

CACHE = WORK_DIR / "cache"
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.0f}s] {m}", flush=True)


def anchor_table(df, anchor):
    has = anchor >= 0
    a = np.where(has, anchor, np.arange(len(df)))
    return pd.DataFrame({
        "l_src": df.src.to_numpy()[a], "l_num": df.b_num.to_numpy()[a],
        "b_num": df.b_num.to_numpy(), "src": df.src.to_numpy(),
        "sn": np.float32(0), "sa": np.float32(0), "sj": np.float32(0),
    }), has


def anchor_features(pf, df, anchor):
    tab, has = anchor_table(df, anchor)
    F = pf.compute(tab)
    cols = [BASE_COLS.index(c) for c in ANCHOR_COLS]
    F = F[:, cols]
    F[~has] = np.nan
    return F


def main():
    truth, n_true = load_truth()
    meta = pd.read_parquet(CACHE / "meta.parquet")
    hp = pd.read_parquet(WORK_DIR / "hold_pred.parquet")
    hold_mask = meta.hold.to_numpy()
    mh = meta[hold_mask].reset_index(drop=True)
    assert (mh.s1_num.to_numpy() == hp.s1_num.to_numpy()).all() and (mh.b_num.to_numpy() == hp.b_num.to_numpy()).all()
    mh["p"] = hp.p.to_numpy()
    # row offsets of hold rows inside each country's cached X
    parts, offsets = [], {}
    start = 0
    for c in sorted(meta.country.unique()):
        n_c = int((meta.country == c).sum())
        offsets[c] = (start, n_c)
        start += n_c
    X2_l, meta_l = [], []
    cached = (CACHE / "X2.npy").exists() and (CACHE / "meta2.parquet").exists()
    for c in ([] if cached else sorted(meta.country.unique())):
        cdir = WORK_DIR / "train" / c
        ra = pd.read_parquet(cdir / "records.parquet")
        recs = {s: ra[ra.src == s].reset_index(drop=True) for s in (1, 2, 3)}
        Xc = np.load(CACHE / f"X_{c}.npy", mmap_mode="r")
        mc = meta[meta.country == c].reset_index(drop=True)
        hold_c = mc.hold.to_numpy()
        df = mh[mh.country == c].reset_index(drop=True)
        ctx, anchor = context_features(df[["s1_num", "b_num", "src", "p"]])
        ent = df.hold_s1.to_numpy()
        pf = PairFeaturizer(str(cdir / "idf.pkl"), recs)
        sub = df[ent].reset_index(drop=True)
        anc_full = anchor[ent]
        # anchor indices refer to rows of df; rebuild a lookup df aligned with df
        tab_src = df.src.to_numpy()
        tab_num = df.b_num.to_numpy()
        a = np.where(anc_full >= 0, anc_full, np.flatnonzero(ent))
        tab = pd.DataFrame({"l_src": tab_src[a], "l_num": tab_num[a], "b_num": sub.b_num.to_numpy(), "src": sub.src.to_numpy(),
                            "sn": np.float32(0), "sa": np.float32(0), "sj": np.float32(0)})
        F = pf.compute(tab)[:, [BASE_COLS.index(k) for k in ANCHOR_COLS]]
        F[anc_full < 0] = np.nan
        pf.close()
        X1 = np.asarray(Xc[np.flatnonzero(hold_c)[ent]])
        X2 = np.concatenate([X1, ctx[ent].to_numpy(np.float32), F], axis=1)
        X2_l.append(X2)
        meta_l.append(sub[["country", "s1_num", "b_num", "src", "y", "p"]])
        log(f"{c}: stage-2 rows {len(sub):,}, features {X2.shape[1]}")
        del Xc, ctx, F, X1
    if cached:
        X2 = np.load(CACHE / "X2.npy")
        m2 = pd.read_parquet(CACHE / "meta2.parquet")
    else:
        X2 = np.concatenate(X2_l)
        m2 = pd.concat(meta_l, ignore_index=True)
    names = ALL_COLS + CTX_COLS + ["anc_" + k for k in ANCHOR_COLS]
    y = m2.y.to_numpy().astype(np.float32)
    if not cached:
        np.save(CACHE / "X2.npy", X2)
        m2.to_parquet(CACHE / "meta2.parquet")
    nn_path = WORK_DIR / "nn" / "nn_hold.parquet"
    if nn_path.exists():  # neural re-scorer score as extra features (NaN outside the uncertain zone)
        keys = ["country", "s1_num", "b_num", "src"]
        nnp = m2[keys].merge(pd.read_parquet(nn_path), on=keys, how="left").nn.to_numpy(np.float32)
        X2 = np.concatenate([X2, nn_features(nnp)], axis=1)
        names = names + NN_COLS
        log(f"neural re-scorer features added ({np.isfinite(nnp).mean():.1%} of rows scored)")

    fold = (pd.util.hash_array(m2.s1_num.to_numpy().astype(np.uint64)) % 5).astype(int)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=100, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=12)
    oof = np.zeros(len(y))
    tr_pred = np.zeros(len(y))
    train_gap = []
    imp = np.zeros(len(names))
    for k in range(5):
        tr, va = fold != k, fold == k
        b = lgb.train(params, lgb.Dataset(X2[tr], y[tr], feature_name=names), num_boost_round=300)
        oof[va] = b.predict(X2[va])
        train_gap.append((np.mean(np.abs(b.predict(X2[tr][:300000]) - y[tr][:300000])), np.mean(np.abs(oof[va] - y[va]))))
        imp += b.feature_importance("gain")
        log(f"fold {k} done")
    print("mean |p-y| train-part vs held-out fold:", np.mean(train_gap, axis=0))
    print((pd.Series(imp, index=names) / imp.sum()).sort_values(ascending=False).head(20).round(4).to_string())

    m2["p2"] = oof
    m2.to_parquet(CACHE / "meta2_oof.parquet")
    cidx = {c: i for i, c in enumerate(sorted(m2.country.unique()))}
    ents = {c: g.s1_num.unique() for c, g in m2.groupby("country")}
    ck = (m2.b_num.to_numpy().astype(np.int64) * 2 + (m2.src.to_numpy() == 3)) * 4 + m2.country.map(cidx).to_numpy()
    country_arr = m2.country.to_numpy()
    s1_arr, y_arr = m2.s1_num.to_numpy(), m2.y.to_numpy()

    def owners(col):
        p = m2[col].to_numpy()
        idx = np.flatnonzero(p >= 0.05)
        order = idx[np.lexsort((-p[idx], ck[idx]))]
        first = np.r_[True, ck[order][1:] != ck[order][:-1]]
        return order[first]

    def score(col, thr, own):
        p = m2[col].to_numpy()
        rows = own if own is not None else np.arange(len(p))
        rows = rows[p[rows] >= thr]
        tot = n = 0
        for c, ids in ents.items():
            r = rows[country_arr[rows] == c]
            f = macro_f05(ids, n_true, s1_arr[r], y_arr[r])
            tot += f.sum()
            n += len(f)
        return tot / n

    own1, own2 = owners("p"), owners("p2")
    res = []
    for thr in np.arange(0.3, 0.96, 0.05):
        res.append({"threshold": round(thr, 2), "stage1 thr-only": score("p", thr, None), "stage1 + owner": score("p", thr, own1),
                    "stage2 thr-only": score("p2", thr, None), "stage2 + owner": score("p2", thr, own2)})
    out = pd.DataFrame(res).set_index("threshold")
    print(out.round(4).to_string())
    best = out["stage2 + owner"].idxmax()
    log(f"best stage-2 threshold {best}: {out.loc[best].round(4).to_dict()}")

    # alternative decision rule: expected-F0.5-optimal candidate set per entity (over the one-owner rows)
    p2 = m2.p2.to_numpy()
    cand_rows = own2[p2[own2] >= 0.05]
    rule, rule_score, best_prior = "threshold", float(out.loc[best, "stage2 + owner"]), None
    for prior in (0.0, 0.1, 0.2, 0.3):
        keep = cand_rows[expected_f_select(s1_arr[cand_rows], p2[cand_rows], prior=prior)]
        tot = n = 0
        for c, ids in ents.items():
            r = keep[country_arr[keep] == c]
            f = macro_f05(ids, n_true, s1_arr[r], y_arr[r])
            tot += f.sum()
            n += len(f)
        log(f"expected-F0.5 rule (prior {prior}): {tot / n:.5f}")
        if tot / n > rule_score + 1e-5:
            rule, rule_score, best_prior = "expected_f", tot / n, prior
    log(f"chosen decision rule: {rule} ({rule_score:.5f})")

    final = lgb.train(params, lgb.Dataset(X2, y, feature_name=names), num_boost_round=300)
    final.save_model(str(WORK_DIR / "model2.txt"))
    with open(WORK_DIR / "decision2.json", "w") as f:
        json.dump({"threshold": float(best), "rule": rule, "expected_f_prior": best_prior, "cv_macro_f05": float(rule_score),
                   "cv_macro_f05_threshold_rule": float(out.loc[best, "stage2 + owner"]), "stage1_reference": float(out["stage1 + owner"].max()),
                   "features": names}, f)
    out.to_csv(WORK_DIR / "stage2_cv.csv")


if __name__ == "__main__":
    main()
