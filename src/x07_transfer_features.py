"""Which feature groups stop the model from transferring to an unseen country (the France situation)?

Same setup as x06_unseen_country.py (train on SOURCE only, evaluate on TARGET held-out entities, threshold 0.70), but
with feature groups removed:
  country-statistic features: IDF-weighted blocking scores / coverages, core-token counts (depend on a country's own
                              word frequencies)
  competition features      : ranks and counts among candidates (depend on how crowded a country's names are)
  token-count features      : raw token / number counts (depend on a country's address format)
"""
import time

import lightgbm as lgb  # keep before pandas/numpy on this Windows setup
import numpy as np
import pandas as pd

from common import WORK_DIR
from features import ALL_COLS
from s03_train import load_truth
from x06_unseen_country import N_TRAIN, PARAMS, evaluate, load, rows

T0 = time.time()
GROUPS = {
    "country-statistic": ["sn", "sa", "sj", "s_tot", "n_cov_a", "n_cov_b", "n_core_jac", "n_core_ov", "n_core_a", "n_core_b",
                          "a_cov_a", "a_cov_b"],
    "competition": ["rank_s1", "ratio_best_s1", "n_cand_s1", "rank_b", "ratio_best_b", "n_s1_for_b", "gap_b", "s1_name_dup"],
    "token-count": ["a_tok_a", "a_tok_b", "a_num_a", "a_num_b", "n_untrans_b", "n_nonlatin_b"],
}
SETUPS = [
    ("all features", []),
    ("without country-statistic", ["country-statistic"]),
    ("without competition", ["competition"]),
    ("without token-count", ["token-count"]),
    ("string similarity only", ["country-statistic", "competition", "token-count"]),
]


def log(m):
    print(f"[{time.time() - T0:7.0f}s] {m}", flush=True)


def main():
    truth, n_true = load_truth()
    D = load()
    rng = np.random.default_rng(0)
    res = []
    for src, tgt in (("US", "India"), ("India", "US")):
        S, T = D[src], D[tgt]
        s_tr = np.sort(rng.choice(np.flatnonzero(~S["meta"].hold.to_numpy()), size=N_TRAIN, replace=False))
        t_tr = np.sort(rng.choice(np.flatnonzero(~T["meta"].hold.to_numpy()), size=N_TRAIN, replace=False))
        t_hold = np.flatnonzero(T["meta"].hold.to_numpy())
        mh = T["meta"].iloc[t_hold].reset_index(drop=True)
        Xs, ys = rows(S["X"], s_tr), S["meta"].y.to_numpy()[s_tr].astype(np.float32)
        Xt, yt = rows(T["X"], t_tr), T["meta"].y.to_numpy()[t_tr].astype(np.float32)
        for tag, drop in SETUPS:
            keep = [i for i, c in enumerate(ALL_COLS) if not any(c in GROUPS[g] for g in drop)]
            for mode, (Xa, ya) in (("unseen", (Xs, ys)), ("in-country", (Xt, yt))):
                if mode == "in-country" and tag not in ("all features", "string similarity only"):
                    continue
                b = lgb.train(PARAMS, lgb.Dataset(Xa[:, keep], ya), num_boost_round=400)
                p = np.empty(len(t_hold), np.float32)
                for lo in range(0, len(t_hold), 2_000_000):
                    p[lo:lo + 2_000_000] = b.predict(np.asarray(T["X"][t_hold[lo:lo + 2_000_000]])[:, keep])
                f = evaluate(p, mh, T["hold_ents"], n_true)
                res.append({"direction": f"{src} -> {tgt}", "features": tag, "trained on": mode, "macro F0.5": f})
                log(f"{src} -> {tgt} | {tag:<28} | {mode:<10} {f:.5f}  ({len(keep)} features)")
        del Xs, Xt
    df = pd.DataFrame(res).pivot_table(index=["features", "trained on"], columns="direction", values="macro F0.5")
    print(df.round(5).to_string())
    df.to_csv(WORK_DIR / "transfer_features.csv")


if __name__ == "__main__":
    main()
