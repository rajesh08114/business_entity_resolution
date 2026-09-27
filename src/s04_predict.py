"""Step 4: score the test candidates, apply the decision rule and write output/matching_results.tsv + candidate_pairs.tsv."""
import gc
import json
import sys
import time

import lightgbm as lgb  # keep before pandas/numpy on this Windows setup
import numpy as np
import pandas as pd

from common import DATA_DIR, OUT_DIR, WORK_DIR, ids_to_str
from features import COMP_COLS, PairFeaturizer, competition_features

T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.0f}s] {m}", flush=True)


def score_country(cdir, boosters):
    cand = pd.read_parquet(cdir / "cand.parquet")
    recs_all = pd.read_parquet(cdir / "records.parquet")
    recs = {s: recs_all[recs_all.src == s].reset_index(drop=True) for s in (1, 2, 3)}
    del recs_all
    dup = recs[1].groupby("name_n").num.transform("size")
    comp = competition_features(cand, pd.Series(dup.to_numpy(), index=recs[1].num.to_numpy()))[COMP_COLS].to_numpy(np.float32)
    pf = PairFeaturizer(str(cdir / "idf.pkl"), recs)
    p = np.empty(len(cand), np.float32)
    for lo, hi, base in pf.iter_chunks(cand):
        X = np.concatenate([base, comp[lo:hi]], axis=1)
        p[lo:hi] = np.mean([b.predict(X) for b in boosters], axis=0)
        if (lo // pf.chunk) % 10 == 0:
            log(f"  {cdir.name}: scored {hi:,}/{len(cand):,}")
    pf.close()
    cand["p"] = p
    cand[["s1_num", "b_num", "src", "p"]].to_parquet(cdir / "pred.parquet", index=False)
    return cand[["s1_num", "b_num", "src", "p"]]


def join_ids(df):
    """df: s1_num, b_num, src -> Series s1_num -> 'S2-..,S3-..' (ids sorted by source then number)."""
    if len(df) == 0:
        return pd.Series(dtype=object)
    df = df.sort_values(["s1_num", "src", "b_num"], kind="stable")
    ids = pd.Series(ids_to_str(df.b_num.to_numpy(), df.src.to_numpy()).astype(object))
    return ids.groupby(df.s1_num.to_numpy()).agg(",".join)


def write_tsv(path, header, s1_ids, s1_nums, lookup):
    vals = pd.Series(s1_nums).map(lookup).fillna("").to_numpy()
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(header + "\n")
        for sid, v in zip(s1_ids, vals):
            f.write(f"{sid}\t{v}\n")


def main(rescore=True):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    decision = json.load(open(WORK_DIR / "decision.json"))
    boosters = [lgb.Booster(model_file=str(WORK_DIR / n)) for n in decision.get("models", ["model.txt"])]
    thr = json.load(open(WORK_DIR / "decision.json"))["threshold"]
    log(f"threshold {thr}")
    s1 = pd.read_csv(DATA_DIR / "test" / "test_source1.tsv", sep="\t", dtype=str, usecols=["entity_id"])
    s1_nums = s1.entity_id.str.slice(3).astype(np.int64).to_numpy()
    matches, cands = [], []
    for cdir in sorted((WORK_DIR / "test").iterdir()):
        if rescore or not (cdir / "pred.parquet").exists():
            pred = score_country(cdir, boosters)
        else:
            pred = pd.read_parquet(cdir / "pred.parquet")
        cand = pd.read_parquet(cdir / "cand.parquet", columns=["s1_num", "b_num", "src"])
        cands.append(join_ids(cand))
        del cand
        keep = pred[pred.p >= 0.05]
        bk = keep.b_num.to_numpy().astype(np.int64) * 2 + (keep.src.to_numpy() == 3)
        win = keep.iloc[pd.Series(keep.p.to_numpy()).groupby(bk).idxmax().to_numpy()]
        matches.append(join_ids(win[win.p >= thr]))
        log(f"{cdir.name}: {len(pred):,} scored, {int((win.p >= thr).sum()):,} accepted pairs")
        del pred, keep, win
        gc.collect()
    match_map = pd.concat(matches)
    cand_map = pd.concat(cands)
    write_tsv(OUT_DIR / "matching_results.tsv", "source1_entity_id\tmatched_entity_ids", s1.entity_id.to_numpy(), s1_nums, match_map)
    write_tsv(OUT_DIR / "candidate_pairs.tsv", "source1_entity_id\tcandidate_entity_ids", s1.entity_id.to_numpy(), s1_nums, cand_map)
    log(f"written; S1 rows {len(s1):,}, entities with >=1 match {len(match_map):,} ({len(match_map) / len(s1):.1%})")


if __name__ == "__main__":
    main(rescore="--reuse" not in sys.argv)
