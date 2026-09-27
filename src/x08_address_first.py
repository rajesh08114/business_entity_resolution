"""Address-first blocking test (no name required): for held-out S1 entities, join directly on RARE address keys
(specific street/locality tokens + exact house number) against the full S2/S3 pool, with no top-K cutoff. Measures how
many true pairs this recovers beyond today's candidate set, and at what candidate-count cost.

Two key families, both already used elsewhere in the pipeline but capped by top-K there:
  house-number compound : "<rare_token>_<house_number>"  (rare_token = an address token below a document-frequency cap)
  postal / long number   : any 5+ digit token, taken alone (PIN / ZIP codes are already near-unique)
Rarity is controlled by a document-frequency cap (a token used by more than CAP records is not "rare").
"""
import time
from collections import Counter

import numpy as np
import pandas as pd

from common import WORK_DIR, pack_pair
from s03_train import load_truth, u01

HF = 0.15
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.0f}s] {m}", flush=True)


def addr_first_keys(strings, dfc, cap):
    """rows, keys for the house-number-compound + postal key families (no top-K, exact match only)."""
    rows, keys = [], []
    for i, s in enumerate(strings):
        toks = s.split()
        nums = [t for t in toks if t.isdigit()]
        rare = [t for t in toks if not t.isdigit() and dfc.get(t, 0) <= cap and len(t) >= 3]
        ks = set()
        for n in nums:
            if len(n) >= 5:
                ks.add("P" + n)
            for r in rare:
                ks.add("H" + r + "_" + n)
        keys.extend(ks)
        rows.extend([i] * len(ks))
    return np.asarray(rows, dtype=np.int32), np.asarray(keys, dtype=object)


def main():
    truth, n_true = load_truth()
    results = []
    for country in ("India", "US"):
        cdir = WORK_DIR / "train" / country
        R = pd.read_parquet(cdir / "records.parquet", columns=["src", "num", "addr_n"])
        s1 = R[R.src == 1].reset_index(drop=True)
        u = u01(s1.num.to_numpy())
        hold_idx = np.flatnonzero((u >= HF * 0.7) & (u < HF))
        hold_nums = s1.num.to_numpy()[hold_idx]
        tr = truth[np.isin(truth >> 33, hold_nums)]
        log(f"{country}: held-out S1 {len(hold_idx):,}, true pairs {len(tr):,}")
        cand = pd.read_parquet(cdir / "cand.parquet", columns=["s1_num", "b_num", "src"])
        cand = cand[np.isin(cand.s1_num.to_numpy(), hold_nums)]
        today_keys = pack_pair(cand.s1_num.to_numpy(), cand.b_num.to_numpy(), cand.src.to_numpy())
        today_miss = tr[~np.isin(tr, today_keys)]
        log(f"{country}: today's candidates miss {len(today_miss):,} true pairs")

        all_addr = R.addr_n.to_numpy()
        dfc = Counter()
        for s in all_addr:
            dfc.update(set(s.split()))
        n_a = len(all_addr)
        b = {src: R[R.src == src].reset_index(drop=True) for src in (2, 3)}
        # document-frequency cap = share of the country's records a token may appear in and still count as "rare"
        for cap in (int(0.00005 * n_a), int(0.0002 * n_a), int(0.001 * n_a)):
            cap = max(cap, 5)
            a_r, a_k = addr_first_keys(s1.addr_n.to_numpy()[hold_idx], dfc, cap)
            n_cand = 0
            found_keys = []
            for src, bdf in b.items():
                b_r, b_k = addr_first_keys(bdf.addr_n.to_numpy(), dfc, cap)
                df = pd.merge(pd.DataFrame({"i": a_r, "k": a_k}), pd.DataFrame({"j": b_r, "k": b_k}), on="k")
                pairs = df[["i", "j"]].drop_duplicates()
                n_cand += len(pairs)
                key = pack_pair(hold_nums[pairs.i.to_numpy()], bdf.num.to_numpy()[pairs.j.to_numpy()], np.full(len(pairs), src))
                found_keys.append(key)
            found = np.unique(np.concatenate(found_keys)) if found_keys else np.zeros(0, np.int64)
            recovered = np.isin(today_miss, found).sum()
            new_beyond_today = np.setdiff1d(found, today_keys)
            results.append({"country": country, "cap": cap, "true pairs": len(tr), "misses recovered": int(recovered),
                            "of misses": len(today_miss), "extra candidates (beyond today)": len(new_beyond_today),
                            "extra cands per S1": len(new_beyond_today) / len(hold_idx)})
            log(f"  cap={cap:>4}: recovers {recovered:,}/{len(today_miss):,} misses, "
                f"+{len(new_beyond_today):,} extra candidates ({len(new_beyond_today) / len(hold_idx):.2f} per S1)")
    df = pd.DataFrame(results)
    pd.set_option("display.width", 200)
    print("\n=== address-first (no name) blocking: recovery vs. candidate cost ===")
    print(df.round(3).to_string(index=False))
    g = df.groupby("cap")[["misses recovered", "of misses", "extra candidates (beyond today)"]].sum()
    g["recovered rate"] = g["misses recovered"] / g["of misses"]
    print("\n=== combined (both countries) ===")
    print(g.round(4).to_string())
    df.to_csv(WORK_DIR / "address_first.csv", index=False)


if __name__ == "__main__":
    main()
