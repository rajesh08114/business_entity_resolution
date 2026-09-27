"""Experiment (round 2): blocking changes measured on the held-out entities against the full S2/S3 pools.

Baseline = production v3 (forward top-15 + reverse top-3, glued keys, number cleanup, address maps).
Variants are applied cumulatively as token-level post-processing of the stored normalized strings:
  V1 split letters from house numbers in addresses  ("no69" -> "no 69", "pno12" -> "pno 12")
  V2 names: digit look-alikes inside words (f0rd, de1hi, 5arl), learned name spelling variants, lnc -> incorporated
  V3 names: honorific prefixes (m/s, sri, smt, dr, the) and "id <number>" removed
  V4 extra address-only candidate list (top-5 per source) - separates same-named entities by address
  V5 sibling expansion: records that are near-duplicates of an entity's 2 best candidates per source are added
Maps learned from labels use entities outside the hold-out only.
"""
import json
import re
import time

import numpy as np
import pandas as pd

from blocking import ADDR_GENERIC_SHARE, NAME_GENERIC_SHARE, _a_chunk, addr_keys, build_index, gen_keys, name_keys, token_df
from common import CAP_FRAC, MIN_CAP, POOL, WORK_DIR, pack_pair
from normalize import learn_alias
from s03_train import load_truth, u01

HF = 0.15
T0 = time.time()
OUT = WORK_DIR / "exp_blocking2"
OUT.mkdir(exist_ok=True)
TOP_K, REV_K, ADDR_K, SEEDS, SIB_K, SIB_MIN_COS = 15, 3, 5, 2, 3, 0.6


def log(m):
    print(f"[{time.time() - T0:7.0f}s] {m}", flush=True)


_ALNUM = re.compile(r"\b([a-z]+)(\d+)\b")


def split_numbers(s):
    s = _ALNUM.sub(r"\1 \2", s)
    return " ".join((t.lstrip("0") or "0") if t.isdigit() else t for t in s.split())


_LOOK = str.maketrans({"0": "o", "1": "l", "5": "s"})


def lookalikes(s):
    out = []
    for t in s.split():
        if any(c.isalpha() for c in t) and any(c.isdigit() for c in t):
            t = t.translate(_LOOK)
        out.append(t)
    return " ".join(out)


HONORIFIC = {"sri", "shri", "smt", "dr", "mr", "mrs", "ms", "the", "messrs", "mess"}
_ID = re.compile(r"\bid \d+\b")


def strip_prefix(s):
    s = _ID.sub(" ", s)
    toks = s.split()
    if len(toks) > 2 and toks[0] == "m" and toks[1] == "s":
        toks = toks[2:]
    while len(toks) > 1 and toks[0] in HONORIFIC:
        toks = toks[1:]
    return " ".join(toks)


def apply_map(s, m):
    return " ".join(m.get(t, t) for t in s.split())


def merge(*parts):
    rows = np.concatenate([p[0] for p in parts])
    keys = np.concatenate([p[1] for p in parts])
    o = np.argsort(rows, kind="stable")
    return rows[o], keys[o]


def subset_keys(fam, rows_sel):
    rows, keys = fam
    m = np.isin(rows, rows_sel)
    r = pd.Index(rows_sel).get_indexer(rows[m]).astype(np.int32)
    o = np.argsort(r, kind="stable")
    return r[o], keys[m][o]


def search(a_keys, n_a, b_keys, n_b, top_k, addr_k=0, row_chunk=2000):
    """Production ranking (raw pool, cosine re-rank, top_k) + optional address-only list. Returns I, J, cosine."""
    cap = max(MIN_CAP, int(CAP_FRAC * n_b))
    BnT, BaT, uniq, keep, idf = build_index(b_keys, n_b, cap)
    V = len(uniq)
    mbn = np.asarray(BnT.power(2).sum(axis=0)).ravel().astype(np.float32)
    mba = np.asarray(BaT.power(2).sum(axis=0)).ravel().astype(np.float32)
    mb = mbn + mba
    arn, akn, ara, aka = a_keys
    I, J, C = [], [], []
    for s in range(0, n_a, row_chunk):
        e = min(s + row_chunk, n_a)
        An = _a_chunk(arn, akn, s, e, uniq, keep, idf, V)
        Aa = _a_chunk(ara, aka, s, e, uniq, keep, idf, V)
        man = np.asarray(An.power(2).sum(axis=1)).ravel().astype(np.float32)
        maa = np.asarray(Aa.power(2).sum(axis=1)).ravel().astype(np.float32)
        Sa = (Aa @ BaT).tocsr()
        P = ((An @ BnT) + Sa).tocsr()
        for i in range(P.shape[0]):
            lo, hi = P.indptr[i], P.indptr[i + 1]
            if hi == lo:
                continue
            d, cols = P.data[lo:hi], P.indices[lo:hi]
            if len(d) > POOL:
                sel = np.argpartition(-d, POOL)[:POOL]
                d, cols = d[sel], cols[sel]
            cos = d / np.sqrt((man[i] + maa[i]) * mb[cols] + 1e-6)
            if len(d) > top_k:
                sel = np.argpartition(-cos, top_k)[:top_k]
                cols, cos = cols[sel], cos[sel]
            if addr_k:
                lo2, hi2 = Sa.indptr[i], Sa.indptr[i + 1]
                if hi2 > lo2:
                    d2, c2 = Sa.data[lo2:hi2], Sa.indices[lo2:hi2]
                    if len(d2) > POOL:
                        sel = np.argpartition(-d2, POOL)[:POOL]
                        d2, c2 = d2[sel], c2[sel]
                    if len(d2) > addr_k:
                        cos2 = d2 / np.sqrt(maa[i] * mba[c2] + 1e-6)
                        c2 = c2[np.argpartition(-cos2, addr_k)[:addr_k]]
                    extra = np.setdiff1d(c2, cols)
                    cols = np.concatenate([cols, extra])
                    cos = np.concatenate([cos, np.zeros(len(extra), np.float32)])
            I.append(np.full(len(cols), i + s, np.int32))
            J.append(cols)
            C.append(cos)
    if not I:
        z = np.zeros(0, np.int64)
        return z, z, np.zeros(0, np.float32)
    return np.concatenate(I), np.concatenate(J), np.concatenate(C)


def learn_name_alias(s1, recs, truth, hold_nums):
    tl = truth[np.isin(truth >> 33, s1.num.to_numpy()) & ~np.isin(truth >> 33, hold_nums)]
    tl = tl[np.random.default_rng(0).permutation(len(tl))[:600_000]]
    n1 = s1.set_index("num").name_n.reindex(tl >> 33).fillna("").to_numpy()
    bn = (tl >> 1) & ((1 << 32) - 1)
    b2 = recs[2].set_index("num").name_n.reindex(bn).fillna("").to_numpy()
    b3 = recs[3].set_index("num").name_n.reindex(bn).fillna("").to_numpy()
    nb = np.where(tl & 1, b3, b2)
    n1 = [lookalikes(x) for x in n1]
    nb = [lookalikes(x) for x in nb]
    return learn_alias(n1, nb, synonym_prec=1.01)  # spelling variants only


def main():
    truth, _ = load_truth()
    results, learned = [], {}
    for country in ("India", "US"):
        R = pd.read_parquet(WORK_DIR / "train" / country / "records.parquet")
        recs = {s: R[R.src == s].reset_index(drop=True) for s in (1, 2, 3)}
        del R
        s1 = recs[1]
        u = u01(s1.num.to_numpy())
        hold_idx = np.flatnonzero((u >= HF * 0.7) & (u < HF))
        hold_nums = s1.num.to_numpy()[hold_idx]
        tr = truth[np.isin(truth >> 33, hold_nums)]
        log(f"{country}: held-out S1 {len(hold_idx):,}, true pairs {len(tr):,}")
        nalias = learn_name_alias(s1, recs, truth, hold_nums)
        nalias["lnc"] = "incorporated"
        learned[country] = nalias
        log(f"{country}: {len(nalias)} name spelling variants: " + ", ".join(f"{k}->{v}" for k, v in list(nalias.items())[:40]))

        names = {"N0": {s: recs[s].name_n.to_numpy() for s in (1, 2, 3)}}
        names["N2"] = {s: np.array([apply_map(lookalikes(x), nalias) for x in names["N0"][s]], dtype=object) for s in (1, 2, 3)}
        names["N3"] = {s: np.array([strip_prefix(x) for x in names["N2"][s]], dtype=object) for s in (1, 2, 3)}
        addrs = {"A0": {s: recs[s].addr_n.to_numpy() for s in (1, 2, 3)}}
        addrs["A1"] = {s: np.array([split_numbers(x) for x in addrs["A0"][s]], dtype=object) for s in (1, 2, 3)}
        ctx = {}
        for v, d in list(names.items()) + list(addrs.items()):
            allv = np.concatenate([d[s] for s in (1, 2, 3)])
            ctx[v] = (token_df(allv), (NAME_GENERIC_SHARE if v[0] == "N" else ADDR_GENERIC_SHARE) * len(allv))
        log(f"{country}: variants built")

        def keys(nv, av, s, rows=None):
            nm, ad = names[nv][s], addrs[av][s]
            if rows is not None:
                nm, ad = nm[rows], ad[rows]
            return gen_keys(nm, name_keys, ctx[nv]) + gen_keys(ad, addr_keys, ctx[av])

        variants = [("V0 baseline (production v3)", "N0", "A0", 0, False),
                    ("V1 + split letters from numbers", "N0", "A1", 0, False),
                    ("V2 + name look-alikes & spelling", "N2", "A1", 0, False),
                    ("V3 + honorific prefixes removed", "N3", "A1", 0, False),
                    ("V4 + address-only list (top-5)", "N3", "A1", ADDR_K, False),
                    ("V5 + sibling expansion", "N3", "A1", ADDR_K, True)]
        cache = {}
        for tag, nv, av, addr_k, sib in variants:
            if (nv, av) not in cache:
                cache.clear()
                cache[(nv, av)] = {"A": keys(nv, av, 1, hold_idx), "S1": keys(nv, av, 1), "B": {s: keys(nv, av, s) for s in (2, 3)}}
                log(f"   keys for {nv}/{av} ready")
            K = cache[(nv, av)]
            found_all, n_cand = [], 0
            fwd = {}
            for src in (2, 3):
                b = recs[src]
                bnum = b.num.to_numpy()
                tr_s = tr[(tr & 1) == (1 if src == 3 else 0)]
                I, J, C = search(K["A"], len(hold_idx), K["B"][src], len(b), TOP_K, addr_k)
                fwd[src] = (I, J, C)
                keyset = pack_pair(hold_nums[I], bnum[J], np.full(len(I), src))
                n_cand += len(I)
                # reverse search restricted to records truly owned by held-out entities (recall is exact; cost estimated below)
                bpos = pd.Index(bnum).get_indexer((tr_s >> 1) & ((1 << 32) - 1))
                sub_rows = np.unique(bpos[bpos >= 0])
                Br = K["B"][src]
                rk = subset_keys((Br[0], Br[1]), sub_rows) + subset_keys((Br[2], Br[3]), sub_rows)
                Ir, Jr, _ = search(rk, len(sub_rows), K["S1"], len(s1), REV_K)
                rkey = pack_pair(s1.num.to_numpy()[Jr], bnum[sub_rows[Ir]], np.full(len(Ir), src))
                found_all.append(tr_s[np.isin(tr_s, keyset) | np.isin(tr_s, rkey)])
            n_cand += 4.3 * len(hold_idx)  # measured production cost of the reverse search per S1 (both sources)
            if sib:
                seeds_by_src = {}
                for src in (2, 3):
                    I, J, C = fwd[src]
                    df = pd.DataFrame({"i": I, "j": J, "c": C}).sort_values(["i", "c"], ascending=[True, False])
                    top = df.groupby("i").head(SEEDS)
                    seeds_by_src[src] = top
                for tsrc in (2, 3):
                    tb = recs[tsrc]
                    tnum = tb.num.to_numpy()
                    tr_t = tr[(tr & 1) == (1 if tsrc == 3 else 0)]
                    add = []
                    for ssrc in (2, 3):
                        top = seeds_by_src[ssrc]
                        seed_rows = np.unique(top.j.to_numpy())
                        Bs = K["B"][ssrc]
                        sk = subset_keys((Bs[0], Bs[1]), seed_rows) + subset_keys((Bs[2], Bs[3]), seed_rows)
                        Is, Js, Cs = search(sk, len(seed_rows), K["B"][tsrc], len(tb), SIB_K + 1)
                        ok = Cs >= SIB_MIN_COS
                        if ssrc == tsrc:
                            ok &= seed_rows[Is] != Js
                        nb = pd.DataFrame({"seed": seed_rows[Is[ok]], "nb": Js[ok]})
                        pairs = top[["i", "j"]].rename(columns={"j": "seed"}).merge(nb, on="seed")
                        add.append(pack_pair(hold_nums[pairs.i.to_numpy()], tnum[pairs.nb.to_numpy()], np.full(len(pairs), tsrc)))
                    add = np.unique(np.concatenate(add))
                    I, J, _ = fwd[tsrc]
                    existing = pack_pair(hold_nums[I], tnum[J], np.full(len(I), tsrc))
                    new = np.setdiff1d(add, existing)
                    n_cand += len(new)
                    found_all.append(tr_t[np.isin(tr_t, new)])
            found = np.unique(np.concatenate(found_all))
            results.append({"variant": tag, "country": country, "true pairs": len(tr), "found": len(found), "S1": len(hold_idx), "candidates": n_cand})
            log(f"   {tag:<38} recall {len(found) / len(tr):.4f}  missed {len(tr) - len(found):,}  cands/S1 ~{n_cand / len(hold_idx):.1f}")
            if tag.startswith("V5"):
                np.save(OUT / f"missed_{country}.npy", np.setdiff1d(tr, found))
        del recs, names, addrs, cache
    df = pd.DataFrame(results)
    g = df.groupby("variant")[["true pairs", "found", "S1", "candidates"]].sum()
    g["recall"] = g.found / g["true pairs"]
    g["missed"] = g["true pairs"] - g.found
    g["missed, scaled to full train"] = (g.missed / g["true pairs"] * 7_638_365).round()
    g["cands per S1"] = g.candidates / g.S1
    pd.set_option("display.width", 250)
    print("\n=== held-out entities, both countries ===")
    print(g[["recall", "missed", "missed, scaled to full train", "cands per S1"]].round(4).to_string())
    print(df.assign(recall=df.found / df["true pairs"]).pivot_table(index="variant", columns="country", values="recall").round(4).to_string())
    g.to_csv(OUT / "results.csv")
    with open(OUT / "name_aliases.json", "w", encoding="utf-8") as f:
        json.dump(learned, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
