"""Experiment (Phase 1): which preprocessing / blocking changes recover missed true pairs?

For the held-out S1 entities (same hash split as s03_train.py) blocking is re-run against the FULL S2/S3 pools of each
country, adding one change at a time, and pair recall + candidates per S1 are measured. Forward top-k for one S1 row does
not depend on the other S1 rows, so blocking only the held-out rows is exact. Maps learned from labels (address lexicon,
aliases) use non-held-out entities only.
"""
import json
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from blocking import ADDR_GENERIC_SHARE, NAME_GENERIC_SHARE, _a_chunk, _content, addr_keys, build_index, gen_keys, name_keys, token_df
from common import CAP_FRAC, MIN_CAP, POOL, WORK_DIR, pack_pair
from normalize import learn_lexicon
from s03_train import load_truth, u01

HF = 0.15
T0 = time.time()
OUT = WORK_DIR / "exp_blocking"
OUT.mkdir(exist_ok=True)


def log(m):
    print(f"[{time.time() - T0:7.0f}s] {m}", flush=True)


ORD_EXTRA = {"thirteenth": "13", "fourteenth": "14", "fifteenth": "15", "sixteenth": "16", "seventeenth": "17", "eighteenth": "18",
             "nineteenth": "19", "twentieth": "20", "thirtieth": "30", "fortieth": "40", "fiftieth": "50", "sixtieth": "60",
             "seventieth": "70", "eightieth": "80", "ninetieth": "90", "hundredth": "100"}
NULLS = {"null", "nan", "none", "na", "nil"}


def fix_numbers(s):
    out = []
    for t in s.split():
        if t in NULLS:
            continue
        t = ORD_EXTRA.get(t, t)
        if t.isdigit():
            t = t.lstrip("0") or "0"
        out.append(t)
    return " ".join(out)


def learn_alias(s1_addrs, b_addrs, min_count=50, min_prec=0.5):
    """B token t absent from the paired S1 address -> the S1 token most often absent from the B address in those pairs."""
    cb, co = Counter(), defaultdict(Counter)
    for a, b in zip(s1_addrs, b_addrs):
        A, B = set(a.split()), set(b.split())
        ub = [t for t in B - A if t.isascii() and t.isalpha() and len(t) >= 3]
        ua = [t for t in A - B if t.isalpha() and len(t) >= 3]
        if not ub or not ua:
            continue
        for t in ub:
            cb[t] += 1
            for l in ua:
                co[t][l] += 1
    alias = {}
    for t, n in cb.items():
        if n < min_count:
            continue
        l, c = co[t].most_common(1)[0]
        if c / n >= min_prec and l != t:
            alias[t] = l
    return alias


def apply_map(s, m):
    return " ".join(m.get(t, t) for t in s.split())


LEGAL = {"private", "limited", "incorporated", "llc", "company", "corporation", "llp", "the", "and", "of", "ltd", "pvt", "inc", "co", "com"}
VOWELS = set("aeiou")


def glued_keys(s, ctx):
    toks = [t for t in s.split() if t not in LEGAL]
    ns = "".join(toks)
    if len(ns) >= 8 and ns.isascii():
        return {"g" + ns[:10]}
    return set()


def skeleton(t):
    sk = t[0] + "".join(c for c in t[1:] if c not in VOWELS)
    return "".join(c for i, c in enumerate(sk) if i == 0 or c != sk[i - 1])


def typo_keys(s, ctx):
    dfc, max_df = ctx
    keys = set()
    for t in _content(s.split(), dfc, max_df, 2):
        if 5 <= len(t) <= 12 and t.isascii() and t.isalpha():
            sk = skeleton(t)
            if len(sk) >= 4:
                keys.add("k" + sk)
            keys.add("d" + t)
            keys.update("d" + t[:i] + t[i + 1:] for i in range(len(t)))
    return keys


def merge(*parts):
    rows = np.concatenate([p[0] for p in parts])
    keys = np.concatenate([p[1] for p in parts])
    o = np.argsort(rows, kind="stable")
    return rows[o], keys[o]


def block(a_keys, n_a, b_keys, n_b, top_k, kn=0, ka=0, row_chunk=2000):
    """Current blocking (raw pool of POOL, cosine re-rank, top_k) plus optional name-only / address-only channels."""
    cap = max(MIN_CAP, int(CAP_FRAC * n_b))
    BnT, BaT, uniq, keep, idf = build_index(b_keys, n_b, cap)
    V = len(uniq)
    mbn = np.asarray(BnT.power(2).sum(axis=0)).ravel().astype(np.float32)
    mba = np.asarray(BaT.power(2).sum(axis=0)).ravel().astype(np.float32)
    mb = mbn + mba
    arn, akn, ara, aka = a_keys
    I, J = [], []
    for s in range(0, n_a, row_chunk):
        e = min(s + row_chunk, n_a)
        An = _a_chunk(arn, akn, s, e, uniq, keep, idf, V)
        Aa = _a_chunk(ara, aka, s, e, uniq, keep, idf, V)
        man = np.asarray(An.power(2).sum(axis=1)).ravel().astype(np.float32)
        maa = np.asarray(Aa.power(2).sum(axis=1)).ravel().astype(np.float32)
        Sn = (An @ BnT).tocsr()
        Sa = (Aa @ BaT).tocsr()
        P = (Sn + Sa).tocsr()
        for i in range(P.shape[0]):
            got = []
            lo, hi = P.indptr[i], P.indptr[i + 1]
            if hi > lo:
                d, cols = P.data[lo:hi], P.indices[lo:hi]
                if len(d) > POOL:
                    sel = np.argpartition(-d, POOL)[:POOL]
                    d, cols = d[sel], cols[sel]
                if len(d) > top_k:
                    sj = d / np.sqrt((man[i] + maa[i]) * mb[cols] + 1e-6)
                    cols = cols[np.argpartition(-sj, top_k)[:top_k]]
                got.append(cols)
            for S, k, ma, mbx in ((Sn, kn, man, mbn), (Sa, ka, maa, mba)):
                if k <= 0:
                    continue
                lo, hi = S.indptr[i], S.indptr[i + 1]
                if hi == lo:
                    continue
                d, cols = S.data[lo:hi], S.indices[lo:hi]
                if len(d) > POOL:
                    sel = np.argpartition(-d, POOL)[:POOL]
                    d, cols = d[sel], cols[sel]
                if len(d) > k:
                    sj = d / np.sqrt(ma[i] * mbx[cols] + 1e-6)
                    cols = cols[np.argpartition(-sj, k)[:k]]
                got.append(cols)
            if got:
                cols = np.unique(np.concatenate(got))
                I.append(np.full(len(cols), i + s, np.int32))
                J.append(cols)
    if not I:
        return np.zeros(0, np.int32), np.zeros(0, np.int32)
    return np.concatenate(I), np.concatenate(J)


def subset_keys(fam, rows_sel):
    rows, keys = fam
    m = np.isin(rows, rows_sel)
    r = pd.Index(rows_sel).get_indexer(rows[m]).astype(np.int32)
    o = np.argsort(r, kind="stable")
    return r[o], keys[m][o]


def main():
    truth, n_true = load_truth()
    results, learned = [], {}
    for country in ("India", "US"):
        cdir = WORK_DIR / "train" / country
        R = pd.read_parquet(cdir / "records.parquet")
        recs = {s: R[R.src == s].reset_index(drop=True) for s in (1, 2, 3)}
        del R
        s1 = recs[1]
        u = u01(s1.num.to_numpy())
        hold = (u >= HF * 0.7) & (u < HF)
        hold_idx = np.flatnonzero(hold)
        hold_nums = s1.num.to_numpy()[hold_idx]
        tr = truth[np.isin(truth >> 33, hold_nums)]
        log(f"{country}: held-out S1 {len(hold_idx):,}, true pairs {len(tr):,}")

        tl = truth[np.isin(truth >> 33, s1.num.to_numpy()[~hold])]
        tl = tl[np.random.default_rng(0).permutation(len(tl))[:600_000]]
        s1a = [fix_numbers(x) for x in s1.set_index("num").addr_n.reindex(tl >> 33).fillna("").to_numpy()]
        bn = (tl >> 1) & ((1 << 32) - 1)
        b2 = recs[2].set_index("num").addr_n.reindex(bn).fillna("").to_numpy()
        b3 = recs[3].set_index("num").addr_n.reindex(bn).fillna("").to_numpy()
        ba = [fix_numbers(x) for x in np.where(tl & 1, b3, b2)]
        keep = [i for i, (a, b) in enumerate(zip(s1a, ba)) if a and b]
        addr_lex = learn_lexicon([s1a[i] for i in keep], [ba[i] for i in keep], min_j=0.0)
        alias = learn_alias([s1a[i] for i in keep], [ba[i] for i in keep])
        amap = {**addr_lex, **alias}
        learned[country] = {"address_lexicon_size": len(addr_lex), "address_lexicon_sample": dict(list(addr_lex.items())[:40]), "aliases": alias}
        log(f"{country}: address lexicon {len(addr_lex)} tokens, aliases {len(alias)}: " + ", ".join(f"{k}->{v}" for k, v in list(alias.items())[:30]))
        del s1a, ba, b2, b3

        addr = {"N0": {s: recs[s].addr_n.to_numpy() for s in (1, 2, 3)}}
        addr["N1"] = {s: np.array([fix_numbers(x) for x in addr["N0"][s]], dtype=object) for s in (1, 2, 3)}
        addr["N2"] = {s: np.array([apply_map(x, amap) for x in addr["N1"][s]], dtype=object) for s in (1, 2, 3)}
        names = {s: recs[s].name_n.to_numpy() for s in (1, 2, 3)}
        all_names = np.concatenate([names[s] for s in (1, 2, 3)])
        ctx_n = (token_df(all_names), NAME_GENERIC_SHARE * len(all_names))
        del all_names
        ctx_a = {}
        for v in ("N0", "N1", "N2"):
            aa = np.concatenate([addr[v][s] for s in (1, 2, 3)])
            ctx_a[v] = (token_df(aa), ADDR_GENERIC_SHARE * len(aa))
        del aa
        log(f"{country}: address variants built")

        h_names = names[1][hold_idx]
        A = {"name": gen_keys(h_names, name_keys, ctx_n), "glued": gen_keys(h_names, glued_keys, None), "typo": gen_keys(h_names, typo_keys, ctx_n)}
        for v in ("N0", "N1", "N2"):
            A["addr_" + v] = gen_keys(addr[v][1][hold_idx], addr_keys, ctx_a[v])
        S1full = None

        for src in (2, 3):
            b = recs[src]
            bnum = b.num.to_numpy()
            B = {"name": gen_keys(names[src], name_keys, ctx_n), "glued": gen_keys(names[src], glued_keys, None),
                 "typo": gen_keys(names[src], typo_keys, ctx_n)}
            for v in ("N0", "N1", "N2"):
                B["addr_" + v] = gen_keys(addr[v][src], addr_keys, ctx_a[v])
            log(f"{country} S{src}: keys ready (|B|={len(b):,})")
            tr_s = tr[(tr & 1) == (1 if src == 3 else 0)]

            def run(tag, nfam, av, top_k=15, kn=0, ka=0):
                ak = merge(*[A[f] for f in nfam]) + A["addr_" + av]
                bk = merge(*[B[f] for f in nfam]) + B["addr_" + av]
                I, J = block(ak, len(hold_idx), bk, len(b), top_k, kn, ka)
                key = pack_pair(hold_nums[I], bnum[J], np.full(len(I), src))
                hit = np.isin(tr_s, key)
                results.append({"variant": tag, "country": country, "source": f"S{src}", "true pairs": len(tr_s), "found": int(hit.sum()),
                                "candidates": len(I), "S1": len(hold_idx)})
                log(f"   {tag:<42} recall {hit.mean():.4f}  cands/S1 {len(I) / len(hold_idx):.1f}")
                return tr_s[hit], len(I)

            run("A  baseline (current)", ["name"], "N0")
            run("A2 baseline with top-20", ["name"], "N0", top_k=20)
            run("B  + number cleanup", ["name"], "N1")
            run("C  + address lexicon & aliases", ["name"], "N2")
            run("D  + glued-name keys", ["name", "glued"], "N2")
            run("E  + typo keys", ["name", "glued", "typo"], "N2")
            found_f, n_f = run("F  + name/address channels 15+5+5", ["name", "glued", "typo"], "N2", 15, 5, 5)

            nf = ["name", "glued", "typo"]
            if S1full is None:
                S1full = merge(gen_keys(names[1], name_keys, ctx_n), gen_keys(names[1], glued_keys, None), gen_keys(names[1], typo_keys, ctx_n)) + \
                    gen_keys(addr["N2"][1], addr_keys, ctx_a["N2"])
            bown = (tr_s >> 1) & ((1 << 32) - 1)
            bpos = pd.Index(bnum).get_indexer(bown)
            ok = bpos >= 0
            sub_rows = np.unique(bpos[ok])
            bn_keys = subset_keys(merge(*[B[f] for f in nf]), sub_rows)
            ba_keys = subset_keys(B["addr_N2"], sub_rows)
            I, J = block(bn_keys + ba_keys, len(sub_rows), S1full, len(s1), 3)
            rkey = pack_pair(s1.num.to_numpy()[J], bnum[sub_rows[I]], np.full(len(I), src))
            found_g = np.union1d(found_f, tr_s[np.isin(tr_s, rkey)])
            add_c = 3 * len(b) / len(s1)
            results.append({"variant": "G  F + reverse top-3 (max extra cands)", "country": country, "source": f"S{src}", "true pairs": len(tr_s),
                            "found": len(found_g), "candidates": int(n_f + add_c * len(hold_idx)), "S1": len(hold_idx)})
            log(f"   {'G  F + reverse top-3':<42} recall {len(found_g) / len(tr_s):.4f}  (at most +{add_c:.1f} cands/S1)")
            del B
        del recs, addr, A, S1full

    df = pd.DataFrame(results)
    df.to_csv(OUT / "results_by_slice.csv", index=False)
    g = df.groupby("variant")[["true pairs", "found", "candidates", "S1"]].sum()
    g["S1"] = g["S1"] / 2
    g["pair recall"] = g.found / g["true pairs"]
    g["missed"] = g["true pairs"] - g.found
    g["misses recovered vs baseline"] = g.loc["A  baseline (current)", "missed"] - g.missed
    g["cands per S1"] = g.candidates / g.S1
    pd.set_option("display.width", 250)
    print("\n=== overall (both countries, both sources) ===")
    print(g[["pair recall", "missed", "misses recovered vs baseline", "cands per S1"]].round(4).to_string())
    piv = df.assign(recall=df.found / df["true pairs"]).pivot_table(index="variant", columns=["country", "source"], values="recall")
    print("\n=== pair recall by country / source ===")
    print(piv.round(4).to_string())
    with open(OUT / "learned_maps.json", "w", encoding="utf-8") as f:
        json.dump(learned, f, ensure_ascii=False, indent=1)
    g.to_csv(OUT / "results_overall.csv")


if __name__ == "__main__":
    main()
