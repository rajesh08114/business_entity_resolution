"""Pairwise features for (S1, candidate) pairs + competition features between candidates."""
import pickle
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

GENERIC_IDF = 2.5

BASE_COLS = [
    "sn", "sa", "sj", "s_tot", "is_s3",
    "n_ratio", "n_tsort", "n_tset", "n_partial", "n_jw", "n_lev", "n_nonlatin_b", "n_untrans_b",
    "a_missing", "a_ratio", "a_tsort", "a_tset",
    "n_jac", "n_core_jac", "n_core_ov", "n_core_a", "n_core_b", "n_cov_a", "n_cov_b", "n_first_eq", "n_last_eq", "n_len_ratio",
    "a_jac", "a_cov_a", "a_cov_b", "a_num_ov", "a_num_jac", "a_num_a", "a_num_b", "a_first_num_eq", "a_postal_eq", "a_last_eq",
    "a_tok_a", "a_tok_b",
    "a_num_split_jac", "a_first_num_sim", "a_num_best_sim", "a_first_num_logdiff",
]
COMP_COLS = ["rank_s1", "ratio_best_s1", "n_cand_s1", "rank_b", "ratio_best_b", "n_s1_for_b", "gap_b", "s1_name_dup"]
ALL_COLS = BASE_COLS + COMP_COLS
_A_COLS = [c for c in BASE_COLS if c.startswith("a_") and c != "a_missing"]

_IDF = {}


def _init(idf_path):
    # idf.pkl is written by s02_block.py in this same pipeline run (trusted local cache)
    with open(idf_path, "rb") as f:
        d = pickle.load(f)
    _IDF["name"], _IDF["addr"] = d["name"], d["addr"]


def _base_chunk(args):
    na, nb, aa, ab, nonlatin_b, sn, sa, sj, is_s3, workers = args
    n = len(na)
    F = np.full((n, len(BASE_COLS)), np.nan, dtype=np.float32)
    col = {c: i for i, c in enumerate(BASE_COLS)}

    def cp(x, y, scorer):
        return process.cpdist(x, y, scorer=scorer, workers=workers).astype(np.float32)

    F[:, col["sn"]] = sn
    F[:, col["sa"]] = sa
    F[:, col["sj"]] = sj
    F[:, col["s_tot"]] = sn + sa
    F[:, col["is_s3"]] = is_s3
    F[:, col["n_ratio"]] = cp(na, nb, fuzz.ratio) / 100
    F[:, col["n_tsort"]] = cp(na, nb, fuzz.token_sort_ratio) / 100
    F[:, col["n_tset"]] = cp(na, nb, fuzz.token_set_ratio) / 100
    F[:, col["n_partial"]] = cp(na, nb, fuzz.partial_ratio) / 100
    F[:, col["n_jw"]] = cp(na, nb, JaroWinkler.normalized_similarity)
    F[:, col["n_lev"]] = cp(na, nb, Levenshtein.normalized_similarity)
    F[:, col["n_nonlatin_b"]] = nonlatin_b
    miss = np.array([(x == "" or y == "") for x, y in zip(aa, ab)])
    F[:, col["a_missing"]] = miss
    ok = ~miss
    if ok.any():
        aa_ok, ab_ok = list(np.asarray(aa, dtype=object)[ok]), list(np.asarray(ab, dtype=object)[ok])
        F[ok, col["a_ratio"]] = cp(aa_ok, ab_ok, fuzz.ratio) / 100
        F[ok, col["a_tsort"]] = cp(aa_ok, ab_ok, fuzz.token_sort_ratio) / 100
        F[ok, col["a_tset"]] = cp(aa_ok, ab_ok, fuzz.token_set_ratio) / 100

    nidf, aidf = _IDF["name"], _IDF["addr"]
    c = col
    for r in range(n):
        sa_, sb_ = na[r].split(), nb[r].split()
        A, B = set(sa_), set(sb_)
        inter = A & B
        F[r, c["n_jac"]] = len(inter) / (len(A | B) or 1)
        ia = {t: nidf.get(t, 0.0) for t in A}
        ta = sum(ia.values())
        tb = sum(nidf.get(t, 0.0) for t in B)
        CA = {t for t, v in ia.items() if v > GENERIC_IDF}
        CB = {t for t in B if nidf.get(t, 0.0) > GENERIC_IDF}
        ci = CA & CB
        F[r, c["n_core_jac"]] = len(ci) / (len(CA | CB) or 1)
        F[r, c["n_core_ov"]] = len(ci)
        F[r, c["n_core_a"]] = len(CA)
        F[r, c["n_core_b"]] = len(CB)
        sh = sum(ia[t] for t in inter)
        F[r, c["n_cov_a"]] = sh / ta if ta > 0 else 0.0
        F[r, c["n_cov_b"]] = sh / tb if tb > 0 else 0.0
        F[r, c["n_first_eq"]] = 1.0 if sa_ and sb_ and sa_[0] == sb_[0] else 0.0
        F[r, c["n_last_eq"]] = 1.0 if sa_ and sb_ and sa_[-1] == sb_[-1] else 0.0
        F[r, c["n_len_ratio"]] = min(len(na[r]), len(nb[r])) / max(len(na[r]), len(nb[r]), 1)
        F[r, c["n_untrans_b"]] = sum(not t.isascii() for t in sb_)
        if miss[r]:
            continue
        qa, qb = aa[r].split(), ab[r].split()
        A, B = set(qa), set(qb)
        inter = A & B
        F[r, c["a_jac"]] = len(inter) / (len(A | B) or 1)
        ta = sum(aidf.get(t, 0.0) for t in A)
        tb = sum(aidf.get(t, 0.0) for t in B)
        sh = sum(aidf.get(t, 0.0) for t in inter)
        F[r, c["a_cov_a"]] = sh / ta if ta > 0 else 0.0
        F[r, c["a_cov_b"]] = sh / tb if tb > 0 else 0.0
        NA = {t for t in A if any(ch.isdigit() for ch in t)}
        NB = {t for t in B if any(ch.isdigit() for ch in t)}
        both = NA & NB
        F[r, c["a_num_ov"]] = len(both)
        F[r, c["a_num_jac"]] = len(both) / (len(NA | NB) or 1)
        F[r, c["a_num_a"]] = len(NA)
        F[r, c["a_num_b"]] = len(NB)
        fa = next((t for t in qa if t in NA), None)
        F[r, c["a_first_num_eq"]] = 1.0 if fa is not None and fa in NB else 0.0
        F[r, c["a_postal_eq"]] = 1.0 if any(t.isdigit() and len(t) in (5, 6) for t in both) else 0.0
        F[r, c["a_last_eq"]] = 1.0 if qa[-1] == qb[-1] else 0.0
        F[r, c["a_tok_a"]] = len(qa)
        F[r, c["a_tok_b"]] = len(qb)
        # house numbers after splitting "no69"/"pno12" and with graded (not only exact) agreement: 421 vs 21, 608 vs 610
        la, lb = _numbers(aa[r]), _numbers(ab[r])
        SA, SB = set(la), set(lb)
        F[r, c["a_num_split_jac"]] = len(SA & SB) / (len(SA | SB) or 1)
        if la and lb:
            F[r, c["a_first_num_sim"]] = Levenshtein.normalized_similarity(la[0], lb[0])
            F[r, c["a_num_best_sim"]] = max(Levenshtein.normalized_similarity(x, y) for x in la[:6] for y in lb[:6])
            F[r, c["a_first_num_logdiff"]] = np.log1p(abs(int(la[0][:9]) - int(lb[0][:9])))
    return F


_ALNUM = re.compile(r"([a-z]+)(\d+)")


def _numbers(addr):
    return [(t.lstrip("0") or "0") for t in _ALNUM.sub(r"\1 \2", addr).split() if t.isdigit()]


class PairFeaturizer:
    """Computes BASE_COLS for candidate rows of one (split, country) using a process pool."""

    def __init__(self, idf_path, recs, workers=6, chunk=250_000):
        self.recs, self.chunk, self.workers = recs, chunk, workers
        self.idx = {s: pd.Index(recs[s].num.to_numpy()) for s in (1, 2, 3)}
        self.pool = ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(idf_path,))
        self.arr = {s: (recs[s].name_n.to_numpy(object), recs[s].addr_n.to_numpy(object), recs[s].name_nonlatin.to_numpy()) for s in (1, 2, 3)}

    def close(self):
        self.pool.shutdown()

    def _chunk_args(self, cand, lo, hi):
        c = cand.iloc[lo:hi]
        src = c.src.to_numpy()
        if "l_num" in c.columns:
            lsrc, lnum = c.l_src.to_numpy(), c.l_num.to_numpy()
            na = np.empty(len(c), object)
            aa = np.empty(len(c), object)
            for s in (1, 2, 3):
                m = lsrc == s
                if m.any():
                    il = self.idx[s].get_indexer(lnum[m])
                    na[m], aa[m] = self.arr[s][0][il], self.arr[s][1][il]
        else:
            ia = self.idx[1].get_indexer(c.s1_num.to_numpy())
            n1, a1, _ = self.arr[1]
            na, aa = n1[ia], a1[ia]
        nb = np.empty(len(c), object)
        ab = np.empty(len(c), object)
        nl = np.zeros(len(c), np.float32)
        for s in (2, 3):
            m = src == s
            if m.any():
                ib = self.idx[s].get_indexer(c.b_num.to_numpy()[m])
                nn, aad, nlat = self.arr[s]
                nb[m], ab[m], nl[m] = nn[ib], aad[ib], nlat[ib]
        return (list(na), list(nb), list(aa), list(ab), nl, c.sn.to_numpy(np.float32), c.sa.to_numpy(np.float32),
                c.sj.to_numpy(np.float32), (src == 3).astype(np.float32), 1)

    def iter_chunks(self, cand, window=None):
        """Yield (lo, hi, feature_matrix) in order; keeps `window` chunks in flight."""
        window = window or self.workers * 2
        bounds = [(lo, min(lo + self.chunk, len(cand))) for lo in range(0, len(cand), self.chunk)]
        futs = []
        it = iter(bounds)
        done = False
        while futs or not done:
            while not done and len(futs) < window:
                b = next(it, None)
                if b is None:
                    done = True
                    break
                futs.append((b, self.pool.submit(_base_chunk, self._chunk_args(cand, *b))))
            if futs:
                (lo, hi), f = futs.pop(0)
                yield lo, hi, f.result()

    def compute(self, cand):
        out = np.empty((len(cand), len(BASE_COLS)), np.float32)
        for lo, hi, F in self.iter_chunks(cand):
            out[lo:hi] = F
        return out


def competition_features(cand, s1_name_dup):
    """Label-free features from the candidate graph. cand: s1_num, b_num, src, sn, sa. Returns DataFrame (COMP_COLS)."""
    t = cand.sj.to_numpy().astype(np.float32)
    s1k = cand.s1_num.to_numpy().astype(np.int64) * 2 + (cand.src.to_numpy() == 3)
    bk = cand.b_num.to_numpy().astype(np.int64) * 2 + (cand.src.to_numpy() == 3)
    g = pd.DataFrame({"s1k": s1k, "bk": bk, "t": t})
    g1 = g.groupby("s1k", sort=False).t
    f = pd.DataFrame(index=cand.index)
    f["rank_s1"] = g1.rank(ascending=False, method="first").to_numpy(np.float32)
    f["ratio_best_s1"] = (g.t / g1.transform("max")).to_numpy(np.float32)
    f["n_cand_s1"] = g1.transform("size").to_numpy(np.float32)
    gb = g.groupby("bk", sort=False).t
    rank_b = gb.rank(ascending=False, method="first")
    f["rank_b"] = rank_b.to_numpy(np.float32)
    f["ratio_best_b"] = (g.t / gb.transform("max")).to_numpy(np.float32)
    f["n_s1_for_b"] = gb.transform("size").to_numpy(np.float32)
    second = pd.Series(g.t.to_numpy()[rank_b.to_numpy() == 2], index=g.bk.to_numpy()[rank_b.to_numpy() == 2])
    f["gap_b"] = (gb.transform("max") - g.bk.map(second).fillna(0)).to_numpy(np.float32)
    f["s1_name_dup"] = cand.s1_num.map(s1_name_dup).to_numpy(np.float32)
    return f
