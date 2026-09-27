"""Shared paths, IO helpers and id packing for the entity-resolution pipeline."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get("ER_DATA_DIR", r"D:\Dataset_ML_C\student_resource\dataset"))
WORK_DIR = Path(os.environ.get("ER_WORK_DIR", r"D:\Dataset_ML_C\work\full"))
OUT_DIR = Path(os.environ.get("ER_OUT_DIR", r"D:\Dataset_ML_C\submission\output"))

TOP_K = 15          # candidates kept per S1 entity per source
CAP_FRAC = 0.0002   # a blocking key is dropped when it occurs in more than CAP_FRAC * |source| records
MIN_CAP = 20
POOL = 600          # best pairs by raw shared-IDF sum that are re-ranked by cosine before keeping TOP_K
RERANK = "cos"
ADDR_K_CROWDED, CROWDED_NAME = 15, 10  # entities whose name is shared by >= 10 S1 entities get 15 address-ranked slots
ADDR_K = 5          # forward search also keeps the best ADDR_K records by address alone
REV_K = 3          # reverse search: each S2/S3 record also proposes its best REV_K S1 entities


def read_split(split):
    d = DATA_DIR / split
    return {n: pd.read_csv(d / f"{split}_source{i}.tsv", sep="\t", dtype=str) for n, i in (("s1", 1), ("s2", 2), ("s3", 3))}


def id_num(series):
    """'S2-123456789' -> 123456789 (int64)."""
    return series.str.slice(3).astype(np.int64).to_numpy()


def pack_pair(s1_num, b_num, src):
    """Unique int64 key for an (S1, B) pair. src is 2 or 3. ids are < 2**30."""
    return (s1_num.astype(np.int64) << 33) | (b_num.astype(np.int64) << 1) | (np.asarray(src) == 3).astype(np.int64)


def ids_to_str(num, src):
    return np.char.add(np.where(np.asarray(src) == 3, "S3-", "S2-"), np.asarray(num).astype(str))


def country_dir(split, country):
    p = WORK_DIR / split / country
    p.mkdir(parents=True, exist_ok=True)
    return p
