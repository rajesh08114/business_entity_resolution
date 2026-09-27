"""Neural pair re-scorer: a small pretrained multilingual transformer (paraphrase-multilingual-MiniLM-L12-v2, Apache-2.0,
118M parameters) fine-tuned as a cross-encoder on "(S1 name | address) [SEP] (record name | address)".

It only rescores the uncertain candidates (stage-1 probability between LO and HI); its output becomes an extra stage-2
feature. Usage:
  python nn_rescorer.py train              fine-tune on training-entity rows of the stage-1 cache
  python nn_rescorer.py score_hold         score uncertain rows of the held-out entities -> nn_hold.parquet
  python nn_rescorer.py score_test         score uncertain test rows (needs pred.parquet from s04) -> test/<c>/nn.parquet
"""
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import WORK_DIR

BASE_MODEL = Path(r"D:\Dataset_ML_C\models\multilingual_minilm")
NN_DIR = WORK_DIR / "nn"
MAX_LEN, BATCH, LR, EPOCHS = 96, 64, 3e-5, 1
N_POS, N_HARD, N_RAND = 400_000, 400_000, 200_000
LO, HI = float(os.environ.get("NN_LO", 0.005)), float(os.environ.get("NN_HI", 0.995))
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.0f}s] {m}", flush=True)


def texts(split, country):
    R = pd.read_parquet(WORK_DIR / split / country / "records.parquet", columns=["src", "num", "name_n", "addr_n"])
    t = R.name_n + " | " + R.addr_n.where(R.addr_n != "", "-")
    return {s: pd.Series(t[R.src == s].to_numpy(), index=R.num[R.src == s].to_numpy()) for s in (1, 2, 3)}


def pair_text(T, s1, b, src):
    a = T[1].reindex(s1).to_numpy()
    bt = np.where(src == 3, T[3].reindex(b).to_numpy(), T[2].reindex(b).to_numpy())
    return list(a), list(bt)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_training_rows():
    """Positives, hard negatives (similar name or address) and random negatives from training-entity rows."""
    cache = WORK_DIR / "cache"
    pf = pq.ParquetFile(cache / "meta.parquet")
    names = ["n_tset", "a_tset", "a_missing"]
    from features import ALL_COLS
    cols = [ALL_COLS.index(c) for c in names]
    X = {}
    offs, rng = {}, np.random.default_rng(0)
    out = []
    start = 0
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["country", "s1_num", "b_num", "src", "hold", "y"]).to_pandas()
        t["gidx"] = np.arange(start, start + len(t))
        start += len(t)
        t = t[~t.hold]
        out.append(t.drop(columns="hold"))
    m = pd.concat(out, ignore_index=True)
    del out
    counts = pq.read_table(cache / "meta.parquet", columns=["country"]).to_pandas().country.value_counts().sort_index()
    off = 0
    for c, n in counts.items():
        offs[c] = off
        off += n
        X[c] = np.load(cache / f"X_{c}.npy", mmap_mode="r")
    pos = m[m.y]
    neg = m[~m.y]
    pos = pos.sample(min(N_POS, len(pos)), random_state=0)
    hard_parts = []
    for c in X:
        nc = neg[neg.country == c]
        loc = np.sort((nc.gidx - offs[c]).to_numpy())
        feats = np.asarray(X[c][loc][:, cols])
        hard = (feats[:, 0] >= 0.7) | ((feats[:, 2] < 0.5) & (feats[:, 1] >= 0.7))
        hard_parts.append(nc.set_index(nc.gidx - offs[c]).loc[loc[hard]])
    hard = pd.concat(hard_parts)
    hard = hard.sample(min(N_HARD, len(hard)), random_state=0)
    rand = neg.sample(N_RAND, random_state=1)
    s = pd.concat([pos, hard, rand], ignore_index=True).sample(frac=1, random_state=2).reset_index(drop=True)
    log(f"training sample: {len(s):,} rows (positives {int(s.y.sum()):,}, hard negatives {len(hard):,})")
    return s


def train():
    NN_DIR.mkdir(parents=True, exist_ok=True)
    s = sample_training_rows()
    a, b = [], []
    for c, g in s.groupby("country"):
        T = texts("train", c)
        x, y = pair_text(T, g.s1_num.to_numpy(), g.b_num.to_numpy(), g.src.to_numpy())
        s.loc[g.index, "ta"] = x
        s.loc[g.index, "tb"] = y
    val = s.iloc[:20000]
    tr = s.iloc[20000:]
    dev = device()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * math.ceil(len(tr) / BATCH)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, k / (0.05 * steps)) * max(0.0, (steps - k) / (steps * 0.95)))
    scaler = torch.cuda.amp.GradScaler()
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    k = 0
    for ep in range(EPOCHS):
        order = np.random.default_rng(ep).permutation(len(tr))
        for i in range(0, len(tr), BATCH):
            idx = order[i:i + BATCH]
            enc = tok(list(tr.ta.to_numpy()[idx]), list(tr.tb.to_numpy()[idx]), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(dev)
            yb = torch.tensor(tr.y.to_numpy()[idx], dtype=torch.float32, device=dev)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = lossf(model(**enc).logits.squeeze(-1), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            k += 1
            if k % 1000 == 0:
                log(f"step {k:,}/{steps:,} loss {loss.item():.4f}")
    model.save_pretrained(NN_DIR / "model")
    tok.save_pretrained(NN_DIR / "model")
    p = predict(model, tok, list(val.ta), list(val.tb))
    from sklearn.metrics import average_precision_score, roc_auc_score
    log(f"validation (training-entity rows not used for training): AUC {roc_auc_score(val.y, p):.4f}  AP {average_precision_score(val.y, p):.4f}")


@torch.no_grad()
def predict(model, tok, ta, tb, batch=256):
    dev = device()
    model.eval()
    out = np.empty(len(ta), np.float32)
    order = np.argsort([len(x) + len(y) for x, y in zip(ta, tb)])  # length bucketing = less padding
    for i in range(0, len(ta), batch):
        idx = order[i:i + batch]
        enc = tok([ta[j] for j in idx], [tb[j] for j in idx], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            out[idx] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
    return out


def load_trained():
    name = os.environ.get("NN_MODEL", "model")
    tok = AutoTokenizer.from_pretrained(NN_DIR / name)
    model = AutoModelForSequenceClassification.from_pretrained(NN_DIR / name).to(device())
    return model, tok


def score_hold():
    h = pd.read_parquet(WORK_DIR / "hold_pred.parquet", columns=["country", "s1_num", "b_num", "src", "hold_s1", "p"])
    h = h[h.hold_s1 & (h.p >= LO) & (h.p <= HI)].reset_index(drop=True)
    log(f"held-out uncertain rows to score: {len(h):,}")
    model, tok = load_trained()
    h["nn"] = np.nan
    for c, g in h.groupby("country"):
        T = texts("train", c)
        a, b = pair_text(T, g.s1_num.to_numpy(), g.b_num.to_numpy(), g.src.to_numpy())
        h.loc[g.index, "nn"] = predict(model, tok, a, b)
        log(f"{c}: scored {len(g):,}")
    h[["country", "s1_num", "b_num", "src", "nn"]].to_parquet(NN_DIR / os.environ.get("NN_HOLD_FILE", "nn_hold.parquet"), index=False)


def score_test():
    model, tok = load_trained()
    for cdir in sorted((WORK_DIR / "test").iterdir()):
        pr = pd.read_parquet(cdir / "pred.parquet")
        g = pr[(pr.p >= LO) & (pr.p <= HI)].reset_index(drop=True)
        T = texts("test", cdir.name)
        a, b = pair_text(T, g.s1_num.to_numpy(), g.b_num.to_numpy(), g.src.to_numpy())
        g["nn"] = predict(model, tok, a, b)
        g[["s1_num", "b_num", "src", "nn"]].to_parquet(cdir / "nn.parquet", index=False)
        log(f"{cdir.name}: scored {len(g):,} uncertain test rows")


if __name__ == "__main__":
    {"train": train, "score_hold": score_hold, "score_test": score_test}[sys.argv[1]]()
