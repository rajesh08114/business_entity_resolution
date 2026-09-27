"""Neural matcher v2: continue fine-tuning the v1 cross-encoder on ~4.5M training-entity pairs, with examples mined from the
stage-1 model's own mistakes (hard negatives: wrong pairs it scores >= 0.02; hard positives: true pairs it scores < 0.9).
Output: nn/model_v2. Held-out entities are never used."""
import json
import math
import time

import lightgbm as lgb  # keep before pandas/numpy on this Windows setup
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import WORK_DIR
from features import ALL_COLS
from nn_rescorer import BATCH, MAX_LEN, NN_DIR, device, pair_text, predict, texts

N_SCAN = 12_000_000
QUOTA = {"hard_pos": 400_000, "pos": 1_400_000, "hard_neg": 1_500_000, "similar_neg": 700_000, "rand_neg": 500_000}
LR = 2e-5
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.0f}s] {m}", flush=True)


def mine():
    cache = WORK_DIR / "cache"
    meta = pd.read_parquet(cache / "meta.parquet", columns=["country", "s1_num", "b_num", "src", "hold", "y"])
    dec = json.load(open(WORK_DIR / "decision.json"))
    boosters = [lgb.Booster(model_file=str(WORK_DIR / n)) for n in dec["models"]]
    ci = [ALL_COLS.index(c) for c in ("n_tset", "a_tset", "a_missing")]
    rng = np.random.default_rng(7)
    parts = []
    for c in sorted(meta.country.unique()):
        mc = meta[meta.country == c].reset_index(drop=True)
        X = np.load(cache / f"X_{c}.npy", mmap_mode="r")
        tr = np.flatnonzero(~mc.hold.to_numpy())
        pick = np.sort(rng.choice(tr, size=min(len(tr), N_SCAN * len(mc) // len(meta)), replace=False))
        p = np.empty(len(pick), np.float32)
        f = np.empty((len(pick), 3), np.float32)
        for lo in range(0, len(pick), 1_000_000):
            A = np.asarray(X[pick[lo:lo + 1_000_000]])
            p[lo:lo + 1_000_000] = np.mean([b.predict(A) for b in boosters], axis=0)
            f[lo:lo + 1_000_000] = A[:, ci]
        d = mc.iloc[pick][["country", "s1_num", "b_num", "src", "y"]].reset_index(drop=True)
        d["p1"], d["n_tset"], d["a_tset"], d["a_missing"] = p, f[:, 0], f[:, 1], f[:, 2]
        parts.append(d)
        log(f"{c}: scanned {len(d):,} training rows with the stage-1 model")
    d = pd.concat(parts, ignore_index=True)
    y = d.y.to_numpy()
    groups = {
        "hard_pos": d[y & (d.p1 < 0.9)],
        "pos": d[y & (d.p1 >= 0.9)],
        "hard_neg": d[~y & (d.p1 >= 0.02)],
        "similar_neg": d[~y & (d.p1 < 0.02) & ((d.n_tset >= 0.7) | ((d.a_missing < 0.5) & (d.a_tset >= 0.7)))],
        "rand_neg": d[~y & (d.p1 < 0.02)],
    }
    s = []
    for g, frame in groups.items():
        k = min(QUOTA[g], len(frame))
        s.append(frame.sample(k, random_state=1).assign(group=g))
        log(f"  {g}: {k:,} of {len(frame):,} available")
    s = pd.concat(s).drop_duplicates(["country", "s1_num", "b_num", "src"]).sample(frac=1, random_state=3).reset_index(drop=True)
    return s


def main():
    s = mine()
    ta = np.empty(len(s), object)
    tb = np.empty(len(s), object)
    for c, g in s.groupby("country"):
        T = texts("train", c)
        a, b = pair_text(T, g.s1_num.to_numpy(), g.b_num.to_numpy(), g.src.to_numpy())
        ta[g.index.to_numpy()], tb[g.index.to_numpy()] = a, b
    yv = s.y.to_numpy().astype(np.float32)
    n_val = 30_000
    val = (ta[:n_val], tb[:n_val], yv[:n_val])
    ta, tb, yv = ta[n_val:], tb[n_val:], yv[n_val:]
    log(f"training pairs {len(yv):,} (positives {yv.mean():.3f})")
    dev = device()
    tok = AutoTokenizer.from_pretrained(NN_DIR / "model")
    model = AutoModelForSequenceClassification.from_pretrained(NN_DIR / "model").to(dev)
    from sklearn.metrics import average_precision_score
    p0 = predict(model, tok, list(val[0]), list(val[1]))
    log(f"v1 on the v2 validation set (hard examples): AP {average_precision_score(val[2], p0):.4f}")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = math.ceil(len(yv) / BATCH)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, k / (0.03 * steps)) * max(0.0, (steps - k) / (steps * 0.97)))
    scaler = torch.amp.GradScaler("cuda")
    lossf = torch.nn.BCEWithLogitsLoss()
    order = np.random.default_rng(0).permutation(len(yv))
    for k, i in enumerate(range(0, len(yv), BATCH), 1):
        idx = order[i:i + BATCH]
        enc = tok(list(ta[idx]), list(tb[idx]), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(dev)
        yb = torch.tensor(yv[idx], device=dev)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = lossf(model(**enc).logits.squeeze(-1), yb)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if k % 2000 == 0:
            log(f"step {k:,}/{steps:,} loss {loss.item():.4f}")
        if k % 20000 == 0:
            model.save_pretrained(NN_DIR / "model_v2")
            tok.save_pretrained(NN_DIR / "model_v2")
    model.save_pretrained(NN_DIR / "model_v2")
    tok.save_pretrained(NN_DIR / "model_v2")
    p1 = predict(model, tok, list(val[0]), list(val[1]))
    log(f"v2 on the same validation set: AP {average_precision_score(val[2], p1):.4f}  (v1 {average_precision_score(val[2], p0):.4f})")


if __name__ == "__main__":
    main()
