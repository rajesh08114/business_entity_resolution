"""Neural matcher v3 — standalone training for a second machine (only needs the transfer bundle, not the dataset).

Bundle layout (copy from the main machine):
  <bundle>/model_v2/            fine-tuned cross-encoder v2 (starting point)
  <bundle>/pairs_pool.parquet   labelled training-entity pairs: ta, tb, y, group   (held-out entities excluded)
  <bundle>/val_hard.parquet     fixed hard validation pairs: ta, tb, y

Steps (all on this machine's GPU):
  1. score the whole pool with v2
  2. keep what v2 still gets wrong or is unsure about, plus a random share of the rest (so it does not forget)
  3. continue fine-tuning v2 on that set -> <bundle>/model_v3 (checkpoints every 10k steps; re-run resumes)
  4. report validation AP of v2 and v3 -> <bundle>/v3_report.json

Usage:  python train_v3_remote.py --bundle D:/transfer_v3 [--batch 128] [--epochs 1]
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_LEN = 96
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.0f}s] {m}", flush=True)


@torch.no_grad()
def predict(model, tok, ta, tb, batch=512):
    model.eval()
    out = np.empty(len(ta), np.float32)
    order = np.argsort([len(x) + len(y) for x, y in zip(ta, tb)])
    for i in range(0, len(ta), batch):
        idx = order[i:i + batch]
        enc = tok([ta[j] for j in idx], [tb[j] for j in idx], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            out[idx] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
        if (i // batch) % 2000 == 0 and i:
            log(f"   scored {i:,}/{len(ta):,}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1.5e-5)
    ap.add_argument("--keep_easy", type=int, default=1_500_000)
    a = ap.parse_args()
    B = Path(a.bundle)
    assert torch.cuda.is_available(), "CUDA GPU required"
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    tok = AutoTokenizer.from_pretrained(B / "model_v2")
    v2 = AutoModelForSequenceClassification.from_pretrained(B / "model_v2").cuda()
    val = pd.read_parquet(B / "val_hard.parquet")
    p_val_v2 = predict(v2, tok, list(val.ta), list(val.tb))
    log(f"v2 validation AP {average_precision_score(val.y, p_val_v2):.5f}")

    sel_path = B / "v3_training_set.parquet"
    if sel_path.exists():
        tr = pd.read_parquet(sel_path)
        log(f"reusing selected training set: {len(tr):,}")
    else:
        pool = pd.read_parquet(B / "pairs_pool.parquet")
        log(f"pool: {len(pool):,} pairs")
        pool["p_v2"] = predict(v2, tok, list(pool.ta), list(pool.tb))
        err = (pool.p_v2 - pool.y.astype(float)).abs()
        hard = pool[(err > 0.2) | ((pool.p_v2 > 0.05) & (pool.p_v2 < 0.95))]
        easy = pool.drop(hard.index).sample(min(a.keep_easy, len(pool) - len(hard)), random_state=0)
        tr = pd.concat([hard, hard[err[hard.index] > 0.5], easy]).sample(frac=1, random_state=1).reset_index(drop=True)  # clear mistakes twice
        log(f"v2 mistakes/uncertain: {len(hard):,} (clear mistakes {int((err > 0.5).sum()):,}); training set {len(tr):,}")
        tr[["ta", "tb", "y"]].to_parquet(sel_path, index=False)
        del pool

    ckpt = B / "model_v3_ckpt"
    start_model = ckpt if (ckpt / "config.json").exists() else B / "model_v2"
    state = json.load(open(ckpt / "state.json")) if (ckpt / "state.json").exists() else {"step": 0}
    model = AutoModelForSequenceClassification.from_pretrained(start_model).cuda()
    del v2
    torch.cuda.empty_cache()
    ta, tb, y = tr.ta.to_numpy(), tr.tb.to_numpy(), tr.y.to_numpy().astype(np.float32)
    steps = a.epochs * math.ceil(len(y) / a.batch)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, k / max(1, 0.03 * steps)) * max(0.0, (steps - k) / (steps * 0.97)))
    for _ in range(state["step"]):
        sched.step()
    scaler = torch.amp.GradScaler("cuda")
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    k = 0
    for ep in range(a.epochs):
        order = np.random.default_rng(ep).permutation(len(y))
        for i in range(0, len(y), a.batch):
            k += 1
            if k <= state["step"]:
                continue
            idx = order[i:i + a.batch]
            enc = tok(list(ta[idx]), list(tb[idx]), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                loss = lossf(model(**enc).logits.squeeze(-1), torch.tensor(y[idx], device="cuda"))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if k % 1000 == 0:
                log(f"step {k:,}/{steps:,} loss {loss.item():.4f}")
            if k % 10_000 == 0:
                model.save_pretrained(ckpt)
                tok.save_pretrained(ckpt)
                json.dump({"step": k}, open(ckpt / "state.json", "w"))
                model.train()
    model.save_pretrained(B / "model_v3")
    tok.save_pretrained(B / "model_v3")
    p_val_v3 = predict(model, tok, list(val.ta), list(val.tb))
    rep = {"val_pairs": len(val), "v2_ap": float(average_precision_score(val.y, p_val_v2)), "v3_ap": float(average_precision_score(val.y, p_val_v3)),
           "v2_auc": float(roc_auc_score(val.y, p_val_v2)), "v3_auc": float(roc_auc_score(val.y, p_val_v3)), "train_pairs": len(y), "steps": steps}
    json.dump(rep, open(B / "v3_report.json", "w"), indent=1)
    log(f"DONE: {rep}")


if __name__ == "__main__":
    main()
