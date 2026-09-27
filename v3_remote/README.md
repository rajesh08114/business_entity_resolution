# Neural matcher v3 on a second machine

Only the GPU fine-tuning runs here. Data preparation, evaluation on held-out entities and test scoring stay on the main
machine (they need the full dataset and a ~20 GB feature cache).

## 1. Setup (second machine, NVIDIA GPU)

```
git clone <repo-url>
cd business_entity_resolution/v3_remote
python -m venv .venv && .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_v3.txt
```

## 2. Copy the bundle (not in git: about 1.5 GB)

From the main machine copy `D:\Dataset_ML_C\transfer_v3\` (USB drive / cloud drive) to e.g. `D:\transfer_v3`:

```
transfer_v3/
  model_v2/             neural matcher v2 (starting point)
  pairs_pool.parquet    labelled training pairs (held-out entities excluded)
  val_hard.parquet      fixed validation pairs
```

## 3. Train

```
python train_v3_remote.py --bundle D:/transfer_v3 --batch 128
```

- Scores the pool with v2 (about 20-30 min), keeps v2's mistakes / uncertain pairs + a random share of the rest, then
  fine-tunes from v2 (about 2-4 h on an RTX 4060).
- Checkpoints every 10,000 steps into `model_v3_ckpt/`; if interrupted, run the same command again to resume.
- If you get a CUDA out-of-memory error, use `--batch 64`.

## 4. Bring back

Copy `transfer_v3/model_v3/` and `transfer_v3/v3_report.json` back to the main machine
(`D:\Dataset_ML_C\work\v5\nn\model_v3`). The main machine then scores the held-out entities with v3 and compares it
with v2 (held-out macro F0.5 0.9857) before anything reaches a submission.
