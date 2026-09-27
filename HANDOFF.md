# Handoff: ML Challenge 2026 — Business Entity Resolution

Read this first in a new Claude Code session on the second laptop. Task: train neural matcher v3 (GPU) and return it to the
main machine. Everything else (dataset, feature cache, test scoring) stays on the main machine.

## Challenge (short)
Match Source 1 entities to Source 2/3 records (train: US, India; test adds France). Metric: macro F0.5 per Source 1 entity
(singletons count). Model must be MIT/Apache and <= 8B parameters; no external lookups. Leaderboard = `matching_results.tsv`.

## Pipeline (main machine, `src/`)
normalize -> blocking (sparse IDF keys, reverse search, address lists) -> stage-1 LightGBM (2 models averaged, 47 features)
-> neural cross-encoder score (paraphrase-multilingual-MiniLM-L12-v2, Apache-2.0, fine-tuned) on uncertain pairs
-> stage-2 LightGBM with context features + neural score -> threshold + one-owner-per-record rule.
`run_pipeline.py` rebuilds everything; steps s01..s06; neural code in `nn_rescorer.py`, `nn_train_v2.py`.

## Results so far
| Version | Held-out macro F0.5 (100,124 train entities) | Leaderboard |
|---|---|---|
| v4 | 0.9822 | 0.974 |
| v5 | 0.9834 | 0.9754 |
| v5 + French hand rules | 0.9834 | 0.9738 (worse, reverted) |
| v5 + neural v1 | 0.9847 | 0.9775 (best so far) |
| v5 + neural v2 | 0.9857 | pending (`output_v5nn2`) |
Leader: 0.9906. Held-out gains have matched leaderboard gains except France (unseen country, about -0.046 for an unseen country in a labelled test).

## Findings that shape the next steps
- Remaining held-out errors (v5+neural v1): blocking misses 6,014; model rejections 7,297; wrong matches 853 (v2: 622).
- Ambiguous records (no address + name shared by several entities) are unresolvable; guessing lowers F0.5.
- Hand rules that were NOT validated with labels hurt (French maps: -0.0016). Only ship what is measured on held-out.
- Unseen-country test (train India -> test US, and reverse): percentile features and self-training do not help;
  every feature group helps. The multilingual neural model is the only piece that helps France.

## Your job on this machine
1. Follow `v3_remote/README.md` (bundle from the main machine, `train_v3_remote.py`).
2. Bring `model_v3/` and `v3_report.json` back. The main machine compares v3 vs v2 on held-out before any submission.
3. Do NOT commit data, models or outputs (see `.gitignore`).

## Working rules from the user
- Never run `git commit` without the user's explicit go-ahead for that commit.
- Validate every submission file with `student_resource/utils/validate_submission.py` (with `--check-ids`).
- Keep the best submission (`submission/output/`) untouched until a new one beats it on held-out AND validates.
