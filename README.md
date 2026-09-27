# Business Entity Resolution — reproducible pipeline

Blocking (sparse IDF-weighted multi-key candidate generation) followed by a LightGBM pairwise matcher and a
precision-oriented decision rule. Only the provided training data is used; there are no external lookups, APIs or
pretrained models.

## Environment

```
pip install -r requirements.txt      # python 3.10, numpy, pandas, scipy, pyarrow, lightgbm, rapidfuzz
```

Notes:
* `lightgbm` must be imported before `pandas`/`numpy` on the Windows machine this was developed on (otherwise LightGBM
  crashes with an access violation). The scripts already do this.
* RAM: the largest step (US train blocking) peaks at roughly 10 GB. Feature building uses a process pool (6 workers).

## Data / output locations

Set by environment variables (defaults are in `src/common.py`):

| variable | meaning |
| --- | --- |
| `ER_DATA_DIR` | folder that contains `train/` and `test/` (the challenge `dataset/` folder) |
| `ER_WORK_DIR` | cache folder for intermediate artefacts (records, candidates, model) |
| `ER_OUT_DIR` | where `matching_results.tsv` and `candidate_pairs.tsv` are written |

## Run end to end

Run from `src/`:

```
python s01_lexicon.py            # learn the non-Latin -> Latin token lexicon from train labels
python s02_block.py test         # normalize + block the test set   (candidate set = candidate_pairs.tsv)
python s02_block.py train        # normalize + block the train set  (full density, used to train the matcher)
python s03_train.py 0.15         # features + LightGBM + threshold tuning on held-out train entities
python s04_predict.py            # score test candidates, apply the decision rule, write the two TSV files
python ../../../student_resource/utils/validate_submission.py \
    --matching ../../../output/matching_results.tsv \
    --candidate ../../../output/candidate_pairs.tsv \
    --test-dir <dataset>/test    # optional: official validator
```

Each `s02_block.py` run processes one country at a time; countries are read from the data (`country` is an open set,
nothing is hardcoded to US/India, so an unseen country such as France is handled identically).

## Modules (`src/`)

| file | role |
| --- | --- |
| `common.py` | paths, parameters (`TOP_K`, `CAP_FRAC`, `POOL`), id packing helpers |
| `normalize.py` | text normalization (accents, punctuation, abbreviations, US/India state codes, ordinals) and the learned transliteration lexicon |
| `blocking.py` | key generation and the sparse-matrix candidate search (`block_source`) |
| `features.py` | ~47 pairwise features (rapidfuzz similarities, IDF coverage, house-number/postal agreement, ...) and the competition features between candidates |
| `s01_lexicon.py` ... `s04_predict.py` | pipeline steps described above |

## Method in one paragraph

1. **Normalize** names/addresses (NFKD accent stripping, punctuation, `pvt`->`private`, `st`->`street`, state names to
   codes, `Sixth`->`6`). Non-Latin names (Devanagari, Kannada, ...) are mapped token-by-token with a lexicon learned from
   the training labels (co-occurrence with the matched Latin S1 name).
2. **Block** per country and per target source: each record gets hashed keys (name tokens, unordered token pairs of the
   rarest tokens, 4-char prefixes/suffixes, address tokens, address token pairs, postal codes). `S1 @ inverted_index(B)`
   scores all pairs sharing a key by the sum of IDF products; the 600 best are re-ranked by cosine similarity and the top 15 per
   source are kept. This candidate set is written unchanged to `candidate_pairs.tsv`.
3. **Match**: LightGBM scores every candidate from the pairwise features plus competition features (rank of a pair
   among all S1 candidates of the same B record, etc.), trained on full-density train candidates with cross-entity hold-out.
4. **Decide**: every B record is assigned to at most one S1 entity (highest probability), and pairs below the tuned
   threshold are dropped, so uncertain entities and singletons get an empty list.
