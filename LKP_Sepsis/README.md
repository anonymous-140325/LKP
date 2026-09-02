# sepsis_lkp.py

LKP (Latent Knowledge Programming) pipeline for early sepsis detection on
the PhysioNet/CinC 2019 Sepsis Challenge dataset. Runs the balanced case
then the unbalanced case sequentially in one execution.

## Data

`Dataset.csv` is the combination of the Challenge's `training_setA` and
`training_setB` directories (each a set of per-patient `.psv`
pipe-separated files) merged into a single CSV: every patient's hourly
rows from both sets are concatenated into one long-format table (one row
per patient-hour) with a `Patient_ID` column added to identify which
`.psv` file each row came from. 40,336 patients, 1,552,210 patient-hours
total, 34 clinical variables (8 vital signs + 26 laboratory values) plus
demographics (`Age`, `Gender`, `Unit1`/`Unit2`, `HospAdmTime`, `ICULOS`)
and the `SepsisLabel` target. Prevalence: 7.27% of patients are septic.

## Pipeline

1. **Snapshot (P=0h).** One row per patient: for septic patients, the true
   clinical onset hour (first `SepsisLabel==1` hour + 6, per the
   Challenge's own label convention), clipped to the patient's last
   recorded hour; for non-septic patients, their last recorded hour.
   Values are causally forward-filled beforehand.
2. **Class balancing.** Two passes, run sequentially:
   - **Balanced pass.** Non-septic patients are downsampled (without
     replacement, seeded) to match the septic patient count, then
     shuffled.
   - **Unbalanced pass.** No balancing — all 40,336 patients are used
     as-is, at natural ~7.27% prevalence.
3. **Feature extraction.** Per clinical variable: last observed value,
   hours since last observation, an ever-observed flag, raw values at
   1–6 hours before the snapshot (local lags), and a GRU-D-style decay
   feature (`last_value * exp(-dt/tau) + population_mean * (1 - exp(-dt/tau))`,
   `tau=24h`) — 34 variables × 9 views = 306 candidates. A candidate
   column is dropped if more than 50% of its training values are missing.
   Population means and the median-imputation values are computed on the
   training split only.
4. **Split.** 80:20 train:test, stratified.
5. **Teacher (M_T).** One of 8 model families (see below), fit on the
   training split.
6. **Knowledge extraction (Ψ: M_T → K).** A surrogate
   `DecisionTreeClassifier` (`max_depth=15`, `min_samples_leaf=100`,
   `criterion="entropy"`, `class_weight="balanced"`) is fit on the SAME
   training features against the teacher's own training-set predictions
   (never ground truth) — identical procedure for every model family,
   including ones with a closed-form alternative. Each root-to-leaf path
   becomes one IF-THEN rule in K; a leaf's predicted label is the
   `class_weight`-weighted majority (`tree_.value` argmax — exactly what
   `surrogate.predict()` would return, not a raw unweighted vote).
7. **Student (M_S).** K alone (no teacher, no raw model) classifies the
   held-out test set via `classify_one()`: a patient matching exactly one
   rule's conditions is classified by that rule; on no match, the rule
   with least total threshold violation is used as a nearest-rule
   fallback (not expected to fire on a well-formed tree partition).

## Models

`tree`, `logreg`, `svm_linear`, `lda`, `gnb`, `random_forest`, `xgboost`,
`autoencoder_latent` (a supervised autoencoder mapping the feature table
to a small latent `z`, with a `DecisionTree` teacher fit on `z`).

## Usage

```bash
python sepsis_lkp.py --data Dataset.csv
```

Optional flags: `--models` (comma-separated subset or `all`),
`--max-patients` (subsample for a quick smoke test), `--out-dir` (where
`knowledge_object_*.json` files are written), `--quiet` (suppress
per-model diagnostic output, keep one summary line per model).
