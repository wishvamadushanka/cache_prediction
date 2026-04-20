# TODO

Goal: reproduce the pipeline and experiment design from the paper
`Machine learning-based cache miss prediction` as closely as possible.

## Current Status

- [x] Keep DB preprocessing under `cache_prediction/preprocess`
- [x] Build a manifest-driven pipeline for runs and settings
- [x] Implement multi-run subsequence dataset loading from SQLite
- [x] Use non-overlapping subsequences
- [x] Use `sequence_length = 200`
- [x] Use tokenizer max length `15`
- [x] Implement LSTM-based regression model
- [x] Train with `MSE`
- [x] Add train / val / test split support
- [x] Add validation loss during training
- [x] Save best model by validation loss
- [x] Add test evaluation with `MSE`, `RMSE`, `R2`, `REP`
- [x] Export per-window actual vs predicted CSV
- [x] Align dataset instruction input with tokenizer-cleaned representation

## Must Do For Paper Reproduction

- [ ] Generate or collect a richer dataset with many runs across cache configurations
- [ ] Include multiple cache size combinations for `L1D`, `L1I`, and `LL`
- [ ] Include multiple core-count settings where applicable
- [ ] Build training data from seen cache configurations
- [ ] Build test data from unseen cache configurations
- [ ] Use program families closer to the paper benchmarks or equivalent target workloads
- [ ] Verify the trace-generation process matches the paper’s intended feature semantics
- [ ] Confirm all feature columns exactly match the final paper input design
- [ ] Verify tokenizer training pipeline is fully reproducible end to end
- [ ] Add Optuna-based hyperparameter optimization
- [ ] Add reduced-data hyperparameter search workflow
- [ ] Add time-limited hyperparameter trials similar to the paper

## Code / Pipeline Improvements

- [ ] Document the full pipeline order: preprocess DBs -> train tokenizer -> train model -> evaluate
- [ ] Decide whether preprocessing should stay as a manual step or be wrapped by a helper script
- [ ] Improve training/evaluation artifact naming and output organization
- [ ] Add config validation for `runs.json` and `settings.json`
- [ ] Add clearer logging for long runs
- [ ] Improve SQLite data loading efficiency for larger datasets
- [ ] Consider caching or preprocessing windows if full-scale runs are too slow
- [ ] Decide whether to keep or remove `max_rows` after sanity-run phase
- [ ] Add a reproducible command sequence for train / evaluate / predict in documentation

## Evaluation Improvements

- [ ] Add plots for per-window actual vs predicted cache misses
- [ ] Add plots similar to the paper’s execution trace figures
- [ ] Add benchmark/run summary tables
- [ ] Add heatmap-style summaries if we later have enough cache/core combinations
- [ ] Add error-vs-core-count plots if/when multicore data is available
- [ ] Add a report summarizing results for each run and cache level

## Data / Experiment Checks

- [ ] Confirm the real cache metadata for every final run
- [ ] Confirm the real core-count metadata for every final run
- [ ] Confirm train / val / test split policy for the final experiment
- [ ] Remove sanity-run limitations when moving to final experiments
- [ ] Re-run training and evaluation on the full intended datasets
- [ ] Compare resulting metrics and behavior against the paper

## Nice To Have

- [ ] Add command-line arguments to override config values
- [ ] Add checkpoint versioning
- [ ] Add seed control for reproducibility
- [ ] Add a small README section for how to reproduce the experiment
