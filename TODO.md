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
- [x] Split train/val from a shared `seen` DB pool at the window level
- [x] Use `unseen` DB pool as test-only evaluation data
- [x] Generate manifest splits from `seenDBpool` / `unseenDBpool`

## Current Dataset Status

- [x] Confirm `unseenDBpool` contains a complete `3 x 3 x 3` cache-configuration grid for `matmult`
- [x] Confirm `unseenDBpool` has one `.txt` metadata file per `.db` file
- [x] Confirm `seenDBpool` contains a complete `3 x 3 x 3` cache-configuration grid for `matmult`
- [x] Confirm `seenDBpool` has one `.txt` metadata file per `.db` file
- [x] Preprocess all DBs in `db/seenDBpool` to add `preprocessed_instruction`
- [x] Preprocess all DBs in `db/unseenDBpool` to add `preprocessed_instruction`
- [x] Generate a combined `runs.json` from `db/seenDBpool` and `db/unseenDBpool`
- [x] Use `seenDBpool` for train/val and `unseenDBpool` for test
- [x] Train and evaluate on the matrix-multiplication seen/unseen dataset split
- [x] Record a baseline unseen-pool evaluation for the matrix-multiplication experiment

## Alignment Snapshot

### Already Aligned

- [x] Use SQLite trace DBs as the model input source
- [x] Use the paper-style numeric feature family, including cache-size metadata
- [x] Use non-overlapping subsequences
- [x] Use subsequence-level regression targets for `L1D`, `L1I`, and `LL`
- [x] Use `sequence_length = 200`
- [x] Use tokenizer max length `15`
- [x] Use an embedding + LSTM + linear output model
- [x] Train with `MSE`
- [x] Use seen configurations for train/val and unseen configurations for test
- [x] Report `MSE`, `RMSE`, `R2`, and `REP`
- [x] Export per-window predictions for later analysis

### Needs Code Change

- [ ] Train the tokenizer in a paper-disciplined way using the seen pool only
- [x] Fix tokenizer corpus statistics bug in [`tokenizer/train_assembly_tokenizer.py`](/home/vm/Desktop/ML/cache_prediction/tokenizer/train_assembly_tokenizer.py)
- [ ] Move tokenizer training into the same config-driven workflow as train/evaluate
- [ ] Add plots for per-window actual vs predicted cache misses
- [ ] Add plots similar to the paper’s execution trace figures
- [ ] Add benchmark/run summary tables
- [ ] Add heatmap-style summaries if we later have enough cache/core combinations
- [ ] Add a report summarizing results for each run and cache level
- [ ] Add Optuna-based hyperparameter optimization
- [ ] Add reduced-data hyperparameter search workflow
- [ ] Add time-limited hyperparameter trials similar to the paper

### Needs Experiment / Data Change

- [ ] Use program families closer to the paper benchmarks or equivalent target workloads
- [ ] Add multiple core-count settings where applicable
- [ ] Confirm the trace-generation process matches the paper’s intended feature semantics
- [ ] Confirm all feature columns exactly match the final paper input design
- [ ] Re-run the full workflow on the final paper-style benchmark family
- [ ] Compare the final results against the paper’s reported behavior

## Must Do For Paper Reproduction

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

- [x] Confirm the real cache metadata for the current matrix-multiplication runs
- [x] Confirm the train / val / test split policy for the current seen/unseen experiment
- [ ] Confirm the real cache metadata for every final paper-style run
- [ ] Confirm the real core-count metadata for every final run
- [ ] Confirm train / val / test split policy for the final paper-style experiment
- [ ] Remove sanity-run limitations when moving to final experiments
- [ ] Re-run training and evaluation on the full intended datasets
- [ ] Compare resulting metrics and behavior against the paper

## Nice To Have

- [ ] Add command-line arguments to override config values
- [ ] Add checkpoint versioning
- [ ] Add seed control for reproducibility
- [ ] Add a small README section for how to reproduce the experiment
