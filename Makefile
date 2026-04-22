PYTHON ?= python

.PHONY: help setup tokenizer test-tokenizer train eval predict preprocess-seen preprocess-unseen manifest

help:
	@echo "Available targets:"
	@echo "  make setup             - Create/update the local virtual environment"
	@echo "  make preprocess-seen   - Preprocess DBs under ../db/seenDBpool"
	@echo "  make preprocess-unseen - Preprocess DBs under ../db/unseenDBpool"
	@echo "  make manifest          - Generate config/runs.json from the DB pools"
	@echo "  make tokenizer         - Train the tokenizer from config/settings.json"
	@echo "  make test-tokenizer    - Test the configured tokenizer"
	@echo "  make train             - Train the cache-miss prediction model"
	@echo "  make eval              - Evaluate the model on the test split"
	@echo "  make predict           - Run a one-sample prediction"

setup:
	bash scripts/setup_env.sh

preprocess-seen:
	$(PYTHON) -m preprocess.preprocessing /home/vm/Desktop/ML/db/seenDBpool

preprocess-unseen:
	$(PYTHON) -m preprocess.preprocessing /home/vm/Desktop/ML/db/unseenDBpool

manifest:
	$(PYTHON) -m config.generate_runs_manifest

tokenizer:
	$(PYTHON) -m tokernizer.train_assembly_tokenizer

test-tokenizer:
	$(PYTHON) -m tokernizer.test_tokenizer

train:
	$(PYTHON) -m train.train

eval:
	$(PYTHON) -m evaluate.evaluate

predict:
	$(PYTHON) -m predict.predict
