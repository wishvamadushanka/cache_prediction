# cache_prediction

Reproduction pipeline for the paper:
`Machine learning-based cache miss prediction`

## Environment Setup

From the project root:

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

This will:
- create `.venv` if needed
- upgrade `pip`
- install dependencies from [`requirements.txt`](/home/vm/Desktop/ML/cache_prediction/requirements.txt)

You can also use:

```bash
make setup
source .venv/bin/activate
```

## Common Commands

List the available shortcuts:

```bash
make help
```

Common `make` commands:

```bash
make preprocess-seen
make preprocess-unseen
make manifest
make tokenizer
make test-tokenizer
make train
make eval
make predict
```

Equivalent direct Python commands:

Train the tokenizer from the shared config:

```bash
python -m tokenizer.train_assembly_tokenizer
```

Test the tokenizer:

```bash
python -m tokenizer.test_tokenizer
```

Train the model:

```bash
python -m train.train
```

Evaluate on the test split:

```bash
python -m evaluate.evaluate
```

## Pipeline Order

The intended pipeline is:

1. Preprocess DBs
2. Train tokenizer
3. Train model
4. Evaluate on unseen DBs
