# Ensemble Conformal Forecasting (Agra dataset)

This repository reproduces the results from the provided notebook/script (published). It is split into `published/` (reproducible code matched to paper) and `dev/` (ongoing improvements).

## Quickstart

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Put `your_dataset.csv` into `data/sample/` (or pass the path to `--data-path` in `configs/default.yaml`).

3. Reproduce the published experiment:

```bash
python run_experiment.py --config configs/default.yaml
```

Results (metrics + diagnostic plots) will be printed and shown. To save models, update the config.
