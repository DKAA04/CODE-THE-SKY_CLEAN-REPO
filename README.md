# Code the Sky — Structural Integrity Classifier

**Vibration-signal classification with explicit evaluation of train/test split choices.**

This project extracts time-domain and FFT-based features from three-axis vibration captures, trains classifiers for bolt-condition labels, and provides a Streamlit interface for inspecting a capture and its prediction. The code includes local scikit-learn training and a separate BigQuery ML path.

## Technical focus

- Feature extraction from 27 kHz sensor recordings: spectral peaks, frequency-band energy and time-domain statistics.
- Comparison of random, replicate, sensor-board and excitation-based splits.
- Gradient boosting, random forest and logistic-regression training options.
- A Streamlit view with waveforms, frequency spectra and predicted class probabilities.

The interesting engineering question is how performance changes when the held-out data comes from different sensor boards rather than randomly selected recordings.

## Reproducibility and results

The previous project notes reported **82.6% accuracy for a local by-board split** and **82.0% for BigQuery ML**. These are historical, author-reported results: the public repository does not include the raw dataset, trained model or evaluation artifacts needed to independently reproduce them. They are not fresh benchmark results from this checkout.

The `combined` split holds out the intersection of selected boards and replicates; other recordings from those boards can remain in training. It should not be interpreted as a stricter unseen-board holdout. The final demo model is trained on all available features; its training score is not a held-out score.

## Run locally

Use Python 3.11+ and obtain a dataset you are authorized to use. The training pipeline cannot run from the public checkout alone.

```bash
git clone https://github.com/DKAA04/CODE-THE-SKY_CLEAN-REPO.git
cd CODE-THE-SKY_CLEAN-REPO
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

Expected input layout:

```text
data/raw/Sensor Board Update Initial Test/
├── Test.csv
└── <A|B|C>/<test_id>/<board>/<timestamp>.csv
```

See [data_loader.py](src/data_loader.py) for metadata columns. Capture CSVs contain the three sensor axes. The demo upload expects `X-axis`, `Y-Axis` and `Z-Axis` columns.

```bash
python scripts/extract_all_features.py
python scripts/baseline_eval.py
python scripts/train_final_model.py
python -m streamlit run app/streamlit_app.py
```

Generated features and the model are written to `data/processed/`. Retain evaluation outputs and dataset/split identifiers if reporting new scores.

## Optional BigQuery ML path

Review the project and dataset identifiers in [src/config.py](src/config.py) and [sql/train_bqml.sql](sql/train_bqml.sql), configure your own Google Cloud authentication, then use [scripts/upload_features.py](scripts/upload_features.py). Cloud operations require your own authorized project and may incur charges. The local path does not require cloud credentials.

## Repository guide

| Area | Files |
| --- | --- |
| Dataset loading | [src/data_loader.py](src/data_loader.py) |
| Feature extraction | [src/features.py](src/features.py) |
| Split definitions | [src/splitter.py](src/splitter.py) |
| Training and evaluation | [src/train.py](src/train.py) |
| Interactive demonstration | [app/streamlit_app.py](app/streamlit_app.py) |
| Exploratory commercial proposal | [docs/commercial_bonus.md](docs/commercial_bonus.md) |

## Boundaries

This is a classification prototype, not a validated aviation inspection system. The demo contains static historical performance and cost labels; these are not live measurements. Commercial estimates are proposal material rather than validated operating results. Dataset redistribution rights must be checked before adding raw captures or derived artifacts.

**Stack:** Python, NumPy, pandas, SciPy, scikit-learn, Streamlit, Plotly and BigQuery ML.
