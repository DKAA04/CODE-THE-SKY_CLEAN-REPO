# Code the Sky · Case 1 — Structural Integrity Classifier

> **3-class structural fault classifier on airfield-lighting vibration data.**  
> Built solo in 6 hours for the ADB Safegate · GDG Aviation Hacking Challenge (KU Leuven, May 11 2026).

**Honest test accuracy:** 82.6% on unseen sensor boards (sklearn GBM) · 82.0% in BQML (ROC-AUC 0.952). Two independent training stacks, same conclusion.

---

## What this does

Given a 27 kHz · 3-axis vibration capture from a sensor board mounted in an airfield-lighting fixture, predict one of three bolt conditions:
- `30NM` — healthy (30 N·m torque baseline)
- `Loose` — loose mounting hardware
- `Mix- 45 Deg` — partial-stiffness fault

**Why this matters:** structural problems in airfield lighting are currently caught only after failure (FOD risk, runway closure) or during expensive blanket inspections. This classifier enables predictive triage at ~$0.01 per scan.

---

## Headline numbers

| Split strategy | Test accuracy | What it measures |
|---|---|---|
| Random (naive) | 97.2% | Leaky baseline |
| By replicate | 98.2% | New replicates of seen tests |
| By excitation pattern | 95.7% | New excitation, seen boards |
| **By sensor board** | **82.6%** | **Unseen boards — deployment scenario** |
| Strictest combined | 98.7%* | ~200 samples, high variance |

The **14-point gap between random and by-board** is the methodology result — it shows what generalizes vs what was leakage.

---

## How to run

```bash
git clone 
cd code-the-sky-cs1
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Place the dataset at: data/raw/Sensor Board Update Initial Test/...
# Then:
python scripts/extract_all_features.py     # ~2 min — produces features.parquet
python scripts/baseline_eval.py            # ~1 min — prints the multi-split table
python scripts/train_final_model.py        # ~30 sec — saves model_gbm.joblib
streamlit run app/streamlit_app.py         # opens demo at localhost:8501
```

For the BQML path, configure `src/config.py` with your GCP project ID and run `scripts/upload_features.py` then the queries in `sql/train_bqml.sql`.

---

## Repo layout

```
src/
├── config.py           # Constants — paths, sample rate, GCP IDs
├── data_loader.py      # Walks Run/TestID/SensorBoardID tree, joins Test.csv
├── features.py         # FFT spectral features (104 per capture, per-axis)
├── splitter.py         # 5 leakage-aware train/test splits
├── train.py            # Sklearn baseline + multi-split evaluator
└── upload_to_bq.py     # Push features DataFrame to BigQuery
sql/
└── train_bqml.sql      # BQML BOOSTED_TREE_CLASSIFIER + evaluation queries
app/
└── streamlit_app.py    # Demo UI: upload CSV → prediction + spectrum
scripts/                # Orchestration scripts run in order
docs/
├── pitch_outline.md
└── commercial_bonus.md # The optional commercial roadmap deliverable
```
---

## Approach

1. **Physics-grounded features.** 104 per capture: peak frequencies, energy in 8 frequency bands, spectral centroid, half-power bandwidth, time-domain stats. Per X/Y/Z axis. Loose bolts shift dominant resonance from ~1500 Hz down to ~150 Hz — an order-of-magnitude signature we can read before we even train.

2. **Leakage-aware evaluation.** The dataset has 15 sensor-board recordings per test. Random splitting puts copies of the same test in train and test, inflating accuracy by ~14 points. We reported all 5 split strategies side by side so judges can see what generalizes.

3. **Cross-validated training.** Same features trained both BQML (cloud) and sklearn (local). 82.0% vs 82.6% on by_board — they agree, so the signal is real.

---

## Stack

Python · NumPy · SciPy · scikit-learn · Streamlit · pandas · pyarrow · Google BigQuery · BQML · Looker Studio · GitHub

---

## Deliverables

- ✅ Working pipeline (8,221 captures → trained model)
- ✅ Multi-split accuracy table
- ✅ BQML training in Google Cloud
- ✅ Streamlit demo (live + screen-recorded backup)
- ✅ Looker Studio dashboard (live URL)
- ✅ Commercial roadmap (optional bonus deliverable — see `docs/commercial_bonus.md`)

---

*Built solo by Angel Aparicio Pastor · KU Leuven Brussels · May 11 2026*