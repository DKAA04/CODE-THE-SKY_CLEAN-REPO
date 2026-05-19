"""Central configuration. Edit these once, used everywhere."""
from pathlib import Path

# Data
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "Sensor Board Update Initial Test"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TEST_CSV = DATA_ROOT / "Test.csv"

# Signal
SAMPLE_RATE_HZ = 27_000
N_SAMPLES = 8_192
NYQUIST_HZ = SAMPLE_RATE_HZ // 2  # 13,500 Hz

# Class mapping (folder name -> human label)
RUN_TO_LABEL = {
    "A": "30NM",      # healthy
    "B": "Loose",     # fault 1
    "C": "Mix-45",    # fault 2
}
LABEL_TO_INT = {"30NM": 0, "Loose": 1, "Mix-45": 2}

# Feature engineering
FREQ_BANDS_HZ = [
    (0, 100), (100, 250), (250, 500), (500, 1000),
    (1000, 2500), (2500, 5000), (5000, 8000), (8000, 13500),
]
N_TOP_PEAKS = 5  # number of peak frequencies to extract per axis

# GCP
GCP_PROJECT_ID = "code-the-sky-cs1"
GCP_REGION = "europe-west1"
BQ_DATASET = "cs1"
BQ_FEATURES_TABLE = "vibration_features"
