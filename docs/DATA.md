# Data provenance

The source is the Code the Sky 2026 ADB Safegate Case Study 1 structural-integrity dataset: controlled three-axis vibration recordings from sensor boards under three bolt conditions. The repository owner confirmed that the hackathon organizers supplied the data publicly and permitted participants to include it in their repositories.

The reviewed local archive was `Case_1_Structural.integrity.Seonsor.Board.data.zip`. This release contains derived signal features for all 8,221 usable captures, a checksum manifest of the source CSVs, exact evaluation partitions and three original sample captures. It does not upload the approximately 1.2 GB raw archive into Git history. Sensor board identifiers describe lab hardware, not people. No local filesystem paths or cloud credentials are included in the derived table.

Attribution: ADB Safegate, Code the Sky hackathon, 2026. Permission to include challenge data is recorded here based on the participant's confirmation; it is not a claim that the organizers granted an unrestricted open-data licence or endorsed this implementation.

## Included artifacts

- `data/features.parquet`: 104 signal features plus experimental metadata, one row per capture.
- `data/feature_manifest.json`: checksum, shape and provenance of the derived table.
- `data/examples/`: one measured capture from each condition, for local interface inspection.
- `reports/dataset_manifest.csv`: relative source paths and SHA-256 hashes.
- `reports/split_assignments.csv`: exact membership for the five evaluated holdouts.
- `reports/evaluation.json`: results, dependency versions and code hashes recorded at evaluation time.

Normalize `Mix- 45 Deg` and `Mix-45 Deg` to the same `Mix-45` class. The supplied metadata otherwise makes these look like two classes. Models use signal features only; run, board, timestamp, replicate and excitation metadata are excluded from model inputs.

## Full raw-data reproduction

If you have the original authorized challenge archive, extract its `Sensor Board Update Initial Test` directory under `data/raw/`. Preserve the `Test.csv` metadata file and the A/B/C capture directories. Run feature extraction and the recorded-evaluation script as documented in the README. Hashes in the manifest identify the exact CSVs used for the published run.

The included feature table allows model-training and split evaluation without the raw archive. It cannot independently prove that feature extraction was correct; use the original CSVs and the dataset manifest for that check.
