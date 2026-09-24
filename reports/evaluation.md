# Reproduced evaluation

Recorded locally on the authorized hackathon dataset. No cloud services were called.

8,221 captures; 104 signal features; three normalized condition labels; seed 42.

| Split | Train captures | Test captures | Test accuracy |
| --- | ---: | ---: | ---: |
| random | 6604 | 1617 | 97.46% |
| by_replicate | 5108 | 3113 | 98.23% |
| by_board | 7007 | 1214 | 82.62% |
| by_excitation | 7399 | 822 | 95.74% |
| combined | 7762 | 459 | 98.69% |

![Confusion matrices](confusion_matrices.png)

## Interpretation

The board split holds out whole sensor boards. The combined split holds out only the intersection of selected boards and replicates; it is not an unseen-board evaluation. Random splits share experimental conditions between training and test data. These are fixed single-seed holdouts, not cross-validation or independent field validation.

The original metadata uses two spellings of the partially loosened condition. Both are normalized to Mix-45 here, giving the three classes specified by the challenge. These results therefore supersede the unverified historical local score; they do not reproduce the original label handling. BigQuery ML was not rerun.

The lab runs correspond to bolt conditions, so run-specific effects can be confounded with condition. Generalization to other fixtures, mounting environments, unseen faults or operational airports is unproven. Class probabilities are uncalibrated.

## Reproduction

For a fresh checkout, run `python scripts/prepare_demo.py` then `python scripts/evaluate_reproducible.py`. To also verify feature extraction, obtain the authorized raw archive and run `python scripts/extract_all_features.py` before evaluation. Without raw CSVs, the script preserves the published manifest and explicitly records that it did not reverify the source files. See [data provenance](../docs/DATA.md).

[Machine-readable results and environment](evaluation.json) · [Exact split assignments](split_assignments.csv) · [Dataset checksums](dataset_manifest.csv)

This portfolio maintenance and evaluation pass was performed with Codex assistance on 24 September 2026; it is separate from the original hackathon work.
