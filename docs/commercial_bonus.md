Code the Sky · CS1 Structural Integrity
Commercial roadmap — From algorithm to product
Angel Aparicio Pastor — Solo participant, ADB Safegate Aviation Hacking Challenge, 11 May 2026

TL;DR
We built a 3-class structural fault classifier from 8,221 sensor-board captures of bolt-condition vibration data. The honest test accuracy on unseen sensor boards is 82.6% (ROC-AUC 0.952). The classifier is not the product. The product is a built-in capability inside CORTEX Service that, paired with EFD Localization (Case 3), forms a CORTEX Predictive Maintenance Package — sold per-fixture-per-year, tier-priced, with a v1 → v3 roadmap explicitly tied to data ADB Safegate is already on track to collect.

1. Commercialization strategy
Where it lives
Inside CORTEX Service as a built-in capability — not as a separate SKU.
Airports buying CORTEX Service are buying operational confidence. They don't want another procurement contract; they want CORTEX to surface the right insight at the right time. Structural Integrity is one more lens through which CORTEX delivers that. Selling it standalone would force airports through procurement twice for what they perceive as one capability — and create internal competition between ADB Safegate sales motions.
Bundling
We bundle Structural Integrity (Case 1) + EFD Localization (Case 3) as the CORTEX Predictive Maintenance Package. Both are reactive-to-predictive transitions on the same physical asset (the airfield series circuit and its fixtures). One purchase decision, two algorithms shipping under one banner. LED Degradation (Case 2) joins at v3 once it's production-ready.
Why not standalone

A standalone fault-classifier SKU forces airports to evaluate three separate algorithms against three separate budgets. Friction without buyer value.
Bundling lets CORTEX Service salespeople pitch outcome categories (structural integrity, electrical health, asset lifecycle) instead of algorithm names.
Bundle pricing is easier to anchor against a competitor's "blanket maintenance contract" alternative.


2. Pricing model & tier roadmap

Tier / What ships/ Pricing model/ Indicative annual
v1 — Today
Bolt-condition classifier (82.6% honest accuracy on by-board split). 6-month field calibration package included on launch.
Per-fixture-per-year, included in CORTEX Premium tier
Bundled in Premium pricing

v2 — 12 months+ 
EFD Localization for series-circuit fault triage. + Field-noise corrections (calibrated against pilot data).
Per-fixture-per-year, +20% premium over CORTEX Premium
Premium × 1.2

v3 — 24+ months+
LED Degradation indicator (Case 2 once production-ready). + Replacement budget forecasting per airport.
Value-based pricing tied to documented downtime reduction
Custom — outcome-indexed

3. Value articulation
What the customer is buying — concretely

Failures avoided — fixtures replaced before they become FOD hazards on active runways
Inspections deferred — moves blanket structural inspections from scheduled sweeps to risk-targeted dispatches. Estimated 30-50% reduction in inspection labor hours (validation pending pilot data)
Faster fault response (paired with EFD) — runway downtime per fault event drops from full-circuit-walk to known-zone investigation
Asset planning — budget forecasting for fixture replacement across the airport's lifecycle, instead of reactive emergency capex

What it does NOT do (intellectual honesty)

Does not predict time-to-failure in hours
Does not detect fault types outside the labeled training set (catastrophic damage, corrosion-only modes)
Does not replace physical inspection — it prioritizes it
Does not generalize to fixtures with different mounting geometries than the training rig (yet)


4. Algorithm-package lifecycle
v1 (today's data, today's capability)

Lab-validated structural classifier (3 classes: 30NM healthy / Loose / Mix-45°)
104 spectral features per capture, gradient-boosted-tree model
Disclosed accuracy under leakage-aware splits (the honest 82.6%, not the inflated 97.2% from random splitting)
Cross-validated in BQML on Google Cloud (82.0%, ROC-AUC 0.952) — same conclusion, different stack
Field calibration tooling (gather a "this fixture is healthy" baseline at install)
Per-fixture inference: ~$0.01 / scan, 200ms latency on Cloud Run

v2 unlocked by

Paired field-and-lab captures from 3–5 representative airports over 12 months
Ambient/traffic vibration capture for noise-robust feature engineering
Integration with EFD Localization on the same hardware

v3 unlocked by

LED Degradation telemetry production-ready (Case 2 outputs)
24+ months of failure-event ground truth from v1+v2 deployments
Photometric output measurements (currently absent; requires sensor revision)


5. The single highest-value data investment

If ADB Safegate is going to invest in one additional data stream to unlock v2, it should be paired field-and-lab captures from 5–10 representative airports across 12 months.

Without this, the lab benchmark stays a lab benchmark. With it, the gap between random-split inflation (97.2%) and the honest by-board number (82.6%) closes, the per-fixture pricing tier becomes defensible to airport CTOs, and the v1 → v2 transition timeline shortens.
Second-priority: photometric output sensors on Axon EQ fixtures to unblock LED degradation. But that's a hardware revision on a longer cycle.

6. What we built today, in 6 hours, solo

Full data pipeline: 8,221 captures parsed, joined with Test.csv metadata, leakage-aware splits
104 spectral features per capture (peak frequencies, band energies, half-power bandwidth, time-domain stats)
Sklearn baseline trained across 5 split strategies → multi-split accuracy table
BQML model trained on Google Cloud (same data, same accuracy — cross-validation)
Streamlit demo: upload CSV → instant prediction + spectrum + confidence
Looker Studio dashboard: class distribution, per-board accuracy heatmap, frequency-by-class chart
Repository: https://github.com/DKAA04/CODE-THE-SKY-CS1
Live dashboard: https://datastudio.google.com/reporting/af47dfc3-458d-4dcb-b631-6c0cbd8c9e99


The technical work is the proof. The commercial framework above is the path to scale.
