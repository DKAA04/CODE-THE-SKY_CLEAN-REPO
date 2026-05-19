-- ===== Run after upload_to_bq.py finishes =====
-- Replace `code-the-sky-cs1` with your actual project ID if different

-- 1. Train the BOOSTED_TREE_CLASSIFIER on the train split
CREATE OR REPLACE MODEL `code-the-sky-cs1.cs1.bolt_classifier`
OPTIONS(
  MODEL_TYPE = 'BOOSTED_TREE_CLASSIFIER',
  INPUT_LABEL_COLS = ['label'],
  AUTO_CLASS_WEIGHTS = TRUE,
  MAX_ITERATIONS = 50
) AS
SELECT
  * EXCEPT (path, run, test_id, board, ts, replicate, split, start_freq_1, end_freq_1, sweep_time_1, amp_1, start_freq_2, end_freq_2)
FROM `code-the-sky-cs1.cs1.vibration_features`
WHERE split = 'train';

-- 2. Evaluate on the test split
SELECT *
FROM ML.EVALUATE(
  MODEL `code-the-sky-cs1.cs1.bolt_classifier`,
  (SELECT * EXCEPT (path, run, test_id, board, ts, replicate, split, start_freq_1, end_freq_1, sweep_time_1, amp_1, start_freq_2, end_freq_2)
   FROM `code-the-sky-cs1.cs1.vibration_features`
   WHERE split = 'test')
);

-- 3. Confusion matrix
SELECT *
FROM ML.CONFUSION_MATRIX(
  MODEL `code-the-sky-cs1.cs1.bolt_classifier`,
  (SELECT * EXCEPT (path, run, test_id, board, ts, replicate, split, start_freq_1, end_freq_1, sweep_time_1, amp_1, start_freq_2, end_freq_2)
   FROM `code-the-sky-cs1.cs1.vibration_features`
   WHERE split = 'test')
);

-- 4. Feature importance
SELECT *
FROM ML.FEATURE_IMPORTANCE(MODEL `code-the-sky-cs1.cs1.bolt_classifier`)
ORDER BY importance_weight DESC
LIMIT 30;

-- 5. Predictions for the demo
SELECT
  path,
  predicted_label,
  predicted_label_probs
FROM ML.PREDICT(
  MODEL `code-the-sky-cs1.cs1.bolt_classifier`,
  (SELECT * FROM `code-the-sky-cs1.cs1.vibration_features` WHERE split = 'test' LIMIT 100)
);
