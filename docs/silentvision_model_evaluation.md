# SilentVision Model Evaluation

This stage is for research comparison only. The external repository does not include a license file, so do not ship its code or weights in a commercial customer version unless you get written permission from the owner.

## What This Model Does

The GitHub model is a clip-level action classifier. It predicts:

- `bag_concealment`
- `clothes_concealment`
- `normal`

It does not know your product zones, customer zone, shelf return logic, or item visibility logic. For our project, it is best tested as a second-stage action model after the current detector finds a possible suspicious moment.

## Local Folders

Put videos here:

- Normal videos: `local_review_workspace/normal_videos/`
- Suspicious videos: `local_review_workspace/suspicious_videos/`

The folder name is only the evaluation truth. It does not tell the model the answer during prediction.

## Run A Dependency Check

From the repository root:

```bash
python tools/evaluate_silentvision_model.py --check-only
```

If TensorFlow is missing locally, run this in Google Colab instead.

## Run Evaluation

```bash
python tools/evaluate_silentvision_model.py --weights path/to/weights.h5 --limit 20
```

The script writes:

- `reports/silentvision_evaluation/<run>/silentvision_predictions.csv`
- `reports/silentvision_evaluation/<run>/silentvision_metrics.json`
- `reports/silentvision_evaluation/<run>/silentvision_clip_predictions.json`

## Metrics To Compare

Use these first:

- `precision`: when it alerts, how often it is right
- `recall`: how many suspicious videos it catches
- `false_alarm_rate`: how often normal videos cause false alerts
- `f1`: balanced score between precision and recall
- `avg_runtime_seconds`: speed per video
- `avg_processing_fps`: rough processing performance

For a sellable retail product, false alarms matter a lot. A model with high recall but many false alerts will not feel reliable to store users.

## Recommendation

Do not replace the current hybrid logic with this model immediately. Compare it first. If it performs well, use it as a second opinion:

1. Current YOLO/pose/zone logic detects product interaction and possible concealment.
2. SilentVision action model checks the clip around the event.
3. Alert only when both systems agree, or show the second-stage score in the dashboard.

This should reduce useless alarms while keeping the system explainable for human review.

## Latest Local Result

Run date: 2026-06-20

Dataset found in the local review folders:

- Normal videos: 231
- Suspicious videos: 134
- Total evaluated: 365

Command:

```bash
.\.venv\Scripts\python.exe .\tools\evaluate_silentvision_model.py --max-windows 20 --target-fps 6 --min-suspicious-windows 1 --threshold 0.50
```

Result:

- TP: 64
- FP: 168
- FN: 70
- TN: 63
- Precision: 0.276
- Recall: 0.478
- Specificity: 0.273
- Accuracy: 0.348
- F1: 0.350
- False alarm rate: 0.727
- Miss rate: 0.522
- Average runtime per video: 1.22 seconds
- Average processing speed: 71.7 sampled FPS

Best threshold tuning result from the same raw scores was still weak:

- Threshold: 0.375
- Minimum suspicious windows: 1
- F1: 0.350
- Precision: 0.272
- Recall: 0.493
- False positives: 177
- False negatives: 68

Conclusion: the external model runs and is fast enough for experimentation, but it is not accurate enough to replace the current project logic on this local dataset. It may still be useful as a research baseline or a weak second opinion, but it should not be the main customer-facing detector.
