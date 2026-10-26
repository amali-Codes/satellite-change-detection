# Satellite Image Change Detection

A deep learning project that identifies pixel-level changes between satellite images captured at different points in time using PyTorch.

## Problem

Given two satellite images of the same geographical location from different time periods, the goal is to identify regions where changes have occurred.

The project builds a pixel-level change detection pipeline using a U-Net-style convolutional neural network.

## Dataset

This project uses the **LEVIR-CD+** dataset.

Each sample contains:

- **Image A:** Earlier satellite image
- **Image B:** Later satellite image
- **Label:** Ground-truth pixel-level change mask

Images are resized to `256 × 256` during preprocessing.

Dataset split:

- Training samples: 637
- Test samples: 348

## Current Pipeline

```text
Satellite Image A ─┐
                   ├──> Change Detection Model
Satellite Image B ─┘             │
                                 ▼
                       Predicted Change Mask
                                 │
                                 ▼
                      Evaluation and Analysis
```

## Implemented Features

- Dataset loading and preprocessing
- CNN-based change detection model
- Training and evaluation pipelines
- BCE and Dice-based loss experiments
- Prediction mask visualization
- IoU, Dice/F1, precision and recall evaluation
- Longer-training experiment
- Probability-threshold analysis

## Experiment Results

### Phase 4B — Longer Training

The model was trained on all 637 training samples for 10 epochs and evaluated on the 348 test samples.

Recorded results at threshold `0.50`:

| Metric | Result |
|---|---:|
| IoU | 0.3007 |
| Dice/F1 | 0.4164 |
| Precision | 0.4960 |
| Recall | 0.4375 |

Prediction visualization: `outputs/phase4b_prediction.png`

### Phase 4C — Threshold Analysis

The Phase 4B model was evaluated on the test set using five probability thresholds.

| Threshold | IoU | Dice/F1 | Precision | Recall |
|---:|---:|---:|---:|---:|
| 0.50 | 0.3510 | 0.5196 | 0.5930 | 0.4623 |
| 0.40 | 0.3558 | 0.5249 | 0.5808 | 0.4787 |
| 0.30 | 0.3608 | 0.5303 | 0.5679 | 0.4973 |
| 0.20 | 0.3660 | 0.5359 | 0.5510 | 0.5215 |
| 0.10 | 0.3733 | 0.5436 | 0.5236 | 0.5652 |

Among the tested thresholds, `0.10` produced the highest IoU, Dice/F1 and recall. Threshold `0.50` produced the highest precision.

**Evaluation note:** Thresholds were explored on the test set, so these results are exploratory. A separate validation split should be used to select the final threshold before an unbiased final test evaluation. Phase 4B and Phase 4C metrics also use different aggregation methods, so their scores should not be directly compared as evidence of improvement.

## Project Roadmap

- [x] Dataset pipeline
- [x] Baseline model
- [x] Training pipeline
- [x] Evaluation pipeline
- [x] Prediction visualization
- [x] Full-dataset training experiment
- [x] Longer-training experiment
- [x] Probability-threshold analysis
- [ ] Validation-based threshold selection
- [ ] Model architecture experiments
- [ ] Error analysis
- [ ] Inference API
- [ ] Docker deployment
- [ ] Interactive demo

## Tech Stack

- Python
- PyTorch
- NumPy
- OpenCV
- Pillow
- scikit-learn

## Project Structure

```text
satellite-change-detection/
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── evaluate_thresholds.py
│   └── utils.py
├── outputs/
│   └── phase4b_prediction.png
├── models/
├── README.md
└── requirements.txt
```

## Project Status

**Completed:** Baseline pipeline, full-dataset training, longer-training experiment and threshold analysis.

**Next:** Establish a validation split, select a threshold using validation data, and evaluate the selected configuration on the test set.
