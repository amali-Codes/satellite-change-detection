# Satellite Image Change Detection

A deep learning project for detecting changes between two satellite images captured at different points in time.

## Problem

Given two satellite images of the same geographical location from different time periods, the goal is to identify pixels where meaningful changes have occurred.

The project focuses on building a pixel-level change detection pipeline using deep learning.

## Dataset

The project uses the LEVIR-CD+ dataset, which contains pairs of high-resolution satellite images and corresponding ground-truth change masks.

Each sample contains:

* Image A — earlier satellite image
* Image B — later satellite image
* Label — pixel-level change mask

The images are resized to `256 × 256` during preprocessing.

## Current Pipeline

```text
Satellite Image A ──┐
                    ├──> Baseline Model ──> Change Mask
Satellite Image B ──┘
```

The current implementation includes:

* Dataset loading and preprocessing
* Baseline CNN model
* Model training
* Model evaluation
* IoU, Dice, precision and recall metrics
* Prediction visualization

## Baseline Results

The initial baseline model produces predictions but currently fails to detect the positive/change class effectively.

This baseline will be used as the reference point for subsequent model improvements.

## Project Roadmap

* [x] Dataset pipeline
* [x] Baseline model
* [x] Training pipeline
* [x] Evaluation pipeline
* [x] Baseline prediction visualization
* [ ] Address class imbalance
* [ ] Improve model architecture
* [ ] Compare model variants
* [ ] Perform error analysis
* [ ] Build inference API
* [ ] Dockerize the application
* [ ] Create an interactive demo

## Tech Stack

* Python
* PyTorch
* NumPy
* OpenCV
* Pillow
* scikit-learn

## Project Status

**Phase 1 — Baseline implementation completed.**

The next phase will focus on improving the model's ability to detect changed regions.
## Baseline Experiment

The initial baseline experiment was trained on a subset of the training data to validate the end-to-end pipeline before scaling up training.

The first evaluation showed that the model predicted no positive change pixels, establishing a clear baseline for the next iteration.