# 🛰️ Land Change Insights

**Deep Learning-Based Satellite Image Change Detection**

Land Change Insights is a computer vision project that uses a custom U-Net-style convolutional neural network to identify pixel-level changes between satellite images captured at different points in time.

The project includes a complete workflow: image preprocessing, model training, evaluation, error analysis, and an interactive web application for running predictions.

**[🚀 Live Demo](https://land-change-insights.streamlit.app)** | **[💻 GitHub Repository](https://github.com/amali-Codes/satellite-change-detection)**

---

## Project Overview

Satellite imagery provides valuable information about how geographical areas change over time. Detecting these changes can support applications such as urban development monitoring, land-use analysis, and environmental observation.

Given two satellite images of the same location:

- **Image A:** Earlier image
- **Image B:** Later image

The model predicts a binary segmentation mask identifying pixels where changes are detected.

## Live Application

The project includes a Streamlit application where users can:

- Upload two satellite images.
- Run the trained PyTorch model.
- Generate a pixel-level change mask.
- View a red overlay highlighting predicted changes on the later image.
- See the percentage of pixels classified as changed.

**Try it here:** https://land-change-insights.streamlit.app

## Dataset

This project uses the **LEVIR-CD+** dataset.

Each sample contains:

| Component | Description |
|---|---|
| Image A | Earlier satellite image |
| Image B | Later satellite image |
| Label | Ground-truth binary change mask |

Dataset split:

- Training samples: 637
- Test samples: 348

### Preprocessing

- Convert images to RGB.
- Resize images to 256 × 256 pixels.
- Convert images to PyTorch tensors.
- Resize segmentation masks using nearest-neighbour interpolation.
- Convert masks into binary labels: `0` = no change, `1` = change.

## Model Architecture

The project uses a custom U-Net-style convolutional neural network implemented in PyTorch.

The model:

1. Concatenates the two RGB images into a six-channel input.
2. Extracts image features using convolutional encoder blocks.
3. Processes deeper features through a bottleneck.
4. Reconstructs spatial information using transposed convolutions and skip connections.
5. Produces a single-channel pixel-level prediction map.

The model returns logits, which are converted into probabilities using a sigmoid function.

A probability threshold of `0.10`, selected using validation data, is used to generate the binary prediction mask.

## Training and Experiments

The project was developed through several experimental phases:

| Phase | Work completed |
|---|---|
| Phase 1 | Baseline change detection model |
| Phase 2 | Class imbalance experiment |
| Phase 3 | Longer training experiment |
| Phase 4 | Full-dataset and extended training |
| Phase 5 | Validation-based threshold selection and final test evaluation |
| Phase 6 | Validation error analysis and per-image performance analysis |
| Phase 7 | Streamlit application and deployment |

### Loss Functions

The training experiments explored Binary Cross-Entropy with Logits and Dice Loss.

The combined loss is:

`Combined Loss = BCE Loss + Dice Loss`

Dice Loss helps address the challenge of segmenting relatively small changed regions within large satellite images.

## Evaluation Results

### Validation Performance

The Phase 4B checkpoint was evaluated on a held-out validation split of 128 images, created from the original training dataset using a fixed random seed.

The following threshold comparison was used to select the operating threshold:

| Threshold | IoU | Dice | Precision | Recall |
|---:|---:|---:|---:|---:|
| 0.50 | 0.5604 | 0.7183 | 0.8287 | 0.6338 |
| 0.40 | 0.5697 | 0.7258 | 0.8155 | 0.6539 |
| 0.30 | 0.5777 | 0.7323 | 0.8002 | 0.6750 |
| 0.20 | 0.5854 | 0.7385 | 0.7804 | 0.7008 |
| **0.10** | **0.5908** | **0.7428** | **0.7461** | **0.7395** |

The threshold of `0.10` was selected based on validation performance.

### Final Test Performance

The selected configuration was evaluated on the separate test set of 348 images.

| Metric | Result |
|---|---:|
| IoU | 0.3733 |
| Dice / F1 | 0.5436 |
| Precision | 0.5236 |
| Recall | 0.5652 |

The difference between validation and test performance highlights the model's generalisation challenges. The test results are reported separately and were not used to select the threshold.

### Error Analysis

Phase 6 examined predictions on the 128-image validation split.

Validation results at threshold `0.10`:

| Metric | Result |
|---|---:|
| IoU | 0.5908 |
| Dice | 0.7428 |
| True Positives | 297,780 |
| False Positives | 101,322 |
| False Negatives | 104,891 |
| True Negatives | 7,884,615 |

The error analysis also ranked individual validation images by IoU and examined cases involving missed changes, false detections, and mixed segmentation errors.

This analysis helps identify where the model can be improved.

## Technology Stack

- **Python** — programming language
- **PyTorch** — deep learning framework
- **Torchvision** — image preprocessing
- **Pillow** — image loading and processing
- **NumPy** — numerical operations
- **Streamlit** — interactive web application
- **Git and GitHub** — version control and source code management

## Project Structure

```text
satellite-change-detection/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   └── phase4b_longer_training_model.pth
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── evaluate_thresholds.py
│   ├── phase6_error_analysis.py
│   ├── phase6_rank_errors.py
│   ├── phase6_error_summary.py
│   └── phase6_visualize_top_errors.py
│
└── outputs/
    └── phase6_error_analysis/
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/amali-Codes/satellite-change-detection.git
cd satellite-change-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

The trained model checkpoint is included in the repository, so the inference application does not require the original training dataset.

**Note:** Running the training and evaluation scripts requires the LEVIR-CD+ dataset and the corresponding local dataset path to be configured in `src/dataset.py`.

## Limitations

- The model operates on resized 256 × 256 images.
- Predictions depend on image quality, alignment, and similarity between the input images and the training dataset.
- False positives and missed changes can occur.
- The predicted changed-pixel percentage is not a verified real-world land-area measurement.
- Test performance indicates that further model improvements are needed.

## Future Improvements

- Experiment with deeper segmentation architectures.
- Explore stronger image augmentation techniques.
- Improve generalisation to different geographical regions.
- Investigate more robust approaches to class imbalance.
- Add geospatial data support and area-based analysis.
- Improve visualisation and model explainability.

## Project Status

**Implemented:**
- Dataset preparation and preprocessing
- Custom U-Net-style model
- Training and evaluation pipeline
- Threshold selection using validation data
- Final test-set evaluation
- Validation error analysis
- Interactive Streamlit application
- Public deployment

**Live application:** https://land-change-insights.streamlit.app

---

*Built as a hands-on computer vision and deep learning project using satellite imagery.*
