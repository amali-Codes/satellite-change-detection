
import torch
from torch.utils.data import DataLoader

from dataset import LEVIRCDDataset
from model import ChangeDetectionModel


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Test dataset: all 348 test images
dataset = LEVIRCDDataset("test")

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False
)

print("Test samples:", len(dataset))


# Load the Phase 4B model
model = ChangeDetectionModel()

model.load_state_dict(
    torch.load(
        "models/phase4b_longer_training_model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

print("Phase 4B model loaded successfully!")


# Probability thresholds to compare
thresholds = [0.50, 0.40, 0.30, 0.20, 0.10]

# Accumulate pixel counts for each threshold
results = {
    threshold: {
        "tp": 0,
        "fp": 0,
        "fn": 0
    }
    for threshold in thresholds
}


# Evaluate the same model and test images at each threshold
with torch.no_grad():
    for image_a, image_b, label in loader:

        image_a = image_a.to(device)
        image_b = image_b.to(device)
        label = (label.to(device) >= 0.5)

        logits = model(image_a, image_b)
        probability = torch.sigmoid(logits)

        for threshold in thresholds:
            prediction = probability >= threshold

            results[threshold]["tp"] += (
                (prediction & label).sum().item()
            )
            results[threshold]["fp"] += (
                (prediction & ~label).sum().item()
            )
            results[threshold]["fn"] += (
                (~prediction & label).sum().item()
            )


# Calculate metrics across all test pixels
print("\n====================================")
print("PHASE 4C: THRESHOLD ANALYSIS")
print("====================================")

for threshold in thresholds:

    tp = results[threshold]["tp"]
    fp = results[threshold]["fp"]
    fn = results[threshold]["fn"]

    iou = tp / (tp + fp + fn + 1e-8)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    print(f"\nThreshold:  {threshold:.2f}")
    print(f"TP:         {tp}")
    print(f"FP:         {fp}")
    print(f"FN:         {fn}")
    print(f"IoU:        {iou:.4f}")
    print(f"Dice/F1:    {dice:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
