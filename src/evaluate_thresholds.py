import torch
from torch.utils.data import DataLoader, Subset

from dataset import LEVIRCDDataset
from model import ChangeDetectionModel


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

full_dataset = LEVIRCDDataset("train")

# Same 100 training samples used during Phase 3B
dataset = Subset(full_dataset, range(100))

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = ChangeDetectionModel()

model.load_state_dict(
    torch.load(
        "models/phase3_longer_training_model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

print("Phase 3B model loaded successfully!")


# --------------------------------------------------
# Thresholds
# --------------------------------------------------

thresholds = [0.50, 0.40, 0.30, 0.20, 0.10]


# --------------------------------------------------
# Store results
# --------------------------------------------------

results = {
    threshold: {
        "intersection": 0,
        "predicted_positive": 0,
        "actual_positive": 0
    }
    for threshold in thresholds
}


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

with torch.no_grad():

    for image_a, image_b, label in loader:

        image_a = image_a.to(device)
        image_b = image_b.to(device)
        label = label.to(device)

        output = model(image_a, image_b)

        probability = torch.sigmoid(output)

        for threshold in thresholds:

            prediction = (probability >= threshold).float()

            intersection = (
                prediction * label
            ).sum().item()

            predicted_positive = prediction.sum().item()
            actual_positive = label.sum().item()

            results[threshold]["intersection"] += intersection
            results[threshold]["predicted_positive"] += predicted_positive
            results[threshold]["actual_positive"] += actual_positive


# --------------------------------------------------
# Calculate metrics
# --------------------------------------------------

print("\n==============================")
print("THRESHOLD ANALYSIS")
print("==============================")

for threshold in thresholds:

    intersection = results[threshold]["intersection"]
    predicted_positive = results[threshold]["predicted_positive"]
    actual_positive = results[threshold]["actual_positive"]

    union = (
        predicted_positive
        + actual_positive
        - intersection
    )

    iou = intersection / (union + 1e-8)

    dice = (
        2 * intersection
        / (predicted_positive + actual_positive + 1e-8)
    )

    precision = (
        intersection
        / (predicted_positive + 1e-8)
    )

    recall = (
        intersection
        / (actual_positive + 1e-8)
    )

    print(f"\nThreshold: {threshold}")
    print(f"Predicted positive pixels: {predicted_positive}")
    print(f"Actual positive pixels:    {actual_positive}")
    print(f"IoU:       {iou:.4f}")
    print(f"Dice/F1:   {dice:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")