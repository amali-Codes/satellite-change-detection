import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from dataset import LEVIRCDDataset
from model import ChangeDetectionModel


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# --------------------------------------------------
# Load test dataset
# --------------------------------------------------

test_dataset = LEVIRCDDataset("test")

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)

print("Test samples:", len(test_dataset))


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model = ChangeDetectionModel().to(device)

model.load_state_dict(
    torch.load(
        "models/phase4a_full_dataset_model.pth",
        map_location=device
    )
)

model.eval()

print("Model loaded successfully!")


# --------------------------------------------------
# Evaluation metrics
# --------------------------------------------------

total_iou = 0
total_dice = 0
total_precision = 0
total_recall = 0

num_samples = 0


# --------------------------------------------------
# Evaluate
# --------------------------------------------------

with torch.no_grad():

    for image_a, image_b, label in test_loader:

        image_a = image_a.to(device)
        image_b = image_b.to(device)
        label = label.to(device)

        # Model prediction
        output = model(image_a, image_b)

        # Convert logits to probabilities
        probability = torch.sigmoid(output)

        # Convert probability to binary mask
        prediction = (probability > 0.5).float()
        if num_samples == 0:
            print("Probability min:", probability.min().item())
            print("Probability max:", probability.max().item())
            print("Probability mean:", probability.mean().item())
            print("Predicted positive pixels:", prediction.sum().item())
            print("Actual positive pixels:", label.sum().item())

        # Flatten
        prediction_flat = prediction.view(-1)
        label_flat = label.view(-1)

        # True positives
        tp = (prediction_flat * label_flat).sum().item()

        # False positives
        fp = (prediction_flat * (1 - label_flat)).sum().item()

        # False negatives
        fn = ((1 - prediction_flat) * label_flat).sum().item()

        # IoU
        iou = tp / (tp + fp + fn + 1e-8)

        # Dice
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)

        # Precision
        precision = tp / (tp + fp + 1e-8)

        # Recall
        recall = tp / (tp + fn + 1e-8)

        total_iou += iou
        total_dice += dice
        total_precision += precision
        total_recall += recall

        num_samples += 1


# --------------------------------------------------
# Final metrics
# --------------------------------------------------

average_iou = total_iou / num_samples
average_dice = total_dice / num_samples
average_precision = total_precision / num_samples
average_recall = total_recall / num_samples


print("\n==============================")
print("TEST SET RESULTS")
print("==============================")

print(f"IoU:       {average_iou:.4f}")
print(f"Dice/F1:   {average_dice:.4f}")
print(f"Precision: {average_precision:.4f}")
print(f"Recall:    {average_recall:.4f}")


# --------------------------------------------------
# Visualize one prediction
# --------------------------------------------------

image_a, image_b, label = test_dataset[0]

image_a_input = image_a.unsqueeze(0).to(device)
image_b_input = image_b.unsqueeze(0).to(device)

with torch.no_grad():

    output = model(
        image_a_input,
        image_b_input
    )

    probability = torch.sigmoid(output)

    prediction = (probability > 0.5).float()


# Convert tensors to images

image_a_display = image_a.permute(1, 2, 0).numpy()
image_b_display = image_b.permute(1, 2, 0).numpy()

label_display = label.squeeze().numpy()
prediction_display = prediction.squeeze().cpu().numpy()


# --------------------------------------------------
# Plot
# --------------------------------------------------

plt.figure(figsize=(16, 4))


plt.subplot(1, 4, 1)
plt.imshow(image_a_display)
plt.title("Before")
plt.axis("off")


plt.subplot(1, 4, 2)
plt.imshow(image_b_display)
plt.title("After")
plt.axis("off")


plt.subplot(1, 4, 3)
plt.imshow(label_display, cmap="gray")
plt.title("Ground Truth")
plt.axis("off")


plt.subplot(1, 4, 4)
plt.imshow(prediction_display, cmap="gray")
plt.title("Prediction")
plt.axis("off")


plt.tight_layout()

plt.savefig(
    "outputs/phase4a_prediction.png",
    dpi=150
)

plt.show()

print("\nPrediction image saved to:")
print("outputs/phase4a_prediction.png")