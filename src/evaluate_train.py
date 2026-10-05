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
# Load the same 100 training samples
# --------------------------------------------------

full_dataset = LEVIRCDDataset("train")

train_dataset = Subset(
    full_dataset,
    range(100)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=False
)

print("Training samples evaluated:", len(train_dataset))


# --------------------------------------------------
# Load Phase 2 model
# --------------------------------------------------

model = ChangeDetectionModel().to(device)

model.load_state_dict(
    torch.load(
        "models/phase2_class_imbalance_model.pth",
        map_location=device
    )
)

model.eval()

print("Phase 2 model loaded successfully!")


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

total_iou = 0
total_dice = 0
total_precision = 0
total_recall = 0

num_samples = 0


with torch.no_grad():

    for image_a, image_b, label in train_loader:

        image_a = image_a.to(device)
        image_b = image_b.to(device)
        label = label.to(device)

        # Model prediction
        output = model(image_a, image_b)

        # Convert logits to probabilities
        probability = torch.sigmoid(output)

        # Convert probability to binary mask
        prediction = (probability > 0.5).float()

        # Print diagnostic for first image
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

        # Metrics
        iou = tp / (tp + fp + fn + 1e-8)

        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)

        precision = tp / (tp + fp + 1e-8)

        recall = tp / (tp + fn + 1e-8)

        total_iou += iou
        total_dice += dice
        total_precision += precision
        total_recall += recall

        num_samples += 1


# --------------------------------------------------
# Final results
# --------------------------------------------------

average_iou = total_iou / num_samples
average_dice = total_dice / num_samples
average_precision = total_precision / num_samples
average_recall = total_recall / num_samples


print("\n==============================")
print("TRAINING SET RESULTS")
print("==============================")

print(f"IoU:       {average_iou:.4f}")
print(f"Dice/F1:   {average_dice:.4f}")
print(f"Precision: {average_precision:.4f}")
print(f"Recall:    {average_recall:.4f}")