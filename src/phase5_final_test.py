import csv
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from dataset import LEVIRCDDataset
from model import ChangeDetectionModel


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold = 0.10

    dataset = LEVIRCDDataset("test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = ChangeDetectionModel().to(device)
    model.load_state_dict(torch.load(
        "models/phase4b_longer_training_model.pth",
        map_location=device
    ))
    model.eval()

    tp = fp = fn = 0

    with torch.no_grad():
        for image_a, image_b, label in loader:
            image_a = image_a.to(device)
            image_b = image_b.to(device)
            label = label.to(device) >= 0.5

            probability = torch.sigmoid(model(image_a, image_b))
            prediction = probability >= threshold

            tp += (prediction & label).sum().item()
            fp += (prediction & ~label).sum().item()
            fn += (~prediction & label).sum().item()

    iou = tp / (tp + fp + fn + 1e-8)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    print("\nFINAL TEST SET RESULTS")
    print("Checkpoint: Phase 4B")
    print("Test images:", len(dataset))
    print("Threshold:", threshold)
    print(f"IoU:       {iou:.4f}")
    print(f"Dice/F1:   {dice:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")

    output_path = Path("outputs/phase5_final_test_metrics.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "checkpoint", "test_images", "threshold",
            "IoU", "Dice", "Precision", "Recall"
        ])
        writer.writerow([
            "phase4b_longer_training_model.pth",
            len(dataset), threshold,
            f"{iou:.6f}", f"{dice:.6f}",
            f"{precision:.6f}", f"{recall:.6f}"
        ])

    print("Metrics saved to:", output_path)


if __name__ == "__main__":
    main()
