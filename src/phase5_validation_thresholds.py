import torch
from torch.utils.data import DataLoader, random_split

from dataset import LEVIRCDDataset
from model import ChangeDetectionModel


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = LEVIRCDDataset("train")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    _, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    model = ChangeDetectionModel().to(device)
    model.load_state_dict(torch.load(
        "models/phase4b_longer_training_model.pth",
        map_location=device
    ))
    model.eval()

    thresholds = [0.50, 0.40, 0.30, 0.20, 0.10]
    results = {
        t: {"tp": 0, "fp": 0, "fn": 0}
        for t in thresholds
    }

    with torch.no_grad():
        for image_a, image_b, label in loader:
            image_a = image_a.to(device)
            image_b = image_b.to(device)
            label = label.to(device) >= 0.5

            probability = torch.sigmoid(model(image_a, image_b))

            for t in thresholds:
                prediction = probability >= t
                results[t]["tp"] += (prediction & label).sum().item()
                results[t]["fp"] += (prediction & ~label).sum().item()
                results[t]["fn"] += (~prediction & label).sum().item()

    print("\nPHASE 5: VALIDATION THRESHOLD ANALYSIS")
    print("Validation samples:", len(val_dataset))

    for t in thresholds:
        tp = results[t]["tp"]
        fp = results[t]["fp"]
        fn = results[t]["fn"]

        iou = tp / (tp + fp + fn + 1e-8)
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        print(f"\nThreshold: {t:.2f}")
        print(f"IoU:       {iou:.4f}")
        print(f"Dice/F1:   {dice:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")


if __name__ == "__main__":
    main()
