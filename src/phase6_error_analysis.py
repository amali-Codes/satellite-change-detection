
import csv
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader, random_split
from dataset import LEVIRCDDataset
from model import ChangeDetectionModel


# Configuration
CHECKPOINT = "models/phase4b_longer_training_model.pth"
OUTPUT_DIR = Path("outputs/phase6_error_analysis")
THRESHOLD = 0.10
SEED = 42


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Recreate the Phase 5 validation split
    full_dataset = LEVIRCDDataset("train")

    train_size = int(0.8 * len(full_dataset))
    validation_size = len(full_dataset) - train_size

    _, validation_dataset = random_split(
        full_dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False
    )

    # Load trained model
    model = ChangeDetectionModel().to(device)

    model.load_state_dict(
        torch.load(CHECKPOINT, map_location=device)
    )

    model.eval()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    # Run predictions without training
    with torch.inference_mode():

        for index, (image_a, image_b, label) in enumerate(loader):

            image_a = image_a.to(device)
            image_b = image_b.to(device)
            label = label.to(device) >= 0.5

            probability = torch.sigmoid(model(image_a, image_b))
            prediction = probability >= THRESHOLD

            tp = (prediction & label).sum().item()
            fp = (prediction & ~label).sum().item()
            fn = (~prediction & label).sum().item()
            tn = (~prediction & ~label).sum().item()

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

            results.append({
                "image": index,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn
            })

            # Convert tensors into images
            image_a_np = image_a[0].cpu().permute(1, 2, 0).numpy()
            image_b_np = image_b[0].cpu().permute(1, 2, 0).numpy()

            label_np = label[0, 0].cpu().numpy()
            prediction_np = prediction[0, 0].cpu().numpy()

            # Error map:
            # Green = correct change
            # Red = false positive
            # Yellow = missed change

            error_map = np.zeros(
                (*label_np.shape, 3),
                dtype=np.uint8
            )

            error_map[(prediction_np == 1) & (label_np == 1)] = [0, 255, 0]
            error_map[(prediction_np == 1) & (label_np == 0)] = [255, 0, 0]
            error_map[(prediction_np == 0) & (label_np == 1)] = [255, 255, 0]

            # Save selected visual examples
            if index < 10:

                fig, axes = plt.subplots(1, 4, figsize=(16, 4))

                axes[0].imshow(image_a_np)
                axes[0].set_title("Image A")

                axes[1].imshow(image_b_np)
                axes[1].set_title("Image B")

                axes[2].imshow(label_np, cmap="gray")
                axes[2].set_title("Actual Change")

                axes[3].imshow(error_map)
                axes[3].set_title("Prediction Error Map")

                for ax in axes:
                    ax.axis("off")

                plt.tight_layout()

                plt.savefig(
                    OUTPUT_DIR / f"validation_{index:03d}.png",
                    dpi=150
                )

                plt.close()

    # Save per-image error statistics
    csv_path = OUTPUT_DIR / "error_analysis_metrics.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=["image", "TP", "FP", "FN", "TN"]
        )

        writer.writeheader()
        writer.writerows(results)

    # Print aggregate metrics
    iou = total_tp / (total_tp + total_fp + total_fn + 1e-8)
    dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn + 1e-8)

    print("\nPHASE 6 ERROR ANALYSIS")
    print("Validation images:", len(validation_dataset))
    print("Threshold:", THRESHOLD)
    print("True Positives:", total_tp)
    print("False Positives:", total_fp)
    print("False Negatives:", total_fn)
    print("True Negatives:", total_tn)
    print(f"IoU: {iou:.4f}")
    print(f"Dice: {dice:.4f}")

    print("\nResults saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
