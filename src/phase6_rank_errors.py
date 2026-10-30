
from pathlib import Path
import csv

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, random_split

from dataset import LEVIRCDDataset
from model import ChangeDetectionModel


# Configuration
CHECKPOINT = "models/phase4b_longer_training_model.pth"
OUTPUT_DIR = Path("outputs/phase6_error_analysis")
WORST_CASES_DIR = OUTPUT_DIR / "worst_cases"

THRESHOLD = 0.10
SEED = 42
NUMBER_OF_WORST_CASES = 5


def tensor_to_image(tensor):
    """Convert a PyTorch image tensor into a displayable image."""
    image = tensor.detach().cpu().numpy()

    if image.ndim == 3:
        image = np.transpose(image, (1, 2, 0))

    return np.clip(image, 0, 1)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WORST_CASES_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Recreate the exact validation split used in Phase 5 and Phase 6
    full_dataset = LEVIRCDDataset(split="train")

    train_size = 509
    validation_size = 128

    _, validation_dataset = random_split(
        full_dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False
    )

    # Load the existing Phase 4B model
    model = ChangeDetectionModel().to(device)

    checkpoint = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    results = []

    print("\nRanking validation images...")

    with torch.inference_mode():
        for index, (image_a, image_b, mask) in enumerate(validation_loader):

            image_a = image_a.to(device)
            image_b = image_b.to(device)
            mask = mask.to(device)

            logits = model(image_a, image_b)
            probabilities = torch.sigmoid(logits)

            prediction = (probabilities >= THRESHOLD).float()

            tp = ((prediction == 1) & (mask == 1)).sum().item()
            fp = ((prediction == 1) & (mask == 0)).sum().item()
            fn = ((prediction == 0) & (mask == 1)).sum().item()
            tn = ((prediction == 0) & (mask == 0)).sum().item()

            denominator = tp + fp + fn

            iou = tp / denominator if denominator > 0 else float("nan")

            dice_denominator = (2 * tp) + fp + fn
            dice = (
                (2 * tp) / dice_denominator
                if dice_denominator > 0
                else float("nan")
            )

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            results.append({
                "validation_index": index,
                "original_dataset_index": validation_dataset.indices[index],
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
                "TN": int(tn),
                "IoU": iou,
                "Dice": dice,
                "Precision": precision,
                "Recall": recall
            })

    # Rank images from lowest IoU to highest IoU
    results.sort(
        key=lambda item: (
            np.isnan(item["IoU"]),
            item["IoU"] if not np.isnan(item["IoU"]) else 0
        )
    )

    # Save the complete ranking
    csv_path = OUTPUT_DIR / "per_image_metrics.csv"

    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nWORST 10 VALIDATION IMAGES")
    print("-" * 65)

    for item in results[:10]:
        print(
            f"Image {item['validation_index']:03d} | "
            f"IoU: {item['IoU']:.4f} | "
            f"Dice: {item['Dice']:.4f} | "
            f"FP: {item['FP']} | "
            f"FN: {item['FN']}"
        )

    # Save visual comparisons for the five lowest-IoU images
    for item in results[:NUMBER_OF_WORST_CASES]:

        index = item["validation_index"]

        image_a, image_b, mask = validation_dataset[index]

        image_a_batch = image_a.unsqueeze(0).to(device)
        image_b_batch = image_b.unsqueeze(0).to(device)

        with torch.inference_mode():
            probability = torch.sigmoid(
                model(image_a_batch, image_b_batch)
            )

        predicted_mask = (
            probability >= THRESHOLD
        ).float().squeeze().cpu().numpy()

        actual_mask = mask.squeeze().cpu().numpy()

        # Error map: green = correct change, red = false positive,
        # yellow = false negative, black = correct background
        error_map = np.zeros(
            (*actual_mask.shape, 3),
            dtype=np.uint8
        )

        tp_mask = (predicted_mask == 1) & (actual_mask == 1)
        fp_mask = (predicted_mask == 1) & (actual_mask == 0)
        fn_mask = (predicted_mask == 0) & (actual_mask == 1)

        error_map[tp_mask] = [0, 255, 0]
        error_map[fp_mask] = [255, 0, 0]
        error_map[fn_mask] = [255, 255, 0]

        fig, axes = plt.subplots(1, 4, figsize=(16, 5))

        axes[0].imshow(tensor_to_image(image_a))
        axes[0].set_title("Image A")

        axes[1].imshow(tensor_to_image(image_b))
        axes[1].set_title("Image B")

        axes[2].imshow(actual_mask, cmap="gray")
        axes[2].set_title("Actual Change")

        axes[3].imshow(error_map)
        axes[3].set_title(
            f"Prediction Errors\nIoU: {item['IoU']:.4f}"
        )

        for axis in axes:
            axis.axis("off")

        plt.tight_layout()

        output_path = (
            WORST_CASES_DIR /
            f"worst_validation_{index:03d}.png"
        )

        plt.savefig(output_path, dpi=150)
        plt.close(fig)

    print("\nPhase 6 ranking completed!")
    print(f"Full ranking: {csv_path}")
    print(f"Worst-case visuals: {WORST_CASES_DIR}")


if __name__ == "__main__":
    main()
