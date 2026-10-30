
from pathlib import Path
import csv

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import random_split

from dataset import LEVIRCDDataset
from model import ChangeDetectionModel


# Configuration
CHECKPOINT = "models/phase4b_longer_training_model.pth"
METRICS_FILE = Path("outputs/phase6_error_analysis/per_image_metrics.csv")
OUTPUT_DIR = Path("outputs/phase6_error_analysis/top_error_visuals")

THRESHOLD = 0.10
SEED = 42
TOP_N = 5


def tensor_to_image(tensor):
    """Convert an image tensor into a displayable image."""
    image = tensor.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    return np.clip(image, 0, 1)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Recreate the same validation split
    full_dataset = LEVIRCDDataset(split="train")

    _, validation_dataset = random_split(
        full_dataset,
        [509, 128],
        generator=torch.Generator().manual_seed(SEED)
    )

    # Read the per-image metrics
    with open(METRICS_FILE, "r", newline="") as file:
        results = list(csv.DictReader(file))

    # Rank by total error pixels: FP + FN
    results.sort(
        key=lambda row: int(row["FP"]) + int(row["FN"]),
        reverse=True
    )

    top_results = results[:TOP_N]

    # Load the existing model
    model = ChangeDetectionModel().to(device)

    checkpoint = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    print("\nTOP 5 VALIDATION ERROR IMAGES")
    print("-" * 55)

    for item in top_results:
        index = int(item["validation_index"])

        print(
            f"Image {index:03d} | "
            f"IoU: {float(item['IoU']):.4f} | "
            f"FP: {item['FP']} | "
            f"FN: {item['FN']}"
        )

        # Load the corresponding validation image
        image_a, image_b, mask = validation_dataset[index]

        image_a_batch = image_a.unsqueeze(0).to(device)
        image_b_batch = image_b.unsqueeze(0).to(device)

        with torch.inference_mode():
            logits = model(image_a_batch, image_b_batch)
            probability = torch.sigmoid(logits)

            prediction = (
                probability >= THRESHOLD
            ).float()

        actual_mask = mask.squeeze().cpu().numpy()
        predicted_mask = prediction.squeeze().cpu().numpy()

        # Create error map
        error_map = np.zeros(
            (*actual_mask.shape, 3),
            dtype=np.uint8
        )

        tp = (predicted_mask == 1) & (actual_mask == 1)
        fp = (predicted_mask == 1) & (actual_mask == 0)
        fn = (predicted_mask == 0) & (actual_mask == 1)

        error_map[tp] = [0, 255, 0]       # Green: correct detection
        error_map[fp] = [255, 0, 0]       # Red: false positive
        error_map[fn] = [255, 255, 0]     # Yellow: missed change

        # Display all four views
        fig, axes = plt.subplots(1, 4, figsize=(17, 5))

        axes[0].imshow(tensor_to_image(image_a))
        axes[0].set_title("Image A - Earlier")

        axes[1].imshow(tensor_to_image(image_b))
        axes[1].set_title("Image B - Later")

        axes[2].imshow(actual_mask, cmap="gray")
        axes[2].set_title("Actual Change")

        axes[3].imshow(error_map)
        axes[3].set_title(
            f"Prediction Errors\n"
            f"IoU: {float(item['IoU']):.4f}"
        )

        for axis in axes:
            axis.axis("off")

        plt.suptitle(
            f"Validation Image {index:03d} | "
            f"FP: {item['FP']} | FN: {item['FN']}"
        )

        plt.tight_layout()

        output_path = (
            OUTPUT_DIR / f"top_error_{index:03d}.png"
        )

        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("\nVisualisation completed!")
    print(f"Saved images to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
