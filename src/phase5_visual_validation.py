import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from dataset import LEVIRCDDataset
from model import ChangeDetectionModel


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LEVIRCDDataset("train")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    _, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = ChangeDetectionModel().to(device)
    model.load_state_dict(torch.load(
        "models/phase4b_longer_training_model.pth",
        map_location=device
    ))
    model.eval()

    output_dir = Path("outputs/phase5_visuals")
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold = 0.10
    saved = 0

    with torch.no_grad():
        for index, (image_a, image_b, label) in enumerate(loader):
            image_a = image_a.to(device)
            image_b = image_b.to(device)

            probability = torch.sigmoid(model(image_a, image_b))
            prediction = (probability >= threshold).float()

            a = image_a[0].cpu().permute(1, 2, 0).numpy()
            b = image_b[0].cpu().permute(1, 2, 0).numpy()
            gt = label[0, 0].cpu().numpy()
            pred = prediction[0, 0].cpu().numpy()

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(a)
            axes[0].set_title("Image A - Before")
            axes[1].imshow(b)
            axes[1].set_title("Image B - After")
            axes[2].imshow(gt, cmap="gray", vmin=0, vmax=1)
            axes[2].set_title("Ground Truth")
            axes[3].imshow(pred, cmap="gray", vmin=0, vmax=1)
            axes[3].set_title("Prediction (0.10)")

            for axis in axes:
                axis.axis("off")

            fig.tight_layout()
            fig.savefig(output_dir / f"validation_{index:03d}.png", dpi=150)
            plt.close(fig)

            saved += 1
            if saved == 5:
                break

    print(f"Saved {saved} visual comparisons to {output_dir}")


if __name__ == "__main__":
    main()
