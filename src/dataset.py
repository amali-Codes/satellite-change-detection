from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


DATASET_DIR = Path(r"C:\Users\amali akshaya\Desktop\LEVIR-CD+")


class LEVIRCDDataset(Dataset):

    def __init__(self, split="train"):

        self.image_a_dir = DATASET_DIR / split / "A"
        self.image_b_dir = DATASET_DIR / split / "B"
        self.label_dir = DATASET_DIR / split / "label"

        self.files = sorted(self.image_a_dir.glob("*.png"))

        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        self.label_transform = transforms.Compose([
            transforms.Resize(
                (256, 256),
                interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        image_a_path = self.files[index]
        image_b_path = self.image_b_dir / image_a_path.name
        label_path = self.label_dir / image_a_path.name

        image_a = Image.open(image_a_path).convert("RGB")
        image_b = Image.open(image_b_path).convert("RGB")
        label = Image.open(label_path).convert("L")

        image_a = self.image_transform(image_a)
        image_b = self.image_transform(image_b)
        label = self.label_transform(label)

        # Convert mask to binary:
        # 0 = no change
        # 1 = change
        label = (label > 0).float()

        return image_a, image_b, label


if __name__ == "__main__":

    dataset = LEVIRCDDataset("train")

    print("Number of training samples:", len(dataset))

    image_a, image_b, label = dataset[0]

    print("Image A shape:", image_a.shape)
    print("Image B shape:", image_b.shape)
    print("Label shape:", label.shape)
    print("Label values:", torch.unique(label))