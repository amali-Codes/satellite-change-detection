import torch
from torch.utils.data import random_split
from dataset import LEVIRCDDataset

dataset = LEVIRCDDataset("train")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)
train_set, val_set = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

print("Total training data:", len(dataset))
print("New training subset:", len(train_set))
print("Validation subset:", len(val_set))
print("Test data remains separate:", len(LEVIRCDDataset("test")))
