import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset

from dataset import LEVIRCDDataset
from model import ChangeDetectionModel


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

full_dataset = LEVIRCDDataset("train")

# Temporary small dataset for testing the pipeline
dataset = Subset(
    full_dataset,
    range(100)
)

print("Training samples:", len(dataset))


# --------------------------------------------------
# DataLoader
# --------------------------------------------------

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = ChangeDetectionModel().to(device)


# --------------------------------------------------
# Loss function
# --------------------------------------------------

criterion = nn.BCEWithLogitsLoss()


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# --------------------------------------------------
# Training
# --------------------------------------------------

epochs = 1

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for batch_index, (image_a, image_b, label) in enumerate(dataloader):

        image_a = image_a.to(device)
        image_b = image_b.to(device)
        label = label.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(image_a, image_b)

        # Calculate loss
        loss = criterion(output, label)

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        total_loss += loss.item()

        print(
            f"Batch [{batch_index + 1}/{len(dataloader)}] "
            f"Loss: {loss.item():.4f}"
        )

    average_loss = total_loss / len(dataloader)

    print(
        f"\nEpoch [{epoch + 1}/{epochs}] "
        f"Average Loss: {average_loss:.4f}"
    )


# --------------------------------------------------
# Save model
# --------------------------------------------------

torch.save(
    model.state_dict(),
    "models/baseline_model.pth"
)

print("\nTraining complete!")
print("Model saved to models/baseline_model.pth")