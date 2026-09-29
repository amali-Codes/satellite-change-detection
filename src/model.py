import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ChangeDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Image A (3 channels) + Image B (3 channels)
        self.encoder1 = DoubleConv(6, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(64, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder1 = DoubleConv(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.decoder2 = DoubleConv(64, 32)

        self.output = nn.Conv2d(32, 1, 1)

    def forward(self, image_a, image_b):
        x = torch.cat([image_a, image_b], dim=1)

        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))

        b = self.bottleneck(self.pool2(e2))

        d1 = self.up1(b)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.decoder1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.decoder2(d2)

        return self.output(d2)


if __name__ == "__main__":
    model = ChangeDetectionModel()

    image_a = torch.randn(4, 3, 256, 256)
    image_b = torch.randn(4, 3, 256, 256)

    output = model(image_a, image_b)

    print("Output shape:", output.shape)