import matplotlib.pyplot as plt

from dataset import load_sample


if __name__ == "__main__":
    image_a, image_b, label = load_sample()

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image_a)
    plt.title("Image A - Before")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(image_b)
    plt.title("Image B - After")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(label, cmap="gray")
    plt.title("Ground Truth Change")
    plt.axis("off")

    plt.tight_layout()
    plt.show()