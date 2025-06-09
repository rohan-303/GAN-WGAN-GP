import os
import matplotlib.pyplot as plt

def save_generated_images(images, epoch, folder="generated_images"):
    os.makedirs(folder, exist_ok=True)
    grid = images[:64].detach().cpu().numpy().reshape(-1, 28, 28)
    fig, axs = plt.subplots(8, 8, figsize=(8, 8))
    for i in range(8):
        for j in range(8):
            axs[i, j].imshow(grid[i * 8 + j], cmap='gray')
            axs[i, j].axis('off')
    plt.tight_layout()
    plt.savefig(f"{folder}/epoch_{epoch}.png")
    plt.close()