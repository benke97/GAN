import matplotlib.pyplot as plt
import numpy as np
import torchvision

def plot_batch(data_loader, title):
    # Get a batch of data
    dataiter = iter(data_loader)
    images_batch = next(dataiter)
    images_batch = images_batch[0]
    print(images_batch.shape)
    print(images_batch.dtype)
    # Convert the batch of images to a grid for display
    grid = torchvision.utils.make_grid(images_batch)
    np_grid = grid.numpy()

    # Plot the images with cmap='viridis'
    plt.figure(figsize=(12, 8))
    plt.imshow(np.transpose(np_grid, (1, 2, 0)), cmap='viridis')
    plt.title(title)
    plt.axis('off')
    plt.show()