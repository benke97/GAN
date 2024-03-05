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

import torch
import numpy as np
from torch.utils.data import Dataset

# Function to calculate global min and max FFT magnitudes
def calculate_global_fft_min_max(dataset):
    global_min = np.inf
    global_max = -np.inf

    for idx in range(len(dataset)):
        # Get the image and its FFT
        _, fft_complex = dataset[idx]
        
        # Compute magnitude of the FFT
        magnitude = torch.abs(fft_complex)
        
        # Update global min and max using torch operations
        global_min = min(global_min, torch.min(magnitude).item())
        global_max = max(global_max, torch.max(magnitude).item())
    
    return global_min, global_max