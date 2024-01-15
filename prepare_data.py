import pickle as pkl
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import re
import matplotlib.pyplot as plt
import random
from skimage import io
from collections import Counter
from torchvision import transforms
from torchvision.transforms import functional as TF
import mrcfile
import torch
from scipy.ndimage import gaussian_filter

def get_data_dicts_and_transforms(simulated_path, experimental_path, apply_probe=False, only_110_za=False):
    """Return the data dictionaries for experimental and simulated data."""
    def extract_number(file_name):
        """Extract the numerical part from the file name."""
        match = re.match(r"(\d+)_HAADF\.mrc", file_name)
        return int(match.group(1)) if match else None

    # Directory containing your files
    directory = simulated_path

    # List all image files and sort them
    image_files = sorted([f for f in os.listdir(directory) if re.match(r"(\d+)_HAADF\.mrc", f)], key=extract_number)
    # Initialize an empty DataFrame for all data
    data_dict = {}

    # Process each image and corresponding .pkl file
    global_min_sim = np.inf
    global_max_sim = -np.inf
    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        
        with mrcfile.open(image_path, permissive=True) as mrc:
            image_data = mrc.data
            global_min_sim = min(global_min_sim, np.min(image_data))
            global_max_sim = max(global_max_sim, np.max(image_data))
        image = Image.fromarray(image_data)
        num = extract_number(image_file)
        pkl_file = os.path.join(directory, f"structure_{num}.pkl")
        
        with open(pkl_file, 'rb') as file:
            df = pkl.load(file)
        
        if apply_probe:
            pixel_size = df['pixel_size'].iloc[0]
            
            # Probe is a Gaussian kernel with HWHM of 0.05 nm with a pixel size of pixel_size
            hwhm_nm = 0.05  # HWHM in nanometers
            hwhm_pixels = hwhm_nm / pixel_size
            sigma = hwhm_pixels / np.sqrt(2 * np.log(2))
            
            probe = sigma
            #print(pixel_size,probe,hwhm_pixels)
            # Convolve the image with the probe
            image = Image.fromarray(gaussian_filter(image_data, sigma=probe))
            #plt.subplot(1, 2, 1)
            #plt.imshow(image_data, cmap='gray')
            #plt.title("Original")
            #plt.subplot(1, 2, 2)
            #plt.imshow(image, cmap='gray')
            #plt.title("Convolved")
            #plt.show()

        data_dict[num] = {'dataframe': df, 'image': image, 'pixel_size': df['pixel_size'].iloc[0]}

    if only_110_za:
        #remove all that dont  have support in 110 zone axis
        filtered_data = {
            num: data
            for num, data in data_dict.items()
            if data['dataframe']['support_interface'].iloc[0] == '111' and data['dataframe']['subset'].iloc[0][:3] != '360'
        }
        data_dict = {num: data for num, data in enumerate(filtered_data.values())}
    #plot all remaining images in data_dict
    #for num, data in data_dict.items():
    #    image = data['image']
    #    image_data = np.array(image)
    #    plt.imshow(image_data, cmap='gray')
    #    plt.title(f"Image {num}")
    #    plt.show()

    pixel_sizes = [data['pixel_size'] for data in data_dict.values()]
    bin_centers = [0.02, 0.025, 0.03, 0.035]

    def closest_bin(pixel_size, bins):
        return bins[np.argmin([abs(pixel_size - bin_val) for bin_val in bins])]

    assigned_bins = [closest_bin(pixel_size, bin_centers) for pixel_size in pixel_sizes]

    bin_counts = Counter(assigned_bins)

    counts = [bin_counts.get(center, 0) for center in bin_centers]

    # Create a bar plot

    bin_dir_map = {0.02: "20 pm", 0.025: "25 pm", 0.03: "30 pm", 0.035: "35 pm"}
    selected_images_dict = {}

    for count, bin_dir in zip(counts, bin_dir_map.values()):
        dir_path = os.path.join(experimental_path, bin_dir)
        tif_files = [f for f in os.listdir(dir_path) if f.endswith('.tif') and re.match(r'\d+\.tif', f)]

        # Randomly select 'count' number of files
        selected_files = random.sample(tif_files, min(count, len(tif_files)))

        selected_images_dict[bin_dir] = [os.path.join(dir_path, f) for f in selected_files]

    bin_dir_map_reverse = {"20 pm": 0.02, "25 pm": 0.025, "30 pm": 0.03, "35 pm": 0.035}

    experimental_data_dict = {}
    counter = 0
    global_min_exp = np.inf
    global_max_exp = -np.inf
    for bin_dir, image_paths in selected_images_dict.items():
        for image_path in image_paths:
            pixel_size = bin_dir_map_reverse[bin_dir]
            image = Image.open(image_path).convert('F')
            image_data = np.array(image)
            global_min_exp = min(global_min_exp, np.min(image_data))
            global_max_exp = max(global_max_exp, np.max(image_data))
            # Store data in the dictionary
            experimental_data_dict[counter] = {
                'dataframe': None,  # Assuming this is to be filled later
                'image': image,
                'pixel_size': pixel_size,
            }
            counter += 1

    transform_sim = transforms.Compose([
        RandomRotation90(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        MinMaxNormalize(global_min_sim, global_max_sim)
    ])

    transform_exp = transforms.Compose([
        RandomRotation90(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        MinMaxNormalize(global_min_exp, global_max_exp)
    ])

    return data_dict, experimental_data_dict, transform_sim, transform_exp


class RandomRotation90:
    """Rotate by one of the given angles."""

    def __init__(self):
        self.angles = [0, 90, 180, 270]

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

class MinMaxNormalize(object):
    def __init__(self, global_min, global_max):
        
        self.global_min = global_min
        self.global_max = global_max

    def __call__(self, img_tensor):
        
        normalized_tensor = (img_tensor - self.global_min) / (self.global_max - self.global_min)

        normalized_tensor = normalized_tensor.clamp(0, 1)

        return normalized_tensor

class SimulatedDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        self.data_dict = data_dict
        self.transform = transform

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        dataframe, simulated_image, pixel_size = self.data_dict[idx].values()
        if self.transform:
            simulated_image = self.transform(simulated_image)
            
        return simulated_image
    
class ExperimentalDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        self.data_dict = data_dict
        self.transform = transform

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        dataframe, experimental_image, pixel_size = self.data_dict[idx].values()
        if self.transform:
            experimental_image = self.transform(experimental_image)

        return experimental_image

# %%
