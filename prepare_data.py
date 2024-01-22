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
import tifffile as tiff

def get_data_dicts_and_transforms(simulated_path, experimental_path, apply_probe=False, only_random=False):
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
            hwhm_nm = 0.05  # HWHM in nanometers
            hwhm_pixels = hwhm_nm / pixel_size
            sigma = hwhm_pixels / np.sqrt(2 * np.log(2))
            probe = sigma
            image = Image.fromarray(gaussian_filter(image_data, sigma=probe))

        data_dict[num] = {'dataframe': df, 'image': image}
    # Create a bar plot

    # if only_random=True make a new data_dict with all entries in the old data_dict that have df['support_interface'] == 'random' or df['particle_interface'] == 'random'
    if only_random:
        new_data_dict = {}
        i = 0
        for key, value in data_dict.items():
            if value['dataframe']['support_interface'].iloc[0] == 'random' or value['dataframe']['particle_interface'].iloc[0] == 'random':
                new_data_dict[i] = value
                i += 1
        data_dict = new_data_dict

    number_of_simulated_images = len(data_dict)
    experimental_path = 'data/experimental'  # Adjust this path as needed

    # Find the number of unique nanostructures
    nanostructure_files = os.listdir(os.path.join(experimental_path, 'new_set/unstacked'))
    number_of_experimental_nanostructures = len(Counter([int(re.match(r"(\d+)_(\d+)\.tif", f).group(2)) for f in nanostructure_files if re.match(r"(\d+)_(\d+)\.tif", f)]))

    # Find frames per nanostructure
    frames_dict = {}
    for f in nanostructure_files:
        match = re.match(r"(\d+)_(\d+)\.tif", f)
        if match:
            nanostructure_idx = int(match.group(2))
            frame_idx = int(match.group(1))
            if nanostructure_idx not in frames_dict:
                frames_dict[nanostructure_idx] = []
            frames_dict[nanostructure_idx].append(frame_idx)

    # Initialize images_per_nanostructure
    images_per_nanostructure = [1] * number_of_experimental_nanostructures
    number_of_simulated_images -= number_of_experimental_nanostructures

    # Distribute remaining images
    while number_of_simulated_images > 0:
        nanostructure = random.randint(1, number_of_experimental_nanostructures)
        if len(frames_dict[nanostructure]) > images_per_nanostructure[nanostructure - 1]:
            images_per_nanostructure[nanostructure - 1] += 1
            number_of_simulated_images -= 1

    # Select images
    image_paths = []
    for idx, num_images in enumerate(images_per_nanostructure):
        nanostructure_idx = idx + 1
        selected_frames = random.sample(frames_dict[nanostructure_idx], num_images)
        for frame in selected_frames:
            file_path = os.path.join(experimental_path, 'new_set/unstacked', f"{frame}_{nanostructure_idx}.tif")
            image_paths.append(file_path)

    # Output
    print("Images per nanostructure:", images_per_nanostructure)
    print("Image paths:", image_paths)



    experimental_data_dict = {}
    counter = 0
    global_min_exp = np.inf
    global_max_exp = -np.inf
    for image_path in image_paths:
        image_data = tiff.imread(image_path)
        global_min_exp = min(global_min_exp, np.min(image_data))
        global_max_exp = max(global_max_exp, np.max(image_data))
        image = Image.fromarray(image_data)
        # Store data in the dictionary
        experimental_data_dict[counter] = {
            'dataframe': None,  # Assuming this is to be filled later
            'image': image,
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
        dataframe, simulated_image = self.data_dict[idx].values()
        if self.transform:
            simulated_image = self.transform(simulated_image)
            
        return simulated_image
    
    def plot_histogram(self):
        """Plot the histogram of the simulated images."""
        simulated_images = [self.data_dict[idx]['image'] for idx in range(len(self.data_dict))]
        simulated_images = torch.stack(simulated_images)
        simulated_images = simulated_images.numpy()
        plt.hist(simulated_images.flatten(), bins=100)
        plt.title("Histogram of Simulated Images")
        plt.show()
    
class ExperimentalDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        self.data_dict = data_dict
        self.transform = transform

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        dataframe, experimental_image = self.data_dict[idx].values()
        if self.transform:
            experimental_image = self.transform(experimental_image)

        return experimental_image
    
    def plot_histogram(self):
        """Plot the histogram of the experimental images."""
        experimental_images = [self.data_dict[idx]['image'] for idx in range(len(self.data_dict))]
        experimental_images = torch.stack(experimental_images)
        experimental_images = experimental_images.numpy()
        plt.hist(experimental_images.flatten(), bins=100)
        plt.title("Histogram of Experimental Images")
        plt.show()

# %%

if __name__ == '__main__':
    # Find all tif stacks in experimental data/new_set
    experimental_path = 'data/experimental/new_set'
    save_path = 'data/experimental/new_set/unstacked'
    list_of_files = os.listdir(experimental_path)

    # Ensure that only files are included, not directories
    list_of_files = [f for f in list_of_files if os.path.isfile(os.path.join(experimental_path, f))]

    # Sort the list of files based on the first four digits
    list_of_files.sort(key=lambda f: int(re.sub('\D', '', f)[:4]) if re.search('\d', f) else 0)

    j = 1
    i = 1
    for file in list_of_files:
        tif_stack = tiff.imread(os.path.join(experimental_path, file))
        print(file, tif_stack.shape)
        for k in range(tif_stack.shape[0]):
            tiff.imsave(os.path.join(save_path, f"{j}_{i}.tif"), tif_stack[k])
            j += 1
            print(j)
        i += 1
# %%
