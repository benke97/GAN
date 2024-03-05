import os
import numpy as np
import mrcfile
import random
import pandas as pd
from preprocess_dataset import preprocess_dataset
import pickle as pkl
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from math import inf
import tifffile as tiff

"""
Adds Gaussian and Poisson noise to the simulated data and populates the folders 
data/noiser
│
├── exp
│
├── sim
│
├── sim_noisy
│
├── train
│   ├── clean
│   ├── noisy
│   └── exp
│
└── val
    ├── clean
    ├── noisy
    └── exp
"""
def plot_im_with_hist(image):
    #plot image and histogram in a 1x2 subfigure:
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot the image on the first subplot
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')  # Hide axes ticks
    
    # Plot the histogram on the second subplot
    axs[1].hist(image.ravel(), bins=100, range=[np.min(image),np.max(image)], fc='k', ec='k')
    axs[1].set_title('Histogram')
    
    # Display the plot
    plt.show()

def add_noise(clean_dir, save_dir):
    #files
    file_names = os.listdir(clean_dir)
    file_paths = [os.path.join(clean_dir,f"{file_name}") for file_name in file_names]
    save_paths = [os.path.join(save_dir, f"{file_name}") for file_name in file_names]
    #dataframe_paths = [os.path.join(dataframe_dir, f"structure_{file_name.split('_')[0].split('.')[0]}.pkl") for file_name in file_names]
    for file_path,save_path in zip(file_paths,save_paths):
        
        image = np.load(file_path)['arr_0']

        #Add gaussian noise
        gaussian_noise = np.random.normal(0, 0.0001, image.shape)
        noisy_image = np.clip(image+gaussian_noise,0,np.max(image+gaussian_noise))

        #Apply Poisson noise
        noisy_image = np.random.poisson(image*500)/500
        noisy_image = gaussian_filter(noisy_image, sigma=0.5)

        #Save noisy image
        np.savez_compressed(save_path, noisy_image)

def convert_exp_to_npz(raw_dir, save_dir):
        
        def convert_tiff_to_npz(tiff_path, npz_path):
            image = tiff.imread(tiff_path)
            np.savez_compressed(npz_path, image)
    
        file_names = os.listdir(raw_dir)
        file_paths = [os.path.join(raw_dir,f"{file_name}") for file_name in file_names]
        save_paths = [os.path.join(save_dir, f"{file_name.split('.')[0]}.npz") for file_name in file_names]
        
        for file_path,save_path in zip(file_paths,save_paths):
            convert_tiff_to_npz(file_path,save_path)

def convert_raw_to_npz(raw_dir, save_dir, apply_probe=False,dataframe_dir = "data/noiser/dataframes/"):

    def convert_mrc_to_npz(mrc_path, npz_path, apply_probe=False, structure_df = None):
        with mrcfile.open(mrc_path, permissive=True) as mrc:
            img_data = mrc.data
            if apply_probe:
                pixel_size = structure_df['pixel_size'].iloc[0]
                hwhm_nm = 0.05  # HWHM in nanometers
                hwhm_pixels = hwhm_nm / pixel_size
                sigma = hwhm_pixels / np.sqrt(2 * np.log(2))
                probe = sigma
                img_data = gaussian_filter(img_data, sigma=probe)
            np.savez_compressed(npz_path, img_data)    




    file_names = os.listdir(raw_dir)
    file_paths = [os.path.join(raw_dir,f"{file_name}") for file_name in file_names]
    save_paths = [os.path.join(save_dir, f"{file_name.split('_')[0]}.npz") for file_name in file_names]
    
    for file_name, file_path, save_path in zip(file_names, file_paths, save_paths):
        idx = file_name.split('_')[0]

        with open(os.path.join(dataframe_dir,f"structure_{idx}.pkl"), "rb") as f:
            structure_df = pkl.load(f)

        convert_mrc_to_npz(file_path,save_path,apply_probe=True, structure_df=structure_df)

def normalize(data_dir, normalization="minmax"):
    #Assumes data_dir is filled with only .npz files

    def min_max_normalize(image,min,max):
        return (image-min)/(max-min)

    if normalization != "minmax":
        NotImplementedError
    files = os.listdir(data_dir)
    global_max = -inf
    global_min = inf
    for file in files:
        
        image = np.load(os.path.join(data_dir,file))["arr_0"]
        max_val = np.max(image)
        min_val = np.min(image)
        
        if max_val > global_max:
            global_max = max_val
        if min_val < global_min:
            global_min = min_val
        
    for file in files: 
        image = np.load(os.path.join(data_dir,file))["arr_0"]
        image = min_max_normalize(image,global_min,global_max)
        np.savez_compressed(os.path.join(data_dir,file), image)  

def normalize_exp(data_dir, normalization="minmax"):
    #Assumes data_dir is filled with only .npz files
    
    def min_max_normalize(image,min,max):
        return (image-min)/(max-min)
    
    files = os.listdir(data_dir)
    global_max = -inf
    for file in files:
        
        image = np.load(os.path.join(data_dir,file))["arr_0"]
        image = image-(np.min(image)+300)#Get rid of the background
        
        #set negative values to 0
        image[image<=0] = 0

        max_val = np.max(image)
        
        if max_val > global_max:
            global_max = max_val

    for file in files: 
        image = np.load(os.path.join(data_dir,file))["arr_0"]
        image = image-(np.min(image)+300) #Get rid of the background
        image = min_max_normalize(image,0,global_max)
        image[image<=0] = 0
        np.savez_compressed(os.path.join(data_dir,file), image)  

def split_and_save(split_ratio, data_dir, domain_name, return_split=False, split="random"):
    #Assumes data_dir is filled with only .npz files
    files = os.listdir(sim_dir)
    num_train = int(len(files)*split_ratio)
    num_val = len(files) - num_train
    if split == "random":
        train_files = random.sample(files, num_train)
        val_files = [file for file in files if file not in train_files]
    else:
        train_files = split[0]
        val_files = split[1]
    
    # navigate back one directory
    root = os.path.dirname(data_dir.rstrip('/'))
    root = os.path.join(root, '')

    train_dir = os.path.join(root, "train", domain_name)
    val_dir = os.path.join(root, "val", domain_name)
    for file in train_files:
        image = np.load(os.path.join(data_dir,file))["arr_0"]
        np.savez_compressed(os.path.join(train_dir,file), image)
    
    for file in val_files:
        image = np.load(os.path.join(data_dir,file))["arr_0"]
        np.savez_compressed(os.path.join(val_dir,file), image)
    
    if return_split:
        return train_files, val_files

def split_and_save_exp(exp_dir,sim_dir,split=0.8,train_path="data/noiser/train/exp", val_path = "data/noiser/val/exp"):
    files = os.listdir(exp_dir)
    num_sim_files = len(os.listdir(sim_dir))
    num_sim_train = int(num_sim_files*split)
    num_sim_val = num_sim_files - num_sim_train
    num_total_files = len(files)
    num_train_images = int(num_total_files*0.8)
    num_series = len(np.unique([file.split('_')[1] for file in files]))
    
    #calc number of images per series by counting how many files end with the same series number (1_1.npz, 2_1.npz, 3_1.npz, 4_2.npz ...)
    images_per_series = np.zeros(num_series)
    for file in files:
        series_number = int(file.split('_')[1].split('.')[0])
        images_per_series[series_number-1] += 1

    #pick random series until num_train_images is reached
    train_series = []
    val_series = []
    train_images = 0
    val_images = 0
    while train_images < num_train_images:
        series = random.randint(1,num_series)
        if series not in train_series:
            train_series.append(series)
            train_images += images_per_series[series-1]
    for i in range(1,num_series+1):
        if i not in train_series:
            val_series.append(i)
            val_images += images_per_series[i-1]

    selected_train_images_per_series = np.zeros(num_series)
    selected_val_images_per_series = np.zeros(num_series)
    #ensure all series are represented in train and val
    for series in train_series:
        selected_train_images_per_series[series-1] = 1
    for series in val_series:
        selected_val_images_per_series[series-1] = 1

    #print(train_series)
    #number of images to select from each series
    if train_images > num_sim_train and val_images > num_sim_val:
        
        while sum(selected_train_images_per_series) < num_sim_train:
            series = random.choice(train_series)
            if selected_train_images_per_series[series-1] < images_per_series[series-1]:
                selected_train_images_per_series[series-1] += 1

        while sum(selected_val_images_per_series) < num_sim_val:
            series = random.choice(val_series)
            if selected_val_images_per_series[series-1] < images_per_series[series-1]:
                selected_val_images_per_series[series-1] += 1
    else:
        NotImplementedError("More sim than exp")

    #make a list of lists where each sublist contains the file names of the images in the series
    train_series_files = []
    val_series_files = []
    for i in range(1,num_series+1):
        series_files = [file for file in files if file.split('_')[1].split('.')[0] == str(i)]
        if i in train_series:
            train_series_files.append(series_files)
        else:
            val_series_files.append(series_files)
    
    train_series_files = [item for sublist in train_series_files for item in sublist]
    val_series_files = [item for sublist in val_series_files for item in sublist]

    train_files = []
    val_files = []
    #for each series in train_series, randomly choose selected_train_images_per_series[i] images from the series and append to train_files
    for idx, num_images in enumerate(selected_train_images_per_series):
        if num_images > 0 and idx+1 in train_series:
            train_files.extend(random.sample(train_series_files, int(num_images)))
            
    for idx, num_images in enumerate(selected_val_images_per_series):
        if num_images > 0 and idx+1 in val_series:
            val_files.extend(random.sample(val_series_files, int(num_images)))

    # save them as npz with name "idx.npz" in train/exp and val/exp
    file_name_mapping = {}

    for idx,file in enumerate(train_files):
        image = np.load(os.path.join(exp_dir,file))["arr_0"]
        np.savez_compressed(os.path.join(train_path,f"{idx}.npz"), image)
        file_name_mapping[file] = f"train/{idx}.npz"

    for idx,file in enumerate(val_files):
        image = np.load(os.path.join(exp_dir,file))["arr_0"]
        np.savez_compressed(os.path.join(val_path,f"{idx}.npz"), image)
        file_name_mapping[file] = f"val/{idx}.npz"

    #save file_name_mapping as a pickle file
    with open("data/noiser/file_name_mapping.pkl", "wb") as f:
        pkl.dump(file_name_mapping, f)

    
if __name__ == "__main__":

    # Read all mrc from noiser/raw_sim and save as npz in sim
    raw_sim_dir = "data/noiser/raw_sim/"
    save_dir = "data/noiser/sim/"
    raw_exp_dir = "data/noiser/raw_exp/"
    exp_save_dir = "data/noiser/exp/"


    if len(os.listdir(raw_sim_dir)) == len(os.listdir(save_dir)) and len(os.listdir(save_dir)) != 0:
        print("Raw files already converted")
    else:
        convert_raw_to_npz(raw_sim_dir,save_dir, apply_probe=True)

    if len(os.listdir(raw_exp_dir)) == len(os.listdir(exp_save_dir)) and len(os.listdir(exp_save_dir)) != 0:
        print("Raw files already converted")
    else:
        convert_exp_to_npz(raw_exp_dir,exp_save_dir)

    # Add noise to the simulated images and save in sim_noisy
    sim_dir = "data/noiser/sim/"
    noisy_dir = "data/noiser/sim_noisy/"
    exp_dir = "data/noiser/exp/"
    if len(os.listdir(sim_dir)) == len(os.listdir(noisy_dir)) and len(os.listdir(noisy_dir)) != 0:
        print("Noisy data already created")
    else:
        add_noise(sim_dir, noisy_dir)

    # Normalize sets by min max normalization within set
    normalize(sim_dir)
    normalize(noisy_dir)
    normalize_exp(exp_dir)    
    print("Normalization complete")
    #Split into train and val
    assert len(os.listdir(sim_dir)) == len(os.listdir(noisy_dir))
    train_files, val_files = split_and_save(0.8, sim_dir, "clean",return_split=True)
    
    split_and_save(0.8, noisy_dir, "noisy", split = [train_files, val_files])

    split_and_save_exp(exp_dir, sim_dir, split=0.8)

    




    

    