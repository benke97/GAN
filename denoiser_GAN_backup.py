#%%
import pickle as pkl
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
import re
import matplotlib.pyplot as plt
import random
from skimage import io
from collections import Counter
import tifffile as tiff
import time
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm
#%%
def is_valid_file(file_name):
    """Check if the file name matches the pattern i_HAADF.tif where i is in the specified range."""
    pattern = re.compile(r"(\d+)_HAADF\.tif")
    match = pattern.match(file_name)

    if not match:
        return False

    number = int(match.group(1))
    return 0 <= number <= 1865 and number != 1234

def extract_number(file_name):
    """Extract the numerical part from the file name."""
    match = re.match(r"(\d+)_HAADF\.tif", file_name)
    return int(match.group(1)) if match else None

# Directory containing your files
directory = "data/simulated"

# List all image files and sort them
image_files = sorted([f for f in os.listdir(directory) if re.match(r"(\d+)_HAADF\.tif", f)], key=extract_number)

# Initialize an empty DataFrame for all data
data_dict = {}

# Process each image and corresponding .pkl file
for image_file in image_files:
    # Load image
    print(image_file)
    image_path = os.path.join(directory, image_file)
    image = Image.open(image_path).convert('F')
    print(image.mode)
    # Extract number from image file name to find corresponding .pkl file
    num = extract_number(image_file)
    pkl_file = os.path.join(directory, f"structure_{num}.pkl")
    # Load dataframe from .pkl file
    with open(pkl_file, 'rb') as file:
        df = pkl.load(file)
    # Add the image to the dataframe
    data_dict[num] = {'dataframe': df, 'image': image, 'pixel_size': df['pixel_size'].iloc[0]}
# %%
# go through all the dataframes and plot a histogram of the pixel sizes
pixel_sizes = [data['pixel_size'] for data in data_dict.values()]
# Define the target bin centers
bin_centers = [0.02, 0.025, 0.03, 0.035]

# Function to determine closest bin center for a given pixel size
def closest_bin(pixel_size, bins):
    return bins[np.argmin([abs(pixel_size - bin_val) for bin_val in bins])]

# Assign each pixel size to its closest bin
assigned_bins = [closest_bin(pixel_size, bin_centers) for pixel_size in pixel_sizes]


# Count how many pixel sizes are closest to each of the four values
bin_counts = Counter(assigned_bins)

# Extract counts for each bin center, ensuring all bin centers are present
counts = [bin_counts.get(center, 0) for center in bin_centers]

# Create a bar plot
plt.bar(bin_centers, counts, width=0.0025, color='blue', align='center')
plt.xlabel('Pixel Size')
plt.ylabel('Count')
plt.title('Count of Pixel Sizes Grouped by Closest Bin Center')
plt.xticks(bin_centers)
plt.show()
# %%
bin_dir_map = {0.02: "20 pm", 0.025: "25 pm", 0.03: "30 pm", 0.035: "35 pm"}
for i in range(len(bin_centers)):
    j = 0
    dir = os.path.join("data/experimental", bin_dir_map[bin_centers[i]])
    print(dir)
    if len(os.listdir(dir)) < 10:
        #print("yup")
        list_of_files = os.listdir(dir)
        #sort the list of files based on first number (until "_")
        list_of_files.sort(key=lambda f: int(re.sub('\D', '', f)))
        #print(list_of_files)
        for file in list_of_files:
            tif_stack = tiff.imread(os.path.join(dir, file))
            print(file, tif_stack.shape)
            for k in range(tif_stack.shape[0]):
                tiff.imsave(os.path.join(dir, f"{j}.tif"), tif_stack[k])
                j += 1
            #read tif stack and save in same folder as single frames

print(counts)
selected_images_dict = {}

for count, bin_dir in zip(counts, bin_dir_map.values()):
    dir_path = os.path.join("data/experimental", bin_dir)
    tif_files = [f for f in os.listdir(dir_path) if f.endswith('.tif') and re.match(r'\d+\.tif', f)]

    # Randomly select 'count' number of files
    selected_files = random.sample(tif_files, min(count, len(tif_files)))

    selected_images_dict[bin_dir] = [os.path.join(dir_path, f) for f in selected_files]
print(len(selected_images_dict.get("25 pm")))
# reshape into dict with keys "pixel size", "image_path"
experimental_data_dict = {}
counter = 0
for bin_dir, image_paths in selected_images_dict.items():
    for image_path in image_paths:
        bin_dir_map_2 = {"20 pm": 0.02, "25 pm": 0.025, "30 pm": 0.03, "35 pm": 0.035}
        pixel_size = bin_dir_map_2[bin_dir]
        image = Image.open(image_path).convert('F')
        experimental_data_dict[counter] = {'dataframe': None, 'image': image, 'pixel_size': pixel_size}
        counter += 1

# %%
#print(experimental_data_dict[0])
#print(data_dict[0])
class RandomRotation90:
    """Rotate by one of the given angles."""

    def __init__(self):
        self.angles = [0, 90, 180, 270]

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

class MinMaxNormalize(object):
    def __init__(self):
        # Additional initialization can be added here if necessary
        pass

    def __call__(self, img_tensor):
        # img_tensor is expected to be a PyTorch tensor

        # Normalize: set min to 0
        img_tensor = (img_tensor - img_tensor.min() / (img_tensor.max() - img_tensor.min()))

        return img_tensor

class SimulatedDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        self.data_dict = data_dict
        self.transform = transform

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        dataframe, simulated_image, pixel_size = self.data_dict[idx].values()

        height, width = simulated_image.size
        pixel_size_tensor = torch.full((1, height, width), pixel_size, dtype=torch.float32)

        if self.transform:
            simulated_image = self.transform(simulated_image)

        
        return torch.cat((simulated_image, pixel_size_tensor), dim=0)
    

class ExperimentalDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        self.data_dict = data_dict
        self.transform = transform

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):

        dataframe, experimental_image, pixel_size = self.data_dict[idx].values()

        height, width = experimental_image.size
        pixel_size_tensor = torch.full((1, height, width), pixel_size, dtype=torch.float32)

        if self.transform:
            experimental_image = self.transform(experimental_image)

        return torch.cat((experimental_image, pixel_size_tensor), dim=0)


transform = transforms.Compose([
    RandomRotation90(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    MinMaxNormalize()
])

simulated_dataset = SimulatedDataset(data_dict, transform=transform)
simulated_loader = DataLoader(simulated_dataset, batch_size=8, shuffle=True)

experimental_dataset = ExperimentalDataset(experimental_data_dict, transform=transform)
experimental_loader = DataLoader(experimental_dataset, batch_size=8, shuffle=True)

# %%
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def conv_block(in_channels, out_channels, dropout_rate=0.5):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
        )

        def upconv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels)
        )

        # Downsampling
        self.conv1 = conv_block(2, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Bridge
        self.conv4 = conv_block(256, 512)

        # Upsampling
        self.upconv5 = upconv_block(512, 256)
        self.conv5 = conv_block(512, 256)

        self.upconv6 = upconv_block(256, 128)
        self.conv6 = conv_block(256, 128)

        self.upconv7 = upconv_block(128, 64)
        self.conv7 = conv_block(128, 64)

        self.output = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_channel, pixel_size_channel = torch.split(img, [1, 1], dim=1)
        x = img
        x1 = self.conv1(x)
        x2 = self.pool1(x1)
        x3 = self.conv2(x2)
        x4 = self.pool2(x3)
        x5 = self.conv3(x4)
        x6 = self.pool3(x5)

        x7 = self.conv4(x6)

        x8 = self.upconv5(x7)
        x9 = torch.cat([x8, x5], dim=1)
        x10 = self.conv5(x9)

        x11 = self.upconv6(x10)
        x12 = torch.cat([x11, x3], dim=1)
        x13 = self.conv6(x12)

        x14 = self.upconv7(x13)
        x15 = torch.cat([x14, x1], dim=1)
        x16 = self.conv7(x15)

        out = torch.cat([self.output(x16), pixel_size_channel], dim=1)

        return out
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_channels, out_channels, normalization=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        # Specify the input and output channels
        input_channels = 2  # for grayscale images

        # Discriminator architecture
        self.conv1 = discriminator_block(input_channels, 64, normalization=False)
        self.conv2 = discriminator_block(64, 128)
        self.conv3 = discriminator_block(128, 256)
        self.conv4 = discriminator_block(256, 512)

        # The following padding is used to adjust the shape of the output
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.final_conv = nn.Conv2d(512, 1, kernel_size=4, padding=1)

    def forward(self, img):
        # Forward pass through the discriminator layers
        x = self.conv1(img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pad(x)
        x = self.final_conv(x)
        return x
    

G_AB = Generator()  # Translates images from domain A to domain B
G_BA = Generator()  # Translates images from domain B to domain A
D_A = Discriminator()  # Discriminator for domain A
D_B = Discriminator()  # Discriminator for domain B


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
G_AB = Generator().to(device)
G_BA = Generator().to(device)
D_A = Discriminator().to(device)
D_B = Discriminator().to(device)
# Adversarial loss
criterion_GAN = nn.MSELoss()

# Cycle consistency loss
criterion_cycle = nn.L1Loss()

# Identity loss (optional, can help with training stability)
criterion_identity = nn.L1Loss()

optimizer_G = torch.optim.Adam(
    list(G_AB.parameters()) + list(G_BA.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

loader_A = experimental_loader
loader_B = simulated_loader
lambda_cycle = 5
lambda_id = 2

losses_G = []
losses_D_A = []
losses_D_B = []

num_epochs = 300
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    epoch_loss_G = 0
    epoch_loss_D_A = 0
    epoch_loss_D_B = 0

    loader_A = tqdm(experimental_loader, total=len(experimental_loader))
    loader_B = tqdm(simulated_loader, total=len(simulated_loader))
    for real_A, real_B in zip(loader_A, loader_B):

        real_A = real_A.to(device)
        real_B = real_B.to(device)
        # real_A and real_B are batches from the two domains

        # Generators G_AB and G_BA
        optimizer_G.zero_grad()

        # Identity loss
        loss_id_A = criterion_identity(G_BA(real_A), real_A)
        loss_id_B = criterion_identity(G_AB(real_B), real_B)

        # GAN loss
        fake_B = G_AB(real_A)
        discriminator_output = D_B(fake_B)
        target_tensor = torch.ones_like(discriminator_output).to(device)  
        loss_GAN_AB = criterion_GAN(discriminator_output, target_tensor)

        fake_A = G_BA(real_B)
        discriminator_output = D_A(fake_A)
        target_tensor = torch.ones_like(discriminator_output).to(device)
        loss_GAN_BA = criterion_GAN(discriminator_output, target_tensor)

        # Cycle loss
        recovered_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recovered_A, real_A)
        recovered_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recovered_B, real_B)

        # Total loss
        print(loss_GAN_AB.item(), loss_GAN_BA.item(), loss_cycle_A.item(), loss_cycle_B.item(), loss_id_A.item(), loss_id_B.item(), lambda_cycle*(loss_cycle_A.item() + loss_cycle_B.item()), lambda_id*(loss_id_A.item() + loss_id_B.item()))
        loss_G = loss_GAN_AB + loss_GAN_BA + lambda_cycle * (loss_cycle_A + loss_cycle_B) + lambda_id * (loss_id_A + loss_id_B)
        loss_G.backward()
        optimizer_G.step()

        # Discriminator A
        optimizer_D_A.zero_grad()

        real_A_output = D_A(real_A)
        target_real = torch.ones_like(real_A_output).to(device)
        loss_real = criterion_GAN(real_A_output, target_real)

        fake_A_output = D_A(fake_A.detach())
        target_fake = torch.zeros_like(fake_A_output).to(device)
        loss_fake = criterion_GAN(fake_A_output, target_fake)

        loss_D_A = (loss_real + loss_fake) / 2
        loss_D_A.backward()
        optimizer_D_A.step()

        # Discriminator B
        optimizer_D_B.zero_grad()

        real_B_output = D_B(real_B)
        target_real_B = torch.ones_like(real_B_output).to(device)
        loss_real_B = criterion_GAN(real_B_output, target_real_B)

        fake_B_output = D_B(fake_B.detach())
        target_fake_B = torch.zeros_like(fake_B_output).to(device) 
        loss_fake_B = criterion_GAN(fake_B_output, target_fake_B)

        # Combine the losses
        loss_D_B = (loss_real_B + loss_fake_B) / 2
        loss_D_B.backward()
        optimizer_D_B.step()

        epoch_loss_G += loss_G.item()
        epoch_loss_D_A += loss_D_A.item()
        epoch_loss_D_B += loss_D_B.item()

    avg_loss_G = epoch_loss_G / len(loader_A)
    avg_loss_D_A = epoch_loss_D_A / len(loader_A)
    avg_loss_D_B = epoch_loss_D_B / len(loader_A)

    losses_G.append(avg_loss_G)
    losses_D_A.append(avg_loss_D_A)
    losses_D_B.append(avg_loss_D_B)

    # Print average losses
    print(f"Generator Loss: {avg_loss_G:.4f}, Discriminator A Loss: {avg_loss_D_A:.4f}, Discriminator B Loss: {avg_loss_D_B:.4f}")

    # Visualizing Generated Images after each epoch
    with torch.no_grad():
        # Take a batch of images from loader_A or loader_B
        sample_A = next(iter(loader_A))[0].to(device)
        sample_B = next(iter(loader_B))[0].to(device)

        # Add batch dimension
        sample_A = sample_A.unsqueeze(0)
        sample_B = sample_B.unsqueeze(0)

        # Generate images
        fake_B = G_AB(sample_A)
        fake_A = G_BA(sample_B)

        # Move images to CPU for plotting and remove batch dimension
        sample_A, sample_B, fake_A, fake_B = sample_A.squeeze(0).cpu(), sample_B.squeeze(0).cpu(), fake_A.squeeze(0).cpu(), fake_B.squeeze(0).cpu()

        # Plotting
        plt.figure(figsize=(10, 4))

        # Plot only the image part of the tensors (assuming it's the first channel)
        plt.subplot(1, 4, 1)
        plt.title("Real A")
        plt.imshow(sample_A[0], cmap='gray')  # Grayscale image from the first channel

        plt.subplot(1, 4, 2)
        plt.title("Fake B")
        plt.imshow(fake_B[0], cmap='gray')  # Grayscale image from the first channel

        plt.subplot(1, 4, 3)
        plt.title("Real B")
        plt.imshow(sample_B[0], cmap='gray')  # Grayscale image from the first channel

        plt.subplot(1, 4, 4)
        plt.title("Fake A")
        plt.imshow(fake_A[0], cmap='gray')  # Grayscale image from the first channel

        plt.show()
# %%
