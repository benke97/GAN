import torch
import torchvision
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt
import random
import pickle as pkl
from models import UNet128_3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class size_dataset(Dataset):
    def __init__(self, data_dir,domain_names, transform=None):
        self.data_dir = data_dir 
        self.transform = transform
        self.domain_names = domain_names
        self.data = self.load_data()

    def load_data(self):
        path_clean = self.data_dir + f"{self.domain_names[0]}/"
        path_noisy = self.data_dir + f"{self.domain_names[1]}/"
        clean_files = os.listdir(path_clean)
        noisy_files = os.listdir(path_noisy)
        structure_dfs= [os.path.join("data/noiser/dataframes",f"structure_{clean_file.split('.')[0]}.pkl") for clean_file in clean_files]

        data = []
        for clean_file, noisy_file, structure_df in zip(clean_files, noisy_files, structure_dfs):
            assert clean_file == noisy_file #same name
            clean_image = np.load(os.path.join(path_clean,clean_file))["arr_0"]
            noisy_image = np.load(os.path.join(path_noisy,noisy_file))["arr_0"]
            #count all atoms with label "Pt" in structure_df
            particle_size = 0
            
            with open(structure_df, "rb") as f:
                structure_df = pkl.load(f)
                particle_size = structure_df[structure_df.label == "Pt"].shape[0]

            data.append((clean_image,noisy_image,particle_size))

        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        clean_image, noisy_image, particle_size = self.data[idx]
        clean_image = Image.fromarray(clean_image)
        noisy_image = Image.fromarray(noisy_image)

        if self.transform:
            clean_image = self.transform(clean_image)
            noisy_image = self.transform(noisy_image)

        if random.random() > 0.5:
            clean_image = TF.hflip(clean_image)
            noisy_image = TF.hflip(noisy_image)
        if random.random() > 0.5:
            clean_image = TF.vflip(clean_image)
            noisy_image = TF.vflip(noisy_image)

        return clean_image, noisy_image, particle_size

class size_dataset_exp(Dataset):
    def __init__(self, data_dir,domain_names, transform=None):
        self.data_dir = data_dir 
        self.transform = transform
        self.domain_names = domain_names
        self.data = self.load_data()

    def load_data(self):
        path_clean = self.data_dir + f"{self.domain_names[0]}/"
        clean_files = os.listdir(path_clean)
        structure_dfs= [os.path.join("data/noiser/dataframes",f"structure_{clean_file.split('.')[0]}.pkl") for clean_file in clean_files]
        G_AB = UNet128_3(1,1,ngf=16).to(device)
        checkpoint = torch.load("exp_checkpoints/lambda_GP_0.03_cyc_20/checkpoint_epoch_150.pth")
        G_AB.load_state_dict(checkpoint["G_AB_state_dict"])
        G_AB.eval()
        data = []
        for clean_file, structure_df in zip(clean_files, structure_dfs):
            clean_image = np.load(os.path.join(path_clean,clean_file))["arr_0"]
            clean_tensor = torch.from_numpy(clean_image).float().unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                fake_exp = G_AB(clean_tensor)
            
            fake_exp = fake_exp.squeeze(0).squeeze(0).cpu().detach().numpy()
            #count all atoms with label "Pt" in structure_df
            particle_size = 0
            
            with open(structure_df, "rb") as f:
                structure_df = pkl.load(f)
                particle_size = structure_df[structure_df.label == "Pt"].shape[0]
            data.append((clean_image,fake_exp,particle_size))

        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        clean_image, noisy_image, particle_size = self.data[idx]
        clean_image = Image.fromarray(clean_image)
        noisy_image = Image.fromarray(noisy_image)

        if self.transform:
            clean_image = self.transform(clean_image)
            noisy_image = self.transform(noisy_image)

        if random.random() > 0.5:
            clean_image = TF.hflip(clean_image)
            noisy_image = TF.hflip(noisy_image)
        if random.random() > 0.5:
            clean_image = TF.vflip(clean_image)
            noisy_image = TF.vflip(noisy_image)

        return clean_image, noisy_image, particle_size
    
if __name__ == "__main__":
    data_dir = "data/noiser/train/"
    domain_names = ["clean","noisy"]
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    print(1)
    dataset = size_dataset(data_dir,domain_names,transform)
    print(2)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    print(3)
    d_exp = size_dataset_exp(data_dir,domain_names,transform)
    print(4)
    d_exp_loader = torch.utils.data.DataLoader(d_exp, batch_size=1, shuffle=True)
    print(5)
    for clean, noisy, particle_size in d_exp_loader:
        #plot images
        plt.imshow(clean[0][0],cmap="gray")
        plt.show()
        plt.imshow(noisy[0][0],cmap="gray")
        plt.show()
        

    for clean, noisy, particle_size in dataloader:
        #plot images
        plt.imshow(clean[0][0],cmap="gray")
        plt.show()
        plt.imshow(noisy[0][0],cmap="gray")
        plt.show()
        break
    

