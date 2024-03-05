import torch
import torchvision
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt
import random

class Pretrain_dataset(Dataset):
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
        
        data = []
        for clean_file, noisy_file in zip(clean_files, noisy_files):
            if self.domain_names == ["clean","noisy"]:
                assert clean_file == noisy_file
            clean_image = np.load(os.path.join(path_clean,clean_file))["arr_0"]
            noisy_image = np.load(os.path.join(path_noisy,noisy_file))["arr_0"]
            data.append((clean_image,noisy_image))
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        clean_image, noisy_image = self.data[idx]
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

        return clean_image, noisy_image
    
if __name__ == "__main__":
    data_dir = "data/noiser/train/"
    domain_names = ["clean","exp"]
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    dataset = Pretrain_dataset(data_dir,domain_names,transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for clean, noisy in dataloader:
        #plot images
        plt.imshow(clean[0][0],cmap="gray")
        plt.show()
        plt.imshow(noisy[0][0],cmap="gray")
        plt.show()
        break
    