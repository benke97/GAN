import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from size_dataset import size_dataset, size_dataset_exp
from models import UNet_size_estimator
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from math import inf

random.seed(1337)
"""Trains a U-Net to estimate size of a nanoparticle"""

# Prepare dataset, read in data and corresponding particle size
BATCH_SIZE = 256
data_dir_train = "data/noiser/train/"
data_dir_val = "data/noiser/val/"
domain_names = ["clean","noisy"]
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_dataset = size_dataset_exp(data_dir_train, domain_names, transform)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = size_dataset_exp(data_dir_val, domain_names, transform)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet_size_estimator(1,1,ngf=64).to(device)

class MSLELoss(nn.Module):
    def __init__(self):
        super(MSLELoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Adding a small constant to ensure the logarithm is computable
        # when the inputs contain zeros
        epsilon = 1e-10
        # Compute the mean squared logarithmic error
        loss = torch.mean((torch.log(y_pred + 1 + epsilon) - torch.log(y_true + 1 + epsilon)) ** 2)
        return loss
    
criterion = MSLELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)




num_epochs = 200
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
best_loss = inf

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    loader_train = tqdm(train_dataloader, total=len(train_dataloader))

    for i, (clean_image, noisy_image, particle_size) in enumerate(loader_train):
        clean_image, noisy_image, particle_size = clean_image.to(device), noisy_image.to(device), particle_size.to(device)
        optimizer.zero_grad()
        outputs = model(noisy_image)
        loss = criterion(outputs, particle_size.float()/1000)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loader_train.set_postfix(loss=running_loss/(i+1))


    model.eval()
    running_loss = 0.0

    loader_val = tqdm(val_dataloader, total=len(val_dataloader))

    for i, (clean_image, noisy_image, particle_size) in enumerate(loader_val):
        clean_image, noisy_image, particle_size = clean_image.to(device), noisy_image.to(device), particle_size.to(device)
        outputs = model(noisy_image)
        loss = criterion(outputs, particle_size.float()/1000)
        running_loss += loss.item()
        loader_val.set_postfix(loss=running_loss/(i+1))

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(val_dataloader)}")

    # Test case
    # Plot with matplotlib image, real size, predicted size
    if running_loss < best_loss:
        best_loss = running_loss
        clean_image, noisy_image, particle_size = next(iter(val_dataloader))
        clean_image, noisy_image, particle_size = clean_image.to(device), noisy_image.to(device), particle_size.to(device) / 1000
        outputs = model(noisy_image)
        #print(outputs, particle_size)
        plt.figure()
        plt.imshow(noisy_image[0][0].cpu().detach().numpy(), cmap="gray")
        plt.title(f"Predicted size: {outputs[0].item()*1000}, Real size: {particle_size[0].item()*1000}")
        plt.show()
        #save model in directory "best_size_estimator"
        if not os.path.exists("best_size_estimator"):
            os.makedirs("best_size_estimator")
        torch.save(model.state_dict(), "best_size_estimator/size_estimator.pth")
        #print("Predicted size: ", outputs[0].item(), "Real size: ", particle_size[0].item())
    
    lr_scheduler.step()
    