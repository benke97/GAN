#%%
import torch
import torch.nn as nn
from models import Generator, Discriminator
from models import ResnetGenerator, PatchGANDiscriminator
from prepare_data import SimulatedDataset, ExperimentalDataset, get_data_dicts_and_transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from helpers import plot_batch, calculate_global_fft_min_max
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from image_pool import ImagePool, NuImagePool

random.seed(1337)

mrc_path = Path('data') / 'mrc'
experimental_path = Path('data') / 'experimental'
sim_data_dict, exp_data_dict, transform_sim, transform_exp = get_data_dicts_and_transforms(mrc_path, experimental_path, apply_probe=True, only_random=True)

simulated_dataset = SimulatedDataset(sim_data_dict, transform=transform_sim)
experimental_dataset = ExperimentalDataset(exp_data_dict, transform=transform_exp)

train_size_sim = int(0.8 * len(simulated_dataset))
val_size_sim = len(simulated_dataset) - train_size_sim

BATCH_SIZE = 1

train_dataset_sim, val_dataset_sim = random_split(simulated_dataset, [train_size_sim, val_size_sim])
train_dataset_exp, val_dataset_exp = random_split(experimental_dataset, [train_size_sim, val_size_sim])
simulated_loader_train = DataLoader(train_dataset_sim, batch_size=BATCH_SIZE, shuffle=True)
simulated_loader_val = DataLoader(val_dataset_sim, batch_size=BATCH_SIZE, shuffle=True)
experimental_loader_train = DataLoader(train_dataset_exp, batch_size=BATCH_SIZE, shuffle=True)
experimental_loader_val = DataLoader(val_dataset_exp, batch_size=BATCH_SIZE, shuffle=True)

def adjust_learning_rate(optimizer, epoch, initial_lr, num_epochs):
    """ Adjusts learning rate each epoch: constant for first half, linear decay in second half. """
    if epoch < num_epochs / 2:
        lr = initial_lr
    else:
        lr = initial_lr * float(num_epochs - epoch) / (num_epochs / 2)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

norm = "instance"
G_AB = ResnetGenerator(norm, n_blocks=6, use_dropout=True).to(device)
G_BA = ResnetGenerator(norm, n_blocks=6, use_dropout=True).to(device)
D_A = PatchGANDiscriminator(norm).to(device)
D_B = PatchGANDiscriminator(norm).to(device)

criterion_GAN = nn.BCELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

initial_lr_G = 0.00001
initial_lr_D_A = 0.00005
initial_lr_D_B = 0.00005

optimizer_G = torch.optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()), lr=initial_lr_G, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=initial_lr_D_A, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=initial_lr_D_B, betas=(0.5, 0.999))

lambda_cycle = 20
lambda_id = 0.5 * lambda_cycle

num_epochs = 300
buffer_D_B = ImagePool(100)
buffer_D_A = ImagePool(100)

def get_sample_image(loader):
    for image_data in loader:
        return image_data[0][0]

sample_A = get_sample_image(experimental_loader_train).to(device)
sample_B = get_sample_image(simulated_loader_train).to(device)

add_noise = lambda x, noise_std: torch.clamp(x + torch.randn_like(x).to(x.device) * noise_std,0,1)

writer = SummaryWriter()

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    adjust_learning_rate(optimizer_G, epoch, initial_lr_G, num_epochs)
    adjust_learning_rate(optimizer_D_A, epoch, initial_lr_D_A, num_epochs)
    adjust_learning_rate(optimizer_D_B, epoch, initial_lr_D_B, num_epochs)

    G_AB.train()
    G_BA.train()
    D_A.train()
    D_B.train()
    train_loss_G = 0
    train_loss_D_A = 0
    train_loss_D_B = 0

    loader_A_train = tqdm(experimental_loader_train, total=len(experimental_loader_train))
    loader_B_train = tqdm(simulated_loader_train, total=len(simulated_loader_train))
    
    for real_A_data, real_B_data in zip(loader_A_train, loader_B_train):
        
        optimizer_G.zero_grad()

        real_A = real_A_data[0].to(device)
        real_B = real_B_data[0].to(device)
        fake_B = G_AB(real_A)
    
        discriminator_output = D_B(add_noise(fake_B,0.001))
        target_tensor = torch.ones_like(discriminator_output).to(device)  
        loss_GAN_AB = criterion_GAN(discriminator_output, target_tensor)

        fake_A = G_BA(real_B)

        discriminator_output = D_A(add_noise(fake_A,0.001))
        target_tensor = torch.ones_like(discriminator_output).to(device)
        #print(discriminator_output.shape)
        loss_GAN_BA = criterion_GAN(discriminator_output, target_tensor)

        recovered_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recovered_A, real_A)
        recovered_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recovered_B, real_B)

        loss_id_A = criterion_identity(G_BA(real_A), real_A)
        loss_id_B = criterion_identity(G_AB(real_B), real_B)
        
        loss_G = loss_GAN_AB + loss_GAN_BA + lambda_cycle * (loss_cycle_B + loss_cycle_A) + lambda_id * (loss_id_A + loss_id_B)
        loss_G.backward()
        optimizer_G.step()

        optimizer_D_A.zero_grad()
        real_A_preds = D_A(add_noise(real_A,0.001))
        target_real = torch.ones_like(real_A_preds).to(device)
        loss_real = criterion_GAN(real_A_preds, target_real)

        fake_A_buffered = buffer_D_A.query(fake_A).to(device)
        fake_A_preds = D_A(add_noise(fake_A.detach(),0.001))
        target_fake = torch.zeros_like(fake_A_preds).to(device)
        loss_fake = criterion_GAN(fake_A_preds, target_fake)

        loss_D_A = (loss_real + loss_fake) / 2
        loss_D_A.backward()  
        optimizer_D_A.step()

        optimizer_D_B.zero_grad()

        real_B_preds = D_B(add_noise(real_B,0.001))
        target_real = torch.ones_like(real_B_preds).to(device)
        loss_real = criterion_GAN(real_B_preds, target_real)

        fake_B_buffered = buffer_D_B.query(fake_B).to(device)
        fake_B_preds = D_B(add_noise(fake_B_buffered.detach(),0.001))
        target_fake = torch.zeros_like(fake_B_preds).to(device)
        loss_fake = criterion_GAN(fake_B_preds, target_fake)

        loss_D_B = (loss_real + loss_fake) / 2
    
        loss_D_B.backward()
        optimizer_D_B.step()

        train_loss_G += loss_G.item()
        train_loss_D_A += loss_D_A.item()
        train_loss_D_B += loss_D_B.item()

    writer.add_scalar('Loss/train_G', train_loss_G/len(experimental_loader_train), epoch)
    writer.add_scalar('Loss/train_D_A', train_loss_D_A/len(experimental_loader_train), epoch)
    writer.add_scalar('Loss/train_D_B', train_loss_D_B/len(experimental_loader_train), epoch)
    writer.add_scalar('Learning rate/G', optimizer_G.param_groups[0]['lr'], epoch)

    print(f"Loss G: {train_loss_G/len(experimental_loader_train)}, Loss D_A: {train_loss_D_A/len(experimental_loader_train)}, Loss D_B: {train_loss_D_B/len(experimental_loader_train)}")

    #plot an ABA cycle and a BAB cycle
    G_AB.eval()
    G_BA.eval()
    with torch.no_grad():
            # Generate ABA cycle
            fake_B = G_AB(sample_A)
            recovered_A = G_BA(fake_B)

            # Generate BAB cycle
            fake_A = G_BA(sample_B)
            recovered_B = G_AB(fake_A)

            # Convert tensors to numpy for plotting
            sample_A_np = sample_A.cpu().numpy()
            fake_B_np = fake_B.cpu().numpy()
            recovered_A_np = recovered_A.cpu().numpy()

            sample_B_np = sample_B.cpu().numpy()
            fake_A_np = fake_A.cpu().numpy()
            recovered_B_np = recovered_B.cpu().numpy()

            # Plotting
            fig, axs = plt.subplots(2, 3, figsize=(12, 8))
            
            # ABA Cycle
            axs[0, 0].imshow(np.transpose(sample_A_np, (1, 2, 0)), cmap='gray')
            axs[0, 0].set_title('Original A')
            axs[0, 1].imshow(np.transpose(fake_B_np, (1, 2, 0)), cmap='gray')
            axs[0, 1].set_title('Fake B')
            axs[0, 2].imshow(np.transpose(recovered_A_np, (1, 2, 0)), cmap='gray')
            axs[0, 2].set_title('Recovered A')

            # BAB Cycle
            axs[1, 0].imshow(np.transpose(sample_B_np, (1, 2, 0)), cmap='gray')
            axs[1, 0].set_title('Original B')
            axs[1, 1].imshow(np.transpose(fake_A_np, (1, 2, 0)), cmap='gray')
            axs[1, 1].set_title('Fake A')
            axs[1, 2].imshow(np.transpose(recovered_B_np, (1, 2, 0)), cmap='gray')
            axs[1, 2].set_title('Recovered B')

            for ax in axs.flat:
                ax.axis('off')

            #plt.show()   
            writer.add_figure('ABA cycle', fig, epoch)



# %%
