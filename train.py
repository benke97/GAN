#%%
import torch
import torch.nn as nn
from models import Generator, Discriminator
from models import ResnetGenerator, PatchGANDiscriminator
from prepare_data import SimulatedDataset, ExperimentalDataset, get_data_dicts_and_transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from helpers import plot_batch
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from image_pool import ImagePool

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

#plot a batch of images from both sets
plot_batch(simulated_loader_train, 'Simulated Images')
plot_batch(experimental_loader_train, 'Experimental Images')

print(len(simulated_dataset), len(experimental_dataset))
#plot the average histograms of the two datasets in the same plot

#doesnt work yet
#simulated_dataset.plot_histogram()
#experimental_dataset.plot_histogram()

#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

norm = "instance"
G_AB = ResnetGenerator(norm, n_blocks=6, use_dropout=False).to(device)
G_BA = ResnetGenerator(norm, n_blocks=6, use_dropout=False).to(device)
D_A = PatchGANDiscriminator(norm).to(device)
D_B = PatchGANDiscriminator(norm).to(device)
D_A_FFT = PatchGANDiscriminator(norm).to(device)
D_B_FFT = PatchGANDiscriminator(norm).to(device)

# Adversarial loss
criterion_GAN = nn.BCELoss()

# Cycle consistency loss
criterion_cycle = nn.L1Loss()

# Identity loss (optional, can help with training stability)
criterion_identity = nn.L1Loss()

optimizer_G_AB = torch.optim.Adam(G_AB.parameters(), lr=0.0002)
optimizer_G_BA = torch.optim.Adam(G_BA.parameters(), lr=0.0002)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=0.0002)
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=0.0002)
optimizer_D_A_FFT = torch.optim.Adam(D_A_FFT.parameters(), lr=0.0002)
optimizer_D_B_FFT = torch.optim.Adam(D_B_FFT.parameters(), lr=0.0002)


#0.001,0.001,0.0003, 0.00001 works but discriminators are trash

lambda_cycle = 10
lambda_id = 0.5 * lambda_cycle

losses_G_AB = []
losses_G_BA = []
losses_D_A = []
losses_D_B = []

log_file_path = "training_log.txt"

writer = SummaryWriter()

from skimage.metrics import structural_similarity as ssim
fixed_sample_A = next(iter(experimental_loader_val))[0].to(device)  # Fixed batch from domain A
fixed_sample_B = next(iter(simulated_loader_val))[0].to(device)     # Fixed batch from domain B

fixed_sample_A = fixed_sample_A.unsqueeze(0)
fixed_sample_B = fixed_sample_B.unsqueeze(0)
num_epochs = 300
buffer_D_B = ImagePool(50,(1, 128, 128))
buffer_D_A = ImagePool(50,(1, 128, 128))

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    epoch_loss_G_AB = 0
    epoch_loss_G_BA = 0
    epoch_loss_D_A = 0
    epoch_loss_D_B = 0
    epoch_loss_D_A_FFT = 0
    epoch_loss_D_B_FFT = 0

    G_AB.train()
    G_BA.train()
    D_A.train()
    D_B.train()
    D_A_FFT.train()
    D_B_FFT.train()

    loader_A_train = tqdm(experimental_loader_train, total=len(experimental_loader_train))
    loader_B_train = tqdm(simulated_loader_train, total=len(simulated_loader_train))
    for real_A_data, real_B_data in zip(loader_A_train, loader_B_train):

        #real_A_img, real_A_fft = real_A_data[:, 0, :, :], real_A_data[:, 1, :, :]
        #real_B_img, real_B_fft = real_B_data[:, 0, :, :], real_B_data[:, 1, :, :]
        real_A = real_A_data[0]
        real_B = real_B_data[0]
        real_A_fft = real_A_data[1]
        real_B_fft = real_B_data[1]

        real_A = real_A.to(device)
        real_B = real_B.to(device)
        real_A_fft = real_A_fft.to(device)
        real_B_fft = real_B_fft.to(device)
        #add gaussian noise to real_B with mean 0 and std 0.001
        noise_B = torch.randn(real_B.size(), device=device) * 0.001
        noise_A = torch.randn(real_A.size(), device=device) * 0.001
        real_B = real_B.clone() + noise_B
        real_A = real_A.clone() + noise_A

        real_A_fft = real_A_fft.clone() + noise_A
        real_B_fft = real_B_fft.clone() + noise_B

        # Generators G_AB and G_BA
        optimizer_G_AB.zero_grad()
        optimizer_G_BA.zero_grad()

        # Identity loss
        loss_id_A = criterion_identity(G_BA(real_A), real_A)
        loss_id_B = criterion_identity(G_AB(real_B), real_B)


        # GAN loss
        fake_B = G_AB(real_A)
        fake_B_buffered = buffer_D_B.query(fake_B).to(device)
        
        #plot fake_B
        #plt.figure(figsize=(8, 4))
        #plt.subplot(1, 2, 1)
        #plt.title("Fake B pre buffer")
        #plt.imshow(fake_B[0].cpu().detach().numpy()[0], cmap='gray')  # Grayscale image from the first channel
        #plt.subplot(1, 2, 2)

        #plt.title("Fake B post buffer")
        #plt.imshow(fake_B[0].cpu().detach().numpy()[0], cmap='gray')  # Grayscale image from the first channel
        #plt.show()


        discriminator_output = D_B(fake_B_buffered)
        discriminator_output_fft = D_B_FFT(torch.abs(torch.fft.fftshift(torch.fft.fft2(fake_B_buffered))))


        target_tensor = torch.ones_like(discriminator_output).to(device)  
        loss_GAN_AB = (criterion_GAN(discriminator_output, target_tensor) + criterion_GAN(discriminator_output_fft, target_tensor))/2

        fake_A = G_BA(real_B)
        fake_A_buffered = buffer_D_A.query(fake_A).to(device)

        #print("Before query:")
        #print(f"Datatype: {fake_A.dtype}, Range: {fake_A.min(), fake_A.max()}, Shape: {fake_A.shape}")
        #print("After query:")
        #print(f"Datatype: {fake_A.dtype}, Range: {fake_A.min(), fake_A.max()}, Shape: {fake_A.shape}")        


        discriminator_output = D_A(fake_A_buffered)
        discriminator_output_fft = D_A_FFT(torch.abs(torch.fft.fftshift(torch.fft.fft2(fake_A_buffered))))
        target_tensor = torch.ones_like(discriminator_output).to(device)
        loss_GAN_BA = (criterion_GAN(discriminator_output, target_tensor) + criterion_GAN(discriminator_output_fft, target_tensor))/2

        # Cycle loss
        recovered_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recovered_A, real_A)
        recovered_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recovered_B, real_B)

        # Total loss
        #print(loss_GAN_AB.item(), loss_GAN_BA.item(), loss_cycle_A.item(), loss_cycle_B.item(), loss_id_A.item(), loss_id_B.item(), lambda_cycle*(loss_cycle_A.item() + loss_cycle_B.item()), lambda_id*(loss_id_A.item() + loss_id_B.item()))
        loss_G_AB = loss_GAN_AB + lambda_cycle * loss_cycle_B + lambda_id * loss_id_B
        loss_G_AB.backward(retain_graph=True)
        optimizer_G_AB.step()
        
        loss_G_BA = loss_GAN_BA + lambda_cycle * loss_cycle_A + lambda_id * loss_id_A
        loss_G_BA.backward()
        optimizer_G_BA.step()

        # Discriminator A
        optimizer_D_A.zero_grad()

        real_A_output = D_A(real_A)
        target_real = torch.ones_like(real_A_output).to(device)
        loss_real = criterion_GAN(real_A_output, target_real)

        fake_A_output = D_A(fake_A_buffered.detach())
        target_fake = torch.zeros_like(fake_A_output).to(device)
        loss_fake = criterion_GAN(fake_A_output, target_fake)

        loss_D_A = (loss_real + loss_fake) / 2
        loss_D_A.backward()
        #for name, param in D_A.named_parameters():
        #    if param.requires_grad:
        #        print(f"{name}, gradient before step: {param.grad}")
        optimizer_D_A.step()

        # Discriminator B
        optimizer_D_B.zero_grad()

        real_B_output = D_B(real_B)
        target_real_B = torch.ones_like(real_B_output).to(device)
        loss_real_B = criterion_GAN(real_B_output, target_real_B)

        fake_B_output = D_B(fake_B_buffered.detach())
        target_fake_B = torch.zeros_like(fake_B_output).to(device) 
        loss_fake_B = criterion_GAN(fake_B_output, target_fake_B)

        # Combine the losses
        loss_D_B = (loss_real_B + loss_fake_B) / 2
        loss_D_B.backward()
        optimizer_D_B.step()

        #discriminator FFT A
        optimizer_D_A_FFT.zero_grad()

        real_A_output_fft = D_A_FFT(torch.abs(torch.fft.fftshift(torch.fft.fft2(real_A))))
        target_real = torch.ones_like(real_A_output_fft).to(device)
        loss_real = criterion_GAN(real_A_output_fft, target_real)

        fake_A_output_fft = D_A_FFT(torch.abs(torch.fft.fftshift(torch.fft.fft2(fake_A_buffered.detach()))))
        target_fake = torch.zeros_like(fake_A_output_fft).to(device)
        loss_fake = criterion_GAN(fake_A_output_fft, target_fake)

        loss_D_A_FFT = (loss_real + loss_fake) / 2
        loss_D_A_FFT.backward()
        optimizer_D_A_FFT.step()

        #discriminator FFT B
        optimizer_D_B_FFT.zero_grad()

        real_B_output_fft = D_B_FFT(torch.abs(torch.fft.fftshift(torch.fft.fft2(real_B))))
        target_real = torch.ones_like(real_B_output_fft).to(device)
        loss_real = criterion_GAN(real_B_output_fft, target_real)

        fake_B_output_fft = D_B_FFT(torch.abs(torch.fft.fftshift(torch.fft.fft2(fake_B_buffered.detach()))))
        target_fake = torch.zeros_like(fake_B_output_fft).to(device)
        loss_fake = criterion_GAN(fake_B_output_fft, target_fake)

        loss_D_B_FFT = (loss_real + loss_fake) / 2
        loss_D_B_FFT.backward()
        optimizer_D_B_FFT.step()

        batch = epoch * len(loader_A_train) + loader_A_train.n
        writer.add_scalar("loss_G_AB", loss_G_AB.item(), batch)
        writer.add_scalar("loss_G_BA", loss_G_BA.item(), batch)
        writer.add_scalar("loss_D_A", loss_D_A.item(), batch)
        writer.add_scalar("loss_D_B", loss_D_B.item(), batch)
        writer.add_scalar("loss_id_A", loss_id_A.item(), batch)
        writer.add_scalar("loss_id_B", loss_id_B.item(), batch)
        writer.add_scalar("loss_GAN_AB", loss_GAN_AB.item(), batch)
        writer.add_scalar("loss_GAN_BA", loss_GAN_BA.item(), batch)
        writer.add_scalar("loss_cycle_A", loss_cycle_A.item(), batch)
        writer.add_scalar("loss_cycle_B", loss_cycle_B.item(), batch)
        writer.add_scalar("loss_D_A_FFT", loss_D_A_FFT.item(), batch)
        writer.add_scalar("loss_D_B_FFT", loss_D_B_FFT.item(), batch)


        epoch_loss_G_AB += loss_G_AB.item()
        epoch_loss_G_BA += loss_G_BA.item()
        epoch_loss_D_A += loss_D_A.item()
        epoch_loss_D_B += loss_D_B.item()
        epoch_loss_D_A_FFT += loss_D_A_FFT.item()
        epoch_loss_D_B_FFT += loss_D_B_FFT.item()

        loss_dict = {
            "loss_G_AB": loss_G_AB.item(),
            "loss_G_BA": loss_G_BA.item(),
            "loss_D_A": loss_D_A.item(),
            "loss_D_B": loss_D_B.item(),
            "loss_id_A": loss_id_A.item(),
            "loss_id_B": loss_id_B.item(),
            "loss_GAN_AB": loss_GAN_AB.item(),
            "loss_GAN_BA": loss_GAN_BA.item(),
            "loss_cycle_A": loss_cycle_A.item(),
            "loss_cycle_B": loss_cycle_B.item(),
            "loss_D_A_FFT": loss_D_A_FFT.item(),
            "loss_D_B_FFT": loss_D_B_FFT.item()
        }

        with open(log_file_path, 'a') as log_file:
            log_file.write(str(loss_dict) + "\n")


    avg_loss_G_AB = epoch_loss_G_AB / len(loader_A_train)
    avg_loss_G_BA = epoch_loss_G_BA / len(loader_A_train)
    avg_loss_D_A = epoch_loss_D_A / len(loader_A_train)
    avg_loss_D_B = epoch_loss_D_B / len(loader_A_train)
    avg_loss_D_A_FFT = epoch_loss_D_A_FFT / len(loader_A_train)
    avg_loss_D_B_FFT = epoch_loss_D_B_FFT / len(loader_A_train)


    losses_G_AB.append(avg_loss_G_AB)
    losses_G_BA.append(avg_loss_G_BA)
    losses_D_A.append(avg_loss_D_A)
    losses_D_B.append(avg_loss_D_B)

    # Print average losses
    print(f"TRAINING: Generator AB Loss: {avg_loss_G_AB:.5f}, Generator BA Loss: {avg_loss_G_BA:.5f}, Discriminator A Loss: {avg_loss_D_A:.5f}, Discriminator B Loss: {avg_loss_D_B:.5f}, Discriminator A FFT Loss: {avg_loss_D_A_FFT:.5f}, Discriminator B FFT Loss: {avg_loss_D_B_FFT:.5f} ")

    # Calcualte validation losses
    G_AB.eval()
    G_BA.eval()
    D_A.eval()
    D_B.eval()
    D_A_FFT.eval()
    D_B_FFT.eval()

    val_loss_G_AB = 0
    val_loss_G_BA = 0
    val_loss_D_A = 0
    val_loss_D_B = 0
    val_loss_id_A = 0
    val_loss_id_B = 0
    val_loss_GAN_AB = 0
    val_loss_GAN_BA = 0
    val_loss_cycle_A = 0
    val_loss_cycle_B = 0
    val_loss_D_A_FFT = 0
    val_loss_D_B_FFT = 0

    with torch.no_grad():
        loader_A_val = tqdm(experimental_loader_val, total=len(experimental_loader_val))
        loader_B_val = tqdm(simulated_loader_val, total=len(simulated_loader_val))

        for real_A_data, real_B_data in zip(loader_A_val, loader_B_val):
            real_A = real_A_data[0]
            real_B = real_B_data[0]
            real_A_fft = real_A_data[1]
            real_B_fft = real_B_data[1]

            real_A = real_A.to(device)
            real_B = real_B.to(device)

            real_A_fft = real_A_fft.to(device)
            real_B_fft = real_B_fft.to(device)
            #add gaussian noise to real_B with mean 0 and std 0.001
            noise = torch.randn(real_B.size(), device=device) * 0.001
            real_B = real_B.clone() + noise
            real_A = real_A.clone() + noise

            real_A_fft = real_A_fft.clone() + noise
            real_B_fft = real_B_fft.clone() + noise

            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)

            # GAN loss
            fake_B = G_AB(real_A)

            discriminator_output = D_B(fake_B)
            discriminator_output_fft = D_B_FFT(torch.abs(torch.fft.fftshift(torch.fft.fft2(fake_B))))
            target_tensor = torch.ones_like(discriminator_output).to(device)  
            loss_GAN_AB = (criterion_GAN(discriminator_output, target_tensor) + criterion_GAN(discriminator_output_fft, target_tensor))/2

            fake_A = G_BA(real_B)

            discriminator_output = D_A(fake_A)
            discriminator_output_fft = D_A_FFT(torch.abs(torch.fft.fftshift(torch.fft.fft2(fake_A))))
            target_tensor = torch.ones_like(discriminator_output).to(device)
            loss_GAN_BA = (criterion_GAN(discriminator_output, target_tensor) + criterion_GAN(discriminator_output_fft, target_tensor))/2

            # Cycle loss
            recovered_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recovered_A, real_A)
            recovered_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recovered_B, real_B)

            # Total loss
            #print(loss_GAN_AB.item(), loss_GAN_BA.item(), loss_cycle_A.item(), loss_cycle_B.item(), loss_id_A.item(), loss_id_B.item(), lambda_cycle*(loss_cycle_A.item() + loss_cycle_B.item()), lambda_id*(loss_id_A.item() + loss_id_B.item()))
            loss_G_AB = loss_GAN_AB + lambda_cycle * loss_cycle_B + lambda_id * loss_id_B
            
            loss_G_BA = loss_GAN_BA + lambda_cycle * loss_cycle_A + lambda_id * loss_id_A

            # Discriminator A
            real_A_output = D_A(real_A)
            target_real = torch.ones_like(real_A_output).to(device)
            loss_real = criterion_GAN(real_A_output, target_real)

            fake_A_output = D_A(fake_A.detach())
            target_fake = torch.zeros_like(fake_A_output).to(device)
            loss_fake = criterion_GAN(fake_A_output, target_fake)

            loss_D_A = (loss_real + loss_fake) / 2

            # Discriminator B
            real_B_output = D_B(real_B)
            target_real_B = torch.ones_like(real_B_output).to(device)
            loss_real_B = criterion_GAN(real_B_output, target_real_B)

            fake_B_output = D_B(fake_B.detach())
            target_fake_B = torch.zeros_like(fake_B_output).to(device) 
            loss_fake_B = criterion_GAN(fake_B_output, target_fake_B)

            # Combine the losses
            loss_D_B = (loss_real_B + loss_fake_B) / 2

            #discriminator FFT A
            real_A_output_fft = D_A_FFT(torch.abs(torch.fft.fftshift(torch.fft.fft2(real_A))))
            target_real = torch.ones_like(real_A_output_fft).to(device)
            loss_real = criterion_GAN(real_A_output_fft, target_real)

            fake_A_output_fft = D_A_FFT(torch.abs(torch.fft.fftshift(torch.fft.fft2(fake_A.detach()))))
            target_fake = torch.zeros_like(fake_A_output_fft).to(device)
            loss_fake = criterion_GAN(fake_A_output_fft, target_fake)

            loss_D_A_FFT = (loss_real + loss_fake) / 2

            #discriminator FFT B
            real_B_output_fft = D_B_FFT(torch.abs(torch.fft.fftshift(torch.fft.fft2(real_B))))
            target_real = torch.ones_like(real_B_output_fft).to(device)
            loss_real = criterion_GAN(real_B_output_fft, target_real)

            fake_B_output_fft = D_B_FFT(torch.abs(torch.fft.fftshift(torch.fft.fft2(fake_B.detach()))))
            target_fake = torch.zeros_like(fake_B_output_fft).to(device)
            loss_fake = criterion_GAN(fake_B_output_fft, target_fake)

            loss_D_B_FFT = (loss_real + loss_fake) / 2

            val_loss_G_AB += loss_G_AB.item()
            val_loss_G_BA += loss_G_BA.item()
            val_loss_D_A += loss_D_A.item()
            val_loss_D_B += loss_D_B.item()
            val_loss_id_A += loss_id_A.item()
            val_loss_id_B += loss_id_B.item()
            val_loss_GAN_AB += loss_GAN_AB.item()
            val_loss_GAN_BA += loss_GAN_BA.item()
            val_loss_cycle_A += loss_cycle_A.item()
            val_loss_cycle_B += loss_cycle_B.item()
            val_loss_D_A_FFT += loss_D_A_FFT.item()
            val_loss_D_B_FFT += loss_D_B_FFT.item()

        avg_val_loss_G_AB = val_loss_G_AB / val_size_sim
        avg_val_loss_G_BA = val_loss_G_BA / val_size_sim
        avg_val_loss_D_A = val_loss_D_A / val_size_sim
        avg_val_loss_D_B = val_loss_D_B / val_size_sim
        avg_val_loss_D_A_FFT = val_loss_D_A_FFT / val_size_sim
        avg_val_loss_D_B_FFT = val_loss_D_B_FFT / val_size_sim
        
        print(f"VALIDATION: Generator AB Loss: {avg_val_loss_G_AB:.5f}, Generator BA Loss: {avg_val_loss_G_BA:.5f}, Discriminator A Loss: {avg_val_loss_D_A:.5f}, Discriminator B Loss: {avg_val_loss_D_B:.5f}, Discriminator A FFT Loss: {avg_val_loss_D_A_FFT:.5f}, Discriminator B FFT Loss: {avg_val_loss_D_B_FFT:.5f} ")

        iters = epoch * len(loader_A_train)
        writer.add_scalar("val_loss_G_AB", avg_val_loss_G_AB, iters)
        writer.add_scalar("val_loss_G_BA", avg_val_loss_G_BA, iters)
        writer.add_scalar("val_loss_D_A", avg_val_loss_D_A, iters)
        writer.add_scalar("val_loss_D_B", avg_val_loss_D_B, iters)
        writer.add_scalar("val_loss_D_A_FFT", avg_val_loss_D_A_FFT, iters)
        writer.add_scalar("val_loss_D_B_FFT", avg_val_loss_D_B_FFT, iters)
        writer.add_scalar("val_loss_id_A", val_loss_id_A / val_size_sim, iters)
        writer.add_scalar("val_loss_id_B", val_loss_id_B / val_size_sim, iters)
        writer.add_scalar("val_loss_GAN_AB", val_loss_GAN_AB / val_size_sim, iters)
        writer.add_scalar("val_loss_GAN_BA", val_loss_GAN_BA / val_size_sim, iters)
        writer.add_scalar("val_loss_cycle_A", val_loss_cycle_A / val_size_sim, iters)
        writer.add_scalar("val_loss_cycle_B", val_loss_cycle_B / val_size_sim, iters)


    # Visualizing Generated Images after each epoch
    with torch.no_grad():
        # Take a batch of images from loader_A or loader_B
        sample_A = fixed_sample_A[0]
        sample_B = fixed_sample_B[0]

        # Generate images
        fake_B = G_AB(sample_A)
        fake_A = G_BA(sample_B)

        recovered_A = G_BA(fake_B)
        recovered_B = G_AB(fake_A)

        identity_A = G_BA(sample_A)
        identity_B = G_AB(sample_B)

        # Move images to CPU for plotting and remove batch dimension
        sample_A, sample_B, fake_A, fake_B = sample_A.squeeze(0).cpu(), sample_B.squeeze(0).cpu(), fake_A.squeeze(0).cpu(), fake_B.squeeze(0).cpu()
        recovered_A, recovered_B = recovered_A.squeeze(0).cpu(), recovered_B.squeeze(0).cpu()
        identity_A, identity_B = identity_A.squeeze(0).cpu(), identity_B.squeeze(0).cpu()
        
        image_grid_ABA = torch.cat((sample_A, fake_B, recovered_A), 2)
        writer.add_image("image_grid_ABA", image_grid_ABA, epoch)
        image_grid_BAB = torch.cat((sample_B, fake_A, recovered_B), 2)
        writer.add_image("image_grid_BAB", image_grid_BAB, epoch)
        image_grid_identity_A = torch.cat((sample_A, identity_A), 2)
        writer.add_image("image_grid_identity_A", image_grid_identity_A, epoch)
        image_grid_identity_B = torch.cat((sample_B, identity_B), 2)
        writer.add_image("image_grid_identity_B", image_grid_identity_B, epoch)
        image_grid_G_AB_G_BA = torch.cat((sample_A, fake_B, sample_B, fake_A), 2)
        writer.add_image("image_grid_G_AB_G_BA", image_grid_G_AB_G_BA, epoch)



        if epoch % 1 == 0:
            # Plotting
            plt.figure(figsize=(10, 4))

            # Plot only the image part of the tensors (assuming it's the first channel)
            plt.subplot(1, 4, 1)
            plt.title("Real A")
            plt.imshow(sample_A[0], cmap='viridis')  # Grayscale image from the first channel

            plt.subplot(1, 4, 2)
            plt.title("Fake B")
            plt.imshow(fake_B[0], cmap='viridis')  # Grayscale image from the first channel

            plt.subplot(1, 4, 3)
            plt.title("Real B")
            plt.imshow(sample_B[0], cmap='viridis')  # Grayscale image from the first channel

            plt.subplot(1, 4, 4)
            plt.title("Fake A")
            plt.imshow(fake_A[0], cmap='viridis')  # Grayscale image from the first channel

            plt.show()

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



        if False:
            # cycle ABA
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 3, 1)
            plt.title("Real A")
            plt.imshow(sample_A[0], cmap='gray')
            plt.subplot(1, 3, 2)
            plt.title("Fake B")
            plt.imshow(fake_B[0], cmap='gray')
            plt.subplot(1, 3, 3)
            plt.title("Recovered A")
            plt.imshow(recovered_A[0], cmap='gray')
            plt.show()

            # cycle BAB
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 3, 1)
            plt.title("Real B")
            plt.imshow(sample_B[0], cmap='gray')
            plt.subplot(1, 3, 2)
            plt.title("Fake A")
            plt.imshow(fake_A[0], cmap='gray')
            plt.subplot(1, 3, 3)
            plt.title("Recovered B")
            plt.imshow(recovered_B[0], cmap='gray')
            plt.show()

            #identity A
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.title("Real A")
            plt.imshow(sample_A[0], cmap='gray')
            plt.subplot(1, 2, 2)
            plt.title("Identity A")
            plt.imshow(identity_A[0], cmap='gray')
            plt.show()

            #identity B
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1) 
            plt.title("Real B")
            plt.imshow(sample_B[0], cmap='gray')
            plt.subplot(1, 2, 2)
            plt.title("Identity B")
            plt.imshow(identity_B[0], cmap='gray')
            plt.show()
        

        #print(sample_A.shape, fake_B.shape, sample_B.shape, fake_A.shape)
        #assert all values of the tensor are between 0 and 1
        assert sample_A.min() >= 0 and sample_A.max() <= 1
        assert fake_B.min() >= 0 and fake_B.max() <= 1
        assert sample_B.min() >= 0 and sample_B.max() <= 1
        assert fake_A.min() >= 0 and fake_A.max() <= 1
        #assert the second channel of sample_A and fake_B are the same
        ssim_AB = ssim (sample_A[0].numpy(), fake_B[0].numpy(),data_range=1)
        ssim_BA = ssim (sample_B[0].numpy(), fake_A[0].numpy(),data_range=1)
        writer.add_scalar("ssim_AB", ssim_AB, epoch)
        writer.add_scalar("ssim_BA", ssim_BA, epoch)
        #print("ssim_AB, ssim_BA", ssim_AB, ssim_BA)


  # %%