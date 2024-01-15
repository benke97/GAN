#%%
import torch
import torch.nn as nn
from models import Generator, Discriminator
from prepare_data import SimulatedDataset, ExperimentalDataset, get_data_dicts_and_transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import random

random.seed(1337)

sim_data_dict, exp_data_dict, transform_sim, transform_exp = get_data_dicts_and_transforms("data/mrc", "data/experimental",apply_probe=True, only_110_za=True)

simulated_dataset = SimulatedDataset(sim_data_dict, transform=transform_sim)
simulated_loader = DataLoader(simulated_dataset, batch_size=8, shuffle=True)

experimental_dataset = ExperimentalDataset(exp_data_dict, transform=transform_exp)
experimental_loader = DataLoader(experimental_dataset, batch_size=8, shuffle=True)
#%%
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
criterion_GAN = nn.BCELoss()

# Cycle consistency loss
criterion_cycle = nn.L1Loss()

# Identity loss (optional, can help with training stability)
criterion_identity = nn.L1Loss()

optimizer_G_AB = torch.optim.Adam(G_AB.parameters(), lr=0.0001)
optimizer_G_BA = torch.optim.Adam(G_BA.parameters(), lr=0.0005)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=0.005)
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=0.0001)

loader_A = experimental_loader
loader_B = simulated_loader
lambda_cycle = 5
lambda_id = 1

losses_G_AB = []
losses_G_BA = []
losses_D_A = []
losses_D_B = []

log_file_path = "training_log.txt"

from skimage.metrics import structural_similarity as ssim
fixed_sample_A = next(iter(experimental_loader))[0].to(device)  # Fixed batch from domain A
fixed_sample_B = next(iter(simulated_loader))[0].to(device)     # Fixed batch from domain B

fixed_sample_A = fixed_sample_A.unsqueeze(0)
fixed_sample_B = fixed_sample_B.unsqueeze(0)
num_epochs = 100
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    epoch_loss_G_AB = 0
    epoch_loss_G_BA = 0
    epoch_loss_D_A = 0
    epoch_loss_D_B = 0

    loader_A = tqdm(experimental_loader, total=len(experimental_loader))
    loader_B = tqdm(simulated_loader, total=len(simulated_loader))
    for real_A, real_B in zip(loader_A, loader_B):

        real_A = real_A.to(device)
        real_B = real_B.to(device)
        # real_A and real_B are batches from the two domains

        # Generators G_AB and G_BA
        optimizer_G_AB.zero_grad()
        optimizer_G_BA.zero_grad()

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

        epoch_loss_G_AB += loss_G_AB.item()
        epoch_loss_G_BA += loss_G_BA.item()
        epoch_loss_D_A += loss_D_A.item()
        epoch_loss_D_B += loss_D_B.item()
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
            "loss_cycle_B": loss_cycle_B.item()
        }

        with open(log_file_path, 'a') as log_file:
            log_file.write(str(loss_dict) + "\n")


    avg_loss_G_AB = epoch_loss_G_AB / len(loader_A)
    avg_loss_G_BA = epoch_loss_G_BA / len(loader_A)
    avg_loss_D_A = epoch_loss_D_A / len(loader_A)
    avg_loss_D_B = epoch_loss_D_B / len(loader_A)

    losses_G_AB.append(avg_loss_G_AB)
    losses_G_BA.append(avg_loss_G_BA)
    losses_D_A.append(avg_loss_D_A)
    losses_D_B.append(avg_loss_D_B)

    # Print average losses
    print(f"Generator AB Loss: {avg_loss_G_AB:.4f}, Generator BA Loss: {avg_loss_G_BA:.4f}, Discriminator A Loss: {avg_loss_D_A:.4f}, Discriminator B Loss: {avg_loss_D_B:.4f}")

    # Visualizing Generated Images after each epoch
    with torch.no_grad():
        # Take a batch of images from loader_A or loader_B
        sample_A = fixed_sample_A
        sample_B = fixed_sample_B

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

        if epoch % 10 == 0:



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
        

        print(sample_A.shape, fake_B.shape, sample_B.shape, fake_A.shape)
        #assert all values of the tensor are between 0 and 1
        assert sample_A.min() >= 0 and sample_A.max() <= 1
        assert fake_B.min() >= 0 and fake_B.max() <= 1
        assert sample_B.min() >= 0 and sample_B.max() <= 1
        assert fake_A.min() >= 0 and fake_A.max() <= 1
        #assert the second channel of sample_A and fake_B are the same
        ssim_AB = ssim (sample_A[0].numpy(), fake_B[0].numpy(),data_range=1)
        ssim_BA = ssim (sample_B[0].numpy(), fake_A[0].numpy(),data_range=1)
        print("ssim_AB, ssim_BA", ssim_AB, ssim_BA)

# %%