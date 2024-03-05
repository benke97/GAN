import torch
import torch.nn as nn
from models import Generator, UNet128, ResnetGenerator, PatchGANDiscriminator, UNet128_5, UNet128_4, UNet128_3, SNPatchGANDiscriminator
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from helpers import plot_batch, calculate_global_fft_min_max
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from image_pool import ImagePool
from pretrain_dataset import Pretrain_dataset
import torchvision.transforms
import torch.nn.functional as F
from torch.autograd import grad
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance


random.seed(1337)

def save_checkpoint(epoch, G_AB, G_BA, D_A, D_B, optimizer_G, optimizer_D_A, optimizer_D_B, loss, checkpoint_dir="checkpoints"):
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth"
    torch.save({
        'epoch': epoch,
        'G_AB_state_dict': G_AB.state_dict(),
        'G_BA_state_dict': G_BA.state_dict(),
        'D_A_state_dict': D_A.state_dict(),
        'D_B_state_dict': D_B.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_A_state_dict': optimizer_D_A.state_dict(),
        'optimizer_D_B_state_dict': optimizer_D_B.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty """
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(real_samples.device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.shape, requires_grad=False).to(real_samples.device)
    # Get gradient w.r.t. interpolates
    gradients = grad(outputs=d_interpolates, inputs=interpolates,
                     grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

checkpoint_freq = 25
BATCH_SIZE = 1
data_dir_train = "data/noiser/train/"
data_dir_val = "data/noiser/val/"
domain_names = ["clean","noisy"]
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_dataset = Pretrain_dataset(data_dir_train, domain_names, transform)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = Pretrain_dataset(data_dir_val, domain_names, transform)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

def adjust_learning_rate_generator(optimizer, epoch, initial_lr, num_epochs):
    """ Adjusts learning rate each epoch: constant for first half, linear decay in second half. """
    if epoch < num_epochs / 2:
        lr = initial_lr
    else:
        lr = initial_lr * float(num_epochs - epoch) / (num_epochs / 2)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_discriminator(optimizer, epoch, initial_lr, num_epochs):
    """ Adjusts learning rate each epoch: constant for first half, linear decay in second half. """
    if epoch < num_epochs / 2:
        lr = initial_lr
    else:
        lr = initial_lr * float(num_epochs - epoch) / (num_epochs / 2)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def make_3_channel_tensor(tensor):
    return torch.cat((tensor, tensor, tensor), 1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

norm = "instance"
#G_AB = Generator().to(device)
#G_BA = Generator().to(device)

G_AB = UNet128_4(1,1,ngf=32).to(device)
G_BA = UNet128_4(1,1,ngf=32).to(device)
D_A = PatchGANDiscriminator(norm).to(device)
D_B = PatchGANDiscriminator(norm).to(device)

criterion_GAN = nn.BCELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()
criterion_reconstruction = nn.MSELoss()

initial_lr_G = 0.00005
initial_lr_D_A = 0.000075
initial_lr_D_B = 0.000075
optimizer_G = torch.optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()), lr=initial_lr_G, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=initial_lr_D_A, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=initial_lr_D_B, betas=(0.5, 0.999))

lambda_cycle = 5
lambda_id = 0.5 * lambda_cycle
lambda_gp = 0.0001
lambda_reconstruction = 0

num_epochs = 300
buffer_D_B = ImagePool(20)
buffer_D_A = ImagePool(20)

#add_noise = lambda x, noise_std: torch.clamp(x + torch.randn_like(x).to(x.device) * noise_std,0,1)

lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze', normalize=True).to(device)

fixed_sample = next(iter(train_dataloader))
fixed_sample_2 = next(iter(train_dataloader))

writer = SummaryWriter()
batch_id = 0

best_translation_epoch = 0
best_translation_loss = 1000000




for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    adjust_learning_rate_generator(optimizer_G, epoch, initial_lr_G, num_epochs)
    adjust_learning_rate_discriminator(optimizer_D_A, epoch, initial_lr_D_A, num_epochs)
    adjust_learning_rate_discriminator(optimizer_D_B, epoch, initial_lr_D_B, num_epochs)

    G_AB.train()
    G_BA.train()
    D_A.train()
    D_B.train()
    train_loss_G = 0
    train_loss_D_A = 0
    train_loss_D_B = 0

    clean_L1 = 0
    noisy_lpips = 0

    loader_train = tqdm(train_dataloader, total=len(train_dataloader))
    
    #Training
    for real_A, real_B in loader_train:
        #Clean, Noisy
        optimizer_G.zero_grad()

        real_A = real_A.to(device)
        real_B = real_B.to(device)

        fake_B = G_AB(real_A)
    
        discriminator_output = D_B(fake_B)
        target_tensor = torch.ones_like(discriminator_output).to(device)  
        loss_GAN_AB = criterion_GAN(discriminator_output, target_tensor)

        fake_A = G_BA(real_B)

        discriminator_output = D_A(fake_A)
        target_tensor = torch.ones_like(discriminator_output).to(device)
        loss_GAN_BA = criterion_GAN(discriminator_output, target_tensor)

        recovered_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recovered_A, real_A)
        recovered_B = G_AB(fake_A)
        loss_cycle_B = lpips(make_3_channel_tensor(recovered_B), make_3_channel_tensor(real_B))

        loss_id_A = criterion_identity(G_BA(real_A), real_A)
        loss_id_B = lpips(make_3_channel_tensor(G_AB(real_B)), make_3_channel_tensor(real_B))
        
        clean_L1 += criterion_identity(real_A.clone(), fake_A.clone()).item()
        noisy_lpips += lpips(make_3_channel_tensor(real_B.clone()), make_3_channel_tensor(fake_B.clone())).item()


        loss_reconstruction_A = criterion_reconstruction(fake_A, real_A)
        loss_reconstruction_B = criterion_reconstruction(fake_B, real_B)
        #print("reconstruction loss: ", lambda_reconstruction * (loss_reconstruction_A + loss_reconstruction_B))
        loss_G = loss_GAN_AB + loss_GAN_BA + lambda_cycle * (loss_cycle_B + loss_cycle_A) + lambda_id * (loss_id_A + loss_id_B) + lambda_reconstruction *  loss_reconstruction_A

        loss_G.backward()
        optimizer_G.step()

        optimizer_D_A.zero_grad()

        
        real_A_preds = D_A(real_A)
        target_real = torch.ones_like(real_A_preds).to(device)
        #print(real_A_preds, target_real)
        loss_real = criterion_GAN(real_A_preds, target_real)

        fake_A_buffered = buffer_D_A.query(fake_A).to(device)
        fake_A_preds = D_A(fake_A.detach())
        target_fake = torch.zeros_like(fake_A_preds).to(device)
        loss_fake = criterion_GAN(fake_A_preds, target_fake)

        gradient_penalty = compute_gradient_penalty(D_A, real_A.data, fake_A.data)
        #print("Adversarial loss A: ", (loss_real + loss_fake) / 2)
        #print("gradient_penalty_A",lambda_gp * gradient_penalty)
        loss_D_A = (loss_real + loss_fake) / 2 + lambda_gp * gradient_penalty
        loss_D_A.backward()  
        optimizer_D_A.step()

        optimizer_D_B.zero_grad()

        

        real_B_preds = D_B(real_B)
        target_real = torch.ones_like(real_B_preds).to(device)
        loss_real = criterion_GAN(real_B_preds, target_real)

        fake_B_buffered = buffer_D_B.query(fake_B).to(device)
        fake_B_preds = D_B(fake_B_buffered.detach())
        target_fake = torch.zeros_like(fake_B_preds).to(device)
        loss_fake = criterion_GAN(fake_B_preds, target_fake)

        gradient_penalty = compute_gradient_penalty(D_B, real_B.data, fake_B.data)
        #print("Adversarial loss B: ", (loss_real + loss_fake) / 2)
        #print("gradient_penalty_B",lambda_gp * gradient_penalty)
        loss_D_B = (loss_real + loss_fake) / 2 + lambda_gp * gradient_penalty
    
        loss_D_B.backward()
        optimizer_D_B.step()

        train_loss_G += loss_G.item()
        train_loss_D_A += loss_D_A.item()
        train_loss_D_B += loss_D_B.item()

        #turn real_A into a 3 channel image
        #real_A_3 = torch.cat((real_A, real_A, real_A), 1)
        #fake_B_3 = torch.cat((fake_B, fake_B, fake_B), 1)
        #real_B_3 = torch.cat((real_B, real_B, real_B), 1)
        #fake_A_3 = torch.cat((fake_A, fake_A, fake_A), 1)

        #print("LPIPS real_A-real_A: ", lpips(real_A_3, real_A_3))
        #print("LPIPS real_A-fake_A: ", lpips(real_A_3, fake_A_3))
        writer.add_scalar('batch_loss/cycle_clean', loss_cycle_A.item(), batch_id)
        writer.add_scalar('batch_loss/cycle_noisy', loss_cycle_B.item(), batch_id)
        writer.add_scalar('batch_loss/id_clean', loss_id_A.item(), batch_id)
        writer.add_scalar('batch_loss/id_noisy', loss_id_B.item(), batch_id)
        writer.add_scalar('batch_loss/GAN_AB', loss_GAN_AB.item(), batch_id)
        writer.add_scalar('batch_loss/GAN_BA', loss_GAN_BA.item(), batch_id)
        writer.add_scalar('batch_loss/loss_D_A', loss_D_A.item(), batch_id)
        writer.add_scalar('batch_loss/loss_D_B', loss_D_B.item(), batch_id)
        batch_id += 1
    
    translation_score = (clean_L1 / len(train_dataloader)) + (noisy_lpips / len(train_dataloader))
    if translation_score < best_translation_loss:
        print(f"Translation score: {translation_score}")
        best_translation_loss = translation_score
        best_translation_epoch = epoch
        save_checkpoint(epoch, G_AB, G_BA, D_A, D_B, optimizer_G, optimizer_D_A, optimizer_D_B, [train_loss_G/len(train_dataloader), train_loss_D_A/len(train_dataloader), train_loss_D_B/len(train_dataloader), 0, 0, 0], "best_checkpoint")
    writer.add_scalar('Loss/translation_score', translation_score, epoch)

    writer.add_scalar('Loss/train_G', train_loss_G/len(train_dataloader), epoch)
    writer.add_scalar('Loss/train_D_A', train_loss_D_A/len(train_dataloader), epoch)
    writer.add_scalar('Loss/train_D_B', train_loss_D_B/len(train_dataloader), epoch)
    writer.add_scalar('Learning rate/G', optimizer_G.param_groups[0]['lr'], epoch)

    print(f"Loss G: {train_loss_G/len(train_dataloader)}, Loss D_A: {train_loss_D_A/len(train_dataloader)}, Loss D_B: {train_loss_D_B/len(train_dataloader)}")

    G_AB.eval()
    G_BA.eval()
    D_A.eval()
    D_B.eval()
    val_loss_G = 0
    val_loss_D_A = 0
    val_loss_D_B = 0

    loader_val = tqdm(val_dataloader, total=len(val_dataloader))
    
    #Validation
    with torch.no_grad():
        for real_A, real_B in loader_val:
            #Clean, Noisy
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            fake_B = G_AB(real_A)
        
            discriminator_output = D_B(fake_B)
            target_tensor = torch.ones_like(discriminator_output).to(device)  
            loss_GAN_AB = criterion_GAN(discriminator_output, target_tensor)

            fake_A = G_BA(real_B)

            discriminator_output = D_A(fake_A)
            target_tensor = torch.ones_like(discriminator_output).to(device)

            loss_GAN_BA = criterion_GAN(discriminator_output, target_tensor)

            recovered_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recovered_A, real_A)
            recovered_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recovered_B, real_B)

            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)
            
            loss_reconstruction_A = criterion_reconstruction(fake_A, real_A)
            loss_reconstruction_B = criterion_reconstruction(fake_B, real_B)

            loss_G = loss_GAN_AB + loss_GAN_BA + lambda_cycle * (loss_cycle_B + loss_cycle_A) + lambda_id * (loss_id_A + loss_id_B) + lambda_reconstruction * loss_reconstruction_A

            val_loss_G += loss_G.item()

            real_A_preds = D_A(real_A)
            target_real = torch.ones_like(real_A_preds).to(device)
            loss_real = criterion_GAN(real_A_preds, target_real)

            fake_A_preds = D_A(fake_A)
            target_fake = torch.zeros_like(fake_A_preds).to(device)
            loss_fake = criterion_GAN(fake_A_preds, target_fake)

            loss_D_A = (loss_real + loss_fake) / 2
            val_loss_D_A += loss_D_A.item()

            real_B_preds = D_B(real_B)
            target_real = torch.ones_like(real_B_preds).to(device)
            loss_real = criterion_GAN(real_B_preds, target_real)

            fake_B_preds = D_B(fake_B)
            target_fake = torch.zeros_like(fake_B_preds).to(device)
            loss_fake = criterion_GAN(fake_B_preds, target_fake)

            loss_D_B = (loss_real + loss_fake) / 2
            val_loss_D_B += loss_D_B.item()

    writer.add_scalar('Loss/val_G', val_loss_G/len(val_dataloader), epoch)
    writer.add_scalar('Loss/val_D_A', val_loss_D_A/len(val_dataloader), epoch)
    writer.add_scalar('Loss/val_D_B', val_loss_D_B/len(val_dataloader), epoch)

    print(f"Val Loss G: {val_loss_G/len(val_dataloader)}, Val Loss D_A: {val_loss_D_A/len(val_dataloader)}, Val Loss D_B: {val_loss_D_B/len(val_dataloader)}")

    #Visualization
    with torch.no_grad():
        fixed_sample_A = fixed_sample[0].to(device)
        fixed_sample_B = fixed_sample[1].to(device)
        fixed_fake_B = G_AB(fixed_sample_A)
        fixed_fake_A = G_BA(fixed_sample_B)
        fixed_recovered_A = G_BA(fixed_fake_B)
        fixed_recovered_B = G_AB(fixed_fake_A)
        
        def resize_img(img):
            return F.interpolate(img, size=(512, 512), mode='nearest')

        fixed_sample_A_resized = resize_img(fixed_sample_A)
        fixed_fake_B_resized = resize_img(fixed_fake_B)
        fixed_recovered_A_resized = resize_img(fixed_recovered_A)
        fixed_sample_B_resized = resize_img(fixed_sample_B)
        fixed_fake_A_resized = resize_img(fixed_fake_A)
        fixed_recovered_B_resized = resize_img(fixed_recovered_B)

        # Concatenate images for cycle visualization
        fixed_cycle_ABA = torch.cat([fixed_sample_A_resized, fixed_fake_B_resized, fixed_recovered_A_resized], dim=3)
        fixed_cycle_BAB = torch.cat([fixed_sample_B_resized, fixed_fake_A_resized, fixed_recovered_B_resized], dim=3)

        # Log images
        writer.add_image('ABA_cycle', fixed_cycle_ABA.squeeze(0), epoch)
        writer.add_image('BAB_cycle', fixed_cycle_BAB.squeeze(0), epoch)

        fixed_sample_2_A = fixed_sample_2[0].to(device)
        fixed_sample_2_B = fixed_sample_2[1].to(device)
        fixed_fake_B_2 = G_AB(fixed_sample_2_A)
        fixed_fake_A_2 = G_BA(fixed_sample_2_B)
        fixed_recovered_A_2 = G_BA(fixed_fake_B_2)
        fixed_recovered_B_2 = G_AB(fixed_fake_A_2)

        fixed_sample_2_A_resized = resize_img(fixed_sample_2_A)
        fixed_fake_B_2_resized = resize_img(fixed_fake_B_2)
        fixed_recovered_A_2_resized = resize_img(fixed_recovered_A_2)
        fixed_sample_2_B_resized = resize_img(fixed_sample_2_B)
        fixed_fake_A_2_resized = resize_img(fixed_fake_A_2)
        fixed_recovered_B_2_resized = resize_img(fixed_recovered_B_2)

        # Concatenate images for cycle visualization
        fixed_cycle_ABA_2 = torch.cat([fixed_sample_2_A_resized, fixed_fake_B_2_resized, fixed_recovered_A_2_resized], dim=3)
        fixed_cycle_BAB_2 = torch.cat([fixed_sample_2_B_resized, fixed_fake_A_2_resized, fixed_recovered_B_2_resized], dim=3)

        # Log images
        writer.add_image('ABA_cycle_2', fixed_cycle_ABA_2.squeeze(0), epoch)
        writer.add_image('BAB_cycle_2', fixed_cycle_BAB_2.squeeze(0), epoch)
    
    #Save checkpoint
    if epoch % checkpoint_freq == 0:
        save_loss = [train_loss_G/len(train_dataloader), train_loss_D_A/len(train_dataloader), train_loss_D_B/len(train_dataloader), val_loss_G/len(val_dataloader), val_loss_D_A/len(val_dataloader), val_loss_D_B/len(val_dataloader)]
        save_checkpoint(epoch, G_AB, G_BA, D_A, D_B, optimizer_G, optimizer_D_A, optimizer_D_B, save_loss)