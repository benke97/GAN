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
import time
from torchvision.transforms.functional import to_pil_image, to_tensor

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

def prepare_images_for_fid(image_list):
    processed_images = []
    for img_tensor in image_list:
        # Convert each tensor to PIL Image, then back to tensor to ensure correct format
        # This also rescales images to [0, 255] and converts them to uint8
        for img in img_tensor:
            pil_img = to_pil_image(img.cpu())  # Convert to PIL Image to rescale correctly
            processed_img = to_tensor(pil_img).to(device) * 255  # Back to tensor and scale
            processed_images.append(processed_img)
    # Stack all images into a single tensor
    return torch.stack(processed_images).type(torch.uint8)

checkpoint_freq = 1
BATCH_SIZE = 1
data_dir_train = "data/noiser/train/"
data_dir_val = "data/noiser/val/"
domain_names = ["clean","exp"]
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

#G_AB = UNet128_4(1,1,ngf=32).to(device)
#G_BA = UNet128_4(1,1,ngf=32).to(device)
G_AB = UNet128_3(1,1,ngf=16).to(device)
G_BA = UNet128_3(1,1,ngf=16).to(device)
D_A = PatchGANDiscriminator(norm).to(device)
D_B = PatchGANDiscriminator(norm).to(device)

checkpoint = torch.load("best_checkpoint/backup_good/checkpoint_epoch_268.pth")
#checkpoint = torch.load("best_checkpoint/UNet4_ngf32_weights/checkpoint_epoch_178.pth")
G_AB.load_state_dict(checkpoint["G_AB_state_dict"])
G_BA.load_state_dict(checkpoint["G_BA_state_dict"])
#D_A.load_state_dict(checkpoint["D_A_state_dict"])
#D_B.load_state_dict(checkpoint["D_B_state_dict"])


criterion_GAN = nn.BCELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

initial_lr_G = 0.00001
initial_lr_D_A = 0.00001
initial_lr_D_B = 0.00001
optimizer_G = torch.optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()), lr=initial_lr_G, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=initial_lr_D_A, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=initial_lr_D_B, betas=(0.5, 0.999))

lambda_cycle = 20
lambda_id = 0.5 * lambda_cycle
lambda_gp = 0.003

num_epochs = 300
buffer_D_B = ImagePool(5)
buffer_D_A = ImagePool(5)

#add_noise = lambda x, noise_std: torch.clamp(x + torch.randn_like(x).to(x.device) * noise_std,0,1)

lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze', normalize=True).to(device)

fixed_sample = next(iter(val_dataloader))
fixed_sample_2 = next(iter(val_dataloader))

best_fid = 1000

writer = SummaryWriter()
batch_id = 0

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
        
        #assert real_B has values between 0 and 1
        #print(torch.max(real_B), torch.min(real_B))
        assert torch.max(real_B) <= 1
        assert torch.min(real_B) >= 0

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

        loss_G = loss_GAN_AB + loss_GAN_BA + lambda_cycle * (loss_cycle_B + loss_cycle_A) + lambda_id * (loss_id_A + loss_id_B)

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

        writer.add_scalar('batch_loss/cycle_clean', loss_cycle_A.item(), batch_id)
        writer.add_scalar('batch_loss/cycle_noisy', loss_cycle_B.item(), batch_id)
        writer.add_scalar('batch_loss/id_clean', loss_id_A.item(), batch_id)
        writer.add_scalar('batch_loss/id_noisy', loss_id_B.item(), batch_id)
        writer.add_scalar('batch_loss/GAN_AB', loss_GAN_AB.item(), batch_id)
        writer.add_scalar('batch_loss/GAN_BA', loss_GAN_BA.item(), batch_id)
        writer.add_scalar('batch_loss/loss_D_A', loss_D_A.item(), batch_id)
        writer.add_scalar('batch_loss/loss_D_B', loss_D_B.item(), batch_id)
        batch_id += 1

    writer.add_scalar('Loss/train_G', train_loss_G/len(train_dataloader), epoch)
    writer.add_scalar('Loss/train_D_A', train_loss_D_A/len(train_dataloader), epoch)
    writer.add_scalar('Loss/train_D_B', train_loss_D_B/len(train_dataloader), epoch)
    writer.add_scalar('Learning rate/G', optimizer_G.param_groups[0]['lr'], epoch)
    writer.add_scalar('Learning rate/D_A', optimizer_D_A.param_groups[0]['lr'], epoch)
    writer.add_scalar('Learning rate/D_B', optimizer_D_B.param_groups[0]['lr'], epoch)

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
    real_A_list = []
    real_B_list = []
    fake_A_list = []
    fake_B_list = []
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

            loss_G = loss_GAN_AB + loss_GAN_BA + lambda_cycle * (loss_cycle_B + loss_cycle_A) + lambda_id * (loss_id_A + loss_id_B)

            val_loss_G += loss_G.item()

            real_A_preds = D_A(real_A)
            target_real = torch.ones_like(real_A_preds).to(device)
            loss_real = criterion_GAN(real_A_preds, target_real)

            fake_A_preds = D_A(fake_A)
            target_fake = torch.zeros_like(fake_A_preds).to(device)
            loss_fake = criterion_GAN(fake_A_preds, target_fake)

            #gradient_penalty = compute_gradient_penalty(D_A, real_A.data, fake_A.data)
            loss_D_A = (loss_real + loss_fake) / 2 #+ lambda_gp * gradient_penalty
            val_loss_D_A += loss_D_A.item()

            real_B_preds = D_B(real_B)
            target_real = torch.ones_like(real_B_preds).to(device)
            loss_real = criterion_GAN(real_B_preds, target_real)

            fake_B_preds = D_B(fake_B)
            target_fake = torch.zeros_like(fake_B_preds).to(device)
            loss_fake = criterion_GAN(fake_B_preds, target_fake)

            #gradient_penalty = compute_gradient_penalty(D_B, real_B.data, fake_B.data)
            loss_D_B = (loss_real + loss_fake) / 2 #+ lambda_gp * gradient_penalty
            val_loss_D_B += loss_D_B.item()

            real_A_list.append(torch.squeeze(make_3_channel_tensor(real_A),0))
            real_B_list.append(torch.squeeze(make_3_channel_tensor(real_B),0))

            fake_A_list.append(torch.squeeze(make_3_channel_tensor(fake_A),0))
            fake_B_list.append(torch.squeeze(make_3_channel_tensor(fake_B),0))

    writer.add_scalar('Loss/val_G', val_loss_G/len(val_dataloader), epoch)
    writer.add_scalar('Loss/val_D_A', val_loss_D_A/len(val_dataloader), epoch)
    writer.add_scalar('Loss/val_D_B', val_loss_D_B/len(val_dataloader), epoch)

    print(f"Val Loss G: {val_loss_G/len(val_dataloader)}, Val Loss D_A: {val_loss_D_A/len(val_dataloader)}, Val Loss D_B: {val_loss_D_B/len(val_dataloader)}")

    #Visualization
    with torch.no_grad():
        fixed_sample_A = fixed_sample[0][0].unsqueeze(0).to(device)
        fixed_sample_B = fixed_sample[1][0].unsqueeze(0).to(device)
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

        fixed_sample_2_A = fixed_sample_2[0][0].unsqueeze(0).to(device)
        fixed_sample_2_B = fixed_sample_2[1][0].unsqueeze(0).to(device)
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
    

    #FID calculation
    fid_A = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    #print(torch.stack(real_A_list, dim=0).shape,real_A_list[0].shape, len(real_A_list))
    fid_A.update(torch.stack(real_A_list, dim=0), real=True)
    fid_A.update(torch.stack(fake_A_list, dim=0), real=False)
    fid_A_score = fid_A.compute().item()
    writer.add_scalar('FID/FID_A', fid_A_score, epoch)
    fid_A.reset()

    fid_B = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    fid_B.update(torch.stack(real_B_list, dim=0), real=True)
    fid_B.update(torch.stack(fake_B_list, dim=0), real=False)
    fid_B_score = fid_B.compute().item()
    writer.add_scalar('FID/FID_B', fid_B_score, epoch)
    fid_B.reset()

    tot_fid = (fid_A_score + fid_B_score) / 2
    writer.add_scalar('FID/FID', tot_fid, epoch)

    if tot_fid < best_fid:
        best_fid = tot_fid
        save_checkpoint(epoch, G_AB, G_BA, D_A, D_B, optimizer_G, optimizer_D_A, optimizer_D_B, [train_loss_G/len(train_dataloader), train_loss_D_A/len(train_dataloader), train_loss_D_B/len(train_dataloader), val_loss_G/len(val_dataloader), val_loss_D_A/len(val_dataloader), val_loss_D_B/len(val_dataloader)],checkpoint_dir="best_fid_checkpoint")


    #Save checkpoint
    if (epoch+1) % checkpoint_freq == 0:
        save_loss = [train_loss_G/len(train_dataloader), train_loss_D_A/len(train_dataloader), train_loss_D_B/len(train_dataloader), val_loss_G/len(val_dataloader), val_loss_D_A/len(val_dataloader), val_loss_D_B/len(val_dataloader)]
        save_checkpoint(epoch, G_AB, G_BA, D_A, D_B, optimizer_G, optimizer_D_A, optimizer_D_B, save_loss,checkpoint_dir="exp_checkpoints")