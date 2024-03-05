import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from models import UNet128_3
# loads specified GAN model and send the validation set through it, randomly plot about 20 images

G_A2B = UNet128_3(1,1,ngf=16)
G_B2A = UNet128_3(1,1,ngf=16)
checkpoint = torch.load("exp_checkpoints/lambda_GP_0.03_cyc_20/checkpoint_epoch_150.pth")
G_A2B.load_state_dict(checkpoint["G_AB_state_dict"])
G_B2A.load_state_dict(checkpoint["G_BA_state_dict"])
G_A2B.eval()
G_B2A.eval()

val_exp_dir = "data/noiser/train/exp"
val_clean_dir = "data/noiser/train/clean"
val_exp = os.listdir(val_exp_dir)
val_clean = os.listdir(val_clean_dir)

for i in range(1):
    idx = np.random.randint(len(val_exp))
    exp = np.load(os.path.join(val_exp_dir, val_exp[idx]))["arr_0"]
    clean = np.load(os.path.join(val_clean_dir, val_clean[idx]))["arr_0"]
    exp = torch.from_numpy(exp).float().unsqueeze(0).unsqueeze(0)
    clean = torch.from_numpy(clean).float().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        fake_exp = G_A2B(clean)
        fake_clean = G_B2A(exp)
    fig, ax = plt.subplots(2,2)
    ax[0,0].imshow(clean[0,0].numpy(),cmap="gray")
    ax[0,0].set_title("clean")
    ax[0,0].axis("off")
    ax[0,1].imshow(fake_exp[0,0].numpy(),cmap="gray")
    ax[0,1].set_title("fake_exp")
    ax[0,1].axis("off")
    ax[1,0].imshow(exp[0,0].numpy(),cmap="gray")
    ax[1,0].set_title("exp")
    ax[1,0].axis("off")
    ax[1,1].imshow(fake_clean[0,0].numpy(),cmap="gray")
    ax[1,1].set_title("fake_clean")
    ax[1,1].axis("off")
    plt.show()
    plt.close()

