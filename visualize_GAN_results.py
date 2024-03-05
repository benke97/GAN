import os
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import pickle as pkl
import mayavi.mlab as mlab
import mrcfile
import tifffile
from models import UNet128_3, PatchGANDiscriminator
from ObjectGenerator import ObjectGenerator

"""This file plots some stuff related to the GAN project"""

if __name__ == "__main__":

    datadir = "data/noiser/"
    experimental_paths = ...
    
    sim_raw_paths = os.listdir(os.path.join(datadir, "raw_sim"))
    sim_raw_paths = [os.path.join(datadir, "raw_sim", path) for path in sim_raw_paths]
    
    sim_paths = os.listdir(os.path.join(datadir, "sim"))
    sim_paths = [os.path.join(datadir, "sim", path) for path in sim_paths]

    sim_noisy_paths = os.listdir(os.path.join(datadir, "sim_noisy"))
    sim_noisy_paths = [os.path.join(datadir, "sim_noisy", path) for path in sim_noisy_paths]

    assert len(sim_raw_paths) == len(sim_paths) == len(sim_noisy_paths)

    idx = random.randint(0, len(sim_raw_paths))
    raw_sim = os.path.join(datadir, "raw_sim",f"{idx}_HAADF.mrc")
    sim = os.path.join(datadir, "sim",f"{idx}.npz")
    sim_noisy = os.path.join(datadir, "sim_noisy",f"{idx}.npz")
    dataframe = os.path.join(datadir, "dataframes",f"structure_{idx}.pkl")
    
    with open(dataframe, "rb") as f:
        structure_df = pkl.load(f)
    ob_gen = ObjectGenerator()
    
    rotation_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    rotation_matrix_2 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    visu = structure_df.copy()
    visu[['x', 'y', 'z']] = visu[['x', 'y', 'z']]
    visu[["x","y","z"]] = visu[["x","y","z"]] * 10
    visu.label = visu.label.replace({'Ce': 0, 'O': 1, 'Pt': 2})
    #ob_gen.mayavi_atomic_structure(visu)

    with mrcfile.open(raw_sim, permissive=True) as mrc:
        raw_sim = mrc.data
    sim = np.load(sim)["arr_0"]
    sim_noisy = np.load(sim_noisy)["arr_0"]

    fig, ax = plt.subplots(1, 3, figsize=(9, 3))  # Adjusted figsize: Width, Height
    ax[0].imshow(raw_sim, cmap="gray",origin="lower")
    ax[0].set_title("Raw multislice sim")
    ax[0].axis('off')  # Remove axes for cleaner look

    ax[1].imshow(sim, cmap="gray",origin="lower")
    ax[1].set_title("Gaussian probe")
    ax[1].axis('off')  # Remove axes for cleaner look

    ax[2].imshow(sim_noisy, cmap="gray",origin="lower")
    ax[2].set_title("Manual noise")
    ax[2].axis('off')  # Remove axes for cleaner look

    plt.tight_layout()  # Adjust layout to minimize white space
    #plt.show()

    #sample 100 random images from sim and noisy_sim and plot in a 10x10 grid. Tight layout and make fig size big
    sim = []
    noisy_sim = []
    for i in range(100):
        idx = random.randint(0, len(sim_paths))
        sim.append(np.load(sim_paths[idx])["arr_0"])
        noisy_sim.append(np.load(sim_noisy_paths[idx])["arr_0"])

    fig, ax = plt.subplots(10, 10, figsize=(14, 20))  # Adjusted for better fit, can be resized based on actual output
    for i in range(10):
        for j in range(10):
            ax[i, j].imshow(sim[i * 10 + j], cmap="gray")
            ax[i, j].axis("off")  # Remove axes for cleaner look
    plt.tight_layout()  # Adjust layout
    #plt.show()

    fig, ax = plt.subplots(10, 10, figsize=(14, 20))  # Same adjustment as above
    for i in range(10):
        for j in range(10):
            ax[i, j].imshow(noisy_sim[i * 10 + j], cmap="gray")
            ax[i, j].axis("off")  # Remove axes for cleaner look
    plt.tight_layout()  # Adjust layout
    #plt.show()


    best_checkpoint_path = "best_checkpoint/backup_good/checkpoint_epoch_268.pth"
    G_AB = UNet128_3(1, 1, ngf=16)
    G_AB.load_state_dict(torch.load(best_checkpoint_path)["G_AB_state_dict"])
    G_AB.eval()
    G_BA = UNet128_3(1, 1, ngf=16)
    G_BA.load_state_dict(torch.load(best_checkpoint_path)["G_BA_state_dict"])
    G_BA.eval()
    
    
    sample_A = torch.from_numpy(sim[0]).unsqueeze(0).unsqueeze(0).float()
    sample_B = torch.from_numpy(noisy_sim[0]).unsqueeze(0).unsqueeze(0).float()
    print(sample_A.shape)
    print(sample_B.shape)
    fake_B = G_AB(sample_A)
    recovered_A = G_BA(fake_B)

    fake_A = G_BA(sample_B)
    recovered_B = G_AB(fake_A)

    sample_A_np = sample_A.cpu().numpy()
    fake_B_np = fake_B.cpu().detach().numpy()
    recovered_A_np = recovered_A.cpu().detach().numpy()
    
    sample_B_np = sample_B.cpu().numpy()
    fake_A_np = fake_A.cpu().detach().numpy()
    recovered_B_np = recovered_B.cpu().detach().numpy()

    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs[0, 0].imshow(sample_A_np[0, 0], cmap='gray')
    axs[0, 0].set_title('Original A')
    axs[0, 1].imshow(fake_B_np[0, 0], cmap='gray')
    axs[0, 1].set_title('Fake B')
    axs[0, 2].imshow(recovered_A_np[0, 0], cmap='gray')
    axs[0, 2].set_title('Recovered A')

    axs[1, 0].imshow(sample_B_np[0, 0], cmap='gray')
    axs[1, 0].set_title('Original B')
    axs[1, 1].imshow(fake_A_np[0, 0], cmap='gray')
    axs[1, 1].set_title('Fake A')
    axs[1, 2].imshow(recovered_B_np[0, 0], cmap='gray')
    axs[1, 2].set_title('Recovered B')

    for ax in axs.flat:
        ax.axis('off')

    #plt.show()

    idx = 2791
    raw_sim = os.path.join(datadir, "raw_sim",f"{idx}_HAADF.mrc")
    sim = os.path.join(datadir, "sim",f"{idx}.npz")
    sim_noisy = os.path.join(datadir, "sim_noisy",f"{idx}.npz")
    dataframe = os.path.join(datadir, "dataframes",f"structure_{idx}.pkl")

    with open(dataframe, "rb") as f:
        structure_df = pkl.load(f)
    
    with mrcfile.open(raw_sim, permissive=True) as mrc:
        raw_sim = mrc.data

    sim = np.load(sim)["arr_0"]
    sim_noisy = np.load(sim_noisy)["arr_0"]

    #A clean, B noisy

    sim2noisy = G_AB(torch.from_numpy(sim).unsqueeze(0).unsqueeze(0).float())
    
    noisy2sim = G_BA(torch.from_numpy(sim_noisy).unsqueeze(0).unsqueeze(0).float())

    sim2noisy = sim2noisy.cpu().detach().numpy()
    noisy2sim = noisy2sim.cpu().detach().numpy()

    fig, axs = plt.subplots(2, 3, figsize=(20, 20))  # Adjusted figsize: Width, Height
    axs[0, 0].imshow(raw_sim, cmap="gray",origin="lower")

    axs[0, 0].axis('off')  # Remove axes for cleaner look

    axs[0, 1].imshow(sim, cmap="gray",origin="lower")

    axs[0, 1].axis('off')  # Remove axes for cleaner look

    axs[0, 2].imshow(sim_noisy, cmap="gray",origin="lower")

    axs[0, 2].axis('off')  # Remove axes for cleaner look

    axs[1, 1].imshow(sim2noisy[0, 0], cmap="gray",origin="lower")

    axs[1, 1].axis('off')  # Remove axes for cleaner look

    axs[1, 2].imshow(noisy2sim[0, 0], cmap="gray",origin="lower")

    axs[1, 2].axis('off')  # Remove axes for cleaner look

    axs[1, 0].set_visible(False)

    plt.show()

    visu = structure_df.copy()
    visu[['x', 'y', 'z']] = visu[['x', 'y', 'z']]
    visu[["x","y","z"]] = visu[["x","y","z"]] * 10
    visu.label = visu.label.replace({'Ce': 0, 'O': 1, 'Pt': 2})
    ob_gen.mayavi_atomic_structure(visu)

    exp_dir = "data/noiser/raw_exp/"
    exp_files = os.listdir(exp_dir)
    exp_files = [os.path.join(exp_dir, file) for file in exp_files]
    #100 random experimental images and plot in 10x10 like before
    exps = []
    for i in range(100):
        idx = random.randint(0, len(exp_files))
        #tif
        if exp_files[idx].endswith(".tif"):
            exps.append(tifffile.imread(exp_files[idx]))

    fig, ax = plt.subplots(10, 10, figsize=(14, 20))  # Adjusted for better fit, can be resized based on actual output
    for i in range(10):
        for j in range(10):
            ax[i, j].imshow(exps[i * 10 + j], cmap="gray")
            ax[i, j].axis("off")  # Remove axes for cleaner look
    plt.tight_layout()  # Adjust layout
    plt.show()

