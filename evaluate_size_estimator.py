import numpy as np
import os
import pickle as pkl
import matplotlib.pyplot as plt
import torch
from models import UNet_size_estimator, UNet128_3
idx = 3416
with open(os.path.join("data/noiser/dataframes",f"structure_{idx}.pkl"), "rb") as f:
    structure_df = pkl.load(f)
    clean_image = np.load(os.path.join("data/noiser/val/clean", f"{idx}.npz"))["arr_0"]
    noisy_image = np.load(os.path.join("data/noiser/val/noisy", f"{idx}.npz"))["arr_0"]                                      


G_BA = UNet128_3(1,1,ngf=16)
G_AB = UNet128_3(1,1,ngf=16)
checkpoint = torch.load("exp_checkpoints/lambda_GP_0.03_cyc_20/checkpoint_epoch_150.pth")
G_AB.load_state_dict(checkpoint["G_AB_state_dict"])
G_BA.load_state_dict(checkpoint["G_BA_state_dict"])
G_AB.eval()
G_BA.eval()

fake_exp = G_AB(torch.from_numpy(clean_image).unsqueeze(0).unsqueeze(0).float())
fake_exp_image = fake_exp.squeeze(0).squeeze(0).detach().numpy()

exp_idx = 342
def get_key(d, value):
    for key, val in d.items():
        if val == value:
            return key
#open file_name_mapping.pkl and get the file name for the exp_idx
with open("data/noiser/file_name_mapping.pkl", "rb") as f:
    file_name_mapping = pkl.load(f)
    #target : f"val/{exp_idx}.npz"
    file_name = get_key(file_name_mapping, f"train/{exp_idx}.npz")
exp_image = np.load(os.path.join("data/noiser/val/exp", f"{exp_idx}.npz"))["arr_0"]
exp_tensor = torch.from_numpy(exp_image).float().unsqueeze(0).unsqueeze(0)



model = UNet_size_estimator(1,1,ngf=64)
model.load_state_dict(torch.load("best_size_estimator/exp/size_estimator.pth"))
model.eval()
#noisy_image = torch.from_numpy(noisy_image).unsqueeze(0).unsqueeze(0).float()
#clean_image = torch.from_numpy(clean_image).unsqueeze(0).unsqueeze(0).float()
prediction = model(fake_exp)
prediction_exp = model(exp_tensor)
real_size = structure_df[structure_df.label == "Pt"].shape[0]
#round to nearest integer
plt.figure()
plt.imshow(fake_exp_image, cmap='gray')
plt.title(f"Predicted size: {round(prediction.item()*1000)}, File name: {file_name}, Real size: {real_size}")
plt.axis("off")
plt.show()

plt.figure()
plt.imshow(exp_image, cmap='gray')
plt.title(f"Exp image, Predicted size: {round(prediction_exp.item()*1000)}")
plt.axis("off")
plt.show()
#real size is the number of Pt atoms in the structure_df
print(len(file_name_mapping))
#for each image in train/exp and val/exp, get the predicted size
#plot a histogram with the predicted sizes and the average size as title
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
for data_dir in ["data/noiser/train/exp", "data/noiser/val/exp"]:
    predicted_sizes = []
    i = 0
    for file in os.listdir(data_dir):
        exp_image = np.load(os.path.join(data_dir, file))["arr_0"]
        exp_tensor = torch.from_numpy(exp_image).float().unsqueeze(0).unsqueeze(0).to(device)
        prediction = model(exp_tensor)
        prediction = prediction.cpu()
        predicted_sizes.append(prediction.item()*1000)
        print(np.round(prediction.item()*1000))
        i += 1
    plt.hist(predicted_sizes, bins=30)
    plt.title(f"Average size: {np.mean(predicted_sizes)}")
    plt.show()