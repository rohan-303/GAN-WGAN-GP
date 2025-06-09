import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
import os
import sys
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gan_architecture import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = "/home/careinfolab/Dr_Luo/Rohan/Gen_Adver_Nets/Dataset"
results_path = "/home/careinfolab/Dr_Luo/Rohan/Gen_Adver_Nets/Results/Vanilla_GAN_DCNN"
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
pixels = 28*28
channels_dim = 1

def objective(trial):
    latent_dim = trial.suggest_int("latent_dim", 64, 256)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    k = trial.suggest_int("k", 1, 3)
    num_epochs = trial.suggest_int("num_epochs", 50, 150)
    batch_size = trial.suggest_int("batch_size", 32, 256)
    dataloader = DataLoader(torchvision.datasets.MNIST(root=dataset_path,train=True,download=True,transform=transform),batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True)
    generator = DCGenerator(latent_dim=latent_dim,im_chan=channels_dim,hidden_dim=64).to(device)
    discriminator = DCDiscriminator(im_chan=channels_dim, hidden_dim=64).to(device)

    _, loss_g = train(generator, discriminator, latent_dim, dataloader=dataloader, num_epochs=num_epochs, lr=learning_rate, device=device,k=k,results_path=results_path,isLinear=False)
    return sum(loss_g[-10:]) / 10
 
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

best_params = study.best_params
os.makedirs(results_path, exist_ok=True)

with open(os.path.join(results_path, "best_params.txt"), "w") as f:
    for key, val in best_params.items():
        f.write(f"{key}: {val}\n")

best_generator = DCGenerator(latent_dim=best_params["latent_dim"],im_chan=channels_dim,hidden_dim=64).to(device)

best_discriminator = DCDiscriminator(im_chan=channels_dim, hidden_dim=64).to(device)

best_dataloader = DataLoader(torchvision.datasets.MNIST(root=dataset_path,train=True,download=True,transform=transform),batch_size=best_params["batch_size"],shuffle=True, num_workers=4,pin_memory=True)

train_loss_d, train_loss_g = train(best_generator, best_discriminator,latent_dim=best_params["latent_dim"],dataloader=best_dataloader,num_epochs=best_params["num_epochs"],lr=best_params["learning_rate"],device=device,k=5,results_path=results_path,isLinear=False)

torch.save(best_generator.state_dict(), os.path.join(results_path, "generator.pth"))
torch.save(best_discriminator.state_dict(), os.path.join(results_path, "discriminator.pth"))
torch.save(train_loss_d, os.path.join(results_path, "train_loss_d.pth"))
torch.save(train_loss_g, os.path.join(results_path, "train_loss_g.pth"))
print("Training complete. Best parameters and models saved.")




