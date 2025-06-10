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
from wgan_architecture import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = "/home/careinfolab/Dr_Luo/Rohan/Gen_Adver_Nets/Dataset"
results_path = "/home/careinfolab/Dr_Luo/Rohan/Gen_Adver_Nets/Results/WGAN_GP_Linear"
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
pixels = 28*28
channels_dim = 1

def objective(trial):
    latent_dim = trial.suggest_int("latent_dim", 64, 256)
    hidden_dim = trial.suggest_int("hidden_dim", 32, 128)
    num_layers_g = trial.suggest_int("num_layers_g", 3, 10)
    num_layers_c = trial.suggest_int("num_layers_c", 2, 4)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    k = trial.suggest_int("k", 2, 5)
    num_epochs = trial.suggest_int("num_epochs", 50, 150)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    lam = trial.suggest_float("lambda", 1.0, 10.0)
    dataloader = DataLoader(torchvision.datasets.MNIST(root=dataset_path,train=True,download=True,transform=transform),batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True)
    generator = LinearGenerator(latent_dim=latent_dim, output_dim=pixels, hidden_dim=hidden_dim, num_layers=num_layers_g,channels_dim=channels_dim).to(device)
    critic = LinearCritic(hidden_dim=hidden_dim, num_layers=num_layers_c,in_dim=pixels*channels_dim).to(device)

    _, loss_g = train(generator, critic, latent_dim, dataloader=dataloader, num_epochs=num_epochs, lr=learning_rate, device=device,k=k,results_path=results_path,lam=lam)
    return sum(loss_g[-10:]) / 10
 
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

best_params = study.best_params
os.makedirs(results_path, exist_ok=True)

with open(os.path.join(results_path, "best_params.txt"), "w") as f:
    for key, val in best_params.items():
        f.write(f"{key}: {val}\n")

best_generator = LinearGenerator(latent_dim=best_params["latent_dim"],output_dim=pixels,hidden_dim=best_params["hidden_dim"],num_layers=best_params["num_layers_g"],channels_dim=channels_dim).to(device)

best_critic = LinearCritic(hidden_dim=best_params["hidden_dim"],num_layers=best_params["num_layers_c"],in_dim=pixels*channels_dim).to(device)

best_dataloader = DataLoader(torchvision.datasets.MNIST(root=dataset_path,train=True,download=True,transform=transform),batch_size=best_params["batch_size"],shuffle=True, num_workers=4,pin_memory=True)

train_loss_d, train_loss_g = train(best_generator, best_critic,latent_dim=best_params["latent_dim"],dataloader=best_dataloader,num_epochs=best_params["num_epochs"],lr=best_params["learning_rate"],device=device,k=best_params['k'],results_path=results_path,lam=best_params['lambda'])

torch.save(best_generator.state_dict(), os.path.join(results_path, "generator.pth"))
torch.save(best_critic.state_dict(), os.path.join(results_path, "critic.pth"))
torch.save(train_loss_d, os.path.join(results_path, "train_loss_d.pth"))
torch.save(train_loss_g, os.path.join(results_path, "train_loss_g.pth"))
print("Training complete. Best parameters and models saved.")




