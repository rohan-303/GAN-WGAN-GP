import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import matplotlib.pyplot as plt
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LinearGenerator(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=128, num_layers=3, channels_dim=1):
        super(LinearGenerator, self).__init__()
        self.channels_dim = channels_dim
        self.side_length = int(math.sqrt(output_dim))

        layers = []
        in_dim = latent_dim

        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if i != 0:
                layers.append(nn.BatchNorm1d(hidden_dim, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_dim = hidden_dim
            hidden_dim = min(hidden_dim * 2, 1024)

        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).view(-1, self.channels_dim, self.side_length, self.side_length)
    
class LinearCritic(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, num_layers=3):
        super(LinearCritic, self).__init__()
        layers = []

        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_dim = hidden_dim
            hidden_dim = hidden_dim // 2  

        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
class DCGenerator(nn.Module):
    def __init__(self, latent_dim=100, im_chan=1, hidden_dim=64):
        super(DCGenerator, self).__init__()
        self.latent_dim = latent_dim

        self.generator = nn.Sequential(
            self._make_gen_block(latent_dim, hidden_dim * 4, kernel_size=7, stride=1, padding=0),  # 1→7
            self._make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1),  # 7→14
            self._make_gen_block(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1),  # 14→28
            nn.Conv2d(hidden_dim, im_chan, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def _make_gen_block(self, in_channels, out_channels, kernel_size, stride, padding, final_layer=False):
        if final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.Tanh()
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = x.view(len(x), self.latent_dim, 1, 1)
        return self.generator(x)



class DCCritic(nn.Module):

    def __init__(self, im_chan=1, hidden_dim=64):
        super(DCCritic, self).__init__()

        self.critic = nn.Sequential(
            self._make_crit_block(im_chan, hidden_dim, normalize=False),     # [B, 64, 14, 14]
            self._make_crit_block(hidden_dim, hidden_dim * 2),               # [B, 128, 7, 7]
            self._make_crit_block(hidden_dim * 2, hidden_dim * 4),           # [B, 256, 3, 3]
            self._make_crit_block(hidden_dim * 4, 1, final_layer=True),      # [B, 1, 1, 1]
        )

    def _make_crit_block(self, in_channels, out_channels, kernel_size=4, stride=2, normalize=True, final_layer=False):
        if final_layer:
            kernel_size = 3  # Fix: avoid kernel > input
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1)]
        if not final_layer:
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)


    def forward(self, image):
        output = self.critic(image)
        return output.view(len(output), -1)
        
def gradient_penalty(critic, real_data, fake_data, device):
    batch_size = real_data.size(0)
    ε = torch.rand_like(real_data,device=device)
    interpolated = ε * real_data + (1 - ε) * fake_data
    interpolated.requires_grad_(True)

    critic_interpolated = critic(interpolated)
    grad_outputs = torch.ones_like(critic_interpolated)

    gradients = torch.autograd.grad(
        outputs=critic_interpolated,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp
    
    
def train(generator, critic, latent_dim, dataloader, num_epochs, lr, lam, device,results_path,isLinear=True, k=5):
    G = generator.to(device)
    C = critic.to(device)
    optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9))
    optimizer_C = torch.optim.Adam(C.parameters(), lr=lr, betas=(0.0, 0.9))
    train_loss_C, train_loss_g = [], []
    step = 0
    for epoch in range(num_epochs):
        for real_imgs, _ in dataloader:
            batch_size = real_imgs.size(0)

            if isLinear:
                real_imgs = real_imgs.view(batch_size, -1).to(device)
            else:
                real_imgs = real_imgs.to(device)


            z = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = G(z)

            # Train Discriminator
            optimizer_C.zero_grad()
            C_real = C(real_imgs)
            if isLinear:
                fake_imgs = fake_imgs.view(batch_size, -1)
                C_fake = C(fake_imgs)
            else:
                C_fake = C(fake_imgs)
            loss_C = -torch.mean(C_real) + torch.mean(C_fake) +  lam * gradient_penalty(C, real_imgs, fake_imgs.detach(), device)
            loss_C.backward()
            optimizer_C.step()
            train_loss_C.append(loss_C.item())

            # Train Generator every k steps
            if step % k == 0:
                optimizer_G.zero_grad()
                fake_imgs = G(z)
                if isLinear:
                    C_fake = C(fake_imgs.view(batch_size, -1))
                else:
                    C_fake = C(fake_imgs)
                loss_G = -torch.mean(C_fake)
                loss_G.backward()
                optimizer_G.step()
                train_loss_g.append(loss_G.item())

            if step % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step {step}, Loss D: {loss_C.item():.4f}, Loss G: {loss_G.item():.4f}")
            step += 1

        # Save sample images every 10 epochs
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                sample_z = torch.randn(64, latent_dim).to(device)
                fake_imgs = G(sample_z)
                #fake_imgs = (fake_imgs+1)/2  # Rescale to [0, 1]
                save_generated_images(fake_imgs, epoch + 1,folder=results_path+'/generated_images')

    return train_loss_C, train_loss_g