flow repozytorium:
notebook wizualizacje_wplywu_parametrow_i_odszumiania pokazuje czym jest definiowany szum pokazane na 3 przykładowych funkcjach (sin, square_wave i sin_1_over) - pokaż różne rodzaje szumu, przy tych funkcjach skorzystaj z klasy z utils/math_functions.py (poniżej). 
Nastepnie pokazana jest lista parametrów i ich znaczenie i wzór z ddpm- harmonogram beta, kroki T i epoki.
możesz skorzystać w wypisaniu wzorów z klasy modeli dyfuzyjnych z ddpm1d, skorzystaj też z teorii o sdeedit i opisz co to, a także fundps (edm1d).
potem pokaz wplyw poszczegolnych parametrów:
jak wygląda szum przy różnych harmonogramach? (cosine vs linear) -  w markdown wypisz czym sie roznią
jak wygląda szum przy różnych krokach T? czym właściwie jest to T? - opisz w markdown
jak wygląda proces zaszumiania?
jak wygląda proces odszumiania w zależności od tych parametrów. nie ograniczaj się wyłącznie do jednego modelu,wykresy pokazuj dla wszystkich

edm1d/edmdenoiser1d.py:
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

def generate_grf_1d(shape, sigma_kernel=1.0, device='cpu'):
    """Generuje Gładki Szum (Gaussian Random Field) za pomocą konwolucji z jądrem Gaussa."""
    noise = torch.randn(shape, device=device)

    # Tworzenie jądra Gaussa
    kernel_size = int(sigma_kernel * 6 + 1)
    if kernel_size % 2 == 0: kernel_size += 1
    x = torch.arange(kernel_size, device=device, dtype=torch.float32) - (kernel_size - 1) / 2
    kernel = torch.exp(-x**2 / (2 * sigma_kernel**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1)

    noise = noise.unsqueeze(1) # (B, 1, L)
    pad_size = kernel_size // 2
    noise_padded = F.pad(noise, (pad_size, pad_size), mode='reflect')
    smooth_noise = F.conv1d(noise_padded, kernel)

    # Normalizacja wariancji do 1
    smooth_noise = smooth_noise / (smooth_noise.std() + 1e-8)

    return smooth_noise.squeeze(1)


class SigmaEmbedding(nn.Module):
    """Zanurzenie ciągłego poziomu szumu (sigma) zamiast dyskretnego t."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, sigma):
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=sigma.device, dtype=torch.float32) * -emb)
        emb = torch.log(sigma.unsqueeze(1) + 1e-5) * emb.unsqueeze(0)
        return torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)

class EDMDenoiser1D(nn.Module):
    """Sieć bazująca na klasycznym Multi-Layer Perceptron (MLP) dla modelu EDM/FunDPS."""
    def __init__(self, data_dim=128, emb_dim=64):
        super().__init__()
        self.sigma_mlp = nn.Sequential(
            SigmaEmbedding(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU()
        )
        self.net = nn.Sequential(
            nn.Linear(data_dim + emb_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, data_dim)
        )

    def forward(self, y, sigma):
        emb = self.sigma_mlp(sigma)
        x = torch.cat([y, emb], dim=-1)
        return self.net(x)

class ForwardOperator:
    """Operator obserwacji - symuluje utratę danych (zachowuje tylko rzadkie punkty pomiarowe)."""
    def __init__(self, mask_indices):
        self.mask_indices = mask_indices

    def __call__(self, a):
        return a[:, self.mask_indices]

ddpm1d/ddpm1d_mlp.py:

# ==========================================

# ARCHITEKTURA MLP I MODEL DDPM 1D

# ==========================================

import torch

import torch.nn as nn

import numpy as np

import math



class SinusoidalPositionEmbeddings(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.dim = dim



    def forward(self, time):

        device = time.device

        half_dim = self.dim // 2

        embeddings = math.log(10000) / (half_dim - 1)

        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)

        embeddings = time[:, None] * embeddings[None, :]

        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        return embeddings



class DenoiseNet1D_MLP(nn.Module):

    """Sieć MLP przewidująca szum epsilon dla wektora 1D z wbudowanym embeddingiem czasu."""

    def __init__(self, data_dim=128, time_emb_dim=64):

        super().__init__()

        self.time_mlp = nn.Sequential(

            SinusoidalPositionEmbeddings(time_emb_dim),

            nn.Linear(time_emb_dim, time_emb_dim),

            nn.SiLU()

        )

        self.fc1 = nn.Linear(data_dim, 256)

        self.time_proj1 = nn.Linear(time_emb_dim, 256)

        self.fc2 = nn.Linear(256, 256)

        self.time_proj2 = nn.Linear(time_emb_dim, 256)

        self.out = nn.Linear(256, data_dim)

        self.act = nn.SiLU()



    def forward(self, x, t):

        # x: [Batch, Data_Dim]

        t_emb = self.time_mlp(t) # [Batch, Time_Emb_Dim]



        x = self.fc1(x)

        x = x + self.time_proj1(t_emb)

        x = self.act(x)



        x = self.fc2(x)

        x = x + self.time_proj2(t_emb)

        x = self.act(x)



        return self.out(x)



class DDPM1D:

    def __init__(self, model, betas, n_T, device):

        self.model = model.to(device)

        self.n_T = n_T

        self.device = device



        # Inicjalizacja harmonogramu (betas)

        self.betas = torch.tensor(betas, dtype=torch.float32).to(device)

        self.alphas = 1. - self.betas

        self.alphas_bar = torch.cumprod(self.alphas, dim=0)



    def compute_loss(self, x0):

        # Losowanie kroków czasowych t

        t = torch.randint(0, self.n_T, (x0.shape[0],), device=self.device).long()

        noise = torch.randn_like(x0)



        alpha_bar_t = self.alphas_bar[t].view(-1, 1)

        # Forward process (dodawanie szumu) - Równanie Ho et al.

        x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise



        # Przewidywanie szumu

        noise_pred = self.model(x_t, t)

        return nn.MSELoss()(noise_pred, noise)



    @torch.no_grad()

    def sample(self, shape):

        # Samplowanie z czystego szumu Gaussa

        x_t = torch.randn(shape, device=self.device)



        # Reverse process (odszumianie krokowo)

        for i in reversed(range(self.n_T)):

            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)

            noise_pred = self.model(x_t, t)



            alpha_t = self.alphas[t].view(-1, 1)

            alpha_bar_t = self.alphas_bar[t].view(-1, 1)

            beta_t = self.betas[t].view(-1, 1)



            if i > 0:

                noise = torch.randn_like(x_t)

            else:

                noise = torch.zeros_like(x_t)



            # Równanie Ho et al. dla kroku reverse

            x_t = (1 / torch.sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * noise_pred)

            x_t = x_t + torch.sqrt(beta_t) * noise



        return x_t

	

    @torch.no_grad()

    def denoise_signal(self, noisy_x, t_start):

        """

        Podejście SDEdit: Przyjmujemy zaszumiony sygnał jako stan w kroku t_start.

        Wykonujemy proces odwrócony (Reverse Process) tylko od t_start do 0.

        """

        x_t = noisy_x.clone()

        

        # Odszumianie krok po kroku od t_start w dół

        for i in reversed(range(t_start)):

            t = torch.full((x_t.shape[0],), i, device=self.device, dtype=torch.long)

            noise_pred = self.model(x_t, t)



            alpha_t = self.alphas[t].view(-1, 1)

            alpha_bar_t = self.alphas_bar[t].view(-1, 1)

            beta_t = self.betas[t].view(-1, 1)



            if i > 0:

                noise = torch.randn_like(x_t)

            else:

                noise = torch.zeros_like(x_t)



            x_t = (1 / torch.sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * noise_pred)

            x_t = x_t + torch.sqrt(beta_t) * noise



        return x_t



ddpm1d_conv1d.py:

# ==========================================

# ARCHITEKTURA CONV1D

# ==========================================

import torch

import torch.nn as nn

import numpy as np

from ddpm1d.ddpm1d_mlp import SinusoidalPositionEmbeddings # Re-use embeddings



class DenoiseNet1D_Conv(nn.Module):

    def __init__(self, data_dim=128, time_emb_dim=64, channels=32):

        super().__init__()

        self.data_dim = data_dim

        

        self.time_mlp = nn.Sequential(

            SinusoidalPositionEmbeddings(time_emb_dim),

            nn.Linear(time_emb_dim, time_emb_dim),

            nn.SiLU()

        )

        

        # Projekcja czasu na kanały

        self.time_proj = nn.Linear(time_emb_dim, channels)



        # Input: [Batch, 1, Data_Dim]

        self.conv_in = nn.Conv1d(1, channels, kernel_size=3, padding=1)

        

        self.block1 = nn.Sequential(

            nn.Conv1d(channels, channels, kernel_size=3, padding=1),

            nn.BatchNorm1d(channels),

            nn.SiLU()

        )

        

        self.block2 = nn.Sequential(

            nn.Conv1d(channels, channels, kernel_size=3, padding=1),

            nn.BatchNorm1d(channels),

            nn.SiLU()

        )

        

        # Output: [Batch, 1, Data_Dim]

        self.conv_out = nn.Conv1d(channels, 1, kernel_size=3, padding=1)



    def forward(self, x, t):

        # x: [Batch, Data_Dim] -> transform do [Batch, 1, Data_Dim] dla Conv1d

        x = x.unsqueeze(1)

        

        t_emb = self.time_mlp(t) # [Batch, Time_Emb_Dim]

        t_feat = self.time_proj(t_emb).unsqueeze(-1) # [Batch, Channels, 1]



        x = self.conv_in(x)

        

        # Wstrzyknięcie czasu (broadcast po długości sygnału)

        x = x + t_feat 

        x = self.block1(x)

        x = x + t_feat

        x = self.block2(x)

        

        x = self.conv_out(x)

        return x.squeeze(1) # Powrót do [Batch, Data_Dim]



ddpm1d/ddpm1d_unet.py:

# ==========================================

# ARCHITEKTURA U-NET 

# ==========================================

import torch

import torch.nn as nn

from ddpm1d.ddpm1d_mlp import SinusoidalPositionEmbeddings



class UNetBlock1D(nn.Module):

    def __init__(self, in_ch, out_ch, time_emb_dim):

        super().__init__()

        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm1d(out_ch)

        self.bn2 = nn.BatchNorm1d(out_ch)

        self.act = nn.SiLU()



    def forward(self, x, t_emb):

        # x: [B, in_ch, L], t_emb: [B, time_emb_dim]

        h = self.act(self.bn1(self.conv1(x)))

        

        # Time embed projection

        time_feat = self.time_mlp(t_emb).unsqueeze(-1) # [B, out_ch, 1]

        h = h + time_feat

        

        h = self.act(self.bn2(self.conv2(h)))

        return h



class DenoiseNet1D_UNet(nn.Module):

    def __init__(self, data_dim=128, time_emb_dim=64, base_channels=32):

        super().__init__()

        assert data_dim % 4 == 0, "Data dim must be divisible by 4 for this basic UNet"

        

        self.time_embed = nn.Sequential(

            SinusoidalPositionEmbeddings(time_emb_dim),

            nn.Linear(time_emb_dim, time_emb_dim),

            nn.SiLU()

        )

        

        ch = base_channels

        # Encoder

        self.inc = nn.Conv1d(1, ch, kernel_size=3, padding=1)

        self.down1 = UNetBlock1D(ch, ch*2, time_emb_dim) # L -> L/2

        self.down2 = UNetBlock1D(ch*2, ch*4, time_emb_dim) # L/2 -> L/4

        self.pool = nn.AvgPool1d(2)

        

        # Bottleneck

        self.mid = UNetBlock1D(ch*4, ch*4, time_emb_dim)

        

        # Decoder

        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')

        self.up_conv1 = UNetBlock1D(ch*4 + ch*2, ch*2, time_emb_dim) # concat skip

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

        self.up_conv2 = UNetBlock1D(ch*2 + ch, ch, time_emb_dim) # concat skip

        

        self.outc = nn.Conv1d(ch, 1, kernel_size=1)



    def forward(self, x, t):

        # x: [B, L]

        x = x.unsqueeze(1) # [B, 1, L]

        t_emb = self.time_embed(t)

        

        # Encoder

        x1 = self.inc(x) # [B, ch, L]

        x2 = self.pool(x1) 

        x2 = self.down1(x2, t_emb) # [B, ch*2, L/2]

        x3 = self.pool(x2)

        x3 = self.down2(x3, t_emb) # [B, ch*4, L/4]

        

        # Mid

        x3 = self.mid(x3, t_emb)

        

        # Decoder

        x4 = self.up1(x3) 

        x4 = torch.cat([x4, x2], dim=1) # Skip connection

        x4 = self.up_conv1(x4, t_emb) # [B, ch*2, L/2]

        

        x5 = self.up2(x4)

        x5 = torch.cat([x5, x1], dim=1) # Skip connection

        x5 = self.up_conv2(x5, t_emb) # [B, ch, L]

        

        out = self.outc(x5)

        return out.squeeze(1) # [B, L]




na podstawie poniższych modeli i plików, stwórz analogiczny notebook jak załaczony ale dla funkcji 2D/3D:

