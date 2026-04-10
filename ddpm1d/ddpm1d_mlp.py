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