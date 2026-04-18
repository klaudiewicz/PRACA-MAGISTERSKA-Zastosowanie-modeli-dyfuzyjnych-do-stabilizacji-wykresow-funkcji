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
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_bar_t = torch.sqrt(self.alphas_bar[t])[:, None]
        sqrt_one_minus_alphas_bar_t = torch.sqrt(1. - self.alphas_bar[t])[:, None]

        # Równanie: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        return sqrt_alphas_bar_t * x_start + sqrt_one_minus_alphas_bar_t * noise
	
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

    @torch.no_grad()
    def ddim_denoise_signal(self, noisy_x, t_start, skip_steps=10):
        """
        Zoptymalizowane odszumianie algorytmem DDIM, pozwalające przeskakiwać kroki.
        skip_steps: Jeśli = 10, to odszumiamy co 10 krok (np. 300 -> 290 -> 280).
        eta: Dla czystego DDIM (deterministycznego) wynosi 0.
        """
        x_t = noisy_x.clone()
        eta = 0.0 # Deterministyczny skok

        # Tworzymy wektor kroków od t_start w dół, aż do 0, skacząc co skip_steps
        time_steps = list(reversed(range(0, t_start, skip_steps)))
        # Upewniamy się, że krok "0" zawsze będzie na końcu
        if time_steps[-1] != 0:
            time_steps.append(0)

        for i in range(len(time_steps) - 1):
            curr_t = time_steps[i]
            next_t = time_steps[i+1]
            
            # Wektory t dla całego batcha
            t_tensor = torch.full((x_t.shape[0],), curr_t, device=self.device, dtype=torch.long)
            
            # 1. Predykcja szumu przez model
            noise_pred = self.model(x_t, t_tensor)
            
            # 2. Wyciągnięcie parametrów dla aktualnego i następnego kroku
            alpha_bar_t = self.alphas_bar[curr_t].view(-1, 1)
            # Jeśli next_t wynosi -1 (zabezpieczenie), alpha_bar to 1.0 (brak szumu)
            alpha_bar_next = self.alphas_bar[next_t].view(-1, 1) if next_t >= 0 else torch.ones_like(alpha_bar_t)

            # 3. Predykcja "czystego" sygnału (x0_pred)
            # x0_pred = (x_t - sqrt(1 - alpha_bar_t) * epsilon) / sqrt(alpha_bar_t)
            x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)

            # 4. Obliczenie wariancji kroku (zależnej od parametru eta)
            # Dla eta=0 (czyste DDIM), to wynosi 0
            sigma_t = eta * torch.sqrt((1 - alpha_bar_next) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_next))
            
            # 5. Obliczenie kierunku "wskazującego" na zaszumiony sygnał (tzw. direction pointing to x_t)
            dir_xt = torch.sqrt(1 - alpha_bar_next - sigma_t**2) * noise_pred
            
            # 6. Złożenie w nowy krok x_{t-1} (lub x_{t-skip_steps})
            noise = torch.randn_like(x_t) if next_t > 0 and eta > 0 else 0.0
            x_t = torch.sqrt(alpha_bar_next) * x0_pred + dir_xt + sigma_t * noise

        return x_t