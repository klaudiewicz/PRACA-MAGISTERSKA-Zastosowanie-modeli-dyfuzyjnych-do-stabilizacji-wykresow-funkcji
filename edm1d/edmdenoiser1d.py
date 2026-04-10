import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import time
import itertools
import os
import pickle
from tqdm import tqdm

from utils.math_functions import MathFunctions
from utils.metrics import calculate_metrics

def generate_grf_1d(shape, sigma_kernel=1.0, device='cpu'):
    """Generuje Gładki Szum (Gaussian Random Field) za pomocą konwolucji z jądrem Gaussa."""
    noise = torch.randn(shape, device=device)

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



class FunDPSExperimentRunner:
    def __init__(self, noise_type='white', sigma_kernel=2.0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.noise_type = noise_type
        self.sigma_kernel = sigma_kernel
        self.math_funcs = MathFunctions(num_points=128)

    def train_unconditional_prior(self, y_true_tensor, epochs=1000, batch_size=32):
        model = EDMDenoiser1D(data_dim=128).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        model.train()
        t_start = time.time()
        loss_history = [] 
        
        for _ in range(epochs):
            optimizer.zero_grad()
            rnd_normal = torch.randn(batch_size, device=self.device)
            sigma = (rnd_normal * 1.2 - 1.2).exp()

            if self.noise_type == 'grf':
                base_noise = generate_grf_1d((batch_size, 128), self.sigma_kernel, self.device)
            else:
                base_noise = torch.randn((batch_size, 128), device=self.device)
                
            noise = base_noise * sigma.unsqueeze(1)
            y_noisy = y_true_tensor.repeat(batch_size, 1) + noise

            y_pred = model(y_noisy, sigma)

            weight = (sigma ** 2 + 1) / (sigma ** 2 + 1e-5)
            loss = (weight.unsqueeze(1) * (y_pred - y_true_tensor.repeat(batch_size, 1)) ** 2).mean()

            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            
        train_time = time.time() - t_start
        return model, train_time, loss_history 

    def run_ablation_study(self, func_name, n_steps_list=[50, 100], zetas=[100, 500], prior_epochs=1000):
        print(f"\n Trening priora ({self.noise_type.upper()} Noise) dla: {func_name}")
        x, y_true = self.math_funcs.get_function(func_name)
        y_tensor = torch.tensor(y_true, dtype=torch.float32).unsqueeze(0).to(self.device)

        np.random.seed(42)
        num_obs = int(128 * 0.10)
        mask_idx = np.sort(np.random.choice(128, num_obs, replace=False))
        forward_op = ForwardOperator(mask_idx)
        obs_tensor = forward_op(y_tensor)

        model, train_time, loss_hist = self.train_unconditional_prior(y_tensor, epochs=prior_epochs)
        
        sampler = FunDPSSampler(model, self.device)

        best_l2 = float('inf')
        best_pred = None
        best_cfg = {}
        metrics_hist = []

        # Grid Search dla procesu inferencji z Guidance (Zeta i Steps)
        configs = list(itertools.product(n_steps_list, zetas))
        for steps, zeta in tqdm(configs, desc=f"Inferencja FunDPS ({func_name})", leave=False):
            try:
                s_start = time.time()
                pred_tensor = sampler.sample(obs_tensor, forward_op, num_steps=steps, zeta=zeta)
                sample_time = time.time() - s_start
                
                pred_np = pred_tensor.cpu().numpy()[0]

                if np.isnan(pred_np).any() or np.isinf(pred_np).any():
                    raise ValueError("Wartości NaN lub Inf w predykcji.")

                m = calculate_metrics(y_true, pred_np, exec_time=sample_time, train_time=train_time)
                metrics_hist.append({'Steps': steps, 'Zeta': zeta, **m})

                if m['L2_Error'] < best_l2:
                    best_l2 = m['L2_Error']
                    best_pred = pred_np
                    best_cfg = {'Steps': steps, 'Zeta': zeta}
                    
            except Exception as e:
                print(f"Błąd dla {func_name} (Steps={steps}, Zeta={zeta}): {e}")
                metrics_hist.append({'Steps': steps, 'Zeta': zeta, 'L2_Error': float('inf'), 'MSE': float('inf'), 'MAE': float('inf'), 'MAPE': float('inf'), 'Wasserstein': float('inf'), 'Total_Time_s': float('inf')})

        best_m = calculate_metrics(y_true, best_pred, exec_time=0.0, train_time=0.0) if best_pred is not None else None
        if best_m is not None:
             for h in metrics_hist:
                if h['Steps'] == best_cfg['Steps'] and h['Zeta'] == best_cfg['Zeta']:
                    best_m['Total_Time_s'] = h.get('Total_Time_s', 0.0)
                    break

        return {
            'x': x, 'y_true': y_true, 'mask_idx': mask_idx, 
            'best_pred': best_pred, 'best_config': best_cfg,
            'best_metrics': best_m, 'metrics_history': metrics_hist,
            'prior_loss_history': loss_hist
        }

	
class FunDPSSampler:
    """Proces odwrócony z wykorzystaniem nawigacji przez gradient obserwacji (Guidance)."""
    def __init__(self, model, device, sigma_min=0.002, sigma_max=80.0, rho=7.0):
        self.model = model.to(device)
        self.device = device
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def get_sigmas(self, num_steps):
        step_indices = torch.arange(num_steps, dtype=torch.float32, device=self.device)
        t = step_indices / (max(num_steps - 1, 1))
        sigma_max_rho = self.sigma_max ** (1 / self.rho)
        sigma_min_rho = self.sigma_min ** (1 / self.rho)
        sigmas = (sigma_max_rho + t * (sigma_min_rho - sigma_max_rho)) ** self.rho
        return torch.cat([sigmas, torch.zeros_like(sigmas[:1])])

    @torch.enable_grad()
    def sample(
	    self,
	    observation,
	    forward_op,
	    num_steps=200,
	    zeta=10.0,
	    data_dim=128,
	):
	
	    self.model.eval()
	    mse = nn.MSELoss()
	
	    sigmas = self.get_sigmas(num_steps)
	
	    # start z wysokiego noise
	    a_i = torch.randn(1, data_dim, device=self.device) * self.sigma_max
	
	    for i in range(num_steps):
	        sigma_i = sigmas[i].unsqueeze(0)
	        sigma_prev = sigmas[i + 1].unsqueeze(0)
	
	        a_i = a_i.detach().requires_grad_(True)
	
	        # 1. denoising step
	        a_hat_0 = self.model(a_i, sigma_i)
	
	        d_i = (a_i - a_hat_0) / sigma_i
	        a_prev = a_i + (sigma_prev - sigma_i) * d_i
	
	        # 2. DPS guidance
	        if sigma_prev.item() > 0:
	
	            pred_obs = forward_op(a_hat_0)
	            loss = mse(pred_obs, observation)
	            grad_x0 = torch.autograd.grad(loss, a_hat_0)[0]	
	            grad_norm = torch.norm(grad_x0) + 1e-8
	            grad_x0 = grad_x0 / grad_norm
	            guidance_scale = zeta * sigma_i
	            a_prev = a_prev - guidance_scale * grad_x0.detach()
	
	        a_i = a_prev.detach()
	
	    return a_i

    @torch.enable_grad()
    def sample_with_history(self, observation, forward_op, num_steps=100, zeta=10.0, data_dim=128):
        """Wersja dla widżetów - zwraca klatki animacji z poprawnym algorytmem DPS."""
        self.model.eval()
        mse = nn.MSELoss()
        sigmas = self.get_sigmas(num_steps)
        
        a_i = torch.randn(1, data_dim, device=self.device) * self.sigma_max
        
        history = []
        history.append(a_i.detach().cpu().numpy()[0]) # Klatka 0

        for i in range(num_steps):
            sigma_i = sigmas[i].unsqueeze(0)
            sigma_prev = sigmas[i + 1].unsqueeze(0)

            a_i = a_i.detach().requires_grad_(True)

            # 1. Denoising
            a_hat_0 = self.model(a_i, sigma_i)
            d_i = (a_i - a_hat_0) / sigma_i
            a_prev = a_i + (sigma_prev - sigma_i) * d_i

            # 2. Guidance
            if sigma_prev.item() > 0:
                pred_obs = forward_op(a_hat_0)
                loss = mse(pred_obs, observation)

                grad_x0 = torch.autograd.grad(loss, a_hat_0)[0]
                grad_norm = torch.norm(grad_x0) + 1e-8
                grad_x0 = grad_x0 / grad_norm

                guidance_scale = zeta * sigma_i
                a_prev = a_prev - guidance_scale * grad_x0.detach()

            a_i = a_prev.detach()
            history.append(a_i.cpu().numpy()[0])

        return history
