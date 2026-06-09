import torch
import torch.nn as nn
import numpy as np
import math


class DDPM1D:

    def __init__(self, model, betas, n_T, device):

        self.model = model.to(device)
        self.n_T = n_T
        self.device = device

        self.betas = torch.tensor(betas,dtype=torch.float32).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas,dim=0)

    # =====================================================
    # SHAPE HELPER
    # =====================================================

    def _expand_time(self, x, t_tensor):
        """
        Zamienia:
        [B]
        -> [B,1,1]

        dla broadcastingu z:
        [B,1,L]
        """

        return t_tensor.view(-1, 1, 1)

    # =====================================================
    # FORWARD PROCESS
    # =====================================================

    @torch.no_grad()
    def q_sample(self, x_start, t, noise=None):
	    if noise is None:
	        noise = torch.randn_like(x_start)
	
	    sqrt_alpha_bar = torch.sqrt(self.alphas_bar[t])
	    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alphas_bar[t])
	    sqrt_alpha_bar = self._expand_time(x_start, sqrt_alpha_bar)
	    sqrt_one_minus_alpha_bar = self._expand_time(x_start, sqrt_one_minus_alpha_bar)
	    x_t = sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise
	    
	    return x_t

    # =====================================================
    # TRAINING LOSS
    # =====================================================

    def compute_loss(self, x0):

        B = x0.shape[0]
        t = torch.randint(0,self.n_T,(B,),device=self.device).long()
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x_start=x0,t=t,noise=noise)
        noise_pred = self.model(x_t, t)

        return nn.MSELoss()(noise_pred, noise)

    # =====================================================
    # STANDARD DDPM SAMPLING
    # =====================================================

    @torch.no_grad()
    def sample(self, shape):
        x_t = torch.randn(shape,device=self.device)
        for i in reversed(range(self.n_T)):
            t = torch.full((shape[0],),i,device=self.device,dtype=torch.long)
            noise_pred = self.model(x_t, t)
            alpha_t = self._expand_time(x_t,self.alphas[t])
            alpha_bar_t = self._expand_time(x_t,self.alphas_bar[t])
            beta_t = self._expand_time(x_t,self.betas[t])
            if i > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = torch.zeros_like(x_t)
            x_t = ((1 / torch.sqrt(alpha_t))* (x_t- (((1 - alpha_t)/ torch.sqrt(1 - alpha_bar_t))* noise_pred)))
            x_t = x_t + torch.sqrt(beta_t) * noise
        return x_t

    # =====================================================
    # SDEDIT RECONSTRUCTION
    # =====================================================

    @torch.no_grad()
    def denoise_signal(self, noisy_x, t_start):

        x_t = noisy_x.clone()
        for i in reversed(range(t_start)):

            t = torch.full((x_t.shape[0],),i,device=self.device,dtype=torch.long)
            noise_pred = self.model(x_t, t)
            alpha_t = self._expand_time(x_t,self.alphas[t])
            alpha_bar_t = self._expand_time(x_t,self.alphas_bar[t])
            beta_t = self._expand_time(x_t,self.betas[t])
            if i > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = torch.zeros_like(x_t)
            x_t = ((1 / torch.sqrt(alpha_t))* (x_t- (((1 - alpha_t)/ torch.sqrt(1 - alpha_bar_t))* noise_pred)))
            x_t = x_t + torch.sqrt(beta_t) * noise
        return x_t

    # =====================================================
    # DDIM
    # =====================================================

    @torch.no_grad()
    def ddim_denoise_signal(self,noisy_x,t_start,skip_steps=10):
        x_t = noisy_x.clone()
        eta = 0.0
        time_steps = list(reversed(range(0, t_start, skip_steps)))

        if time_steps[-1] != 0:
            time_steps.append(0)

        for i in range(len(time_steps) - 1):

            curr_t = time_steps[i]
            next_t = time_steps[i + 1]

            t_tensor = torch.full((x_t.shape[0],),curr_t,device=self.device,dtype=torch.long)
            noise_pred = self.model(x_t,t_tensor)

            alpha_bar_t = self.alphas_bar[curr_t]
            alpha_bar_next = (
                self.alphas_bar[next_t]
                if next_t >= 0
                else torch.tensor(1.0,device=self.device))

            alpha_bar_t = alpha_bar_t.view(1, 1, 1)
            alpha_bar_next = alpha_bar_next.view(1, 1, 1)
            x0_pred = (x_t- torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
            sigma_t = (eta* torch.sqrt(((1 - alpha_bar_next)/ (1 - alpha_bar_t))* (1- alpha_bar_t/ alpha_bar_next)))

            dir_xt = torch.sqrt(1- alpha_bar_next- sigma_t ** 2) * noise_pred
            if next_t > 0 and eta > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = torch.zeros_like(x_t)
            x_t = (torch.sqrt(alpha_bar_next)* x0_pred+ dir_xt+ sigma_t * noise)

        return x_t


# =========================================================
# POSITIONAL EMBEDDINGS
# =========================================================

class SinusoidalPositionEmbeddings(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim,device=device) * -embeddings)
        embeddings = (time[:, None]* embeddings[None, :])
        embeddings = torch.cat((embeddings.sin(),embeddings.cos()),dim=-1)
        return embeddings


# =========================================================
# BETA SCHEDULE
# =========================================================

def get_beta_schedule(schedule_type,beta_start,beta_end,n_T):

    if schedule_type == 'linear': return np.linspace(beta_start,beta_end,n_T)

    elif schedule_type == 'cosine':
        steps = n_T + 1
        x = np.linspace(0,n_T,steps)
        alphas_cumprod = (np.cos(((x / n_T) + 0.008) / 1.008* np.pi / 2) ** 2)
        alphas_cumprod = (alphas_cumprod/ alphas_cumprod[0])
        betas = (1- (alphas_cumprod[1:]/ alphas_cumprod[:-1]))
        return np.clip(betas,0.0001,0.9999)

    else:
        raise ValueError(f"Unknown schedule: {schedule_type}")