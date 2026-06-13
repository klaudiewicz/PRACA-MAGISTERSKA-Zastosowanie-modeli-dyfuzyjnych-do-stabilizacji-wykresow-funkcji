# ==========================================
# ARCHITEKTURA MLP I MODEL DDPM 1D
# ==========================================
import torch
import torch.nn as nn
import numpy as np
import math
from models.ddpm1d import SinusoidalPositionEmbeddings


class DenoiseNet1D_MLP(nn.Module):
    def __init__(self, data_dim=128, hidden_dim=256, time_emb_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        
        self.fc1 = nn.Linear(data_dim, hidden_dim)
        self.time_proj1 = nn.Linear(time_emb_dim, hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.time_proj2 = nn.Linear(time_emb_dim, hidden_dim)
        
        self.out = nn.Linear(hidden_dim, data_dim)
        self.act = nn.SiLU()

    def forward(self, x, t):
	        is_3d = False
	        if x.dim() == 3:
	            is_3d = True
	            x = x.squeeze(1)
	
	        # x ma teraz [B, L]
	        t_emb = self.time_mlp(t) # [B, Time_Emb_Dim]
	
	        x = self.fc1(x)
	        x = x + self.time_proj1(t_emb)
	        x = self.act(x)
	
	        x = self.fc2(x)
	        x = x + self.time_proj2(t_emb)
	        x = self.act(x)
	
	        out = self.out(x) # [B, L]
	        
	        if is_3d:
	            out = out.unsqueeze(1)
	            
	        return out
