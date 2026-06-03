# # ==========================================
# # ARCHITEKTURA CONV1D
# # ==========================================
import torch
import torch.nn as nn
import numpy as np
from models.ddpm1d import SinusoidalPositionEmbeddings # Re-use embeddings

class ResBlock1D(nn.Module):
    def __init__(self, ch, time_emb_dim, dilation):
        super().__init__()
        
        self.conv1 = nn.Conv1d(ch, ch, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(ch, ch, kernel_size=3, padding=1)
        
        self.norm1 = nn.GroupNorm(8, ch)
        self.norm2 = nn.GroupNorm(8, ch)
        
        self.act = nn.SiLU()
        
        # FiLM conditioning
        self.time_mlp = nn.Linear(time_emb_dim, ch * 2)

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        
        # FiLM
        scale_shift = self.time_mlp(t_emb).unsqueeze(-1)
        scale, shift = scale_shift.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        
        return x + h  # residual
		
class DenoiseNet1D_Conv(nn.Module):
    def __init__(self, data_dim=128, time_emb_dim=64, base_channels=64):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        
        # Updated references to use base_channels
        self.input_proj = nn.Conv1d(1, base_channels, kernel_size=1)

        self.blocks = nn.ModuleList([
            ResBlock1D(base_channels, time_emb_dim, dilation=1),
            ResBlock1D(base_channels, time_emb_dim, dilation=2),
            ResBlock1D(base_channels, time_emb_dim, dilation=4),
            ResBlock1D(base_channels, time_emb_dim, dilation=8),
            ResBlock1D(base_channels, time_emb_dim, dilation=4),
            ResBlock1D(base_channels, time_emb_dim, dilation=2),
        ])
        
        self.out = nn.Conv1d(base_channels, 1, kernel_size=1)

    def forward(self, x, t):
	    if x.dim() == 2:
	        x = x.unsqueeze(1)
	    elif x.dim() == 4:
	        x = x.squeeze(1)
	
	    t_emb = self.time_mlp(t)
	    x = self.input_proj(x)
	
	    for block in self.blocks:
	        x = block(x, t_emb)
	
	    return self.out(x)