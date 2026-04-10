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
