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