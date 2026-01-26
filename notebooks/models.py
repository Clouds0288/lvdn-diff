import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GINEConv

# --- 1. Sinusoidal Time Embeddings ---
class SinusoidalPositionalEmbeddings(nn.Module):
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

# --- 2. Universal GNN Backbone ---
class UniversalGNN(nn.Module):
    def __init__(self, in_channels=7, hidden_dim=64, time_dim=32, edge_dim=2):
        super().__init__()

        # Node Projection
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Edge Projection
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Time Projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, hidden_dim)
        )

        # GNN Layers
        self.conv1 = GINEConv(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        )
        self.conv2 = GINEConv(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        )
        self.conv3 = GINEConv(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        )

        # Output Head
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1) 
        )

    def forward(self, x, edge_index, edge_attr, time_emb, batch):
        h = self.node_mlp(x)
        edge_feat = self.edge_mlp(edge_attr)

        if time_emb is not None:
            t_feat = self.time_mlp(time_emb)
            if batch is not None:
                h = h + t_feat[batch] # Broadcast
            else:
                h = h + t_feat

        h = self.conv1(h, edge_index, edge_attr=edge_feat)
        h = F.silu(h)
        h = self.conv2(h, edge_index, edge_attr=edge_feat)
        h = F.silu(h)
        h = self.conv3(h, edge_index, edge_attr=edge_feat)
        h = F.silu(h)

        return self.final_mlp(h)

# --- 3. Diffusion Scheduler ---
class DiffusionScheduler(nn.Module):
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod))

    def add_noise(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        # Broadcast logic handled by PyTorch tensors if dimensions align
        # Reshaping for safety: [Batch/Nodes, 1]
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_om_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)

        x_noisy = sqrt_alpha_t * x_start + sqrt_om_alpha_t * noise
        return x_noisy, noise

# --- 4. M2 Wrapper (Regressor) ---
class M2_GNN_Regressor(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.time_emb_layer = SinusoidalPositionalEmbeddings(dim=32)
        self.backbone = UniversalGNN(in_channels=7, hidden_dim=hidden_dim, time_dim=32)

    def forward(self, condition, edge_index, edge_attr, batch):
        N = condition.shape[0]
        device = condition.device

        # Placeholder for State (Ch 0)
        x_state_placeholder = torch.zeros((N, 1), device=device)
        x_in = torch.cat([x_state_placeholder, condition], dim=-1)

        # Static Time (t=0)
        if batch is not None:
            batch_size = batch.max().item() + 1
            t = torch.zeros((batch_size,), device=device, dtype=torch.long)
        else:
            t = torch.tensor([0], device=device, dtype=torch.long)

        t_emb = self.time_emb_layer(t)
        return self.backbone(x_in, edge_index, edge_attr, t_emb, batch)

# --- 5. M3 Wrapper (Diffusion) ---
class M3_Physics_Diffusion(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.time_emb_layer = SinusoidalPositionalEmbeddings(dim=32)
        self.backbone = UniversalGNN(in_channels=7, hidden_dim=hidden_dim, time_dim=32)
        self.scheduler = DiffusionScheduler()

    def forward(self, x_t, t, condition, edge_index, edge_attr, batch):
        x_in = torch.cat([x_t, condition], dim=-1)
        t_emb = self.time_emb_layer(t)
        return self.backbone(x_in, edge_index, edge_attr, t_emb, batch)
