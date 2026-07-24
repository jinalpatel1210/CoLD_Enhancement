import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import subset_width

def aggregate(latents, method="mean"):
    if method == "mean":
        return torch.stack(latents, dim=0).mean(dim=0)
    if method == "sum":
        return torch.stack(latents, dim=0).sum(dim=0)
    if method == "max":
        return torch.stack(latents, dim=0).max(dim=0).values
    if method == "min":
        return torch.stack(latents, dim=0).min(dim=0).values
    if method == "concat":
        return torch.cat(latents, dim=1)
    raise ValueError(f"Unknown aggregation: {method!r}")

class HiddenLayers(nn.Module):
    def __init__(self, dims, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(1, len(dims) - 1):
            self.layers.append(nn.Linear(dims[i - 1], dims[i]))
            self.layers.append(nn.LeakyReLU(inplace=False))
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, dropout):
        super().__init__()
        self.body = HiddenLayers([in_dim, hidden_dim, latent_dim], dropout)

    def forward(self, x):
        return self.body(x)

class DecoderBlock(nn.Module):
    def __init__(self, latent_dim, out_dim):
        super().__init__()
        self.net = nn.Linear(latent_dim, out_dim)

    def forward(self, z):
        return self.net(z)

class MultiViewEncoder(nn.Module):
    def __init__(self, n_features, n_subsets, overlap, hidden_dim, latent_dim,
                 dropout, normalize_projection, p_norm):
        super().__init__()
        self.n_features = n_features
        self.in_dim = subset_width(n_features, n_subsets, overlap)
        self.latent_dim = latent_dim
        self.normalize_projection = normalize_projection
        self.p_norm = p_norm

        self.encoder = EncoderBlock(self.in_dim, hidden_dim, latent_dim, dropout)
        self.decoder = DecoderBlock(latent_dim, n_features)
        self.proj1 = nn.Linear(latent_dim, latent_dim)
        self.proj2 = nn.Linear(latent_dim, latent_dim)

    def encode(self, x_subset):
        return self.encoder(x_subset)

    def project(self, latent):
        z = F.leaky_relu(self.proj1(latent))
        z = self.proj2(z)
        if self.normalize_projection:
            z = F.normalize(z, p=self.p_norm, dim=1)
        return z

    def forward(self, x_subset):
        latent = self.encode(x_subset)
        z = self.project(latent)
        recon = self.decoder(latent)
        return z, latent, recon

class MLPClassifierHead(nn.Module):
    def __init__(self, input_dim, n_classes, hidden=(64, 32, 16)):
        super().__init__()
        h1, h2, h3 = hidden
        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, n_classes)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        feat = self.act(self.fc3(x))
        return self.out(feat), feat


