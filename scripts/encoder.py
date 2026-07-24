import itertools
import os

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm

from .config import Config
from .data import DataModule
from .losses import JointLoss
from .model import MultiViewEncoder, aggregate
from .utils import ensure_dir

CHECKPOINT_VERSION = 1

def _swap_features(x):
    """Column-wise random permutation corruption for masked feature augmentation."""
    no, dim = x.shape
    out = np.zeros_like(x)
    for i in range(dim):
        out[:, i] = x[np.random.permutation(no), i]
    return out

class EncoderTrainer:
    def __init__(self, config, datamodule):
        self.cfg = config
        self.dm = datamodule
        self.device = torch.device(config.resolve_device())
        self.model = MultiViewEncoder(
            n_features=config.n_features,
            n_subsets=config.n_subsets,
            overlap=config.overlap,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            dropout=config.dropout,
            normalize_projection=config.normalize_projection,
            p_norm=config.p_norm,
        ).to(self.device)
        self.joint_loss = JointLoss(
            batch_size=config.batch_size,
            tau=config.tau,
            device=self.device,
            contrastive_loss=config.contrastive_loss,
            distance_loss=config.distance_loss,
            use_reconstruction=config.use_reconstruction_loss,
            cosine_similarity=config.cosine_similarity,
        )

    def _generate_subsets(self, x, mode="train"):
        cfg = self.cfg
        col_indices = self.dm.subset_indices
        n_subsets = cfg.n_subsets
        perm = np.random.permutation(n_subsets) if mode == "train" else np.arange(n_subsets)

        x_np = x.cpu().numpy()
        subsets = []
        for i in perm:
            cols = col_indices[i]
            x_bar = x_np[:, cols].copy()
            if mode == "train" and cfg.masking_ratio > 0:
                mask = np.random.binomial(1, cfg.masking_ratio, x_bar.shape)
                x_swapped = _swap_features(x_bar)
                x_bar = x_bar * (1 - mask) + x_swapped * mask
            subsets.append(torch.as_tensor(x_bar, dtype=torch.float32, device=self.device))
        return subsets

    @staticmethod
    def _process_batch(xi, xj):
        return torch.cat([xi, xj], dim=0)

    def _subset_combinations(self, subsets):
        combos = list(itertools.combinations(subsets, 2))
        return [self._process_batch(a, b) for a, b in combos]

    def fit(self):
        cfg = self.cfg
        optimizer = Adam(self.model.parameters(), lr=cfg.learning_rate)
        loader = self.dm.train_loader(shuffle=True, drop_last=True)
        self.model.train()

        for epoch in range(cfg.pretrain_epochs):
            running = 0.0
            pbar = tqdm(
                loader,
                desc=f"[pretrain] epoch {epoch + 1}/{cfg.pretrain_epochs}",
                leave=False,
                disable=not cfg.verbose,
            )
            for x, _, _ in pbar:
                x = x.to(self.device)
                x_orig = self._process_batch(x, x)
                subset_inputs = self._generate_subsets(x, mode="train")
                if cfg.use_contrastive_combinations:
                    subset_inputs = self._subset_combinations(subset_inputs)

                total_loss = x.new_zeros(())
                for xi in subset_inputs:
                    if cfg.use_contrastive_combinations:
                        x_in = xi
                    else:
                        x_in = self._process_batch(xi, xi)
                    z, _, xrecon = self.model(x_in)
                    tloss, _, _, _ = self.joint_loss(z, xrecon, x_orig)
                    total_loss = total_loss + tloss

                total_loss = total_loss / len(subset_inputs)
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                running += total_loss.item()
                pbar.set_postfix(loss=f"{total_loss.item():.4f}")

            n = max(1, len(loader))
            print(f"[pretrain] epoch {epoch + 1:>3}: loss={running / n:.4f}")
        return self

    def _checkpoint_meta(self):
        cfg = self.cfg
        return {
            "version": CHECKPOINT_VERSION,
            "pretrain_checkpoint_id": cfg.pretrain_checkpoint_id,
            "n_features": cfg.n_features,
            "n_subsets": cfg.n_subsets,
            "overlap": cfg.overlap,
            "hidden_dim": cfg.hidden_dim,
            "latent_dim": cfg.latent_dim,
            "dropout": cfg.dropout,
            "normalize_projection": cfg.normalize_projection,
            "p_norm": cfg.p_norm,
        }

    def _validate_checkpoint_meta(self, meta):
        expected = self._checkpoint_meta()
        keys = (
            "n_features", "n_subsets", "overlap", "hidden_dim", "latent_dim",
            "dropout", "normalize_projection", "p_norm",
        )
        mismatches = {
            k: (meta.get(k), expected[k])
            for k in keys
            if meta.get(k) != expected[k]
        }
        if mismatches:
            raise ValueError(
                "Encoder checkpoint incompatible with current config: "
                + ", ".join(f"{k}={got!r} (expected {exp!r})" for k, (got, exp) in mismatches.items())
            )

    def save_encoder(self, path):
        ensure_dir(os.path.dirname(path) or ".")
        torch.save(
            {"model_state_dict": self.model.state_dict(), "meta": self._checkpoint_meta()},
            path,
        )
        print(f"[pretrain] saved encoder -> {path}")

    def load_encoder(self, path):
        try:
            payload = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=self.device)
        if isinstance(payload, dict) and "model_state_dict" in payload:
            meta = payload.get("meta", {})
            if meta:
                self._validate_checkpoint_meta(meta)
            self.model.load_state_dict(payload["model_state_dict"])
        else:
            self.model.load_state_dict(payload)
        self.model.to(self.device)
        print(f"[pretrain] loaded encoder <- {path}")

    def fit_or_load(self):
        path = self.cfg.resolve_encoder_checkpoint_path()
        if (
            self.cfg.load_encoder_if_exists
            and not self.cfg.force_pretrain
            and os.path.isfile(path)
        ):
            self.load_encoder(path)
            return self
        self.fit()
        if self.cfg.save_encoder:
            self.save_encoder(path)
        return self

    @torch.no_grad()
    def embed(self, loader):
        self.model.eval()
        z_global_chunks = []
        z_subset_chunks = []
        y_noisy_chunks, y_clean_chunks = [], []

        for x, y_noisy, y_clean in loader:
            x = x.to(self.device)
            subsets = self._generate_subsets(x, mode="test")
            latents = [self.model.encode(s) for s in subsets]
            g = aggregate(latents, self.cfg.aggregation)

            z_global_chunks.append(g.cpu().numpy())
            z_subset_chunks.append([z.cpu().numpy() for z in latents])
            y_noisy_chunks.append(y_noisy.numpy())
            y_clean_chunks.append(y_clean.numpy())

        z_global = np.concatenate(z_global_chunks, axis=0)
        z_subsets = [
            np.concatenate([c[j] for c in z_subset_chunks], axis=0)
            for j in range(self.cfg.n_subsets)
        ]
        y_noisy = np.concatenate(y_noisy_chunks)
        y_clean = np.concatenate(y_clean_chunks)
        return z_global, z_subsets, y_noisy, y_clean
