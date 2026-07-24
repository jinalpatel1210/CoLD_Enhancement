import dataclasses
from dataclasses import dataclass, asdict
from typing import Optional

import yaml

@dataclass
class Config:
    dataset: str = "maltls"
    data_dir: str = "./data"
    results_dir: str = "./results"
    n_classes: int = 23
    n_features: int = 0

    noise_type: str = "asys"
    noise_rate: float = 0.4
    feature_reordering: bool = True
    use_standard_scaler: bool = False

    n_subsets: int = 4
    overlap: float = 0.75
    aggregation: str = "mean"

    hidden_dim: int = 117
    latent_dim: int = 117
    normalize_projection: bool = True
    p_norm: int = 2
    dropout: float = 0.0
    max_recover_samples: int = 7000

    masking_ratio: float = 0.3
    contrastive_loss: bool = True
    distance_loss: bool = False
    cosine_similarity: bool = False
    use_reconstruction_loss: bool = True
    tau: float = 0.1

    pretrain_epochs: int = 15
    batch_size: int = 32
    learning_rate: float = 1e-3
    seed: int = 787
    eval_seed: int = 1

    encoder_checkpoint: Optional[str] = None
    save_encoder: bool = True
    load_encoder_if_exists: bool = True
    force_pretrain: bool = False

    purify: bool = True
    
    high_threshold: float = 0.80
    low_threshold: float = 0.60

    min_clusters: int = 2
    max_clusters: int = 23
    dynamic_clusters: bool = True

    reliability_weight: float = 0.60
    neighbour_weight: float = 0.40
    cluster_method: str = "gmm"
    cluster_random_state: int = 0
    recover_discarded: bool = True
    classifier: str = "xgboost"
    classifier_epochs: int = 50
    classifier_lr: float = 1e-3
    progressive_thresholds = [0.95, 0.90, 0.85, 0.80]

    confidence_threshold = 0.90

    top_recovery_ratio = 0.30

    use_progressive_retraining = True

    global_weight = 0.40

    view_weight = 0.20
    device: str = "auto"
    num_workers: int = 0
    verbose: bool = True

    def resolve_device(self):
        import torch
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    @property
    def run_name(self):
        return (
            f"{self.dataset}_{self.noise_type}_{self.noise_rate}"
            f"_{self.classifier}_{self.cluster_method}"
        )

    @property
    def pretrain_checkpoint_id(self):
        """Pretrain run id (label noise is not part of encoder training)."""
        reorder = "reord" if self.feature_reordering else "noreord"
        return (
            f"{reorder}_sub{self.n_subsets}_ov{self.overlap}"
            f"_h{self.hidden_dim}_z{self.latent_dim}_ep{self.pretrain_epochs}"
            f"_bs{self.batch_size}_lr{self.learning_rate}_seed{self.seed}"
        )

    def resolve_encoder_checkpoint_path(self):
        import os
        if self.encoder_checkpoint:
            return self.encoder_checkpoint
        return os.path.join(
            self.results_dir,
            "checkpoints",
            self.dataset,
            self.pretrain_checkpoint_id,
            "encoder.pt",
        )

    @property
    def use_contrastive_combinations(self):
        return self.contrastive_loss or self.distance_loss

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

def load_config(path=None, **overrides):
    data = {}
    if path is not None:
        with open(path) as f:
            data.update(yaml.safe_load(f) or {})
    for k, v in overrides.items():
        if v is not None:
            data[k] = v
    valid = {f.name for f in dataclasses.fields(Config)}
    unknown = set(data) - valid
    if unknown:
        raise KeyError(f"Unknown config keys: {sorted(unknown)}")
    return Config(**data)

