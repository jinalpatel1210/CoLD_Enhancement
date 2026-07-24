import os

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from .config import Config
from .feature_reorder import FeatureReorderer

def inject_label_noise(y, noise_rate, n_classes, noise_type="sys"):
    if noise_rate <= 0:
        return y.copy()
    if noise_type == "sys":
        benign_rate = malicious_rate = noise_rate
    elif noise_type == "asys":
        benign_rate, malicious_rate = 0.0, noise_rate
    else:
        raise ValueError(f"noise_type must be 'sys' or 'asys', got {noise_type!r}")

    y_noisy = y.copy()
    for cls in np.unique(y):
        cls = int(cls)
        idx = np.where(y_noisy == cls)[0]
        rate = benign_rate if cls == 0 else malicious_rate
        n_flip = int(len(idx) * rate)
        if n_flip <= 0:
            continue
        flip_idx = np.random.choice(idx, size=n_flip, replace=False)
        if cls == 0:
            y_noisy[flip_idx] = np.random.randint(1, n_classes, size=n_flip)
        else:
            y_noisy[flip_idx] = 0
    return y_noisy

def subset_column_indices(n_features, n_subsets, overlap):
    n_column_subset = int(n_features / n_subsets)
    n_overlap = int(overlap * n_column_subset)
    column_idx = np.arange(n_features)
    out = []
    for i in range(n_subsets):
        if i == 0:
            start, stop = 0, n_column_subset + n_overlap
        else:
            start = i * n_column_subset - n_overlap
            stop = (i + 1) * n_column_subset
        out.append(column_idx[start:stop])
    return out

def subset_width(n_features, n_subsets, overlap):
    n_column_subset = int(n_features / n_subsets)
    n_overlap = int(overlap * n_column_subset)
    return n_column_subset + n_overlap

class TrafficDataset(Dataset):
    def __init__(self, X, y_noisy, y_clean):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y_noisy = torch.as_tensor(y_noisy, dtype=torch.long)
        self.y_clean = torch.as_tensor(y_clean, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_noisy[idx], self.y_clean[idx]

class DataModule:
    def __init__(self, config):
        self.cfg = config
        self.reorderer = FeatureReorderer(enabled=config.feature_reordering)
        self.train_ds = None
        self.test_ds = None
        self.subset_indices = None

    def setup(self):
        X_train, y_train, X_test, y_test = self._load_raw()
        self.cfg.n_features = X_train.shape[1]

        if self.cfg.use_standard_scaler:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        X_train = self.reorderer.fit_transform(X_train)
        X_test = self.reorderer.transform(X_test)

        y_train_noisy = inject_label_noise(
            y_train, self.cfg.noise_rate, self.cfg.n_classes, self.cfg.noise_type,
        )
        actual = float(np.mean(y_train_noisy != y_train))
        print(
            f"[data] {self.cfg.dataset}: train={len(y_train)} test={len(y_test)} "
            f"features={self.cfg.n_features} classes={self.cfg.n_classes} | "
            f"{self.cfg.noise_type} noise rate={actual:.3f}"
        )

        self.train_ds = TrafficDataset(X_train, y_train_noisy, y_train)
        self.test_ds = TrafficDataset(X_test, y_test, y_test)
        self.subset_indices = subset_column_indices(
            self.cfg.n_features, self.cfg.n_subsets, self.cfg.overlap,
        )
        return self

    def train_loader(self, shuffle=True, drop_last=False):
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.cfg.num_workers,
        )

    def test_loader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )

    def _load_raw(self):
        nested = os.path.join(self.cfg.data_dir, self.cfg.dataset)
        if os.path.exists(os.path.join(nested, "X_train.npy")):
            root = nested
        elif os.path.exists(os.path.join(self.cfg.data_dir, "X_train.npy")):
            root = self.cfg.data_dir
        else:
            root = nested
        paths = {
            "X_train": os.path.join(root, "X_train.npy"),
            "y_train": os.path.join(root, "y_train.npy"),
            "X_test": os.path.join(root, "X_test.npy"),
            "y_test": os.path.join(root, "y_test.npy"),
        }
        if all(os.path.exists(p) for p in paths.values()):
            return (
                np.load(paths["X_train"]).astype(np.float32),
                np.load(paths["y_train"]).astype(np.int64),
                np.load(paths["X_test"]).astype(np.float32),
                np.load(paths["y_test"]).astype(np.int64),
            )
        raise FileNotFoundError(
            f"Missing .npy files under {root!r}. "
            f"Expected X_train.npy, y_train.npy, X_test.npy, y_test.npy. "
        )
