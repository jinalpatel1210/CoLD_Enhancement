from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from .config import Config
from .metrics import classification_metrics, full_report
from .model import MLPClassifierHead

VALID_CLASSIFIERS = ("xgboost", "logistic", "mlp")

class BaseClassifier(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

class XGBoostClassifier(BaseClassifier):
    def __init__(self, n_classes):
        self._n_classes = n_classes
        self._clf = None

    @property
    def name(self):
        return "xgboost"

    def _new_estimator(self):
        return XGBClassifier(

    objective="multi:softprob",
    num_class=self._n_classes,
    eval_metric="mlogloss",

    n_estimators=500,
    max_depth=7,
    learning_rate=0.03,

    subsample=0.9,
    colsample_bytree=0.9,

    min_child_weight=3,

    gamma=0.1,

    random_state=57,
)

    def fit(self, X, y):
        y = np.asarray(y, dtype=np.int64)
        X = np.asarray(X, dtype=np.float32)
        present = set(np.unique(y).tolist())
        missing = [c for c in range(self._n_classes) if c not in present]
        if missing:
            X_aug = np.vstack([X] + [np.zeros((1, X.shape[1]), dtype=X.dtype) for _ in missing])
            y_aug = np.concatenate([y] + [np.array([c], dtype=np.int64) for c in missing])
            sw = np.concatenate([
                np.ones(len(y), dtype=np.float32),
                np.zeros(len(missing), dtype=np.float32),
            ])
            self._clf = self._new_estimator()
            self._clf.fit(X_aug, y_aug, sample_weight=sw)
        else:
            self._clf = self._new_estimator()
            self._clf.fit(X, y)
        return self

    def predict(self, X):
        return self._clf.predict(X)

    def predict_proba(self, X):
        return self._clf.predict_proba(X)

class LogisticClassifier(BaseClassifier):
    def __init__(self, seed=57):
        self._seed = seed
        self._clf = None

    @property
    def name(self):
        return "logistic"

    def fit(self, X, y):
        self._clf = LogisticRegression(
            max_iter=1200, solver="lbfgs", C=1.0,
            multi_class="multinomial", random_state=self._seed,
        )
        self._clf.fit(X, np.asarray(y, dtype=np.int64))
        return self

    def predict(self, X):
        return self._clf.predict(X)

    def predict_proba(self, X):
        return self._clf.predict_proba(X)

class TorchMLPClassifier(BaseClassifier):
    def __init__(self, config):
        self.cfg = config
        self.device = torch.device(config.resolve_device())
        self._model = None
        self._n_classes = config.n_classes

    @property
    def name(self):
        return "mlp"

    def fit(self, X, y):
        input_dim = X.shape[1]
        classes = np.unique(y)
        self._n_classes = max(int(classes.max()) + 1, len(classes), self.cfg.n_classes)
        self._model = MLPClassifierHead(input_dim, self._n_classes).to(self.device)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.cfg.classifier_lr)
        criterion = nn.CrossEntropyLoss()

        Xtr = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        ytr = torch.as_tensor(y, dtype=torch.long, device=self.device)
        n = len(ytr)
        bs = max(self.cfg.batch_size, 256)

        self._model.train()
        for _ in range(self.cfg.classifier_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, bs):
                idx = perm[start:start + bs]
                logits, _ = self._model(Xtr[idx])
                loss = criterion(logits, ytr[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        self._model.eval()
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        Xt = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        logits, _ = self._model(Xt)
        return torch.softmax(logits, dim=1).cpu().numpy()

    @torch.no_grad()
    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

def create_classifier(config):
    classifier = config.classifier.lower()
    if classifier == "xgboost":
        return XGBoostClassifier(n_classes=config.n_classes)
    if classifier == "logistic":
        return LogisticClassifier(seed=config.seed)
    if classifier == "mlp":
        return TorchMLPClassifier(config)
    raise ValueError(f"classifier must be one of {VALID_CLASSIFIERS}, got {config.classifier!r}")

def train_classifier(config, X_train, y_train):
    """
    Train the downstream classifier and return the trained model.
    No evaluation is performed.
    """

    clf = create_classifier(config)

    clf.fit(
        np.asarray(X_train, dtype=np.float32),
        np.asarray(y_train, dtype=np.int64),
    )

    return clf

def train_and_evaluate(config, X_train, y_train, X_test, y_test):
    clf = create_classifier(config)
    clf.fit(
    np.asarray(X_train, dtype=np.float32),
    np.asarray(y_train, dtype=np.int64),)
    pred = clf.predict(X_test)
    metrics = classification_metrics(y_test, pred)
    if config.verbose:
        print(f"[classifier={clf.name}]")
        print(full_report(y_test, pred))
    return metrics, pred, clf
