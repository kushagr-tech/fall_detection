"""
models/neural_net.py
====================
Lightweight 1-D CNN + LSTM hybrid suitable for on-device inference.

Architecture
------------
Input  : (batch, window_size=100, channels=6)
│
├── Conv1d(6→32, k=5)  + BatchNorm + ReLU + MaxPool(2)    → (batch, 48, 32)
├── Conv1d(32→64, k=3) + BatchNorm + ReLU + MaxPool(2)    → (batch, 23, 64)
├── LSTM(64→64, layers=1, bidirectional=False)             → last hidden (batch, 64)
├── Dropout(0.3)
└── Linear(64 → n_classes)

Parameter count ≈ 37 k — small enough to run on any modern Android phone via
PyTorch Mobile (or ONNX Runtime if exported).

Training strategy
-----------------
* Focal loss (γ=2) instead of cross-entropy to further boost recall on the
  rare 'falling' class.
* class_weights are used as α in focal loss for double protection.
* AdamW + cosine annealing schedule.
* Early stopping on validation macro-F1 (not accuracy).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

log = logging.getLogger(__name__)

# ── Hyperparameters ────────────────────────────────────────────────────────────

DEFAULT_HPARAMS = dict(
    epochs        = 60,
    batch_size    = 64,
    lr            = 1e-3,
    weight_decay  = 1e-3,
    focal_gamma   = 2.0,
    dropout       = 0.35,
    patience      = 12,    # early-stopping patience (epochs)
)


# ── Model definition ───────────────────────────────────────────────────────────

class HARNet(nn.Module):
    """1-D CNN + LSTM for time-series HAR / fall detection."""

    def __init__(self, n_classes: int, n_channels: int = 6, dropout: float = 0.35):
        super().__init__()

        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout1d(0.15),
            nn.MaxPool1d(kernel_size=2),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout1d(0.15),
            nn.MaxPool1d(kernel_size=2),
        )

        self.lstm = nn.LSTM(
            input_size=64, hidden_size=64,
            num_layers=1, batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, time, channels)  → CNN wants (batch, channels, time)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        # Back to (batch, time, features) for LSTM
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]              # last layer's hidden state (batch, 64)
        return self.classifier(x)


# ── Focal loss ─────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Multi-class focal loss with per-class alpha weighting:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - pt) ** self.gamma
        loss = -focal_weight * log_pt

        if self.alpha is not None:
            at = self.alpha.gather(0, targets)
            loss = at * loss

        return loss.mean()


# ── Trainer ────────────────────────────────────────────────────────────────────

class NeuralNetModel:
    """
    Wraps HARNet with training loop, early stopping, and serialisation.
    The model operates on raw windows (no feature extraction needed).
    """

    def __init__(self, hparams: dict | None = None):
        self.hparams = {**DEFAULT_HPARAMS, **(hparams or {})}
        self.label_encoder = LabelEncoder()
        self.model: HARNet | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._is_fitted = False

    # ── Training ───────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray | None = None,
        y_val:   np.ndarray | None = None,
    ) -> "NeuralNetModel":
        """
        Parameters
        ----------
        X_train : (n, window_size, 6)
        y_train : (n,) string labels
        X_val   : optional validation set for early stopping
        y_val   : optional validation labels
        """
        hp = self.hparams

        y_enc = self.label_encoder.fit_transform(y_train)
        n_classes = len(self.label_encoder.classes_)

        # Class weights for focal loss α
        class_counts = np.bincount(y_enc, minlength=n_classes).astype(np.float32)
        class_weights = (class_counts.sum() / (n_classes * class_counts))
        alpha = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        self.model = HARNet(
            n_classes=n_classes,
            n_channels=X_train.shape[2],
            dropout=hp["dropout"],
        ).to(self.device)

        criterion = FocalLoss(alpha=alpha, gamma=hp["focal_gamma"])
        optimizer = optim.AdamW(
            self.model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"]
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=hp["epochs"]
        )

        train_loader = self._make_loader(X_train, y_enc, shuffle=True)

        best_val_f1   = -1.0
        best_state    = None
        patience_cnt  = 0

        for epoch in range(1, hp["epochs"] + 1):
            train_loss = self._train_epoch(train_loader, criterion, optimizer)
            scheduler.step()

            if X_val is not None and y_val is not None:
                val_f1 = self._val_f1(X_val, y_val)
                if val_f1 >= best_val_f1:
                    best_val_f1  = val_f1
                    best_state   = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_cnt = 0
                else:
                    patience_cnt += 1

                if epoch % 10 == 0 or (epoch >= 25 and patience_cnt >= hp["patience"]):
                    log.info(
                        "Epoch %d/%d  loss=%.4f  val_macro_f1=%.4f  patience=%d",
                        epoch, hp["epochs"], train_loss, val_f1, patience_cnt,
                    )

                if epoch >= 25 and patience_cnt >= hp["patience"]:
                    log.info("Early stopping at epoch %d.", epoch)
                    break
            else:
                if epoch % 10 == 0:
                    log.info("Epoch %d/%d  loss=%.4f", epoch, hp["epochs"], train_loss)

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self._is_fitted = True
        return self

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns string labels."""
        self._check_fitted()
        logits = self._forward(X)
        y_enc = logits.argmax(dim=1).cpu().numpy()
        return self.label_encoder.inverse_transform(y_enc)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        logits = self._forward(X)
        return torch.softmax(logits, dim=1).cpu().numpy()

    @property
    def classes_(self) -> np.ndarray:
        return self.label_encoder.classes_

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "hparams":    self.hparams,
                "le":         self.label_encoder,
                "n_classes":  len(self.label_encoder.classes_),
                "n_channels": 6,
            },
            path,
        )
        log.info("NN model saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "NeuralNetModel":
        obj = cls.__new__(cls)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        obj.hparams       = saved["hparams"]
        obj.label_encoder = saved["le"]
        obj.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obj.model         = HARNet(
            n_classes=saved["n_classes"],
            n_channels=saved["n_channels"],
            dropout=saved["hparams"]["dropout"],
        ).to(obj.device)
        obj.model.load_state_dict(saved["state_dict"])
        obj.model.eval()
        obj._is_fitted = True
        return obj

    # ── TorchScript / mobile export ────────────────────────────────────────────

    def export_torchscript(self, path: str | Path, window_size: int = 100) -> None:
        """Export for PyTorch Mobile (.ptl)."""
        self._check_fitted()
        dummy = torch.zeros(1, window_size, 6)
        scripted = torch.jit.trace(self.model.cpu(), dummy)
        if hasattr(scripted, "_save_for_lite_interpreter"):
            scripted._save_for_lite_interpreter(str(path))
        else:
            scripted.save(str(path))
        log.info("TorchScript mobile model saved → %s", path)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _make_loader(
        self, X: np.ndarray, y_enc: np.ndarray, shuffle: bool
    ) -> DataLoader:
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y_enc, dtype=torch.long)
        return DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.hparams["batch_size"],
            shuffle=shuffle,
        )

    def _augment(self, X_batch: torch.Tensor) -> torch.Tensor:
        """
        Time-series sensor data augmentation:
          1. Jitter (subtle Gaussian noise)
          2. Random amplitude scaling per channel
          3. Occasional channel dropout
        """
        if not self.model.training:
            return X_batch
        noise = torch.randn_like(X_batch) * 0.02
        X_aug = X_batch + noise
        scale = torch.rand(X_batch.size(0), 1, X_batch.size(2), device=X_batch.device) * 0.14 + 0.93
        X_aug = X_aug * scale
        mask = (torch.rand(X_batch.size(0), 1, X_batch.size(2), device=X_batch.device) > 0.05).float()
        return X_aug * mask

    def _train_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
    ) -> float:
        self.model.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            X_batch = self._augment(X_batch)
            optimizer.zero_grad()
            loss = criterion(self.model(X_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(X_batch)
        return total_loss / len(loader.dataset)

    def _forward(self, X: np.ndarray) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            return self.model(X_t)

    def _val_f1(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        from sklearn.metrics import f1_score
        logits = self._forward(X_val)
        y_enc  = logits.argmax(dim=1).cpu().numpy()
        y_pred = self.label_encoder.inverse_transform(y_enc)
        return float(f1_score(y_val, y_pred, average="macro", zero_division=0))

    def _check_fitted(self) -> None:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
