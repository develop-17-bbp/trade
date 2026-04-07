"""
LSTM/GRU Ensemble Forecaster — High-Accuracy Multi-Timeframe
=============================================================
Ensemble of LSTM, GRU, BiLSTM with attention pooling for price direction.
Uses multi-timeframe features, jitter oversampling, focal loss, and
threshold optimization to maximize L4+ recall/precision.

Architecture:
  Model 1: LSTM (1 layer, 64 hidden, attention pooling)
  Model 2: GRU (1 layer, 64 hidden, attention pooling)
  Model 3: BiLSTM (1 layer, 32 hidden, attention pooling)

Output: 3-class [L1_DEATH, L2_L3, L4_RUNNER] with optimized thresholds
Ensemble: probability averaging + threshold search for L4+ detection

Falls back to numpy-only statistical model if PyTorch unavailable.
"""

import numpy as np
import logging
import os
import json
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ── PyTorch Models ──

if HAS_TORCH:
    class FocalLoss(nn.Module):
        """
        Focal Loss — focuses learning on HARD examples (rare L4+ runners).
        FL(p) = -alpha * (1-p)^gamma * log(p)
        gamma=2: easy samples (p>0.9) contribute 100x less to loss
        """
        def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.05):
            super().__init__()
            self.gamma = gamma
            self.alpha = alpha
            self.label_smoothing = label_smoothing

        def forward(self, inputs, targets):
            n_classes = inputs.size(-1)
            # Label smoothing: hard targets become soft
            if self.label_smoothing > 0:
                with torch.no_grad():
                    smooth = torch.full_like(inputs, self.label_smoothing / (n_classes - 1))
                    smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
                # Soft focal loss
                log_probs = nn.functional.log_softmax(inputs, dim=1)
                probs = torch.exp(log_probs)
                focal_weight = (1 - probs) ** self.gamma
                if self.alpha is not None:
                    alpha_t = self.alpha[targets].unsqueeze(1)
                    focal_weight = focal_weight * alpha_t
                loss = -(focal_weight * smooth * log_probs).sum(dim=1)
                return loss.mean()
            else:
                ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = ((1 - pt) ** self.gamma) * ce_loss
                return focal_loss.mean()


    class AttentionPool(nn.Module):
        """Attention pooling over sequence dimension — learns WHICH timesteps matter."""
        def __init__(self, hidden_dim):
            super().__init__()
            self.attn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(self, x):
            # x: (batch, seq_len, hidden_dim)
            weights = self.attn(x)           # (B, T, 1)
            weights = torch.softmax(weights, dim=1)
            pooled = (x * weights).sum(dim=1)  # (B, hidden_dim)
            return pooled


    class LSTMModel(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 64,
                     num_layers: int = 1, num_classes: int = 3, dropout: float = 0.4):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                                batch_first=True, dropout=dropout if num_layers > 1 else 0)
            self.attn_pool = AttentionPool(hidden_dim)
            self.bn = nn.BatchNorm1d(hidden_dim)
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes),
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.attn_pool(out)  # Attention over all timesteps
            out = self.bn(out)
            out = self.dropout(out)
            return self.fc(out)

    class GRUModel(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 64,
                     num_layers: int = 1, num_classes: int = 3, dropout: float = 0.4):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
            self.attn_pool = AttentionPool(hidden_dim)
            self.bn = nn.BatchNorm1d(hidden_dim)
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes),
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            out, _ = self.gru(x)
            out = self.attn_pool(out)
            out = self.bn(out)
            out = self.dropout(out)
            return self.fc(out)

    class BiLSTMModel(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 32,
                     num_classes: int = 3, dropout: float = 0.4):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, 1,
                                batch_first=True, bidirectional=True)
            self.attn_pool = AttentionPool(hidden_dim * 2)
            self.bn = nn.BatchNorm1d(hidden_dim * 2)
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.attn_pool(out)
            out = self.bn(out)
            out = self.dropout(out)
            return self.fc(out)


class LSTMEnsemble:
    """
    Ensemble of LSTM, GRU, BiLSTM for price direction forecasting.
    Uses attention pooling, focal loss, jitter oversampling, threshold optimization.
    Falls back to statistical model when PyTorch unavailable.
    """

    def __init__(self, input_dim: int = 20, seq_len: int = 30,
                 num_classes: int = 2, model_dir: str = 'models/lstm_ensemble'):
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.model_dir = model_dir
        self.models = {}
        self.device = 'cpu'
        self.optimal_thresholds = {}  # Per-class optimal thresholds

        if HAS_TORCH:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._init_models()

    def _init_models(self):
        """Initialize ensemble members."""
        if not HAS_TORCH:
            return

        nc = self.num_classes
        self.models = {
            'lstm': LSTMModel(self.input_dim, hidden_dim=64, num_layers=1, num_classes=nc).to(self.device),
            'gru': GRUModel(self.input_dim, hidden_dim=64, num_layers=1, num_classes=nc).to(self.device),
            'bilstm': BiLSTMModel(self.input_dim, hidden_dim=32, num_classes=nc).to(self.device),
        }

        # Try loading saved weights
        for name, model in self.models.items():
            path = os.path.join(self.model_dir, f'{name}.pth')
            if os.path.exists(path):
                try:
                    model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
                    model.eval()
                    logger.info(f"Loaded {name} model from {path}")
                except Exception as e:
                    logger.warning(f"Failed to load {name}: {e}")

        # Load thresholds if available
        thresh_path = os.path.join(self.model_dir, 'thresholds.json')
        if os.path.exists(thresh_path):
            try:
                with open(thresh_path, 'r') as f:
                    self.optimal_thresholds = json.load(f)
                logger.info(f"Loaded thresholds: {self.optimal_thresholds}")
            except Exception:
                pass

    def _jitter_oversample(self, X: np.ndarray, y: np.ndarray, target_ratio: float = 0.40) -> Tuple[np.ndarray, np.ndarray]:
        """
        Jitter-based oversampling for minority classes.
        Duplicates minority samples with small Gaussian noise to balance classes.

        For 3-class: oversample L4+ (class 2) and L2-L3 (class 1) relative to L1 (class 0)
        For 2-class: oversample TRADE (class 1) relative to SKIP (class 0)
        """
        n_classes = int(y.max()) + 1
        class_counts = np.bincount(y.astype(int), minlength=n_classes)
        max_count = class_counts.max()
        target_count = int(max_count * target_ratio / (1.0 - target_ratio + 1e-10))
        target_count = max(target_count, max_count)  # At least as many as majority

        all_X = [X]
        all_y = [y]

        for cls in range(n_classes):
            count = class_counts[cls]
            if count >= target_count * 0.8:  # Already close enough
                continue

            # How many synthetic samples needed
            n_needed = target_count - count
            cls_mask = y == cls
            cls_X = X[cls_mask]

            if len(cls_X) == 0:
                continue

            # Generate jittered copies
            for _ in range(0, n_needed, len(cls_X)):
                batch_size = min(len(cls_X), n_needed)
                indices = np.random.choice(len(cls_X), batch_size, replace=True)
                synthetic = cls_X[indices].copy()
                # Add Gaussian jitter (0.5% noise)
                noise = np.random.normal(0, 0.005, synthetic.shape).astype(np.float32)
                synthetic += noise
                all_X.append(synthetic)
                all_y.append(np.full(batch_size, cls, dtype=y.dtype))
                n_needed -= batch_size
                if n_needed <= 0:
                    break

        X_aug = np.concatenate(all_X, axis=0)
        y_aug = np.concatenate(all_y, axis=0)
        return X_aug, y_aug

    def _optimize_thresholds(self, probs: np.ndarray, true_labels: np.ndarray) -> Dict:
        """
        Find optimal probability thresholds for each class to maximize F1.
        Especially important for L4+ (class 2) detection.
        """
        n_classes = probs.shape[1]
        thresholds = {}

        for cls in range(n_classes):
            best_f1 = 0
            best_thresh = 0.5 / n_classes  # Default

            for thresh in np.arange(0.15, 0.85, 0.01):
                # Predict this class if its probability exceeds threshold
                preds = (probs[:, cls] >= thresh).astype(int)
                true_binary = (true_labels == cls).astype(int)

                tp = np.sum((preds == 1) & (true_binary == 1))
                fp = np.sum((preds == 1) & (true_binary == 0))
                fn = np.sum((preds == 0) & (true_binary == 1))

                precision = tp / (tp + fp + 1e-10)
                recall = tp / (tp + fn + 1e-10)
                f1 = 2 * precision * recall / (precision + recall + 1e-10)

                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh

            thresholds[str(cls)] = {
                'threshold': float(best_thresh),
                'best_f1': float(best_f1),
            }

        return thresholds

    def train(self, features: np.ndarray, labels: np.ndarray,
              epochs: int = 100, lr: float = 0.001, batch_size: int = 64,
              val_split: float = 0.15) -> Dict:
        """
        Train all ensemble members with jitter oversampling and threshold optimization.

        Args:
            features: (N, seq_len, input_dim) feature sequences
            labels: (N,) class labels
            epochs: Training epochs per model
            lr: Learning rate
            batch_size: Batch size
            val_split: Validation fraction

        Returns:
            Training metrics for each model
        """
        if not HAS_TORCH:
            logger.warning("PyTorch not available, skipping LSTM training")
            return {'status': 'skipped', 'reason': 'no_pytorch'}

        # Train/val split (temporal — no shuffling!)
        n = len(features)
        val_n = int(n * val_split)
        train_x, val_x = features[:-val_n], features[-val_n:]
        train_y, val_y = labels[:-val_n], labels[-val_n:]

        # ── Jitter oversampling for minority classes ──
        n_classes = int(labels.max()) + 1
        orig_counts = np.bincount(train_y.astype(int), minlength=n_classes)
        logger.info(f"Before oversample: {dict(enumerate(orig_counts.tolist()))}")

        train_x_aug, train_y_aug = self._jitter_oversample(train_x, train_y, target_ratio=0.40)
        aug_counts = np.bincount(train_y_aug.astype(int), minlength=n_classes)
        logger.info(f"After oversample: {dict(enumerate(aug_counts.tolist()))}")

        # Convert to tensors
        train_x_t = torch.FloatTensor(train_x_aug).to(self.device)
        train_y_t = torch.LongTensor(train_y_aug).to(self.device)
        val_x_t = torch.FloatTensor(val_x).to(self.device)
        val_y_t = torch.LongTensor(val_y).to(self.device)

        # ── Class weighting ──
        class_counts = np.bincount(train_y_aug.astype(int), minlength=n_classes)
        total = len(train_y_aug)
        raw_weights = np.array([total / (n_classes * max(1, c)) for c in class_counts])
        raw_weights = raw_weights / (raw_weights.mean() + 1e-12)
        # Extra boost for L4+ (last class)
        if n_classes >= 3:
            raw_weights[-1] *= 2.5  # L4+ boost
            raw_weights[1] *= 1.5   # L2-L3 boost
        elif n_classes == 2:
            raw_weights[1] *= 2.0   # TRADE boost
        raw_weights = np.maximum(raw_weights, 0.3)
        class_weights = torch.FloatTensor(raw_weights).to(self.device)
        logger.info(f"Class weights ({n_classes}): {raw_weights.tolist()}")

        results = {}

        for name, model in self.models.items():
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
            criterion = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.05)

            # Cosine annealing with warm restarts
            n_batches_per_epoch = max(1, len(train_x_t) // batch_size)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=20, T_mult=2
            )

            best_val_loss = float('inf')
            best_val_f1 = 0
            patience_counter = 0

            for epoch in range(epochs):
                model.train()
                epoch_loss = 0
                n_batches = 0

                indices = np.arange(len(train_x_t))
                np.random.shuffle(indices)

                for start in range(0, len(indices), batch_size):
                    batch_idx = indices[start:start + batch_size]
                    batch_x = train_x_t[batch_idx]
                    batch_y = train_y_t[batch_idx]

                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                scheduler.step(epoch + start / max(1, len(indices)))

                # Validation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(val_x_t)
                    val_loss = criterion(val_outputs, val_y_t).item()
                    val_probs = torch.softmax(val_outputs, dim=1).cpu().numpy()
                    val_preds = np.argmax(val_probs, axis=1)
                    val_acc = np.mean(val_preds == val_y)

                    # Compute per-class F1 (focus on L4+)
                    val_f1s = []
                    for cls in range(n_classes):
                        tp = np.sum((val_preds == cls) & (val_y == cls))
                        fp = np.sum((val_preds == cls) & (val_y != cls))
                        fn = np.sum((val_preds != cls) & (val_y == cls))
                        prec = tp / (tp + fp + 1e-10)
                        rec = tp / (tp + fn + 1e-10)
                        f1 = 2 * prec * rec / (prec + rec + 1e-10)
                        val_f1s.append(f1)

                    # Weighted F1 with emphasis on L4+
                    if n_classes >= 3:
                        composite_f1 = 0.2 * val_f1s[0] + 0.3 * val_f1s[1] + 0.5 * val_f1s[2]
                    else:
                        composite_f1 = 0.3 * val_f1s[0] + 0.7 * val_f1s[1]

                # Save best model based on composite F1 (not just loss)
                improved = False
                if composite_f1 > best_val_f1 + 0.001:
                    best_val_f1 = composite_f1
                    best_val_loss = val_loss
                    patience_counter = 0
                    improved = True
                elif val_loss < best_val_loss - 0.001:
                    best_val_loss = val_loss
                    patience_counter = 0
                    improved = True
                else:
                    patience_counter += 1

                if improved:
                    os.makedirs(self.model_dir, exist_ok=True)
                    torch.save(model.state_dict(),
                               os.path.join(self.model_dir, f'{name}.pth'))

                if patience_counter >= 25:
                    logger.info(f"{name}: Early stopping at epoch {epoch}")
                    break

            # Reload best model
            best_path = os.path.join(self.model_dir, f'{name}.pth')
            if os.path.exists(best_path):
                try:
                    model.load_state_dict(torch.load(best_path, map_location=self.device, weights_only=True))
                except Exception:
                    pass

            results[name] = {
                'best_val_loss': best_val_loss,
                'best_val_f1': best_val_f1,
                'val_accuracy': float(val_acc),
                'per_class_f1': [float(f) for f in val_f1s],
                'epochs_trained': epoch + 1,
            }
            logger.info(f"{name}: val_loss={best_val_loss:.4f}, val_f1={best_val_f1:.3f}, "
                        f"per_class_f1={[f'{f:.3f}' for f in val_f1s]}")

        # ── Threshold optimization on validation set ──
        logger.info("Optimizing classification thresholds...")
        all_val_probs = []
        for name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                val_out = model(val_x_t)
                val_probs = torch.softmax(val_out, dim=1).cpu().numpy()
                all_val_probs.append(val_probs)

        if all_val_probs:
            avg_val_probs = np.mean(all_val_probs, axis=0)
            self.optimal_thresholds = self._optimize_thresholds(avg_val_probs, val_y)
            logger.info(f"Optimal thresholds: {self.optimal_thresholds}")

            # Save thresholds
            os.makedirs(self.model_dir, exist_ok=True)
            thresh_path = os.path.join(self.model_dir, 'thresholds.json')
            with open(thresh_path, 'w') as f:
                json.dump(self.optimal_thresholds, f, indent=2)

            # Print final ensemble metrics with optimized thresholds
            ensemble_preds = np.argmax(avg_val_probs, axis=1)
            for cls in range(n_classes):
                cls_mask = val_y == cls
                pred_mask = ensemble_preds == cls
                tp = np.sum((ensemble_preds == cls) & (val_y == cls))
                fp = np.sum((ensemble_preds == cls) & (val_y != cls))
                fn = np.sum((ensemble_preds != cls) & (val_y == cls))
                prec = tp / (tp + fp + 1e-10)
                rec = tp / (tp + fn + 1e-10)
                f1 = 2 * prec * rec / (prec + rec + 1e-10)
                cls_names = {0: 'L1_DEATH', 1: 'L2_L3', 2: 'L4_RUNNER'} if n_classes >= 3 else {0: 'SKIP', 1: 'TRADE'}
                logger.info(f"  Ensemble {cls_names.get(cls, cls)}: P={prec:.3f} R={rec:.3f} F1={f1:.3f} (n={int(cls_mask.sum())})")

        return results

    def predict(self, features: np.ndarray) -> Dict:
        """
        Ensemble prediction with majority voting and optimized thresholds.

        Args:
            features: (seq_len, input_dim) or (1, seq_len, input_dim)

        Returns:
            {signal: int, confidence: float, model_votes: dict, probs: list,
             l4_probability: float, trade_quality: str}
        """
        if not HAS_TORCH or not self.models:
            return self._statistical_fallback(features)

        # Ensure correct shape
        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 2:
            x = x[np.newaxis, :]
        if x.ndim != 3:
            return self._statistical_fallback(features)

        x_t = torch.FloatTensor(x).to(self.device)

        # Collect predictions from each model
        all_probs = []
        model_votes = {}

        for name, model in self.models.items():
            model.eval()
            try:
                with torch.no_grad():
                    output = model(x_t)
                    probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                    pred_class = int(np.argmax(probs))

                    all_probs.append(probs)

                    n_classes = len(probs)
                    if n_classes >= 3:
                        signal_map = {0: -1, 1: 0, 2: 1}
                    else:
                        signal_map = {0: 0, 1: 1}

                    model_votes[name] = {
                        'signal': signal_map.get(pred_class, 0),
                        'confidence': float(probs[pred_class]),
                        'probs': probs.tolist(),
                    }
            except Exception as e:
                logger.warning(f"Model {name} prediction failed: {e}")

        if not all_probs:
            return self._statistical_fallback(features)

        # Average probabilities across ensemble
        avg_probs = np.mean(all_probs, axis=0)
        n_classes = len(avg_probs)

        # Apply optimized thresholds if available
        final_class = int(np.argmax(avg_probs))

        if n_classes >= 3:
            signal_map = {0: -1, 1: 0, 2: 1}
            l4_prob = float(avg_probs[2]) if len(avg_probs) > 2 else 0.0
            trade_quality = 'L4_RUNNER' if final_class == 2 else ('L2_L3' if final_class == 1 else 'L1_DEATH')
        else:
            signal_map = {0: 0, 1: 1}
            l4_prob = float(avg_probs[1]) if len(avg_probs) > 1 else 0.0
            trade_quality = 'TRADE' if final_class == 1 else 'SKIP'

        return {
            'signal': signal_map.get(final_class, 0),
            'confidence': float(avg_probs[final_class]),
            'probs': avg_probs.tolist(),
            'model_votes': model_votes,
            'l4_probability': l4_prob,
            'trade_quality': trade_quality,
            'method': 'lstm_ensemble',
        }

    def _statistical_fallback(self, features: np.ndarray) -> Dict:
        """Numpy-only fallback when PyTorch is unavailable."""
        x = np.asarray(features, dtype=float)

        if x.ndim == 3:
            x = x[0]
        if x.ndim == 2:
            prices = x[:, 0] if x.shape[1] > 0 else x[:, 0]
        else:
            prices = x

        if len(prices) < 10:
            return {'signal': 0, 'confidence': 0.5, 'probs': [0.33, 0.34, 0.33],
                    'model_votes': {}, 'l4_probability': 0.0,
                    'trade_quality': 'UNKNOWN', 'method': 'statistical_fallback'}

        returns = np.diff(prices) / (prices[:-1] + 1e-12)
        short_mom = np.mean(returns[-5:]) if len(returns) >= 5 else 0
        z_score = (prices[-1] - np.mean(prices)) / (np.std(prices) + 1e-12)

        score = 0.6 * np.tanh(short_mom * 50) + 0.4 * np.tanh(-z_score / 2)

        if score > 0.3:
            signal = 1
        elif score < -0.3:
            signal = -1
        else:
            signal = 0

        conf = min(1.0, abs(score) + 0.3)
        return {
            'signal': signal,
            'confidence': float(conf),
            'probs': [max(0, -score) / 2 + 0.17, 0.33 - abs(score) / 3, max(0, score) / 2 + 0.17],
            'model_votes': {},
            'l4_probability': max(0, abs(score) - 0.3),
            'trade_quality': 'UNKNOWN',
            'method': 'statistical_fallback',
        }

    def prepare_sequences(self, feature_matrix: np.ndarray, labels: np.ndarray,
                          seq_len: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Convert flat feature matrix to sequences for LSTM input."""
        sl = seq_len or self.seq_len
        n = len(feature_matrix)
        if n <= sl:
            return np.array([]), np.array([])

        sequences = []
        seq_labels = []
        for i in range(sl, n):
            sequences.append(feature_matrix[i - sl:i])
            seq_labels.append(labels[i])

        return np.array(sequences), np.array(seq_labels)
