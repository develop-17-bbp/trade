"""
LSTM/GRU Ensemble Forecaster
=================================
Ensemble of LSTM and GRU networks for short-term price direction prediction.
Uses majority voting across multiple architectures for robust signals.

Architecture:
  Model 1: LSTM (2 layers, 64 hidden) — captures long dependencies
  Model 2: GRU (2 layers, 64 hidden) — faster training, good for short sequences
  Model 3: BiLSTM (1 layer, 32 hidden) — bidirectional context

All models output 3-class probabilities: [SHORT, FLAT, LONG]
Final signal = weighted vote (equal weight unless calibrated).

Falls back to numpy-only statistical model if PyTorch unavailable.

Usage:
    from src.models.lstm_ensemble import LSTMEnsemble
    ensemble = LSTMEnsemble()
    ensemble.train(features, labels)
    signal = ensemble.predict(current_features)
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
    class LSTMModel(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 64,
                     num_layers: int = 2, num_classes: int = 3, dropout: float = 0.3):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                                batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_dim, num_classes)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.dropout(out[:, -1, :])  # Last timestep
            return self.fc(out)

    class GRUModel(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 64,
                     num_layers: int = 2, num_classes: int = 3, dropout: float = 0.3):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                              batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_dim, num_classes)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            out, _ = self.gru(x)
            out = self.dropout(out[:, -1, :])
            return self.fc(out)

    class BiLSTMModel(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 32,
                     num_classes: int = 3, dropout: float = 0.3):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, 1,
                                batch_first=True, bidirectional=True, dropout=0)
            self.fc = nn.Linear(hidden_dim * 2, num_classes)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.dropout(out[:, -1, :])
            return self.fc(out)


class LSTMEnsemble:
    """
    Ensemble of LSTM, GRU, BiLSTM for price direction forecasting.
    Falls back to statistical model when PyTorch unavailable.
    """

    def __init__(self, input_dim: int = 20, seq_len: int = 30,
                 model_dir: str = 'models/lstm_ensemble'):
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.model_dir = model_dir
        self.models = {}
        self.device = 'cpu'

        if HAS_TORCH:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._init_models()

    def _init_models(self):
        """Initialize ensemble members."""
        if not HAS_TORCH:
            return

        self.models = {
            'lstm': LSTMModel(self.input_dim).to(self.device),
            'gru': GRUModel(self.input_dim).to(self.device),
            'bilstm': BiLSTMModel(self.input_dim).to(self.device),
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

    def train(self, features: np.ndarray, labels: np.ndarray,
              epochs: int = 50, lr: float = 0.001, batch_size: int = 64,
              val_split: float = 0.15) -> Dict:
        """
        Train all ensemble members.

        Args:
            features: (N, seq_len, input_dim) feature sequences
            labels: (N,) class labels {0=SHORT, 1=FLAT, 2=LONG}
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

        # Convert to tensors
        train_x_t = torch.FloatTensor(train_x).to(self.device)
        train_y_t = torch.LongTensor(train_y).to(self.device)
        val_x_t = torch.FloatTensor(val_x).to(self.device)
        val_y_t = torch.LongTensor(val_y).to(self.device)

        results = {}

        for name, model in self.models.items():
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            criterion = nn.CrossEntropyLoss()
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=5, factor=0.5
            )

            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(epochs):
                # Mini-batch training
                model.train()
                epoch_loss = 0
                n_batches = 0

                indices = np.arange(len(train_x_t))
                # Shuffle within epoch (ok for training, temporal split already done)
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

                # Validation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(val_x_t)
                    val_loss = criterion(val_outputs, val_y_t).item()
                    val_preds = torch.argmax(val_outputs, dim=1)
                    val_acc = (val_preds == val_y_t).float().mean().item()

                scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    os.makedirs(self.model_dir, exist_ok=True)
                    torch.save(model.state_dict(),
                               os.path.join(self.model_dir, f'{name}.pth'))
                else:
                    patience_counter += 1
                    if patience_counter >= 10:
                        logger.info(f"{name}: Early stopping at epoch {epoch}")
                        break

            results[name] = {
                'best_val_loss': best_val_loss,
                'val_accuracy': val_acc,
                'epochs_trained': epoch + 1,
            }
            logger.info(f"{name}: val_loss={best_val_loss:.4f}, val_acc={val_acc:.3f}")

        return results

    def predict(self, features: np.ndarray) -> Dict:
        """
        Ensemble prediction with majority voting.

        Args:
            features: (seq_len, input_dim) or (1, seq_len, input_dim)

        Returns:
            {signal: int, confidence: float, model_votes: dict, probs: list}
        """
        if not HAS_TORCH or not self.models:
            return self._statistical_fallback(features)

        # Ensure correct shape
        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 2:
            x = x[np.newaxis, :]  # Add batch dimension
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
                    # Map: 0=SHORT, 1=FLAT, 2=LONG → -1, 0, +1
                    signal_map = {0: -1, 1: 0, 2: 1}
                    model_votes[name] = {
                        'signal': signal_map[pred_class],
                        'confidence': float(probs[pred_class]),
                        'probs': probs.tolist(),
                    }
            except Exception as e:
                logger.warning(f"Model {name} prediction failed: {e}")

        if not all_probs:
            return self._statistical_fallback(features)

        # Average probabilities across ensemble
        avg_probs = np.mean(all_probs, axis=0)
        final_class = int(np.argmax(avg_probs))
        signal_map = {0: -1, 1: 0, 2: 1}

        return {
            'signal': signal_map[final_class],
            'confidence': float(avg_probs[final_class]),
            'probs': avg_probs.tolist(),
            'model_votes': model_votes,
            'method': 'lstm_ensemble',
        }

    def _statistical_fallback(self, features: np.ndarray) -> Dict:
        """
        Numpy-only fallback when PyTorch is unavailable.
        Uses momentum + mean-reversion signals from raw features.
        """
        x = np.asarray(features, dtype=float)

        # If 3D, take last sequence
        if x.ndim == 3:
            x = x[0]
        if x.ndim == 2:
            # Use first feature column as price proxy
            prices = x[:, 0] if x.shape[1] > 0 else x[:, 0]
        else:
            prices = x

        if len(prices) < 10:
            return {'signal': 0, 'confidence': 0.5, 'probs': [0.33, 0.34, 0.33],
                    'model_votes': {}, 'method': 'statistical_fallback'}

        # Simple momentum + mean reversion
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
            'probs': [max(0, -score) / 2 + 0.17,
                      0.33 - abs(score) / 3,
                      max(0, score) / 2 + 0.17],
            'model_votes': {},
            'method': 'statistical_fallback',
        }

    def prepare_sequences(self, feature_matrix: np.ndarray, labels: np.ndarray,
                          seq_len: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert flat feature matrix to sequences for LSTM input.

        Args:
            feature_matrix: (N, n_features) raw features
            labels: (N,) labels

        Returns:
            (sequences, seq_labels) of shape (N-seq_len, seq_len, n_features), (N-seq_len,)
        """
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
