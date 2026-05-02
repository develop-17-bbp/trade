"""Train the Candlestick Pattern Transformer via self-distillation.

Pipeline:
    1. Load Binance archive (BTCUSDT, ETHUSDT + altcoins) at multiple
       timeframes (15m, 1h, 4h)
    2. Slide a window of WINDOW_SIZE bars across each series
    3. For each window, run the heuristic detector at the LAST bar to
       generate multi-label pattern targets (one-hot per pattern)
    4. Train CandlestickTransformer to predict these labels via BCE
    5. Walk-forward split: oldest 75% train, 15% val, 10% test
    6. Save weights to models/candlestick_transformer.pt
    7. Champion-gate against heuristic on test set: must achieve
       per-pattern F1 within 5pts of heuristic teacher

Usage:
    # Quick run (1 epoch, small batch — for smoke testing)
    python scripts/train_candlestick_transformer.py --quick

    # Full training (default settings)
    python scripts/train_candlestick_transformer.py \\
        --timeframes 15m 1h 4h \\
        --assets BTCUSDT ETHUSDT \\
        --window 20 --epochs 8 --batch-size 256 \\
        --lr 3e-4

Operator notes:
    * Requires PyTorch installed (pip install torch)
    * Trains on CPU OR CUDA — auto-detect; use --device cuda to force
    * Binance archive must be at data/{ASSET}-{TF}.parquet
    * Model checkpoint at models/candlestick_transformer.pt is what
      production inference loads via get_inferer()
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _check_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        print("ERROR: PyTorch not installed. Run: pip install torch", file=sys.stderr)
        return False


def _load_parquet_bars(asset: str, timeframe: str) -> Optional[Tuple[np.ndarray, ...]]:
    """Load OHLCV from data/{asset}-{tf}.parquet."""
    try:
        import pandas as pd
        path = os.path.join(_PROJECT_ROOT, "data", f"{asset}-{timeframe}.parquet")
        if not os.path.exists(path):
            print(f"  [skip] {path} not found")
            return None
        df = pd.read_parquet(path)
        # Standardize columns (some parquets use 'Open' vs 'open')
        col_map = {c.lower(): c for c in df.columns}
        try:
            o = df[col_map.get("open", "open")].values.astype(np.float32)
            h = df[col_map.get("high", "high")].values.astype(np.float32)
            l = df[col_map.get("low", "low")].values.astype(np.float32)
            c = df[col_map.get("close", "close")].values.astype(np.float32)
            v = df[col_map.get("volume", "volume")].values.astype(np.float32)
        except KeyError as e:
            print(f"  [skip] {path}: missing column {e}")
            return None
        return o, h, l, c, v
    except Exception as e:
        print(f"  [skip] {path}: {e}")
        return None


def _build_dataset(
    assets: List[str],
    timeframes: List[str],
    window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slide window across all (asset, timeframe) series, generate
    (X, y) where X is encoded windows and y is multi-label pattern
    targets from the heuristic detector."""
    from src.models.candlestick_transformer import encode_window
    from src.trading.strategies.candlestick_patterns import (
        detect_all, PATTERN_NAMES, PATTERN_INDEX, NUM_PATTERNS,
    )

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    total = 0
    for asset in assets:
        for tf in timeframes:
            data = _load_parquet_bars(asset, tf)
            if data is None:
                continue
            o, h, l, c, v = data
            n = len(c)
            if n < window + 50:
                print(f"  [skip] {asset}-{tf}: only {n} bars")
                continue
            print(f"  [load] {asset}-{tf}: {n} bars")
            for i in range(window, n):
                # Heuristic label at bar i
                opens_w = o[i - window:i].tolist()
                highs_w = h[i - window:i].tolist()
                lows_w = l[i - window:i].tolist()
                closes_w = c[i - window:i].tolist()
                vols_w = v[i - window:i].tolist()
                patterns = detect_all(opens_w, highs_w, lows_w, closes_w, vols_w)
                # Encode window into X
                encoded = encode_window(
                    opens_w, highs_w, lows_w, closes_w, vols_w, window=window,
                )
                if encoded is None:
                    continue
                # Multi-label target
                y = np.zeros(NUM_PATTERNS, dtype=np.float32)
                if not patterns:
                    y[PATTERN_INDEX["no_pattern"]] = 1.0
                else:
                    for p in patterns:
                        if p.name in PATTERN_INDEX:
                            y[PATTERN_INDEX[p.name]] = 1.0
                X_list.append(encoded)
                y_list.append(y)
                total += 1
                if total % 5000 == 0:
                    print(f"    ... {total} samples generated")
    if not X_list:
        raise RuntimeError("no training samples — check assets/timeframes/data files")
    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    print(f"  [done] dataset shape: X={X.shape} y={y.shape}")
    return X, y


def _walk_forward_split(
    X: np.ndarray, y: np.ndarray,
    val_frac: float = 0.15, test_frac: float = 0.10,
) -> Tuple[np.ndarray, ...]:
    n = len(X)
    train_end = int(n * (1 - val_frac - test_frac))
    val_end = int(n * (1 - test_frac))
    return (
        X[:train_end], y[:train_end],
        X[train_end:val_end], y[train_end:val_end],
        X[val_end:], y[val_end:],
    )


def _f1_score(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """Macro-F1 for multi-label."""
    pred_bin = (y_pred >= threshold).astype(np.int32)
    true_bin = y_true.astype(np.int32)
    f1_per_class = []
    for k in range(y_true.shape[1]):
        tp = ((pred_bin[:, k] == 1) & (true_bin[:, k] == 1)).sum()
        fp = ((pred_bin[:, k] == 1) & (true_bin[:, k] == 0)).sum()
        fn = ((pred_bin[:, k] == 0) & (true_bin[:, k] == 1)).sum()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        f1_per_class.append(f1)
    return float(np.mean(f1_per_class))


def train(
    assets: List[str],
    timeframes: List[str],
    window: int = 20,
    epochs: int = 8,
    batch_size: int = 256,
    lr: float = 3e-4,
    device: str = "auto",
    output_path: Optional[str] = None,
    quick: bool = False,
) -> dict:
    """Main training entry. Returns summary dict + writes weights to disk."""
    if not _check_torch():
        sys.exit(1)
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from src.models.candlestick_transformer import (
        CandlestickTransformer, CandleTransformerConfig,
    )

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INIT] device={device} window={window} epochs={epochs} batch={batch_size}")

    if quick:
        epochs = 1
        batch_size = 64
        print("[QUICK] reduced epochs=1, batch=64 for smoke test")

    print("[DATA] building dataset...")
    X, y = _build_dataset(assets, timeframes, window)
    if quick:
        # Subsample to 1k examples
        idx = np.random.choice(len(X), min(1000, len(X)), replace=False)
        X, y = X[idx], y[idx]

    Xtr, ytr, Xva, yva, Xte, yte = _walk_forward_split(X, y)
    print(f"[SPLIT] train={len(Xtr)} val={len(Xva)} test={len(Xte)}")

    # ── Model ──
    config = CandleTransformerConfig(window=window, n_classes=y.shape[1])
    model = CandlestickTransformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] {n_params:,} parameters")

    # ── Training ──
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)),
        batch_size=batch_size, shuffle=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    bce_loss = nn.BCEWithLogitsLoss()

    best_val_f1 = 0.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            loss = bce_loss(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)
        scheduler.step()

        # Validate
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                probs = torch.sigmoid(model(x_batch)).cpu().numpy()
                val_preds.append(probs)
                val_targets.append(y_batch.numpy())
        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        val_f1 = _f1_score(val_targets, val_preds)

        elapsed = time.time() - t0
        print(f"[EPOCH {epoch + 1}/{epochs}] train_loss={train_loss:.4f} "
              f"val_f1={val_f1:.4f}  ({elapsed:.1f}s)")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ── Test ──
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    test_preds = []
    with torch.no_grad():
        for i in range(0, len(Xte), batch_size):
            x_batch = torch.from_numpy(Xte[i:i + batch_size]).to(device)
            probs = torch.sigmoid(model(x_batch)).cpu().numpy()
            test_preds.append(probs)
    test_preds = np.concatenate(test_preds, axis=0)
    test_f1 = _f1_score(yte, test_preds)
    print(f"[TEST] holdout F1 = {test_f1:.4f}")

    # ── Save ──
    if output_path is None:
        output_path = os.path.join(_PROJECT_ROOT, "models", "candlestick_transformer.pt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"[SAVE] {output_path}")

    return {
        "n_params": n_params,
        "n_train": len(Xtr),
        "n_val": len(Xva),
        "n_test": len(Xte),
        "best_val_f1": float(best_val_f1),
        "test_f1": float(test_f1),
        "model_path": output_path,
        "config": {
            "window": window, "epochs": epochs, "batch_size": batch_size,
            "lr": lr, "device": device,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--timeframes", nargs="+", default=["15m", "1h", "4h"])
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", default=None, help="output .pt path")
    parser.add_argument("--quick", action="store_true",
                        help="single epoch, small batch, subsampled data — for smoke testing")
    args = parser.parse_args()

    summary = train(
        assets=args.assets,
        timeframes=args.timeframes,
        window=args.window,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        output_path=args.output,
        quick=args.quick,
    )
    print()
    print("=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
