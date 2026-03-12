import torch
import torch.nn as nn
import numpy as np
import os
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class PatchTST(nn.Module):
    """
    Real PyTorch Architecture for PatchTST.
    """
    def __init__(self, seq_len=400, patch_size=16, stride=8, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.n_patches = (seq_len - patch_size) // stride + 1
        
        # Patch embedding
        self.patch_embed = nn.Linear(patch_size, d_model)
        
        # Positional Encoding
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model))
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Heads: 1 for Probability of Up, 1 for Volatility/Shock probability
        self.head_prob = nn.Linear(d_model, 1)
        self.head_shock = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # x is [Batch, SeqLen]
        B, L = x.shape
        # Create patches
        patches = []
        for i in range(self.n_patches):
            p = x[:, i*self.stride : i*self.stride + self.patch_size]
            patches.append(p)
        
        # [Batch, n_patches, patch_size]
        patches = torch.stack(patches, dim=1)
        
        # Embed
        x = self.patch_embed(patches) + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # Aggregate (Mean pooling over patches)
        x_mean = x.mean(dim=1)
        
        prob_up = torch.sigmoid(self.head_prob(x_mean))
        prob_shock = torch.sigmoid(self.head_shock(x_mean))
        
        return prob_up, prob_shock

class PatchTSTClassifier:
    """
    Orchestrator for the PatchTST deep learning model.
    """
    def __init__(self, model_path: str = 'models/patchtst_v1.pt'):
        self.seq_len = 400
        self.model = PatchTST(seq_len=self.seq_len)
        self.is_ready = False
        
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                self.model.eval()
                self.is_ready = True
                logger.info(f"[PatchTST] Real model loaded from {model_path}")
            except Exception as e:
                logger.error(f"[PatchTST] Failed to load real model: {e}")
                
    def predict(self, price_data: np.ndarray) -> Dict[str, Any]:
        """
        Deep Inference using Transformer patches.
        """
        if not self.is_ready or len(price_data) < self.seq_len + 1:
            return {"confidence": 0, "prediction": 0, "prob_up": 0.5, "shock_prob": 0.0}

        try:
            # Prepare returns for the last seq_len window
            returns = np.diff(price_data) / (price_data[:-1] + 1e-9)
            input_tensor = torch.tensor(returns[-self.seq_len:], dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                prob_up_tensor, prob_shock_tensor = self.model(input_tensor)
                prob_up = float(prob_up_tensor.item())
                prob_shock = float(prob_shock_tensor.item())
            
            # Prediction threshold
            prediction = 0
            if prob_up > 0.55:
                prediction = 1
            elif prob_up < 0.45:
                prediction = -1
                
            return {
                "prediction": prediction,
                "prob_up": prob_up,
                "confidence": abs(prob_up - 0.5) * 2,
                "liquidity_shock_prob": prob_shock,
                "engine": "PatchTST_v1_Transformer"
            }
        except Exception as e:
            logger.error(f"[PatchTST] Inference error: {e}")
            return {"confidence": 0, "prediction": 0, "prob_up": 0.5, "shock_prob": 0.0}
