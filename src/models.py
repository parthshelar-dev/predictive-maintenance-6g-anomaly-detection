"""
Anomaly detection models and evaluation module for 6G predictive maintenance.
Includes Isolation Forest, PyTorch Autoencoder, and PCA Reconstruction baseline.
All models fit exclusively on past healthy training data (zero data leakage).
"""

import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, roc_auc_score

# Feature column definitions
PHYSICAL_FEATURE_COLS = [
    'Temperature_C_z', 'Vibration_Hz_z', 'Power_Consumption_kW_z', 'Error_Rate_%_z', 'Predictive_Maintenance_Score_z',
    'temp_z_roll_mean_w5', 'temp_roll_std_w5', 'vib_z_roll_mean_w5', 'vib_roll_max_w5', 'power_roll_std_w5',
    'temp_z_roll_mean_w15', 'vib_z_roll_mean_w15', 'power_roll_std_w15',
    'vib_to_power_ratio', 'vib_power_instability_w10',
    'error_rate_slope_w10', 'pdm_decay_slope_w10', 'temp_trend_slope_w10'
]

NETWORK_FEATURE_COLS = [
    'Network_Latency_ms_z', 'Packet_Loss_%_z', 'network_stress_index'
]

ALL_FEATURE_COLS = PHYSICAL_FEATURE_COLS + NETWORK_FEATURE_COLS


class CompactPyTorchAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 12, bottleneck_dim: int = 6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.LeakyReLU(0.1)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class ModelScorer:
    """Wrapper to fit, score, and evaluate Isolation Forest, Autoencoder, and PCA."""
    def __init__(self, feature_cols=None, random_state=42):
        self.feature_cols = feature_cols or ALL_FEATURE_COLS
        self.random_state = random_state
        self.scaler = RobustScaler()
        self.iforest = IsolationForest(
            n_estimators=150,
            contamination=0.01,
            max_features=0.85,
            random_state=random_state,
            n_jobs=-1
        )
        self.pca = PCA(n_components=6, random_state=random_state)
        self.autoencoder = None
        self.score_ranges = {}
        self.is_fitted = False

    def fit(self, df_train_healthy: pd.DataFrame, epochs: int = 15, batch_size: int = 256):
        X_train_raw = df_train_healthy[self.feature_cols].values
        X_train = self.scaler.fit_transform(X_train_raw)

        # 1. Fit Isolation Forest
        self.iforest.fit(X_train)

        # 2. Fit PCA Reconstruction Baseline
        self.pca.fit(X_train)

        # 3. Fit PyTorch Autoencoder
        torch.manual_seed(self.random_state)
        input_dim = X_train.shape[1]
        self.autoencoder = CompactPyTorchAutoencoder(input_dim=input_dim, hidden_dim=12, bottleneck_dim=6)
        
        tensor_x = torch.tensor(X_train, dtype=torch.float32)
        dataset = TensorDataset(tensor_x, tensor_x)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.AdamW(self.autoencoder.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.MSELoss()

        self.autoencoder.train()
        for epoch in range(epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                recon = self.autoencoder(batch_x)
                loss = criterion(recon, batch_y)
                loss.backward()
                optimizer.step()

        self.autoencoder.eval()
        self.is_fitted = True
        return self

    def calibrate_normalizers(self, df_val: pd.DataFrame):
        """Calibrate min-max normalizers for 0-1 scores using validation data."""
        scores_raw = self.score_raw(df_val)
        for model_name, s in scores_raw.items():
            s_min = float(np.percentile(s, 0.5))
            s_max = float(np.percentile(s, 99.5))
            if s_max - s_min < 1e-6:
                s_max = s_min + 1.0
            self.score_ranges[model_name] = {'min': s_min, 'max': s_max}

    def score_raw(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        assert self.is_fitted, "Models must be fitted before scoring."
        X_raw = df[self.feature_cols].values
        X = self.scaler.transform(X_raw)

        # Isolation forest raw score: higher = more anomalous
        if_raw = -self.iforest.score_samples(X)

        # PCA reconstruction error
        X_pca_recon = self.pca.inverse_transform(self.pca.transform(X))
        pca_raw = np.mean((X - X_pca_recon) ** 2, axis=1)

        # Autoencoder reconstruction error
        self.autoencoder.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X, dtype=torch.float32)
            recon = self.autoencoder(tensor_x).numpy()
            ae_raw = np.mean((X - recon) ** 2, axis=1)

        # Z-score composite baseline
        z_cols = [c for c in self.feature_cols if c.endswith('_z')]
        z_raw = np.max(np.abs(df[z_cols].values), axis=1)

        return {
            'isolation_forest': if_raw,
            'autoencoder': ae_raw,
            'pca_reconstruction': pca_raw,
            'zscore_baseline': z_raw
        }

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns 0-1 normalized anomaly scores for all models."""
        scores_raw = self.score_raw(df)
        df_scores = pd.DataFrame(index=df.index)

        for model_name, s in scores_raw.items():
            if model_name in self.score_ranges:
                s_min = self.score_ranges[model_name]['min']
                s_max = self.score_ranges[model_name]['max']
            else:
                s_min = float(np.percentile(s, 0.5))
                s_max = float(np.percentile(s, 99.5))
            norm_s = np.clip((s - s_min) / (s_max - s_min + 1e-6), 0.0, 1.0)
            df_scores[f'{model_name}_score'] = norm_s

        return df_scores

    def evaluate_model(
        self,
        y_true: np.ndarray,
        scores: np.ndarray,
        threshold: float = None
    ) -> dict:
        """
        Calculates Precision, Recall, F1, PR-AUC, and ROC-AUC.
        If threshold is None, selects the threshold maximizing F1.
        """
        pr_auc = float(average_precision_score(y_true, scores))
        roc_auc = float(roc_auc_score(y_true, scores))

        if threshold is None:
            best_f1, best_th, best_prec, best_rec = 0.0, 0.5, 0.0, 0.0
            candidate_thresholds = np.linspace(0.1, 0.9, 81)
            for th in candidate_thresholds:
                preds = (scores >= th).astype(int)
                p, r, f, _ = precision_recall_fscore_support(y_true, preds, average='binary', zero_division=0)
                if f > best_f1:
                    best_f1, best_th, best_prec, best_rec = f, th, p, r
            opt_th = best_th
            prec, rec, f1 = best_prec, best_rec, best_f1
        else:
            opt_th = threshold
            preds = (scores >= opt_th).astype(int)
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average='binary', zero_division=0)

        return {
            'threshold': round(float(opt_th), 4),
            'precision': round(float(prec), 4),
            'recall': round(float(rec), 4),
            'f1_score': round(float(f1), 4),
            'pr_auc': round(float(pr_auc), 4),
            'roc_auc': round(float(roc_auc), 4)
        }

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(model_dir, "scaler.joblib"))
        joblib.dump(self.iforest, os.path.join(model_dir, "isolation_forest.joblib"))
        joblib.dump(self.pca, os.path.join(model_dir, "pca_model.joblib"))
        joblib.dump(self.score_ranges, os.path.join(model_dir, "score_ranges.joblib"))
        joblib.dump(self.feature_cols, os.path.join(model_dir, "model_feature_cols.joblib"))
        torch.save(self.autoencoder.state_dict(), os.path.join(model_dir, "pytorch_autoencoder.pt"))
        print(f"Saved all trained models to {model_dir}")
