"""
Per-machine and mode-conditioned baseline calibration module.
Learns robust medians and IQRs exclusively on early healthy training data (zero leakage).
"""

import os
import joblib
import numpy as np
import pandas as pd

SENSOR_COLS_DEFAULT = [
    'Temperature_C', 'Vibration_Hz', 'Power_Consumption_kW',
    'Network_Latency_ms', 'Packet_Loss_%', 'Quality_Control_Defect_Rate_%',
    'Production_Speed_units_per_hr', 'Predictive_Maintenance_Score', 'Error_Rate_%'
]

class RobustBaselineModel:
    def __init__(self, sensor_cols=None):
        self.sensor_cols = sensor_cols or SENSOR_COLS_DEFAULT
        self.mode_conditioned_ = {}
        self.machine_only_ = {}
        self.global_ = {}
        self.is_fitted = False
        
    def fit(self, df_train: pd.DataFrame):
        """Fit global, machine-only, and mode-conditioned robust baselines on healthy training data."""
        # 1. Global baseline
        for col in self.sensor_cols:
            q25 = float(df_train[col].quantile(0.25))
            median = float(df_train[col].median())
            q75 = float(df_train[col].quantile(0.75))
            iqr = max(q75 - q25, 1e-4)
            self.global_[col] = {'median': median, 'iqr': iqr, 'q25': q25, 'q75': q75}
            
        # 2. Machine-only baseline
        grouped_m = df_train.groupby('Machine_ID')
        for m_id, grp in grouped_m:
            self.machine_only_[m_id] = {}
            for col in self.sensor_cols:
                q25 = float(grp[col].quantile(0.25))
                median = float(grp[col].median())
                q75 = float(grp[col].quantile(0.75))
                iqr = max(q75 - q25, 1e-4)
                self.machine_only_[m_id][col] = {'median': median, 'iqr': iqr, 'q25': q25, 'q75': q75}
                
        # 3. Mode-conditioned baseline (Machine_ID x Operation_Mode)
        grouped_mm = df_train.groupby(['Machine_ID', 'Operation_Mode'])
        for (m_id, mode), grp in grouped_mm:
            key = (m_id, mode)
            self.mode_conditioned_[key] = {}
            for col in self.sensor_cols:
                q25 = float(grp[col].quantile(0.25))
                median = float(grp[col].median())
                q75 = float(grp[col].quantile(0.75))
                iqr = max(q75 - q25, 1e-4)
                self.mode_conditioned_[key][col] = {'median': median, 'iqr': iqr, 'q25': q25, 'q75': q75}
                
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, method: str = 'mode_conditioned') -> pd.DataFrame:
        """
        Transforms raw sensor readings into robust z-scores:
        z = (x - median) / (iqr / 1.34898)
        """
        assert self.is_fitted, "Model must be fitted before transform."
        scale_factor = 1.34898
        df_z = pd.DataFrame(index=df.index)
        
        if method == 'global':
            for col in self.sensor_cols:
                med = self.global_[col]['median']
                denom = self.global_[col]['iqr'] / scale_factor
                df_z[f'{col}_z'] = (df[col] - med) / denom
                
        elif method == 'machine_only':
            for col in self.sensor_cols:
                med_map = {m_id: self.machine_only_[m_id][col]['median'] for m_id in self.machine_only_}
                iqr_map = {m_id: self.machine_only_[m_id][col]['iqr'] / scale_factor for m_id in self.machine_only_}
                
                # Default fallback
                global_med = self.global_[col]['median']
                global_denom = self.global_[col]['iqr'] / scale_factor
                
                med_series = df['Machine_ID'].map(med_map).fillna(global_med)
                denom_series = df['Machine_ID'].map(iqr_map).fillna(global_denom)
                df_z[f'{col}_z'] = (df[col] - med_series) / denom_series
                
        elif method == 'mode_conditioned':
            keys = list(zip(df['Machine_ID'], df['Operation_Mode']))
            for col in self.sensor_cols:
                med_map = {k: self.mode_conditioned_[k][col]['median'] for k in self.mode_conditioned_}
                iqr_map = {k: self.mode_conditioned_[k][col]['iqr'] / scale_factor for k in self.mode_conditioned_}
                
                global_med = self.global_[col]['median']
                global_denom = self.global_[col]['iqr'] / scale_factor
                
                # Fast series mapping
                key_series = pd.Series(keys, index=df.index)
                med_series = key_series.map(med_map).fillna(global_med)
                denom_series = key_series.map(iqr_map).fillna(global_denom)
                df_z[f'{col}_z'] = (df[col] - med_series) / denom_series
        else:
            raise ValueError(f"Unknown baseline method: {method}")
            
        return df_z

    def get_baseline_summary(self) -> pd.DataFrame:
        """Returns tabular summary of the global baselines."""
        rows = []
        for col, stats in self.global_.items():
            rows.append({
                'Sensor': col,
                'Global_Median': round(stats['median'], 3),
                'Global_IQR': round(stats['iqr'], 3),
                'Global_Q25': round(stats['q25'], 3),
                'Global_Q75': round(stats['q75'], 3)
            })
        return pd.DataFrame(rows)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        
    @classmethod
    def load(cls, filepath: str):
        return joblib.load(filepath)
