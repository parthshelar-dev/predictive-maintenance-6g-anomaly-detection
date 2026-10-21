"""
Feature engineering module for 6G-integrated predictive maintenance and anomaly detection.
Generates baseline deviations, rolling count-based statistics per machine, kinematic ratios,
degradation trend slopes, and network quality indices.
"""

import numpy as np
import pandas as pd
from src.baseline import RobustBaselineModel

def compute_rolling_slopes(series: pd.Series, window: int = 10) -> pd.Series:
    """
    Computes linear regression slope over a rolling count window of size `window`.
    Using closed-form OLS formula: slope = cov(x, y) / var(x)
    where x = 0, 1, ..., window-1 (constant x).
    """
    x = np.arange(window)
    x_mean = (window - 1) / 2.0
    x_var = np.sum((x - x_mean) ** 2)
    weights = (x - x_mean) / x_var

    def calc_slope(arr):
        if len(arr) < window or np.isnan(arr).any():
            return 0.0
        return float(np.dot(weights, arr))

    return series.rolling(window=window, min_periods=max(3, window // 2)).apply(
        lambda y: (y[-1] - y[0]) / max(len(y) - 1, 1) if len(y) < window else np.dot(weights, y),
        raw=True
    ).fillna(0.0)


def engineer_features(
    df_injected: pd.DataFrame,
    baseline_model: RobustBaselineModel,
    window_short: int = 5,
    window_long: int = 15
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes all feature categories per machine chronologically:
    1. Baseline deviations and robust z-scores (mode-conditioned)
    2. Count-based rolling statistics per machine
    3. Vibration-to-power kinematic instability ratios
    4. Degradation trend slopes (Error Rate, PDM Score, Temperature)
    5. 6G Network communication quality metrics
    
    Returns:
        df_feats: pd.DataFrame containing all engineered features + metadata + ground truth
        df_dict: pd.DataFrame data dictionary documenting all features
    """
    df = df_injected.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(['Machine_ID', 'datetime']).reset_index(drop=True)
    
    # 1. Mode-conditioned robust z-scores
    df_z = baseline_model.transform(df, method='mode_conditioned')
    for col in df_z.columns:
        df[col] = df_z[col].values

    # 2. Kinematic instability ratios
    df['vib_to_power_ratio'] = df['Vibration_Hz'] / (df['Power_Consumption_kW'] + 1e-4)
    df['temp_to_power_ratio'] = df['Temperature_C'] / (df['Power_Consumption_kW'] + 1e-4)

    # 3. Network quality metrics
    df['network_stress_index'] = np.sqrt(
        np.maximum(0, df['Network_Latency_ms_z'])**2 + 
        np.maximum(0, df['Packet_Loss_%_z'])**2
    )

    # 4. Grouped per-machine rolling features and degradation slopes
    engineered_list = []
    
    for m_id, grp in df.groupby('Machine_ID', sort=False):
        grp = grp.sort_values('datetime').copy()
        
        # Rolling stats - short window (wear inception / dynamic response)
        grp['temp_z_roll_mean_w5'] = grp['Temperature_C_z'].rolling(window_short, min_periods=1).mean()
        grp['temp_roll_std_w5'] = grp['Temperature_C'].rolling(window_short, min_periods=2).std().fillna(0.0)
        grp['vib_z_roll_mean_w5'] = grp['Vibration_Hz_z'].rolling(window_short, min_periods=1).mean()
        grp['vib_roll_max_w5'] = grp['Vibration_Hz'].rolling(window_short, min_periods=1).max()
        grp['power_roll_std_w5'] = grp['Power_Consumption_kW'].rolling(window_short, min_periods=2).std().fillna(0.0)
        
        # Rolling stats - long window (sustained drift / structural wear)
        grp['temp_z_roll_mean_w15'] = grp['Temperature_C_z'].rolling(window_long, min_periods=1).mean()
        grp['vib_z_roll_mean_w15'] = grp['Vibration_Hz_z'].rolling(window_long, min_periods=1).mean()
        grp['power_roll_std_w15'] = grp['Power_Consumption_kW'].rolling(window_long, min_periods=2).std().fillna(0.0)
        
        # Rolling kinematic ratio
        grp['vib_power_instability_w10'] = (
            grp['Vibration_Hz'].rolling(10, min_periods=2).std() / 
            (grp['Power_Consumption_kW'].rolling(10, min_periods=2).std() + 1e-4)
        ).fillna(0.0)
        
        # Degradation trend slopes over last 10 readings (~8 hours operational time)
        grp['error_rate_slope_w10'] = compute_rolling_slopes(grp['Error_Rate_%'], window=10)
        grp['pdm_decay_slope_w10'] = compute_rolling_slopes(grp['Predictive_Maintenance_Score'], window=10)
        grp['temp_trend_slope_w10'] = compute_rolling_slopes(grp['Temperature_C'], window=10)
        
        engineered_list.append(grp)
        
    df_engineered = pd.concat(engineered_list, axis=0)
    # Restore original chronological order across the entire plant
    df_engineered = df_engineered.sort_values('datetime').reset_index(drop=True)
    
    # Feature Data Dictionary
    feature_metadata = [
        ('Temperature_C_z', 'Mode-conditioned robust z-score of machine operating temperature', 'Thermal Baseline Deviation'),
        ('Vibration_Hz_z', 'Mode-conditioned robust z-score of machine vibration frequency', 'Mechanical Baseline Deviation'),
        ('Power_Consumption_kW_z', 'Mode-conditioned robust z-score of electrical power consumption', 'Electrical Baseline Deviation'),
        ('Error_Rate_%_z', 'Mode-conditioned robust z-score of process manufacturing error rate', 'Process Quality Deviation'),
        ('Predictive_Maintenance_Score_z', 'Mode-conditioned robust z-score of embedded PDM score', 'Asset Health Score Deviation'),
        ('temp_z_roll_mean_w5', 'Rolling mean of temperature z-score over 5 readings per machine', 'Short-term Thermal State'),
        ('temp_roll_std_w5', 'Rolling std of temperature over 5 readings per machine', 'Thermal Volatility'),
        ('vib_z_roll_mean_w5', 'Rolling mean of vibration z-score over 5 readings per machine', 'Short-term Bearing Vibration'),
        ('vib_roll_max_w5', 'Rolling peak vibration over 5 readings per machine', 'Kinematic Shock / Peak'),
        ('power_roll_std_w5', 'Rolling std of power consumption over 5 readings (load instability)', 'Electrical Load Jitter'),
        ('temp_z_roll_mean_w15', 'Rolling mean of temperature z-score over 15 readings (~12h)', 'Sustained Thermal Runaway'),
        ('vib_z_roll_mean_w15', 'Rolling mean of vibration z-score over 15 readings (~12h)', 'Sustained Bearing Degradation'),
        ('power_roll_std_w15', 'Rolling std of power consumption over 15 readings (~12h)', 'Long-term Mechanical Friction Volatility'),
        ('vib_to_power_ratio', 'Instantaneous ratio of vibration to power consumption', 'Kinematic Power Coupling'),
        ('vib_power_instability_w10', 'Ratio of rolling vibration std to rolling power std over 10 readings', 'Dynamic Friction Instability'),
        ('error_rate_slope_w10', 'Linear regression slope of Error Rate over last 10 readings', 'Defect Escalation Rate'),
        ('pdm_decay_slope_w10', 'Linear regression slope of PDM score over last 10 readings', 'Asset Health Decay Velocity'),
        ('temp_trend_slope_w10', 'Linear regression slope of temperature over last 10 readings', 'Thermal Runaway Velocity'),
        ('Network_Latency_ms_z', 'Mode-conditioned robust z-score of 6G communication latency', '6G Network Latency Stress'),
        ('Packet_Loss_%_z', 'Mode-conditioned robust z-score of 6G packet loss percentage', '6G Transmission Loss Stress'),
        ('network_stress_index', 'Vector magnitude of positive latency and packet loss deviations', 'Composite 6G Communication Stress')
    ]
    
    df_dict = pd.DataFrame(feature_metadata, columns=['Feature_Name', 'Description', 'Category'])
    return df_engineered, df_dict
