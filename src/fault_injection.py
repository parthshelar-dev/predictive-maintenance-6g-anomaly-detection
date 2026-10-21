"""
Semi-synthetic fault and network anomaly injection for 6G smart manufacturing.
Injects verifiable ground-truth degradation and network events into a copy of the clean data.
"""

import os
import pandas as pd
import numpy as np

def inject_semi_synthetic_anomalies(
    df_clean: pd.DataFrame,
    train_end: str = "2025-02-11 00:00:00",
    val_end: str = "2025-02-25 00:00:00",
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Injects mechanical degradation and 6G network corruption anomalies into validation and test periods.
    Early training data (<= train_end) remains 100% untouched and healthy (no data leakage).
    
    Returns:
        df_injected: pd.DataFrame with injected values and ground truth columns
        df_params: pd.DataFrame detailing injection parameters
    """
    np.random.seed(random_state)
    df = df_clean.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Initialize ground truth columns
    df['is_fault'] = 0
    df['fault_type'] = 'none'
    df['fault_start_time'] = pd.NaT
    df['failure_time'] = pd.NaT
    
    train_cutoff = pd.to_datetime(train_end)
    val_cutoff = pd.to_datetime(val_end)
    
    injection_records = []
    
    # -------------------------------------------------------------
    # 1. MECHANICAL FAULT INJECTION SPECIFICATIONS
    # Validation machines: 7, 14, 28
    # Test machines: 3, 19, 33, 45, 49
    # -------------------------------------------------------------
    mechanical_configs = [
        # Val set faults
        {'machine_id': 7,  'start': '2025-02-14 06:00:00', 'fail': '2025-02-19 12:00:00', 'temp_rise': 18.0, 'vib_rise': 3.2, 'power_instab': 2.8, 'split': 'val'},
        {'machine_id': 14, 'start': '2025-02-17 00:00:00', 'fail': '2025-02-22 18:00:00', 'temp_rise': 22.0, 'vib_rise': 3.8, 'power_instab': 3.5, 'split': 'val'},
        {'machine_id': 28, 'start': '2025-02-19 12:00:00', 'fail': '2025-02-24 16:00:00', 'temp_rise': 16.0, 'vib_rise': 2.9, 'power_instab': 2.5, 'split': 'val'},
        # Test set faults
        {'machine_id': 3,  'start': '2025-02-26 08:00:00', 'fail': '2025-03-03 14:00:00', 'temp_rise': 20.0, 'vib_rise': 3.5, 'power_instab': 3.0, 'split': 'test'},
        {'machine_id': 19, 'start': '2025-02-27 14:00:00', 'fail': '2025-03-04 20:00:00', 'temp_rise': 24.0, 'vib_rise': 4.1, 'power_instab': 3.8, 'split': 'test'},
        {'machine_id': 33, 'start': '2025-03-01 02:00:00', 'fail': '2025-03-06 18:00:00', 'temp_rise': 19.0, 'vib_rise': 3.3, 'power_instab': 3.2, 'split': 'test'},
        {'machine_id': 45, 'start': '2025-03-02 10:00:00', 'fail': '2025-03-08 08:00:00', 'temp_rise': 25.0, 'vib_rise': 4.3, 'power_instab': 4.0, 'split': 'test'},
        {'machine_id': 49, 'start': '2025-03-03 18:00:00', 'fail': '2025-03-09 22:00:00', 'temp_rise': 21.0, 'vib_rise': 3.6, 'power_instab': 3.4, 'split': 'test'},
    ]
    
    for cfg in mechanical_configs:
        m_id = cfg['machine_id']
        t_start = pd.to_datetime(cfg['start'])
        t_fail = pd.to_datetime(cfg['fail'])
        
        # Mask for this machine during degradation window (from start to end of dataset)
        deg_mask = (df['Machine_ID'] == m_id) & (df['datetime'] >= t_start)
        fail_mask = (df['Machine_ID'] == m_id) & (df['datetime'] >= t_fail)
        
        if deg_mask.sum() == 0:
            continue
            
        times = df.loc[deg_mask, 'datetime']
        total_span = (t_fail - t_start).total_seconds()
        
        # Progress tau in [0, 1] up to t_fail, and 1.0+ post-failure
        progress = (times - t_start).dt.total_seconds() / total_span
        progress_clamped = np.minimum(progress, 1.3)
        
        # Non-linear degradation curve (gradual inception, accelerating near failure: tau^1.8)
        deg_curve = np.power(np.maximum(0, progress_clamped), 1.8)
        
        # 1. Thermal creep: Temperature increases smoothly with increasing noise
        temp_drift = cfg['temp_rise'] * deg_curve + np.random.normal(0, 1.2, len(times)) * progress_clamped
        df.loc[deg_mask, 'Temperature_C'] += temp_drift
        
        # 2. Vibration spike: Bearing harmonic excitation (progressive ramp)
        vib_drift = cfg['vib_rise'] * np.power(progress_clamped, 2.0) + np.random.normal(0, 0.25, len(times)) * progress_clamped
        df.loc[deg_mask, 'Vibration_Hz'] += np.maximum(0, vib_drift)
        
        # 3. Power consumption instability: fluctuating load + mechanical friction
        power_noise = np.random.normal(0, cfg['power_instab'], len(times)) * progress_clamped
        df.loc[deg_mask, 'Power_Consumption_kW'] += power_noise + 0.8 * cfg['power_instab'] * deg_curve
        
        # 4. Error rate escalation: quality degradation accelerates
        error_drift = 18.0 * np.power(progress_clamped, 2.2) + np.random.normal(0, 1.5, len(times)) * progress_clamped
        df.loc[deg_mask, 'Error_Rate_%'] = np.clip(df.loc[deg_mask, 'Error_Rate_%'] + error_drift, 0, 100)
        
        # 5. Predictive Maintenance Score decay: drops toward 0.05
        pdm_decay = 0.42 * np.power(progress_clamped, 1.5)
        df.loc[deg_mask, 'Predictive_Maintenance_Score'] = np.clip(df.loc[deg_mask, 'Predictive_Maintenance_Score'] - pdm_decay, 0.01, 1.0)
        
        # Ground truth labels
        df.loc[deg_mask, 'is_fault'] = 1
        df.loc[deg_mask, 'fault_type'] = 'mechanical'
        df.loc[deg_mask, 'fault_start_time'] = t_start
        df.loc[deg_mask, 'failure_time'] = t_fail
        
        injection_records.append({
            'machine_id': m_id,
            'split': cfg['split'],
            'fault_type': 'mechanical',
            'start_time': t_start,
            'end_or_fail_time': t_fail,
            'duration_hours': round(total_span / 3600.0, 1),
            'target_metrics': 'Temp, Vib, Power, Error, PDM_Score',
            'severity_profile': f"Temp +{cfg['temp_rise']}C, Vib +{cfg['vib_rise']}Hz, PwrInstab +{cfg['power_instab']}kW"
        })
        
    # -------------------------------------------------------------
    # 2. 6G NETWORK CORRUPTION INJECTION SPECIFICATIONS
    # Validation machines: 11, 22, 38
    # Test machines: 8, 17, 30, 41
    # -------------------------------------------------------------
    network_configs = [
        # Val set network events
        {'machine_id': 11, 'start': '2025-02-15 08:00:00', 'end': '2025-02-15 20:00:00', 'latency_spike': 95.0,  'packet_loss_spike': 22.0, 'split': 'val'},
        {'machine_id': 22, 'start': '2025-02-18 10:00:00', 'end': '2025-02-19 02:00:00', 'latency_spike': 120.0, 'packet_loss_spike': 28.0, 'split': 'val'},
        {'machine_id': 38, 'start': '2025-02-21 14:00:00', 'end': '2025-02-22 04:00:00', 'latency_spike': 85.0,  'packet_loss_spike': 18.0, 'split': 'val'},
        # Test set network events
        {'machine_id': 8,  'start': '2025-02-27 06:00:00', 'end': '2025-02-27 18:00:00', 'latency_spike': 140.0, 'packet_loss_spike': 32.0, 'split': 'test'},
        {'machine_id': 17, 'start': '2025-03-01 12:00:00', 'end': '2025-03-02 06:00:00', 'latency_spike': 110.0, 'packet_loss_spike': 25.0, 'split': 'test'},
        {'machine_id': 30, 'start': '2025-03-04 04:00:00', 'end': '2025-03-04 22:00:00', 'latency_spike': 160.0, 'packet_loss_spike': 35.0, 'split': 'test'},
        {'machine_id': 41, 'start': '2025-03-06 16:00:00', 'end': '2025-03-07 10:00:00', 'latency_spike': 130.0, 'packet_loss_spike': 30.0, 'split': 'test'},
    ]
    
    for net in network_configs:
        m_id = net['machine_id']
        t_start = pd.to_datetime(net['start'])
        t_end = pd.to_datetime(net['end'])
        
        net_mask = (df['Machine_ID'] == m_id) & (df['datetime'] >= t_start) & (df['datetime'] <= t_end)
        if net_mask.sum() == 0:
            continue
            
        n_pts = net_mask.sum()
        # Network latency and packet loss bursts
        lat_noise = np.random.gamma(shape=2.0, scale=net['latency_spike'] / 2.0, size=n_pts)
        df.loc[net_mask, 'Network_Latency_ms'] += lat_noise
        
        loss_noise = np.random.uniform(net['packet_loss_spike'] * 0.7, net['packet_loss_spike'] * 1.3, size=n_pts)
        df.loc[net_mask, 'Packet_Loss_%'] = np.clip(df.loc[net_mask, 'Packet_Loss_%'] + loss_noise, 0, 100)
        
        # Telemetry packet jitter: slightly corrupted sensor values due to dropped samples / transmission bit flips
        jitter_mask = np.random.binomial(1, 0.35, size=n_pts).astype(bool)
        jitter_idx = df.loc[net_mask].index[jitter_mask]
        df.loc[jitter_idx, 'Temperature_C'] += np.random.normal(0, 4.0, size=len(jitter_idx))
        df.loc[jitter_idx, 'Vibration_Hz'] += np.random.normal(0, 0.8, size=len(jitter_idx))
        
        # Ground truth labels
        df.loc[net_mask, 'is_fault'] = 1
        df.loc[net_mask, 'fault_type'] = 'network'
        df.loc[net_mask, 'fault_start_time'] = t_start
        # Network events do NOT have a mechanical failure time
        df.loc[net_mask, 'failure_time'] = pd.NaT
        
        dur_hrs = (t_end - t_start).total_seconds() / 3600.0
        injection_records.append({
            'machine_id': m_id,
            'split': net['split'],
            'fault_type': 'network',
            'start_time': t_start,
            'end_or_fail_time': t_end,
            'duration_hours': round(dur_hrs, 1),
            'target_metrics': 'Latency, Packet_Loss, Transmission Jitter',
            'severity_profile': f"Latency +{net['latency_spike']}ms, PktLoss +{net['packet_loss_spike']}%"
        })
        
    df_params = pd.DataFrame(injection_records)
    return df, df_params
