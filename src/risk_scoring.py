"""
Risk level classification, temporal risk escalation, and cascading pattern detection module.
Maps continuous anomaly scores into Low/Medium/High risk categories and ranks high-risk assets.
"""

import os
import numpy as np
import pandas as pd

def determine_risk_thresholds(df_val: pd.DataFrame, score_col: str = 'isolation_forest_score') -> dict:
    """
    Derives defensible Low / Medium / High risk thresholds using the validation partition.
    - Low: Below 85th percentile of validation scores (nominal operations)
    - Medium: 85th percentile to 96th percentile (incipient drift / monitoring required)
    - High: 96th percentile and above (confirmed degradation / alert dispatch)
    """
    scores = df_val[score_col].values
    q_med = float(np.percentile(scores, 85.0))
    q_high = float(np.percentile(scores, 96.0))
    
    # Ensure minimum gap
    if q_high <= q_med:
        q_high = q_med + 0.15
        
    return {
        'score_column': score_col,
        'low_threshold_max': round(q_med, 4),
        'medium_threshold_max': round(q_high, 4),
        'high_threshold_min': round(q_high, 4)
    }


def map_risk_levels(df: pd.DataFrame, thresholds: dict, score_col: str = 'isolation_forest_score') -> pd.DataFrame:
    """Maps continuous anomaly score to Low, Medium, High risk tiers."""
    df_out = df.copy()
    s = df_out[score_col].values
    
    th_med = thresholds['low_threshold_max']
    th_high = thresholds['high_threshold_min']
    
    conditions = [
        s < th_med,
        (s >= th_med) & (s < th_high),
        s >= th_high
    ]
    choices = ['Low', 'Medium', 'High']
    df_out['risk_level'] = np.select(conditions, choices, default='Low')
    return df_out


def compute_daily_temporal_risk(df_scored: pd.DataFrame, score_col: str = 'isolation_forest_score') -> pd.DataFrame:
    """Aggregates daily operational risk per machine and computes 7-day escalation trend slope."""
    df = df_scored.copy()
    df['date'] = pd.to_datetime(df['datetime']).dt.date
    
    daily = df.groupby(['Machine_ID', 'date']).agg(
        mean_score=(score_col, 'mean'),
        max_score=(score_col, 'max'),
        readings_count=(score_col, 'count'),
        high_risk_count=('risk_level', lambda x: (x == 'High').sum()),
        fault_type_mode=('fault_type', lambda x: x.mode()[0] if not x.empty else 'none'),
        is_fault_max=('is_fault', 'max')
    ).reset_index()
    
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values(['Machine_ID', 'date']).reset_index(drop=True)
    
    # 7-day rolling escalation slope of mean score per machine
    def calc_slope(s):
        if len(s) < 3:
            return 0.0
        x = np.arange(len(s))
        x_dev = x - x.mean()
        var_x = np.sum(x_dev**2)
        if var_x == 0:
            return 0.0
        return float(np.sum(x_dev * (s - s.mean())) / var_x)

    daily['escalation_slope_7d'] = daily.groupby('Machine_ID')['mean_score'].rolling(7, min_periods=3).apply(calc_slope, raw=False).reset_index(level=0, drop=True).fillna(0.0)
    return daily


def rank_high_risk_machines(
    daily_df: pd.DataFrame,
    df_full: pd.DataFrame,
    recent_days: int = 7,
    score_col: str = 'isolation_forest_score'
) -> pd.DataFrame:
    """
    Ranks industrial assets by priority using:
    1. Recent mean anomaly score
    2. Escalation slope (acceleration of wear)
    3. Total High-Risk observations
    4. Diagnostic categorization (Mechanical vs 6G-Network-Suspected)
    """
    max_date = daily_df['date'].max()
    cutoff_date = max_date - pd.Timedelta(days=recent_days)
    recent_daily = daily_df[daily_df['date'] >= cutoff_date]
    recent_full = df_full[pd.to_datetime(df_full['datetime']).dt.date >= cutoff_date.date()]
    
    machine_summary = []
    for m_id, grp in recent_daily.groupby('Machine_ID'):
        m_full = recent_full[recent_full['Machine_ID'] == m_id]
        recent_mean = float(grp['mean_score'].mean())
        recent_max = float(grp['max_score'].max())
        mean_slope = float(grp['escalation_slope_7d'].mean())
        total_high = int(grp['high_risk_count'].sum())
        
        # Determine diagnostic attribution
        net_stress = float(m_full['network_stress_index'].mean()) if 'network_stress_index' in m_full else 0.0
        vib_mean = float(m_full['Vibration_Hz_z'].mean()) if 'Vibration_Hz_z' in m_full else 0.0
        temp_mean = float(m_full['Temperature_C_z'].mean()) if 'Temperature_C_z' in m_full else 0.0
        
        if net_stress > 5.0 and vib_mean < 1.0 and temp_mean < 1.0:
            diag_label = "6G Network Suspected (Comm Anomaly)"
            action = "Inspect 6G Transceiver / Edge Gateway; Mechanical Maintenance NOT required"
        elif recent_mean > 0.60 or total_high > 10:
            diag_label = "Mechanical Wear Confirmed"
            action = "Dispatch Mechanical Maintenance: Inspect Bearings, Motor Drive & Thermal Cooling"
        else:
            diag_label = "Nominal / Low Drift"
            action = "Routine Monitoring"
            
        priority_index = (recent_mean * 0.45) + (max(0, mean_slope) * 10.0 * 0.35) + (min(total_high, 30) / 30.0 * 0.20)
        
        machine_summary.append({
            'Machine_ID': m_id,
            'Priority_Score': round(priority_index, 4),
            'Recent_Mean_Score': round(recent_mean, 4),
            'Recent_Max_Score': round(recent_max, 4),
            'Escalation_Slope_7d': round(mean_slope, 4),
            'High_Risk_Alert_Count': total_high,
            'Primary_Diagnostic': diag_label,
            'Recommended_Action': action
        })
        
    df_ranked = pd.DataFrame(machine_summary).sort_values('Priority_Score', ascending=False).reset_index(drop=True)
    df_ranked['Priority_Rank'] = df_ranked.index + 1
    return df_ranked


def detect_cascading_risk_patterns(
    daily_df: pd.DataFrame,
    threshold_simultaneous: int = 3
) -> pd.DataFrame:
    """
    Detects cascading risk events where multiple machines experience concurrent risk escalation.
    Identifies whether clusters stem from common network events or factory-wide stress.
    """
    cascades = []
    
    for date, grp in daily_df.groupby('date'):
        high_m = grp[grp['high_risk_count'] > 2]['Machine_ID'].tolist()
        escalating_m = grp[grp['escalation_slope_7d'] > 0.03]['Machine_ID'].tolist()
        
        affected = sorted(list(set(high_m + escalating_m)))
        if len(affected) >= threshold_simultaneous:
            cascades.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Simultaneous_Machines_Count': len(affected),
                'Affected_Machine_IDs': str(affected),
                'Mean_Cluster_Score': round(grp[grp['Machine_ID'].isin(affected)]['mean_score'].mean(), 4),
                'Cascade_Signature': 'Multi-Asset Wear Wave' if len(affected) < 6 else 'Factory-Wide Network / Environmental Wave'
            })
            
    return pd.DataFrame(cascades)
