"""
Novelty experiments and advanced analytical evaluations for 6G predictive maintenance.
Includes baseline ablations, 6G network-aware separation, early warning lead times,
feature explainability (SHAP), and Downtime Prevention Index (DPI).
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, confusion_matrix

def compute_early_warning_lead_times(
    df_risk: pd.DataFrame,
    df_params: pd.DataFrame,
    lead_time_min_hours: float = 12.0
) -> tuple[pd.DataFrame, dict]:
    """
    Computes Early Warning Lead Time = failure_time - first_high_risk_time.
    For each mechanical failure scenario, determines the exact lead time achieved.
    """
    mech_scenarios = df_params[df_params['fault_type'] == 'mechanical'].copy()
    lead_time_records = []
    
    for _, row in mech_scenarios.iterrows():
        m_id = row['machine_id']
        t_start = pd.to_datetime(row['start_time'])
        t_fail = pd.to_datetime(row['end_or_fail_time'])
        
        # Sub-dataframe for this machine during the fault trajectory
        sub = df_risk[(df_risk['Machine_ID'] == m_id) & (df_risk['datetime'] >= t_start) & (df_risk['datetime'] <= t_fail)].sort_values('datetime')
        high_alerts = sub[sub['risk_level'] == 'High']
        
        if not high_alerts.empty:
            t_first_alert = high_alerts['datetime'].iloc[0]
            lead_time_seconds = (t_fail - t_first_alert).total_seconds()
            lead_time_hours = lead_time_seconds / 3600.0
            detected = True
            actionable = (lead_time_hours >= lead_time_min_hours)
        else:
            t_first_alert = pd.NaT
            lead_time_hours = 0.0
            detected = False
            actionable = False
            
        lead_time_records.append({
            'Machine_ID': m_id,
            'Split': row['split'],
            'Fault_Inception': t_start,
            'Catastrophic_Failure_Time': t_fail,
            'First_High_Alert_Time': t_first_alert,
            'Lead_Time_Hours': round(lead_time_hours, 2),
            'Total_Fault_Duration_Hours': round((t_fail - t_start).total_seconds() / 3600.0, 2),
            'Lead_Time_Ratio_%': round((lead_time_hours / ((t_fail - t_start).total_seconds() / 3600.0)) * 100, 1) if (t_fail - t_start).total_seconds() > 0 else 0.0,
            'Detected': detected,
            'Actionable (>= 12h)': actionable
        })
        
    df_lead = pd.DataFrame(lead_time_records)
    
    valid_leads = df_lead.loc[df_lead['Detected'], 'Lead_Time_Hours']
    stats = {
        'total_mechanical_failures': len(df_lead),
        'detected_count': int(df_lead['Detected'].sum()),
        'actionable_count': int(df_lead['Actionable (>= 12h)'].sum()),
        'actionable_rate_%': round(float(df_lead['Actionable (>= 12h)'].mean() * 100), 2),
        'mean_lead_time_hours': round(float(valid_leads.mean()), 2),
        'median_lead_time_hours': round(float(valid_leads.median()), 2),
        'min_lead_time_hours': round(float(valid_leads.min()), 2),
        'max_lead_time_hours': round(float(valid_leads.max()), 2),
        'q25_lead_time_hours': round(float(valid_leads.quantile(0.25)), 2),
        'q75_lead_time_hours': round(float(valid_leads.quantile(0.75)), 2)
    }
    
    return df_lead, stats


def evaluate_network_aware_separation(df_risk: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Evaluates whether network-aware feature attribution separates 6G communication corruption
    from genuine mechanical machine wear, preventing false mechanical dispatches.
    """
    # Mechanical vs Network classification logic:
    # If High Risk:
    #   if network_stress_index >= 3.0 and physical wear indicators (Vib + Temp z-scores) < 2.0 -> Network Anomaly Suspected
    #   else -> Mechanical Wear Confirmed
    df = df_risk.copy()
    
    net_stress = df['network_stress_index'].values
    phys_stress = np.maximum(df['Vibration_Hz_z'].values, df['Temperature_C_z'].values)
    is_high = (df['risk_level'] == 'High').values
    
    pred_type = np.full(len(df), 'none', dtype=object)
    
    # Network alert condition
    net_mask = is_high & (net_stress >= 3.0) & (phys_stress < 2.5)
    # Mechanical alert condition
    mech_mask = is_high & ~net_mask
    
    pred_type[net_mask] = 'network'
    pred_type[mech_mask] = 'mechanical'
    
    df['predicted_fault_type'] = pred_type
    
    # Ground truth vs Predicted cross-tabulation during alert events
    alerts_mask = (df['risk_level'] == 'High')
    ct = pd.crosstab(
        df.loc[alerts_mask, 'fault_type'],
        df.loc[alerts_mask, 'predicted_fault_type'],
        margins=True
    )
    
    # False dispatch reduction metric:
    # How many network anomalies would have been falsely dispatched as mechanical failures without network-aware logic?
    network_events = df[df['fault_type'] == 'network']
    network_alerts = network_events[network_events['risk_level'] == 'High']
    total_net_alerts = len(network_alerts)
    correctly_identified_net = (network_alerts['predicted_fault_type'] == 'network').sum()
    false_mech_dispatches = (network_alerts['predicted_fault_type'] == 'mechanical').sum()
    
    dispatch_reduction_pct = (correctly_identified_net / total_net_alerts * 100) if total_net_alerts > 0 else 100.0
    
    metrics = {
        'total_network_corruption_alerts': int(total_net_alerts),
        'correctly_classified_as_network': int(correctly_identified_net),
        'false_mechanical_dispatches': int(false_mech_dispatches),
        'false_dispatch_reduction_pct': round(float(dispatch_reduction_pct), 2)
    }
    
    return ct, metrics


def compute_downtime_prevention_index(
    lead_time_df: pd.DataFrame,
    df_risk: pd.DataFrame,
    unplanned_repair_hours: float = 16.0,
    planned_maintenance_hours: float = 3.0,
    cost_per_downtime_hour: float = 5000.0,
    cost_per_false_inspection: float = 400.0,
    min_lead_time_hours: float = 12.0
) -> pd.DataFrame:
    """
    Computes Downtime Prevention Index (DPI) and financial impact with transparent assumptions:
    - Unplanned Catastrophic Repair Time = 16.0 hours
    - Planned Proactive Maintenance Time = 3.0 hours
    - Net Downtime Saved per Prevented Breakdown = 13.0 hours
    - Hourly Factory Downtime Cost = $5,000 / hour
    - Cost of Dispatched Inspection = $400 / alert
    - Lead Time Threshold for Actionable Staging = 12.0 hours
    """
    M = len(lead_time_df)
    actionable_failures = lead_time_df[lead_time_df['Lead_Time_Hours'] >= min_lead_time_hours]
    k_prevented = len(actionable_failures)
    
    net_hours_saved_per_failure = unplanned_repair_hours - planned_maintenance_hours
    total_downtime_saved_hours = k_prevented * net_hours_saved_per_failure
    max_possible_downtime = M * unplanned_repair_hours
    
    dpi_percentage = (total_downtime_saved_hours / max_possible_downtime) * 100.0 if max_possible_downtime > 0 else 0.0
    
    gross_financial_savings = total_downtime_saved_hours * cost_per_downtime_hour
    
    # Healthy records falsely alerted as High Risk
    healthy_test = df_risk[(df_risk['datetime'] > '2025-02-25 00:00:00') & (df_risk['fault_type'] == 'none')]
    false_alert_count = int((healthy_test['risk_level'] == 'High').sum())
    total_inspection_cost = false_alert_count * cost_per_false_inspection
    
    net_financial_savings = gross_financial_savings - total_inspection_cost
    roi_ratio = (gross_financial_savings / (total_inspection_cost + 1e-4))
    
    results = [
        {'Metric': 'Total Injected Catastrophic Failures', 'Value': str(M), 'Unit': 'Assets'},
        {'Metric': 'Actionable Failures Prevented (Lead Time >= 12h)', 'Value': str(k_prevented), 'Unit': 'Assets'},
        {'Metric': 'Failure Prevention Rate', 'Value': f"{k_prevented / M * 100:.1f}", 'Unit': '%'},
        {'Metric': 'Unplanned Repair Downtime Assumption', 'Value': f"{unplanned_repair_hours:.1f}", 'Unit': 'Hours/Incident'},
        {'Metric': 'Planned Maintenance Downtime Assumption', 'Value': f"{planned_maintenance_hours:.1f}", 'Unit': 'Hours/Incident'},
        {'Metric': 'Net Downtime Saved Per Prevented Failure', 'Value': f"{net_hours_saved_per_failure:.1f}", 'Unit': 'Hours/Incident'},
        {'Metric': 'Total Factory Downtime Hours Prevented', 'Value': f"{total_downtime_saved_hours:.1f}", 'Unit': 'Hours'},
        {'Metric': 'Downtime Prevention Index (DPI)', 'Value': f"{dpi_percentage:.2f}", 'Unit': '% of Max Downtime'},
        {'Metric': 'Downtime Financial Value Saved ($5,000/hr)', 'Value': f"${gross_financial_savings:,.2f}", 'Unit': 'USD'},
        {'Metric': 'False Inspection Dispatches on Healthy Test Data', 'Value': str(false_alert_count), 'Unit': 'Events'},
        {'Metric': 'Cost of False Inspections ($400/event)', 'Value': f"${total_inspection_cost:,.2f}", 'Unit': 'USD'},
        {'Metric': 'Net Economic Benefit to Manufacturer', 'Value': f"${net_financial_savings:,.2f}", 'Unit': 'USD'},
        {'Metric': 'Economic Benefit-to-Inspection Cost Ratio', 'Value': f"{roi_ratio:.1f}x", 'Unit': 'Ratio'}
    ]
    
    return pd.DataFrame(results)


def compute_shap_feature_explainability(
    df_risk: pd.DataFrame,
    model_dir: str = "../models",
    sample_size: int = 200
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes SHAP feature importance for Isolation Forest alerts to provide root-cause explainability.
    """
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    iforest = joblib.load(os.path.join(model_dir, "isolation_forest.joblib"))
    cols = joblib.load(os.path.join(model_dir, "model_feature_cols.joblib"))
    
    # Select sample of high risk alerts
    high_alerts = df_risk[df_risk['risk_level'] == 'High'].head(sample_size).copy()
    if high_alerts.empty:
        high_alerts = df_risk.head(sample_size).copy()
        
    X_scaled = scaler.transform(high_alerts[cols].values)
    
    explainer = shap.TreeExplainer(iforest)
    shap_vals = explainer.shap_values(X_scaled)
    
    # Mean absolute SHAP value per feature
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    df_shap_summary = pd.DataFrame({
        'Feature': cols,
        'Mean_Absolute_SHAP': mean_abs_shap
    }).sort_values('Mean_Absolute_SHAP', ascending=False).reset_index(drop=True)
    
    # Top 3 drivers for 5 sample alerts
    sample_explanations = []
    for idx in range(min(5, len(high_alerts))):
        row = high_alerts.iloc[idx]
        shaps = shap_vals[idx]
        top_indices = np.argsort(shaps)[::-1][:3]
        top_drivers = [f"{cols[i]} (+{shaps[i]:.3f})" for i in top_indices]
        sample_explanations.append({
            'Machine_ID': row['Machine_ID'],
            'Timestamp': row['datetime'],
            'Fault_Type': row['fault_type'],
            'Anomaly_Score': round(row['isolation_forest_score'], 3),
            'Top_Root_Cause_Drivers': "; ".join(top_drivers)
        })
        
    df_sample_alerts = pd.DataFrame(sample_explanations)
    return df_shap_summary, df_sample_alerts
