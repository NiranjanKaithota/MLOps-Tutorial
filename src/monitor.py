import argparse
import pandas as pd
import os
from clearml import Task
try:
    # Try new API (Evidently 0.7+)
    from evidently import Report
    from evidently.presets import DataDriftPreset
except ImportError:
    try:
        # Try old API (Evidently < 0.7)
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except ImportError as e:
        print(f"❌ Could not import Evidently modules: {e}")
        print("💡 Ensure you have evidently installed: pip install evidently>=0.4.0")
        exit(1)

def run_monitor(reference_path, current_path, report_path, upload):
    # Load Reference Data
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Could not find reference data at {reference_path}")
    
    reference_data = pd.read_parquet(reference_path)
    
    # Handle Current Data (Split or Load)
    if current_path:
        if not os.path.exists(current_path):
            raise FileNotFoundError(f"Could not find current data at {current_path}")
        current_data = pd.read_parquet(current_path)
    else:
        print("ℹ️ No current data provided. Using 50/50 split of reference data for demo.")
        if len(reference_data) < 2:
            raise ValueError("Not enough data for drift detection")
        midpoint = len(reference_data) // 2
        current_data = reference_data.iloc[midpoint:].reset_index(drop=True)
        reference_data = reference_data.iloc[:midpoint].reset_index(drop=True)

    # Filter out columns that naturally drift (Time)
    drop_cols = ['timestamp', 'failure_column'] # failure_column might be static metadata
    reference_data = reference_data.drop(columns=[c for c in drop_cols if c in reference_data.columns], errors='ignore')
    current_data = current_data.drop(columns=[c for c in drop_cols if c in current_data.columns], errors='ignore')

    print(f"🕵️ Monitoring Drift: Reference({len(reference_data)}) vs Current({len(current_data)})")

    # Generate Report
    report = Report(metrics=[
        DataDriftPreset()
    ])

    # Generate Report
    report = Report(metrics=[
        DataDriftPreset()
    ])

    results = report.run(
        reference_data=reference_data,
        current_data=current_data
    )

    # Evidently 0.7+ returns a result object, older versions run in-place
    if results is not None:
        results.save_html(report_path)
    else:
        report.save_html(report_path)
        
    print(f"✅ Data drift report generated: {report_path}")

    # Upload to ClearML if requested
    if upload:
        print("🚀 Initializing ClearML Task...")
        task = Task.init(
            project_name="MetroPT Maintenance",
            task_name="Data Drift Check",
            reuse_last_task_id=False
        )
        task.upload_artifact(
            name="Evidently Data Drift Report",
            artifact_object=report_path
        )
        print("✅ Report uploaded to ClearML")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Evidently Data Drift Monitoring")
    parser.add_argument("--reference", type=str, default="data/engineered_data.parquet", help="Path to reference data")
    parser.add_argument("--current", type=str, default=None, help="Path to current data (optional)")
    parser.add_argument("--report", type=str, default="data_drift_report.html", help="Output path for HTML report")
    parser.add_argument("--upload", action="store_true", help="Upload report to ClearML")
    
    args = parser.parse_args()
    
    run_monitor(args.reference, args.current, args.report, args.upload)
