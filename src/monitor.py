import argparse
import pandas as pd
import numpy as np
import os

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset

# Optional ClearML upload
try:
    from clearml import Task
    CLEARML_AVAILABLE = True
except Exception:
    CLEARML_AVAILABLE = False


def main(reference_path, current_path, report_name, upload):
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference file not found: {reference_path}")
    if not os.path.exists(current_path):
        raise FileNotFoundError(f"Current file not found: {current_path}")

    ref = pd.read_parquet(reference_path) if reference_path.endswith('.parquet') else pd.read_csv(reference_path)
    cur = pd.read_parquet(current_path) if current_path.endswith('.parquet') else pd.read_csv(current_path)

    # Ensure predictions are present in current data
    if 'prediction' not in cur.columns:
        raise ValueError("Current data must contain a 'prediction' column. Provide predictions in the current file or use a different workflow to generate them.")

    report = Report(metrics=[DataDriftPreset(), RegressionPreset()])
    report.run(reference_data=ref.reset_index(drop=True), current_data=cur.reset_index(drop=True))

    report.save_html(report_name)
    print(f"Evidently monitoring report saved to: {report_name}")

    if upload:
        if not CLEARML_AVAILABLE:
            print("ClearML is not available in this environment; skipping upload.")
            return
        try:
            task = Task.init(project_name="Monitoring", task_name=f"Evidently Report - {os.path.basename(report_name)}", reuse_last_task_id=False)
            task.upload_artifact(name=os.path.basename(report_name), artifact_object=report_name)
            print("Report uploaded to ClearML task.")
        except Exception as e:
            print(f"Failed to upload report to ClearML: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', required=True, help='Path to reference dataset (parquet or csv)')
    parser.add_argument('--current', required=True, help='Path to current dataset (must contain a prediction column)')
    parser.add_argument('--report', default='evidently_monitor_report.html', help='Output HTML report name')
    parser.add_argument('--upload', action='store_true', help='If set and ClearML is available, upload the report as an artifact')

    args = parser.parse_args()
    main(args.reference, args.current, args.report, args.upload)
