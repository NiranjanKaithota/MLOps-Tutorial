# MetroPT Predictive Maintenance MLOps

This project predicts the Remaining Useful Life (RUL) of metro train compressors using the MetroPT dataset.

## Architecture
- **Tracking:** ClearML (Hosted)
- **Models:** Lasso, Random Forest, LSTM

## How to Run
1. Clone this repo in Google Colab.
2. Install dependencies: `pip install -r requirements.txt`
3. Set ClearML Credentials using environment variables.
4. Run `src/train.py`.

## DVC (Data Version Control) Quick Start

This project can track large data files and models using DVC. Minimal example commands:

```powershell
# initialize dvc in your repo (one-time)
dvc init

# add dataset to dvc (example)
dvc add data/engineered_data.parquet

# commit the .dvc file and dvc config to git
git add data/engineered_data.parquet.dvc .dvc/config
git commit -m "Track engineered dataset with DVC"

# push data to remote storage (configure remote first, e.g. dvc remote add -d myremote s3://...)
dvc push
```

## Monitoring with Evidently

You can produce monitoring reports after training or from a batch of inference results.

- Training script generates an Evidently report automatically and saves `evidently_report_<model>.html`.
- To run monitoring separately: 

```powershell
python src/monitor.py --reference data/engineered_data.parquet --current data/current_batch.parquet --report my_report.html --upload
```

The `--upload` flag will upload the report to ClearML if ClearML is available in your environment.
