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
