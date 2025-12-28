# 🚇 MetroPulse: MLOps Predictive Maintenance System

![CI/CD Status](https://github.com/NiranjanKaithota/MLOps-Tutorial/actions/workflows/ci-cd.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.9-blue)
![React](https://img.shields.io/badge/React-18-cyan)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)

**MetroPulse** is an end-to-end MLOps platform designed to predict failures in Metro Train Air Production Units (APU). It features a multi-model consensus system (GRU, LSTM, CNN), real-time drift monitoring, and a fully containerized architecture.

## 🏗️ Architecture
The system follows a microservices architecture:
1.  **AI/ML Core:** Trained on the MetroPT-3 dataset using TensorFlow/Keras.
2.  **Backend (FastAPI):** Handles live data simulation, model inference, and drift detection.
3.  **Frontend (React + Vite):** A glassmorphic dashboard for real-time visualization.
4.  **Monitoring:** Integrated **Evidently AI** for data drift detection.
5.  **DevOps:** Dockerized deployment with Nginx reverse proxy and GitHub Actions CI/CD.

## 🚀 Key Features
* **Multi-Model Consensus:** Compares outputs from **GRU (Champion)**, **LSTM**, and **CNN** to reduce false positives.
* **Live Simulation:** Streams test set data to mimic real-time sensor telemetry.
* **Drift Detection:** Automatically generates statistical reports when data distribution shifts.
* **CI/CD Pipeline:** Automated testing and build verification via GitHub Actions.

## 🛠️ Tech Stack
* **Models:** GRU, LSTM (w/ Dropout), CNN 1D
* **Backend:** Python, FastAPI, Pandas, Scikit-Learn
* **Frontend:** TypeScript, React, TailwindCSS, Recharts
* **Ops:** Docker, Docker Compose, Nginx, GitHub Actions

## 📦 Installation & Setup

### Prerequisites
* Docker Desktop installed
* Git installed

### Running the App
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/NiranjanKaithota/MLOps-Tutorial.git
    cd MLOps-Tutorial
    ```

2.  **Launch with Docker:**
    ```bash
    docker-compose up --build
    ```

3.  **Access the Dashboard:**
    Open `http://localhost:3000` in your browser.

## 📊 Models Performance
The system utilizes a consensus mechanism across three distinct architectures. Performance metrics are based on the test simulation set (last 25% of dataset).

| Model | Architecture | Accuracy (@24h) | MAE (Hours) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **LSTM** | Recurrent (w/ Dropout) | **100.0%** | **8.73h** | ⚡ High Precision |
| **GRU v2.3** | Gated Recurrent | 98.5% | 8.99h | 🏆 Robust Champion |
| **CNN v1** | 1D Convolutional | 96.0% | 11.1h | ⚠️ Experimental |

* **Consensus Logic:** If both GRU and LSTM predict failure < 24h, the system triggers a **CRITICAL** alert.

## 🔄 CI/CD Pipeline
The project includes a GitHub Actions workflow that:
1.  **Verifies Backend:** Installs dependencies and runs sanity tests.
2.  **Builds Frontend:** Ensures the React app compiles without errors.
3.  **Validation:** Ensures the codebase is production-ready on every push.

---
*Developed by Niranjan S Kaithota | Nishan U Shetty*
