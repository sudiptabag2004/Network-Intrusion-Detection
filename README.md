# 🔐 Network Intrusion Detection System (NIDS)

This project is a machine learning–based **Network Intrusion Detection System** that classifies network traffic as **Malicious** or **Benign** based on flow-level features. It uses a trained **deep learning model** and a user-friendly **Streamlit web interface** for interactive analysis.

---

## 🚀 Features

- 📁 Upload CSVs of network traffic flows
- 🧠 Predicts **Malicious** or **Benign** using a trained MLP model
- 📊 Real-time interactive visualizations:
  - 3D-style **Bar Chart**
  - Donut-style **Pie Chart**
  - Probability **Histogram**
- 📥 Download predictions as CSV
- 📝 Built-in format guide in the sidebar

---

## 🧠 Model Overview

- **Architecture**: 3-layer MLP with dropout + batch norm
- **Loss Function**: Focal loss (custom implementation)
- **Scaler**: RobustScaler
- **Imbalance Handling**: SMOTE oversampling
- **Dataset**: [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

---

## 📄 CSV Input Format

Your input file should include at least these columns:

- `dur`, `proto`, `service`, `state`
- `sbytes`, `dbytes`, `spkts`, `dpkts`

❗**Do NOT include**: `id`, `label`, `attack_cat` — these will be ignored or removed automatically.

---


## 📦 Setup

Install dependencies:

pip install -r requirements.txt


streamlit run nids_app.py
