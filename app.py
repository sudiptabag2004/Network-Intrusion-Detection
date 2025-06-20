import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px

# === Load model and artifacts ===
MODEL_PATH   = "nids_model_saved/mlp_model.h5"
SCALER_PATH  = "nids_model_saved/scaler.pkl"
TAU_PATH     = "nids_model_saved/threshold.json"
COLS_PATH    = "nids_model_saved/columns.json"

model   = tf.keras.models.load_model(MODEL_PATH, compile=False)
scaler  = joblib.load(SCALER_PATH)

with open(TAU_PATH) as f:
    tau = json.load(f)['tau']
with open(COLS_PATH) as f:
    columns = json.load(f)

# === Streamlit page setup ===
st.set_page_config(page_title="NIDS Dashboard", layout="wide")
st.title("🔐 Network Intrusion Detection System")
st.markdown("Upload a CSV file with network traffic data to classify each flow as **Malicious** or **Benign**.")

# === CSV Format Guide in Sidebar ===
st.sidebar.header("📄 CSV Format Guide")
st.sidebar.markdown("""
Your CSV should include columns like:

- `dur`, `proto`, `service`, `state`
- `sbytes`, `dbytes`, `spkts`, `dpkts`

✅ Do **not** include:
- `label`, `attack_cat`, or `id`

📌 Only `.csv` format supported.
""")

# === Upload CSV ===
uploaded_file = st.file_uploader("📁 Upload Network Flow CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    try:
        # === Feature Engineering ===
        df['byte_ratio'] = df['sbytes'] / (df['dbytes'] + 1)
        df['pkt_ratio']  = df['spkts']  / (df['dpkts']  + 1)
        for col in ['sbytes', 'dbytes', 'spkts', 'dpkts', 'dur']:
            df[f'log_{col}'] = np.log1p(df[col])
        df = df.drop(columns=['id', 'label', 'attack_cat'], errors='ignore')

        df = pd.get_dummies(df)
        df = df.reindex(columns=columns, fill_value=0)

        # === Prediction ===
        X_scaled = scaler.transform(df)
        probs = model.predict(X_scaled).ravel()
        preds = (probs >= tau).astype(int)

        df['Intrusion_Probability'] = probs.round(4)
        df['Prediction'] = np.where(preds == 1, "Malicious", "Benign")

        # === Results Table ===
        st.subheader("🧠 Prediction Results")
        st.dataframe(df[["Prediction", "Intrusion_Probability"]], use_container_width=True)

        # === Summary Counts ===
        summary_df = df["Prediction"].value_counts().rename_axis("Traffic Type").reset_index(name="Count")
        malicious_count = summary_df.loc[summary_df["Traffic Type"] == "Malicious", "Count"].sum()
        benign_count    = summary_df.loc[summary_df["Traffic Type"] == "Benign", "Count"].sum()
        total           = len(df)

        st.markdown(f"""
        ### 📊 Summary
        - 🟥 **Malicious**: {malicious_count}
        - 🟩 **Benign**: {benign_count}
        - 🔢 **Total Records**: {total}
        """)

        # === 3D-STYLE BAR CHART ===
        bar_fig = go.Figure(data=[
            go.Bar(
                x=summary_df["Traffic Type"],
                y=summary_df["Count"],
                marker=dict(
                    color=['#D65A5A' if x == "Malicious" else '#6CA966' for x in summary_df["Traffic Type"]],
                    line=dict(color='rgba(0,0,0,0.6)', width=1.5),
                    opacity=0.85
                ),
                width=0.6
            )
        ])
        bar_fig.update_layout(
            title="📊 Traffic Type Count (3D Bar Style)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            xaxis=dict(title="", tickfont=dict(size=11)),
            yaxis=dict(title="Count", tickfont=dict(size=11)),
            template="plotly_white"
        )
        st.plotly_chart(bar_fig, use_container_width=True)

        # === 3D-STYLE PIE CHART ===
        pie_fig = go.Figure(data=[
            go.Pie(
                labels=summary_df["Traffic Type"],
                values=summary_df["Count"],
                hole=0.3,
                marker=dict(
                    colors=["#6CA966", "#D65A5A"],
                    line=dict(color='white', width=1.5)
                ),
                pull=[0.02, 0],
                rotation=90,
                textinfo="label+percent",
                insidetextorientation="radial"
            )
        ])
        pie_fig.update_layout(
            title="🍩 Traffic Proportion (3D Donut Style)",
            showlegend=True,
            template="plotly_white"
        )
        st.plotly_chart(pie_fig, use_container_width=True)

        # === Download Button ===
        st.download_button("📥 Download Results as CSV", df.to_csv(index=False), file_name="nids_predictions.csv")

    except Exception as e:
        st.error(f"❌ Error during processing:\n\n`{e}`")
