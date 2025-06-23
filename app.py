import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

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

# === Streamlit UI ===
st.set_page_config(page_title="NIDS Dashboard", layout="wide")
st.title("🔐 Network Intrusion Detection System")
st.markdown("Upload a CSV file with network traffic data to classify each flow as **Malicious** or **Benign**.")

st.sidebar.header("📄 CSV Format Guide")
st.sidebar.markdown("""
Your CSV should include columns like:
- `dur`, `proto`, `service`, `state`
- `sbytes`, `dbytes`, `spkts`, `dpkts`

✅ Do **not** include:
- `label`, `attack_cat`, or `id`

📌 Only `.csv` format supported.
""")

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

        # === Display Results
        st.subheader("🧠 Prediction Results")
        st.dataframe(df[["Prediction", "Intrusion_Probability"]], use_container_width=True)

        # === Summary
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

        # === 3D Bar Chart
        bar_fig = go.Figure(data=[
            go.Bar(
                x=summary_df["Traffic Type"],
                y=summary_df["Count"],
                marker=dict(
                    color=['#D65A5A' if x == "Malicious" else '#6CA966' for x in summary_df["Traffic Type"]],
                    line=dict(color='rgba(0,0,0,0.6)', width=1.5),
                    opacity=0.85
                )
            )
        ])
        bar_fig.update_layout(
            title="📊 Traffic Type Count (3D Style)",
            template="plotly_white"
        )
        st.plotly_chart(bar_fig, use_container_width=True)

        # === 3D Pie Chart
        pie_fig = go.Figure(data=[
            go.Pie(
                labels=summary_df["Traffic Type"],
                values=summary_df["Count"],
                hole=0.3,
                marker=dict(colors=["#6CA966", "#D65A5A"], line=dict(color='white', width=1.5)),
                pull=[0.02, 0],
                textinfo="label+percent",
                insidetextorientation="radial"
            )
        ])
        pie_fig.update_layout(
            title="🍩 Traffic Proportion (Donut Style)",
            template="plotly_white"
        )
        st.plotly_chart(pie_fig, use_container_width=True)

        # === Create HTML Report for Download
        report_html = f"""
        <html><head><title>NIDS Report</title></head><body>
        <h2>📊 Network Intrusion Detection Report</h2>
        <p><b>Total Records:</b> {total}</p>
        <p><b>Malicious:</b> {malicious_count} &nbsp;&nbsp;&nbsp; <b>Benign:</b> {benign_count}</p>
        <h3>🔢 Sample Prediction Table</h3>
        {df[['Prediction', 'Intrusion_Probability']].head(10).to_html(index=False)}
        <h3>📊 Bar Chart</h3>
        {bar_fig.to_html(full_html=False, include_plotlyjs='cdn')}
        <h3>🍩 Pie Chart</h3>
        {pie_fig.to_html(full_html=False, include_plotlyjs='cdn')}
        </body></html>
        """
            
        st.download_button("📄 Download Full Report (HTML)", report_html, file_name="nids_report.html", mime="text/html")

        # === Also CSV option
        st.download_button("📥 Download Prediction CSV (numerical values)", df.to_csv(index=False), file_name="nids_predictions.csv")

    except Exception as e:
        st.error(f"❌ Error during processing:\n\n`{e}`")
