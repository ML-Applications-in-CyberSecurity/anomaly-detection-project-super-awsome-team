import socket
import json
import pandas as pd
import joblib
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import csv
from datetime import datetime

try:
    from together import Together
except ImportError:
    print("Error: Together SDK not found. Please install via 'pip install together'.", file=sys.stderr)
    raise

# ---- Configuration ----
HOST = 'localhost'
PORT = 9999
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "anomaly_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.joblib")
TRUST_PATH = os.path.join(BASE_DIR, "models", "trust_params.json")
LOG_FILE = os.path.join(BASE_DIR, "anomaly_log.csv")

# Initialize log file (add 'trust' column, no rating)
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'timestamp', 'src_port', 'dst_port', 'packet_size',
            'duration_ms', 'protocol', 'label', 'reason', 'trust'
        ])

# Load model, scaler, trust params
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'. Please run train_model.ipynb to generate it.")
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler file not found at '{SCALER_PATH}'. Run train_model.ipynb to generate it.")
if not os.path.exists(TRUST_PATH):
    raise FileNotFoundError(f"Trust params file not found at '{TRUST_PATH}'. Run train_model.ipynb to generate it.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(TRUST_PATH) as f:
    trust_params = json.load(f)
score_min = trust_params.get("score_min")
score_max = trust_params.get("score_max")

# Together AI configuration
API_KEY =  "20bc2f3806a1a04f8ae0d0059e91d797857b05ce4c178f3f36d1adbb083e417e"
if not API_KEY:
    raise ValueError("Environment variable TOGETHER_API_KEY not set.")
client = Together(api_key=API_KEY)
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

# After loading model, scaler, trust_params:
with open(os.path.join(BASE_DIR, "models", "feature_cols.json")) as f:
    feature_cols = json.load(f)

# Define preprocess_with_scaler and numeric_cols exactly as in training:
numeric_cols = ['packet_size', 'duration_ms']
def preprocess_with_scaler(df: pd.DataFrame, scaler, feature_cols: list) -> pd.DataFrame:
    df_copy = df.copy()
    # Protocol one-hot
    if 'protocol' in df_copy:
        proto_vals = df_copy['protocol'].apply(lambda x: str(x).upper())
        df_copy['protocol_TCP'] = proto_vals.apply(lambda x: 1 if x == 'TCP' else 0)
        df_copy['protocol_UDP'] = proto_vals.apply(lambda x: 1 if x == 'UDP' else 0)
    else:
        df_copy['protocol_TCP'] = 0
        df_copy['protocol_UDP'] = 0
    # src_port categorical
    COMMON_PORTS = [80, 443, 22, 8080]
    df_copy['is_src_common'] = df_copy['src_port'].apply(lambda x: 1 if x in COMMON_PORTS else 0)
    for p in COMMON_PORTS:
        df_copy[f'src_{p}'] = df_copy['src_port'].apply(lambda x: 1 if x == p else 0)
    # dst_port bins
    df_copy['dst_low'] = df_copy['dst_port'].apply(lambda x: 1 if x < 10240 else 0)
    df_copy['dst_mid'] = df_copy['dst_port'].apply(lambda x: 1 if 10240 <= x < 49152 else 0)
    df_copy['dst_high'] = df_copy['dst_port'].apply(lambda x: 1 if x >= 49152 else 0)
    df_copy['dst_suspicious'] = df_copy['dst_port'].apply(lambda x: 1 if x >= 60000 else 0)
    # Ensure numeric exist
    for col in numeric_cols:
        if col not in df_copy:
            df_copy[col] = 0
    # Ensure all feature_cols exist
    for col in feature_cols:
        if col not in df_copy:
            df_copy[col] = 0
    df_features = df_copy[feature_cols].copy()
    # Scale numeric cols
    df_features[numeric_cols] = scaler.transform(df_features[numeric_cols])
    return df_features

def pre_process_and_scale(data: dict):
    df = pd.DataFrame([data])
    df_feat = preprocess_with_scaler(df, scaler, feature_cols)
    X_scaled = df_feat.values  # shape (1, len(feature_cols))
    # Compute trust 0–10
    score = model.decision_function(X_scaled)[0]
    if score_max is not None and score_min is not None and score_max > score_min:
        trust01 = (score - score_min) / (score_max - score_min)
    else:
        trust01 = 0.0
    trust01 = max(0.0, min(1.0, trust01))
    trust10 = trust01 * 10
    return X_scaled, trust10


def parse_llm_response(content: str):
    """
    Only extract Label and Reason lines. Ignore any 'thinking' or extra detail.
    """
    label, reason = "", ""
    for line in content.splitlines():
        line_stripped = line.strip()
        if line_stripped.lower().startswith("label"):
            parts = line_stripped.split(":", 1)
            if len(parts) > 1:
                label = parts[1].strip()
        elif line_stripped.lower().startswith("reason"):
            parts = line_stripped.split(":", 1)
            if len(parts) > 1:
                reason = parts[1].strip()
    # Fallbacks
    if not label:
        first_line = content.strip().splitlines()[0]
        if len(first_line) < 100:
            label = first_line.strip()
    if not reason:
        reason = content.strip()
    return label, reason

def alert_llm_for_anomaly(data: dict):
    """
    Ask LLM only for Label and Reason. No rating, no step-by-step thinking.
    """
    system_message = {
        "role": "system",
        "content": (
            "You are an AI assistant specialized in network anomaly detection. "
            "Given a JSON data point flagged as anomalous, provide **only**:\n"
            "Label: <short description of anomaly>\n"
            "Reason: <possible cause or explanation>\n"
            "Respond concisely without describing your internal thought process."
        )
    }
    user_message = {
        "role": "user",
        "content": (
            f"Network traffic data point: {json.dumps(data)}.\n"
            "It was flagged as anomalous. Please respond exactly in the format:\n"
            "Label: <...>\n"
            "Reason: <...>"
        )
    }
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[system_message, user_message],
            stream=False,
        )
        content = response.choices[0].message.content
    except Exception as e:
        print(f"Error calling Together AI API: {e}", file=sys.stderr)
        return None, None
    return parse_llm_response(content)

def main():
    # Live PCA plot setup
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Network Traffic PCA Live Visualization")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((HOST, PORT))
        except ConnectionRefusedError:
            print(f"Could not connect to server at {HOST}:{PORT}.", file=sys.stderr)
            return

        print(f"Client connected to server at {HOST}:{PORT}.\n")
        buffer = ""
        X_all, labels_all = [], []
        anomaly_count = 0

        while True:
            try:
                chunk = s.recv(1024).decode()
            except Exception as e:
                print(f"Socket error: {e}", file=sys.stderr)
                break
            if not chunk:
                print("Server closed connection.")
                break
            buffer += chunk

            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    print("Error decoding JSON:", line, file=sys.stderr)
                    continue

                print(f"Data Received: {data}")
                # Preprocess, scale, compute trust
                X_point_scaled, trust = pre_process_and_scale(data)
                try:
                    pred = model.predict(X_point_scaled)[0]
                except Exception as e:
                    print(f"Error during model prediction: {e}", file=sys.stderr)
                    continue

                # Collect all points
                # Note: store unscaled for plotting, trust only used for display/log
                X_all.append(X_point_scaled[0])
                labels_all.append(pred)

                # Print prediction + trust
                if pred == -1:
                    anomaly_count += 1
                    print(f"Model prediction: Anomaly detected (trust={trust:.3f})")
                    label, reason = alert_llm_for_anomaly(data)
                    if label is None:
                        print("🚨 Anomaly detected, but LLM response failed.")
                        # Log with empty label/reason
                        log_label, log_reason = "", ""
                    else:
                        print(f"\n🚨 Anomaly Detected!\nLabel: {label}\nReason: {reason}\nTrust: {trust:.3f}\n")
                        log_label, log_reason = label, reason
                    # Log anomaly
                    with open(LOG_FILE, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            datetime.utcnow().isoformat(),
                            data.get('src_port', ''),
                            data.get('dst_port', ''),
                            data.get('packet_size', ''),
                            data.get('duration_ms', ''),
                            data.get('protocol', ''),
                            log_label,
                            log_reason,
                            trust
                        ])
                else:
                    print(f"Model prediction: Normal (trust={trust:.3f})")

                # Update live PCA plot on every message
                try:
                    df_vis = pd.DataFrame(X_all, columns=feature_cols)
                    df_vis['label'] = ['Anomaly' if l == -1 else 'Normal' for l in labels_all]
                    pca = PCA(n_components=2)
                    components = pca.fit_transform(df_vis.drop(columns=['label']))
                    df_vis['PCA1'], df_vis['PCA2'] = components[:,0], components[:,1]

                    ax.clear()
                    sns.scatterplot(
                    data=df_vis, x='PCA1', y='PCA2',
                    hue='label', palette={'Anomaly':'red', 'Normal':'green'},
                    ax=ax
                    )
                    ax.set_title("Network Traffic PCA Live Visualization")
                    ax.set_xlabel("PCA1")
                    ax.set_ylabel("PCA2")
                    plt.pause(0.01)
                except Exception as e:
                    print(f"Error updating live plot: {e}", file=sys.stderr)

    print("Client exiting.")

if __name__ == "__main__":
    main()
