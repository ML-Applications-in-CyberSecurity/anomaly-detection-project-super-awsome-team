import socket
import json
import pandas as pd
import joblib
import os
import sys

# ---- Together AI SDK import ----
try:
    from together import Together
except ImportError:
    print("Error: Together SDK not found. Please install via 'pip install together'.", file=sys.stderr)
    raise


# ---- Configuration ----
HOST = 'localhost'
PORT = 9999

# Load the trained Isolation Forest model saved in train_model.ipynb (Step 1).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "anomaly_model.joblib")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at '{MODEL_PATH}'. "
        "Please ensure you have run train_model.ipynb and saved the model there."
    )

model = joblib.load(MODEL_PATH)

# Together AI configuration
API_KEY = os.getenv("TOGETHER_API_KEY")
if not API_KEY:
    raise ValueError("Environment variable TOGETHER_API_KEY not set. Please set it to your Together AI API key before running.")

# Initialize Together client
client = Together(api_key=API_KEY)
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"  

def pre_process_data(data: dict) -> pd.DataFrame:
    """
    Convert a single data point (dict) into a DataFrame / array for model prediction.
    Must mirror preprocessing used during training:
      - Numerical columns unchanged.
      - One-hot encode 'protocol', keeping only 'protocol_UDP' to match training.
    """
    # Create DataFrame from single record
    df = pd.DataFrame([data])

    # Example features from server: 'src_port', 'dst_port', 'packet_size', 'duration_ms', 'protocol'
    # During training we one-hot encoded 'protocol' with drop_first=True, so only 'protocol_UDP' column was used:
    #   protocol == 'UDP'  -> protocol_UDP = 1
    #   protocol == 'TCP'  -> protocol_UDP = 0
    # If other protocol values (e.g., 'UNKNOWN'), treat as not-UDP (i.e., protocol_UDP = 0).
    if 'protocol' in df.columns:
        df['protocol_UDP'] = df['protocol'].apply(lambda x: 1 if str(x).upper() == 'UDP' else 0)
        df = df.drop(columns=['protocol'])

    # Ensure the column order matches training. If during training you used columns in a specific order,
    # you may want to reorder here. E.g.:
    # feature_columns = ['src_port', 'dst_port', 'packet_size', 'duration_ms', 'protocol_UDP']
    # df = df[feature_columns]
    # Here we assume training used exactly these columns in this order.
    expected_cols = ['src_port', 'dst_port', 'packet_size', 'duration_ms', 'protocol_UDP']
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    # Select in order
    df = df[expected_cols]

    return df

def parse_llm_response(content: str):
    """
    Parse the LLM response to extract 'Label' and 'Reason'.
    Expects the LLM to respond with something like:
      Label: <some label>
      Reason: <some explanation>
    If not found, returns the whole content as reason and empty label.
    """
    label = ""
    reason = ""
    # Split lines and search for lines starting with Label/Reason (case-insensitive)
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
    if not label:
        # As fallback, take first line as label if succinct
        first_line = content.strip().splitlines()[0]
        if len(first_line) < 100:
            label = first_line.strip()
    if not reason:
        # Fallback: use full content if no explicit Reason found
        reason = content.strip()
    return label, reason

def alert_llm_for_anomaly(data: dict):
    """
    Call the Together AI LLaMA3 70B model to get a human-readable label and reason for the anomaly.
    """
    # Construct system and user messages. Adapt responsibly from sample prompt in spec.
    system_message = {
        "role": "system",
        "content": (
            "You are an AI assistant specialized in network anomaly detection. "
            "Given a single network traffic data point flagged as anomalous by an automated model, "
            "you should provide a concise Label describing the anomaly type and a Reason suggesting a possible cause."
        )
    }
    user_message = {
        "role": "user",
        "content": (
            f"Network traffic reading (JSON): {json.dumps(data)}.\n"
            "The anomaly detection model flagged this as an anomaly. "
            "Please respond in the format:\n"
            "Label: <short description of anomaly>\n"
            "Reason: <possible cause or explanation>\n"
            "Be concise and informative."
        )
    }
    messages = [system_message, user_message]

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            stream=False,
        )
    except Exception as e:
        print(f"Error calling Together AI API: {e}", file=sys.stderr)
        return None, None

    # Extract response content
    # Depending on SDK, this may vary. Example for typical response object:
    try:
        content = response.choices[0].message.content
    except Exception:
        # Fallback: if different attribute names
        content = getattr(response, 'text', '') or str(response)
    label, reason = parse_llm_response(content)
    return label, reason

def main():
    # Connect to server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((HOST, PORT))
        except ConnectionRefusedError:
            print(f"Could not connect to server at {HOST}:{PORT}. Ensure server.py is running.", file=sys.stderr)
            return
        buffer = ""
        print(f"Client connected to server at {HOST}:{PORT}.\n")

        while True:
            try:
                chunk = s.recv(1024).decode()
            except Exception as e:
                print(f"Socket error: {e}", file=sys.stderr)
                break
            if not chunk:
                # Connection closed
                print("Server closed connection.")
                break
            buffer += chunk

            # Process line-delimited JSON messages
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

                # Preprocess for model
                df_point = pre_process_data(data)
                # Convert to numpy array for prediction
                X_point = df_point.values  # shape (1, n_features)
                # Predict: 1 for normal, -1 for anomaly
                try:
                    pred = model.predict(X_point)[0]
                except Exception as e:
                    print(f"Error during model prediction: {e}", file=sys.stderr)
                    continue

                if pred == -1:
                    # Anomaly detected
                    print("Model prediction: Anomaly detected.")
                    label, reason = alert_llm_for_anomaly(data)
                    if label is None and reason is None:
                        print("🚨 Anomaly Detected, but failed to get LLM alert.")
                    else:
                        print(f"\n🚨 Anomaly Detected!\nLabel: {label}\nReason: {reason}\n")
                    # Optionally, you could log anomalies to a CSV or take further action here.
                else:
                    # Normal data
                    print("Model prediction: Normal.")

    print("Client exiting.")

if __name__ == "__main__":
    main()
