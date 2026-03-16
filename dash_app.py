import warnings
from cryptography.utils import CryptographyDeprecationWarning
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

import base64
import io
import time
import threading
import pandas as pd
import numpy as np
from collections import deque, Counter
from scapy.all import sniff, get_if_list, rdpcap, IP, UDP, IPv6
from joblib import load
from dash import Dash, dcc, html, Input, Output, State, ctx
from dash.exceptions import PreventUpdate
import plotly.express as px

# ======================= MODEL AND CONFIG =======================
#MODEL_PATH = "models/randomforest_data1000_N100_BIT8.joblib"
#SCALER_PATH = "models/scaler_data1000_N100_BIT8.joblib"
#ENCODER_PATH = "models/le_data1000_N100_BIT8.joblib"

MODEL_PATH = "models/randomforest_iscxcustom_N100_BIT8.joblib"
SCALER_PATH = "models/scaler_iscxcustom_N100_BIT8.joblib"
ENCODER_PATH = "models/le_iscxcustom_N100_BIT8.joblib"

model = load(MODEL_PATH)
scaler = load(SCALER_PATH)
label_encoder = load(ENCODER_PATH)

N_BYTES = 100
BIT_TYPE = 8
packet_buffer = deque(maxlen=10000)
lock = threading.Lock()

running = False
bpf_filter = ""
pcap_results = {}  # Dictionary: filename -> list of classified packets
pcap_biflows = {}  # filename -> list of biflow keys

# ======================= PROCESSING FUNCTIONS =======================
def extract_features(pkt):
    """
    Extracts byte-level features from a given packet.

    Args:
        pkt: A scapy packet object.

    Returns:
        A NumPy array of size N_BYTES containing padded and cleaned byte values,
        or None if the packet is IPv6.
    """
    if IPv6 in pkt:
        return None
    if IP in pkt:
        raw_bytes = bytes(pkt[IP])[:N_BYTES]
    else:
        raw_bytes = bytes(pkt)[:N_BYTES]
    if len(raw_bytes) > 24:
        raw_bytes = raw_bytes[:12] + raw_bytes[24:]
    if UDP in pkt and len(raw_bytes) > 28:
        raw_bytes = raw_bytes[:28] + b'\x00' * 12 + raw_bytes[28:]
    byte_array = np.frombuffer(raw_bytes, dtype=np.uint8)
    padded_array = np.pad(byte_array, (0, N_BYTES - len(byte_array)), 'constant')
    return padded_array

def bitize(features, bit_type=8):
    """
    Normalizes a feature vector to the [0, 1] range using BITization

    Args:
        features: NumPy array of feature values.
        bit_type: BITization type (only 8 supported).

    Returns:
        Normalized NumPy array.
    """
    return features.astype(np.float32) / 255.0

def classify_packet(pkt):
    """
    Applies the trained model to classify a packet.

    Args:
        pkt: A scapy packet object.

    Returns:
        A dictionary with classification result and metadata (timestamp, src, dst, label, len),
        or None if the packet is not valid for classification.
    """
    feat = extract_features(pkt)
    if feat is None:
        return None
    feat = bitize(feat.reshape(1, -1), BIT_TYPE)
    feat = scaler.transform(feat)
    pred = model.predict(feat)
    label = label_encoder.inverse_transform(pred)[0]

    # Extract ports and protocol
    src_port = pkt.sport if hasattr(pkt, 'sport') else None
    dst_port = pkt.dport if hasattr(pkt, 'dport') else None
    proto = pkt.proto if hasattr(pkt, 'proto') else (pkt[IP].proto if IP in pkt else None)
    proto_str = {6: 'TCP', 17: 'UDP'}.get(proto, str(proto))

    return {
        "timestamp": time.time(),
        "src": pkt[IP].src if IP in pkt else "?",
        "dst": pkt[IP].dst if IP in pkt else "?",
        "sport": src_port,
        "dport": dst_port,
        "proto": proto_str,
        "label": label,
        "len": len(pkt)
    }

# ======================= DASH APP =======================
app = Dash(__name__)
app.title = "Traffic Classifier"

app.layout = html.Div([
    html.H1("📡 Live and PCAP Traffic Classifier"),
    dcc.Tabs([
        dcc.Tab(label="🟢 Live Capture", children=[
            dcc.Input(id="bpf-filter", type="text", placeholder="BPF Filter (e.g., tcp port 80)", style={"width": "100%", "marginBottom": "10px"}),
            html.Button("▶️ Start Capture", id="start-button", n_clicks=0),
            html.Button("⏹️ Stop Capture", id="stop-button", n_clicks=0),
            html.Div(id="status"),
            dcc.Interval(id="update-interval", interval=500, n_intervals=0),
            html.Label("Filter by label:"),
            dcc.Checklist(id="live-label-filter", options=[], inline=True),
            html.Label("Filter by IP (source or destination):"),
            dcc.Dropdown(id="live-ip-filter", options=[], placeholder="Select or type an IP", multi=True, searchable=True),
            html.Br(),
            dcc.Graph(id="live-graph"),
        ]),

        dcc.Tab(label="📂 PCAP", children=[
            dcc.Upload(
                id="upload-pcap",
                children=html.Div(["📁 Drag and drop or click to upload a .pcap or .pcapng file"]),
                multiple=True,
                style={"border": "2px dashed #aaa", "padding": "20px", "marginTop": "20px"}
            ),
            dcc.Loading(
    type="default",
    children=[
        dcc.Dropdown(id="pcap-dropdown", placeholder="Select an uploaded file"),
        html.Div(id="pcap-loading", children="", style={"marginTop": "10px", "color": "green"})
    ]
),
            html.Div([
    html.Label("Filter by label:"),
    dcc.Checklist(id="label-filter", options=[], inline=True),
    html.Label("Filter by IP (source or destination):"),
    dcc.Dropdown(id="ip-filter", options=[], placeholder="Select or type an IP", multi=True, searchable=True),
    html.Label("Filter by biflow (IP:port ⬌ IP:port):"),
    dcc.Dropdown(id="biflow-filter", options=[], placeholder="Select one or more biflows", multi=True),
    html.Br(),
], style={"marginTop": "10px"}),
html.Div(id="pcap-summary"),
dcc.Graph(id="pcap-graph"),
html.Hr(),
html.H4("📊 Label distribution per biflow (independent of filters)"),
dcc.Graph(id="biflow-label-graph"),
        ]),
    ])
])

# ======================= CAPTURE =======================
def capture():
    """
    Starts the packet sniffing process using a global BPF filter.
    Packets are processed and passed to the buffer.
    """
    global running, bpf_filter
    sniff(prn=lambda pkt: store_in_buffer(pkt), store=0, stop_filter=lambda x: not running, filter=bpf_filter)

def store_in_buffer(pkt):
    result = classify_packet(pkt)
    if result:
        with lock:
            packet_buffer.append(result)

def start_capture():
    global running
    if not running:
        running = True
        threading.Thread(target=capture, daemon=True).start()

def stop_capture():
    global running
    running = False

# ======================= CALLBACKS =======================
@app.callback(
    Output("status", "children"),
    Input("start-button", "n_clicks"),
    Input("stop-button", "n_clicks"),
    State("bpf-filter", "value"),
    prevent_initial_call=True
)
def handle_capture(start_clicks, stop_clicks, filter_value):
    global bpf_filter
    action = ctx.triggered_id
    if action == "start-button":
        bpf_filter = filter_value or ""
        start_capture()
        return f"✅ Capture started. Filter: {bpf_filter or 'none'}"
    elif action == "stop-button":
        stop_capture()
        return "⛔ Capture stopped"
    return ""

@app.callback(
    Output("live-graph", "figure"),
    Output("live-label-filter", "options"),
    Output("live-ip-filter", "options"),
    Input("update-interval", "n_intervals"),
    State("live-label-filter", "value"),
    State("live-ip-filter", "value")
)
def update_live_graph(n, selected_labels, selected_ips):
    with lock:
        if not packet_buffer:
            empty_fig = px.scatter(title="Waiting for packets...")
            return empty_fig, [], []
        df = pd.DataFrame(packet_buffer)
    if selected_labels:
        df = df[df["label"].isin(selected_labels)]
    if selected_ips:
        df = df[df["src"].isin(selected_ips) | df["dst"].isin(selected_ips)]

    df["Time"] = pd.to_datetime(df["timestamp"], unit="s")
    df.set_index("Time", inplace=True)
    df_resample = df.groupby("label").resample("100ms").size().reset_index(name="count")
    fig = px.line(df_resample, x="Time", y="count", color="label", )

    # Add percentage info to legend title
    label_counts = df["label"].value_counts()
    total = len(df)
    fig.for_each_trace(lambda t: t.update(name=f"{t.name} ({label_counts[t.name] / total:.1%})"))
    label_options = sorted(df["label"].unique())
    options = [{'label': lbl, 'value': lbl} for lbl in label_options]
    ip_options = sorted(set(df['src']).union(df['dst']))
    ip_dropdown_options = [{'label': ip, 'value': ip} for ip in ip_options]
    return fig, options, ip_dropdown_options

@app.callback(
    Output("pcap-dropdown", "options"),
    Output("pcap-dropdown", "value"),
    Output("pcap-loading", "children"),
    Output("label-filter", "options"),
    Input("upload-pcap", "contents"),
    State("upload-pcap", "filename"),
    prevent_initial_call=True
)
def load_pcap_file(contents, filenames):
    """Loads PCAP files and classifies their packets while displaying progress."""
    global pcap_results, pcap_biflows
    if contents is None:
        raise PreventUpdate

    for c, name in zip(contents, filenames):
        content_type, content_string = c.split(',')
        decoded = base64.b64decode(content_string)
        file = io.BytesIO(decoded)
        packets = rdpcap(file)
        classified = [classify_packet(pkt) for pkt in packets if classify_packet(pkt)]
        pcap_results[name] = classified

        # Extract biflows from classified packets
        biflow_keys = set()
        for pkt in classified:
            try:
                if pkt["sport"] is None or pkt["dport"] is None:
                    continue  # Skip packets without valid ports
                ip1, ip2 = sorted([pkt["src"], pkt["dst"]])
                port1, port2 = sorted([int(pkt["sport"]), int(pkt["dport"])])
                proto = pkt["proto"]
                biflow_keys.add(((ip1, port1), (ip2, port2), proto))
            except Exception as e:
                continue  # Skip if any issue with data


        pcap_biflows[name] = list(biflow_keys)

    labels_set = sorted({pkt["label"] for pkts in pcap_results.values() for pkt in pkts})
    return list(pcap_results.keys()), filenames[-1], "✅ PCAP files loaded successfully.", [{'label': lbl, 'value': lbl} for lbl in labels_set] # select the last uploaded

@app.callback(
    Output("pcap-summary", "children"),
    Output("pcap-graph", "figure"),
    Output("ip-filter", "options"),
    Output("biflow-filter", "options"),
    Output("biflow-label-graph", "figure"),
    Input("pcap-dropdown", "value"),
    Input("label-filter", "value"),
    Input("ip-filter", "value"),
    Input("biflow-filter", "value")
)
def display_pcap(name, selected_labels, selected_ips, selected_biflows):
    if not name or name not in pcap_results:
        raise PreventUpdate

    data = pcap_results[name]
    df = pd.DataFrame(data)

    if selected_labels:
        df = df[df["label"].isin(selected_labels)]
    if selected_ips:
        df = df[df["src"].isin(selected_ips) | df["dst"].isin(selected_ips)]

    def pkt_to_biflow_key(pkt):
        try:
            if pkt["sport"] is None or pkt["dport"] is None:
                return None
            ip1, ip2 = sorted([pkt["src"], pkt["dst"]])
            port1, port2 = sorted([int(pkt["sport"]), int(pkt["dport"])])
            proto = pkt["proto"]
            return ((ip1, port1), (ip2, port2), proto)
        except:
            return None
        
    def biflow_key_to_str(key):
        (ip1, port1), (ip2, port2), proto = key
        return f"[{proto}] {ip1}:{port1} ⬌ {ip2}:{port2}"

    df["biflow_key"] = df.apply(pkt_to_biflow_key, axis=1)
    df = df[df["biflow_key"].notna()]
    df["biflow_str"] = df["biflow_key"].apply(biflow_key_to_str)

    if selected_biflows:
        df = df[df["biflow_str"].isin(selected_biflows)]

    print("Selected biflows:", selected_biflows)
    print("DF after biflow filtering:", df.shape)

    if df.empty:
        return html.Div("⚠️ No data after biflow filter."), px.scatter(title="No data"), [], []


    df["Time"] = pd.to_datetime(df["timestamp"], unit="s")
    df.set_index("Time", inplace=True)
    df_resample = df.groupby("label").resample("100ms").size().reset_index(name="count")
    fig = px.line(df_resample, x="Time", y="count", color="label", title=f"PCAP: {name}")

    summary = Counter(df["label"])
    total = sum(summary.values())
    summary_html = html.Ul([
        html.Li(f"{label}: {count} ({count / total:.1%})") for label, count in summary.items()
    ])

    ip_options = sorted(set(df['src']).union(df['dst']))
    ip_dropdown_options = [{'label': ip, 'value': ip} for ip in ip_options]

    # Count packets per biflow_str
    biflow_counts = df["biflow_str"].value_counts().to_dict()
    biflows = pcap_biflows.get(name, [])
    biflow_options = [{
        "label": f"{biflow_key_to_str(key)} ({biflow_counts.get(biflow_key_to_str(key), 0)} pkts)",
        "value": biflow_key_to_str(key)
    } for key in biflows]


    # === Generate biflow-wise label distribution chart (unfiltered) ===
    df_all = pd.DataFrame(pcap_results[name])
    df_all["biflow_key"] = df_all.apply(pkt_to_biflow_key, axis=1)
    df_all = df_all[df_all["biflow_key"].notna()]
    df_all["biflow_str"] = df_all["biflow_key"].apply(biflow_key_to_str)

    # Group by biflow and label, then count
    grouped = df_all.groupby(["biflow_str", "label"]).size().reset_index(name="count")

    # Create stacked bar chart
    biflow_label_fig = px.bar(
        grouped,
        x="biflow_str",
        y="count",
        color="label",
        title="Label distribution per biflow",
    )

    # Rotate x labels for readability
    biflow_label_fig.update_layout(xaxis_tickangle=-45, barmode="stack", height=500)


    return summary_html, fig, ip_dropdown_options, biflow_options, biflow_label_fig

# ======================= MAIN =======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
