import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from torch.utils.data import TensorDataset, random_split, DataLoader
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from tabulate import tabulate
from sklearn.metrics import confusion_matrix
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from datetime import datetime
from joblib import dump

# -------------------------------
# MLP
# -------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(400, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(300, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(200, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Dropout(0.05),
        )
        self.classifier = nn.Linear(50, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x)
    
# -------------------------------
# SAE
# -------------------------------

class AutoencoderBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AutoencoderBlock, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        encoded = self.activation(self.encoder(x))
        decoded = self.decoder(encoded)
        return encoded, decoded
    
def train_autoencoder_block(block, data_loader, epochs=20, lr=1e-3, device='cpu', layer_idx=None):
    block.to(device)
    optimizer = optim.Adam(block.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        block.train()
        total_loss = 0
        for x, _ in data_loader:
            x = x.to(device)
            _, decoded = block(x)
            loss = criterion(decoded, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        print(f"    📦 Layer {layer_idx} | Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.6f}")

    return block


    
class SAEClassifier(nn.Module):
    def __init__(self, layer_dims, num_classes):
        super(SAEClassifier, self).__init__()
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(layer_dims[i], layer_dims[i+1]),
                nn.ReLU(inplace=True),
                nn.Dropout(0.05)
            ) for i in range(len(layer_dims) - 1)
        ])
        self.classifier = nn.Linear(layer_dims[-1], num_classes)

    def forward(self, x):
        for encoder in self.encoders:
            x = encoder(x)
        return self.classifier(x)

    def encode(self, x):
        for encoder in self.encoders:
            x = encoder(x)
        return x
    

"""
Encodes data_cpu (tensor on CPU) with the block's encoder, in mini-batches.
Returns a CPU tensor with shape = [N, hidden_dim].
"""
def _encode_dataset_batched(block, data_cpu, batch_size=2048, device='cuda'):
    block.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, data_cpu.size(0), batch_size):
            xb = data_cpu[i:i+batch_size].to(device, non_blocking=True)
            zb = block.activation(block.encoder(xb))
            outs.append(zb.detach().cpu())  # return to the CPU so as not to use VRAM.
    return torch.cat(outs, dim=0)

    
def pretrain_sae(model, dataset, layer_dims, batch_size=128, pretrain_epochs=20, device='cpu'):
    current_data = dataset.tensors[0].cpu().contiguous()

    for i in range(len(layer_dims) - 1):
        print(f"\n🧱 Pretraining layer {i+1}/{len(layer_dims) - 1}: {layer_dims[i]} ➝ {layer_dims[i+1]}")
        block = AutoencoderBlock(layer_dims[i], layer_dims[i+1])

        temp_dataset = TensorDataset(current_data, current_data)
        temp_loader = DataLoader(
            temp_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=(str(device).startswith('cuda')),
            num_workers=0
        )

        block = train_autoencoder_block(block, temp_loader, epochs=pretrain_epochs, device=device, layer_idx=i+1)

        # Copy weights to the main model (without extra clone)
        model.encoders[i][0].weight.data.copy_(block.encoder.weight.data)
        model.encoders[i][0].bias.data.copy_(block.encoder.bias.data)

        # Encode in mini-batches on GPU and return to CPU
        encode_bs = max(1024, batch_size)  # ajustable
        current_data = _encode_dataset_batched(block, current_data, batch_size=encode_bs, device=device)

        # Free up VRAM between layers
        del block
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()

    return model





# -------------------------------
# CNN1D
# -------------------------------
class CNN1D(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=200, kernel_size=4, stride=3)
        self.bn1 = nn.BatchNorm1d(200)
        self.conv2 = nn.Conv1d(in_channels=200, out_channels=200, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm1d(200)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.05)

        # Calculate final output of convolutions to connect to dense layer
        conv_output_len = self._compute_conv_output_length(input_dim)

        self.fc1 = nn.Linear(conv_output_len * 200, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 50)
        self.classifier = nn.Linear(50, num_classes)

    def _compute_conv_output_length(self, input_length):
        x = torch.zeros(1, 1, input_length)
        x = self.pool(self.bn2(self.conv2(self.bn1(self.conv1(x)))))
        return x.shape[2]

    def forward(self, x):
        x = x.unsqueeze(1)  # reshape (batch_size, features) → (batch_size, 1, features)
        x = self.dropout(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        return self.classifier(x)
    
def get_cnn1d(input_dim, num_classes):
    return CNN1D(input_dim, num_classes)

# -------------------------------
# Factory model
# -------------------------------
"""
def select_dl_model(model_name, input_dim, num_classes):
    model_name = model_name.lower()
    if model_name == "mlp":
        return MLP(input_dim, num_classes)
    elif model_name == "cnn1d":
        return get_cnn1d(input_dim, num_classes)
    else:
        raise ValueError(f"Unsupported DL model: {model_name}")"""
def select_dl_model(model_name, input_dim, num_classes, dataset=None, device='cpu'):
    model_name = model_name.lower()
    if model_name == "mlp":
        return MLP(input_dim, num_classes)
    elif model_name == "cnn1d":
        return get_cnn1d(input_dim, num_classes)
    elif model_name == "sae":
        layer_dims = [input_dim, 400, 300, 200, 100, 50]
        model = SAEClassifier(layer_dims, num_classes)
        if dataset is None:
            raise ValueError("Dataset is required for pretraining SAE.")
        print("🔧 Pretraining SAE layers...")
        model = pretrain_sae(model, dataset, layer_dims, device=device)
        return model
    else:
        raise ValueError(f"Unsupported DL model: {model_name}")


# -------------------------------
# Training with early stopping
# -------------------------------
def train_model_with_early_stopping(model, dataset, batch_size=128, epochs=50, patience=5, lr=1e-3, device='cpu'):
    dataset_len = len(dataset)
    train_len = int(0.64 * dataset_len)
    val_len = int(0.16 * dataset_len)
    test_len = dataset_len - train_len - val_len

    train_set, val_set, test_set = random_split(
        dataset, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train:.4f}, Val Loss = {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹️ Early stopping triggered.")
                break

    model.load_state_dict(best_state)
    return model, test_loader

def train_model_with_early_stopping_from_loaders(model, train_loader, val_loader, test_loader, epochs=50, patience=5, lr=1e-3, device='cpu'):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item()

        avg_train = train_loss / max(1, len(train_loader))
        avg_val   = val_loss   / max(1, len(val_loader))
        print(f"Epoch {epoch+1}: Train Loss = {avg_train:.4f}, Val Loss = {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹️ Early stopping triggered.")
                break

    model.load_state_dict(best_state)
    return model, test_loader

# -------------------------------
# Final evaluation
# -------------------------------
def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y.numpy())

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    accuracy = report["accuracy"]
    macro = report["macro avg"]
    weighted = report["weighted avg"]

    metrics = {
        "Accuracy": round(accuracy, 4),
        "Macro Precision": round(macro["precision"], 4),
        "Macro Recall": round(macro["recall"], 4),
        "Macro F1": round(macro["f1-score"], 4),
        "Weighted Precision": round(weighted["precision"], 4),
        "Weighted Recall": round(weighted["recall"], 4),
        "Weighted F1": round(weighted["f1-score"], 4),
    }

    return metrics, y_true, y_pred

def save_confusion_and_dendrogram(y_true, y_pred, class_labels, model_name, input_name, out_dir="images"):
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build base name
    safe_input = os.path.splitext(os.path.basename(input_name))[0]
    base_name = f"{model_name.lower()}_{safe_input}_{timestamp}"

    # Heatmap
    heatmap_path = os.path.join(out_dir, f"{base_name}_confusion.png")

    # Dendrogram
    dendro_path = os.path.join(out_dir, f"{base_name}_dendrogram.png")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_df = pd.DataFrame(cm_normalized, index=class_labels, columns=class_labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt=".2f", cmap="Blues", cbar=True)
    plt.title(f"Matriz de confusión normalizada ({model_name})")
    plt.xlabel("Predicción")
    plt.ylabel("Clase real")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()

    # Clustering jerárquico
    distance_matrix = pdist(cm_normalized, metric='euclidean')
    linkage_matrix = linkage(distance_matrix, method='ward')

    plt.figure(figsize=(12, 6))
    dendrogram(
        linkage_matrix,
        labels=class_labels,
        leaf_rotation=90,
        leaf_font_size=10
    )
    plt.title(f"Hierarchical clustering between classes ({model_name})")
    plt.ylabel("Euclidean distance")
    plt.tight_layout()
    plt.savefig(dendro_path)
    plt.close()

    print(f"✅ Saved confusion matrix and dendrogram:\n- {heatmap_path}\n- {dendro_path}")

# -------------------------------
# Export model
# -------------------------------

def export_model_bundle(model, le, input_file, model_name, input_dim, scaler=None, model_save_dir="models"):
    from joblib import dump
    import os
    import json
    from datetime import datetime
    import torch

    os.makedirs(model_save_dir, exist_ok=True)

    # Base names
    base_input = os.path.splitext(os.path.basename(input_file))[0]
    base_name = f"{model_name.lower()}_{base_input}"
    
    # File names
    model_filename = f"{base_name}.pt"
    le_filename = f"le_{base_input}.joblib"
    scaler_filename = f"scaler_{base_input}.joblib"
    config_filename = f"{base_name}.json"

    # Paths
    model_path = os.path.join(model_save_dir, model_filename)
    le_path = os.path.join(model_save_dir, le_filename)
    scaler_path = os.path.join(model_save_dir, scaler_filename)
    config_path = os.path.join(model_save_dir, config_filename)

    # Save model weights
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved in: {model_path}")

    # Save LabelEncoder
    dump(le, le_path)
    print(f"💾 LabelEncoder saved in: {le_path}")

    # Save Scaler (if provided)
    if scaler is not None:
        dump(scaler, scaler_path)
        print(f"💾 Scaler saved in: {scaler_path}")
    else:
        scaler_filename = None

    # Save configuration
    config = {
        "model_name": model_name.lower(),
        "framework": "pytorch",
        "input_file": os.path.basename(input_file),
        "input_dim": input_dim,
        "num_classes": len(le.classes_),
        "class_labels": le.classes_.tolist(),
        "created_at": datetime.now().isoformat(),
        "model_file": model_filename,
        "label_encoder_file": le_filename,
        "scaler_file": scaler_filename
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"📝 Config saved in: {config_path}")



# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train DL models on traffic data")
    parser.add_argument("--input", type=str, required=True, help="Path to .npz file")
    parser.add_argument("--models", type=str, nargs="+", required=True, help="List of DL models (e.g., mlp cnn1d)")
    parser.add_argument("--export", action="store_true", help="If enabled, export the trained model to /models.")


    args = parser.parse_args()
    input_file = args.input
    selected_models = [m.lower() for m in args.models]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"📥 Loading data from: {input_file}")
    data = np.load(input_file)
    X = data["X"]
    y = data["y"]

    print("🧪 Normalizing and preparing dataset...")

    # 1) Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # 2) Random split 64/16/20 (non-stratified)
    N = X.shape[0]
    rng = np.random.default_rng(42)
    perm = rng.permutation(N)
    n_train = int(0.64 * N)
    n_val   = int(0.16 * N)
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:n_train+n_val]
    test_idx  = perm[n_train+n_val:]

    # 3) Fit scaler only with X_train and transform each partition
    scaler = MinMaxScaler().fit(X[train_idx])
    X_train = scaler.transform(X[train_idx])
    X_val   = scaler.transform(X[val_idx])
    X_test  = scaler.transform(X[test_idx])

    # 4) Tensors and DataLoaders
    Xtr_t = torch.tensor(X_train, dtype=torch.float32)
    Xva_t = torch.tensor(X_val,   dtype=torch.float32)
    Xte_t = torch.tensor(X_test,  dtype=torch.float32)
    ytr_t = torch.tensor(y_enc[train_idx], dtype=torch.long)
    yva_t = torch.tensor(y_enc[val_idx],   dtype=torch.long)
    yte_t = torch.tensor(y_enc[test_idx],  dtype=torch.long)

    train_set = TensorDataset(Xtr_t, ytr_t)
    val_set   = TensorDataset(Xva_t, yva_t)
    test_set  = TensorDataset(Xte_t, yte_t)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=128)
    test_loader  = DataLoader(test_set,  batch_size=128)

    # 5) Dataset for pretraining SAE (inputs=targets)
    train_only_sae = TensorDataset(Xtr_t, Xtr_t)

    input_dim = X.shape[1]
    num_classes = len(le.classes_)

    summary = []

    for model_name in selected_models:
        print(f"\n🚀 Training model: {model_name.upper()}")
        start_time = time.time()

        model = select_dl_model(model_name, input_dim, num_classes, dataset=train_only_sae, device=device)
        trained_model, _ = train_model_with_early_stopping_from_loaders(model, train_loader, val_loader, test_loader, device=device)
        metrics, y_true, y_pred = evaluate_model(trained_model, test_loader, device=device)

        if args.export:
            export_model_bundle(trained_model, le, input_file, model_name, input_dim=input_dim, scaler=scaler)

        # Save graphs
        save_confusion_and_dendrogram(
            y_true=y_true,
            y_pred=y_pred,
            class_labels=le.classes_,
            model_name=model_name,
            input_name=input_file
        )


        elapsed = round(time.time() - start_time, 2)
        model_key = model_name.lower()

        # Reorder dictionary to match with ml_train_models.py
        ordered_metrics = {
            "Model": model_key,
            "Time (s)": elapsed,
            "Accuracy": metrics["Accuracy"],
            "Macro Precision": metrics["Macro Precision"],
            "Macro Recall": metrics["Macro Recall"],
            "Macro F1": metrics["Macro F1"],
            "Weighted Precision": metrics["Weighted Precision"],
            "Weighted Recall": metrics["Weighted Recall"],
            "Weighted F1": metrics["Weighted F1"],
        }

        summary.append(ordered_metrics)





    print("\n📊 Summary:")
    print(tabulate(summary, headers="keys", tablefmt="grid"))

if __name__ == "__main__":
    main()
