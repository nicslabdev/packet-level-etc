import argparse
import time
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tabulate import tabulate  # pip install tabulate if not already installed
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
try:
    from thundersvm import SVC as thunderSVC
    THUNDERSVM_OK = True
except ImportError:
    from sklearn.svm import SVC
    thunderSVC = SVC  # fallback
    THUNDERSVM_OK = False


# -------------------------------
# Define classifier factory
# -------------------------------
def select_classifier(model_name):
    model_name = model_name.lower()
    models = {
        'knn': lambda: KNeighborsClassifier(n_neighbors=10),
        'gaussiannb': lambda: GaussianNB(),
        'svm': lambda: thunderSVC(kernel='rbf'),
        'linearsvm': lambda: thunderSVC(kernel='linear'),
        #'svm': lambda: SVC(kernel='rbf'),
        #'linearsvm': lambda: LinearSVC(max_iter=10000),
        'randomforest': lambda: RandomForestClassifier(n_estimators=100),
        'softmax': lambda: SGDClassifier(loss='log_loss', learning_rate='constant', eta0=0.01, max_iter=1000, tol=1e-3),
        'xgboost': lambda: XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0),
        'lgbm': lambda: LGBMClassifier()
    }
    if model_name in models:
        return models[model_name]()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

# -------------------------------
# Train and evaluate a model
# -------------------------------
def test_classifier(model_name, X_train, X_test, y_train, y_test, label_encoder):
    clf = select_classifier(model_name)

    start_time = time.time()
    print(f"Training {model_name.upper()}...")
    clf.fit(X_train, y_train)
    print(f"Training done in {time.time() - start_time:.2f} seconds.")
    y_pred = clf.predict(X_test)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Training + prediction done in {elapsed:.2f} seconds.")

    report = classification_report(
        y_test, y_pred, output_dict=True,
        target_names=label_encoder.classes_,
        zero_division=0
    )
    
    accuracy = report["accuracy"]
    macro = report["macro avg"]
    weighted = report["weighted avg"]

    print(f"\nResults for {model_name.upper()}:")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro Precision: {macro['precision']:.4f}")
    print(f"Macro Recall:    {macro['recall']:.4f}")
    print(f"Macro F1-score:  {macro['f1-score']:.4f}")
    print(f"Weighted Precision: {weighted['precision']:.4f}")
    print(f"Weighted Recall:    {weighted['recall']:.4f}")
    print(f"Weighted F1-score:  {weighted['f1-score']:.4f}")
    print(classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

    return {
        "Model": model_name.lower(),
        "Time (s)": round(elapsed, 2),
        "Accuracy": round(accuracy, 4),
        "Macro Precision": round(macro["precision"], 4),
        "Macro Recall": round(macro["recall"], 4),
        "Macro F1": round(macro["f1-score"], 4),
        "Weighted Precision": round(weighted["precision"], 4),
        "Weighted Recall": round(weighted["recall"], 4),
        "Weighted F1": round(weighted["f1-score"], 4),
    }, clf
    
def export_model_bundle(model, le, scaler, input_file, model_name, input_dim, framework="scikit-learn", model_save_dir="models"):
    from joblib import dump
    import os
    import json
    from datetime import datetime

    os.makedirs(model_save_dir, exist_ok=True)

    # Base name
    base_name = f"{model_name.lower()}_{os.path.splitext(os.path.basename(input_file))[0]}"
    
    # File names
    model_filename = f"{base_name}.joblib"
    le_filename = f"le_{os.path.splitext(os.path.basename(input_file))[0]}.joblib"
    scaler_filename = f"scaler_{os.path.splitext(os.path.basename(input_file))[0]}.joblib"
    config_filename = f"{base_name}.json"

    # Paths
    model_path = os.path.join(model_save_dir, model_filename)
    le_path = os.path.join(model_save_dir, le_filename)
    scaler_path = os.path.join(model_save_dir, scaler_filename)
    config_path = os.path.join(model_save_dir, config_filename)

    # Save model, LabelEncoder, and Scaler
    dump(model, model_path)
    dump(le, le_path)
    dump(scaler, scaler_path)

    print(f"💾 Model saved in: {model_path}")
    print(f"💾 LabelEncoder saved in: {le_path}")
    print(f"💾 Scaler saved in: {scaler_path}")

    # Save config
    config = {
        "model_name": model_name.lower(),
        "input_file": os.path.basename(input_file),
        "framework": framework,
        "input_dim": input_dim,
        "num_classes": len(le.classes_),
        "class_labels": le.classes_.tolist(),
        "model_file": os.path.basename(model_path),
        "label_encoder_file": os.path.basename(le_path),
        "scaler_file": os.path.basename(scaler_path),
        "created_at": datetime.now().isoformat()
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"📝 Config saved in: {config_path}")


# -------------------------------
# Main entry point
# -------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate classifiers using features extracted from a .npz file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", type=str, required=True, help="Path to the .npz file with features and labels")
    parser.add_argument("--models", type=str, nargs="+", required=True,
                        help="List of models to train (e.g., knn svm). Use 'all' to run all supported models.")
    parser.add_argument("--train_fraction", type=float, default=1.0,
                    help="Fraction of the dataset to use for training (0.1 to 1.0)")
    parser.add_argument("--export", action="store_true", help="If enabled, export the trained model to /models.")

    args = parser.parse_args()

    if 'svm' in [m.lower() for m in args.models] and not THUNDERSVM_OK:
        print("WARNING: ThunderSVM is not installed. Using scikit-learn's SVC (CPU).")


    input_file = args.input
    selected_models = args.models

    print(f"Loading data from: {input_file}")
    data = np.load(input_file)
    X = data["X"].astype(np.float32)
    y = data["y"]
    if args.train_fraction < 1.0:
        print(f"Reducing dataset to {int(args.train_fraction * 100)}% using stratified sampling...")
        sss = StratifiedShuffleSplit(n_splits=1, train_size=args.train_fraction, random_state=42)
        idx, _ = next(sss.split(X, y))
        X = X[idx]
        y = y[idx]

    print("Encoding string labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.33, stratify=y_encoded, random_state=42
    )

    print("Normalizing features (fit on TRAIN only)...")
    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)


    available_models = ['knn', 'gaussiannb', 'svm', 'randomforest', 'softmax', 'xgboost', 'lgbm']
    models_to_run = available_models if 'all' in [m.lower() for m in selected_models] else selected_models

    summary = []
    for model in models_to_run:
        result, clf = test_classifier(model, X_train, X_test, y_train, y_test, label_encoder)
        summary.append(result)

        if args.export:
            is_svm_family = model.lower() in ("svm", "linearsvm")
            framework = "thundersvm" if (THUNDERSVM_OK and is_svm_family) else "scikit-learn"
        
            export_model_bundle(
                model=clf,
                le=label_encoder,
                scaler=scaler,
                input_file=input_file,
                model_name=model,
                input_dim=X.shape[1],
                framework=framework
            )

    # Show summary table
    print("\nSummary:")
    print(tabulate(summary, headers="keys", tablefmt="grid"))


if __name__ == "__main__":
    main()
