import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, mean_squared_error, top_k_accuracy_score, classification_report, confusion_matrix, r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from joblib import dump
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

# === CONFIG ===
DATA_DIR = "C:/Users/mruga/Desktop/New folder/Capstone Project/AutoShopIQ_Repair_chatbot/Data/Modeling_Prepared"
MODEL_DIR = "C:/Users/mruga/Desktop/New folder/Capstone Project/AutoShopIQ_Repair_chatbot/Models"
os.makedirs(MODEL_DIR, exist_ok=True)

# === HELPER: Load CSVs ===
def load_csv(name):
    return pd.read_csv(os.path.join(DATA_DIR, name))

def load_numeric_series(name):
    return pd.to_numeric(load_csv(name).squeeze(), errors="coerce").fillna(0).clip(lower=0)

# === FEATURE ENGINEERING ===
def get_vehicle_tier(make):
    luxury_brands = ["BMW", "Mercedes-Benz", "Audi", "Lexus", "Jaguar", "Infiniti", "Tesla", "Acura", "Porsche", "Cadillac"]
    economy_brands = ["Toyota", "Honda", "Hyundai", "Kia", "Chevrolet", "Ford", "Nissan", "Mazda", "Volkswagen"]
    if make in luxury_brands:
        return "luxury"
    elif make in economy_brands:
        return "economy"
    else:
        return "mid"

def prepare_features(X):
    X = X.copy()
    if all(col in X.columns for col in ['make', 'model', 'year']):
        X['vehicle'] = X['make'] + " " + X['model'] + " " + X['year'].astype(str)
        X['vehicle_tier'] = X['make'].apply(get_vehicle_tier)
    else:
        X['vehicle'] = "Unknown"
        X['vehicle_tier'] = "mid"
    X['concern'] = X.get('concern', "no_concern_provided").fillna("no_concern_provided")
    if 'mileage_in' in X.columns:
        X['mileage_in'] = pd.to_numeric(X['mileage_in'], errors="coerce").fillna(0).clip(lower=0)
    else:
        X['mileage_in'] = pd.Series([0] * len(X))
    keywords = ['noise', 'leak', 'brake', 'check', 'oil']
    for kw in keywords:
        X[f'has_{kw}'] = X['concern'].str.contains(kw, case=False, na=False).astype(int)
    return X

# === BERT SETUP ===
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

def encode_bert(texts):
    return bert_model.encode(texts.tolist() if isinstance(texts, pd.Series) else texts)

# === UPDATED TRANSFORM FEATURES ===
dummy_columns = []

def get_transformed_features(X, is_train=True):
    concern_vecs = encode_bert(X['concern'])
    cat_cols = pd.get_dummies(X[['vehicle', 'vehicle_tier']], drop_first=False)
    global dummy_columns
    if is_train:
        dummy_columns = cat_cols.columns.tolist()
    else:
        for col in dummy_columns:
            if col not in cat_cols.columns:
                cat_cols[col] = 0
        cat_cols = cat_cols[dummy_columns]
    scaled_mileage = StandardScaler().fit_transform(X[['mileage_in']])
    keyword_flags = X[[col for col in X.columns if col.startswith('has_')]]
    return np.hstack([concern_vecs, cat_cols.values, scaled_mileage, keyword_flags.values]).astype(np.float32)

# === UPDATED CLASSIFIER ===
def train_lgb_classifier(X_train, y_train, X_test, y_test, name, encoder):
    # Optional: log class distribution
    print("\n🔍 Class distribution (training):")
    print(Counter(y_train))

    # Filter to most common classes (optional debug mode)
    class_counts = Counter(y_train)
    top_classes = [cls for cls, count in class_counts.items() if count >= 20]
    filter_mask = np.isin(y_train, top_classes)
    X_train, y_train = X_train[filter_mask], y_train[filter_mask]

    # Skipping oversampling due to memory limits
    X_train_res, y_train_res = X_train, y_train

    # Filter test set to the same top classes
    filter_mask_test = np.isin(y_test, top_classes)
    X_test, y_test = X_test[filter_mask_test], y_test[filter_mask_test]

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(np.unique(y_train)),
        class_weight="balanced",
        verbosity=1,
        importance_type="gain"
    )
    model.fit(X_train_res, y_train_res)

    # Log feature importance
    print("\n📈 Feature Importances:")
    importances = model.feature_importances_
    for i, imp in enumerate(importances[:10]):
        print(f"Feature {i}: {imp}")

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    top5 = top_k_accuracy_score(y_test, model.predict_proba(X_test), k=5, labels=np.unique(y_test))
    print(f"✅ {name} Accuracy: {acc:.2f} | Top-5 Accuracy: {top5:.2f}")

    print("\n📋 Classification Report:")
    print(classification_report(y_test, preds, labels=np.unique(y_test), target_names=[encoder.classes_[i] for i in np.unique(y_test)]))

    print("\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # Misclassification logging
    mismatches = y_test != preds
    if mismatches.any():
        print("\n🔍 Misclassified Examples:")
        for i in np.where(mismatches)[0][:10]:
            true_label = encoder.classes_[y_test[i]]
            pred_label = encoder.classes_[preds[i]]
            print(f"Example {i}: True={true_label} | Predicted={pred_label}")

    dump(model, os.path.join(MODEL_DIR, f"{name}.joblib"))
    dump(encoder, os.path.join(MODEL_DIR, f"{name}_encoder.joblib"))
    decoded_map = {int(i): label for i, label in enumerate(encoder.classes_)}
    with open(os.path.join(MODEL_DIR, f"{name}_label_mapping.json"), "w") as f:
        json.dump(decoded_map, f, indent=2)

# === FINAL TRAINING CALL ===
# Load and prepare data
X_job_train_raw = load_csv("X_job_train.csv")
X_job_test_raw = load_csv("X_job_test.csv")
y_job_train_raw = load_csv("y_job_train.csv").squeeze()
y_job_test_raw = load_csv("y_job_test.csv").squeeze()

# Label encoding
job_encoder = LabelEncoder()
y_job_train = job_encoder.fit_transform(y_job_train_raw)

# Filter y_job_test to only known labels
test_mask = y_job_test_raw.isin(job_encoder.classes_)
y_job_test_raw = y_job_test_raw[test_mask].reset_index(drop=True)
X_job_test_raw = X_job_test_raw[test_mask].reset_index(drop=True)
y_job_test = job_encoder.transform(y_job_test_raw)

# Feature transformation
X_job_train_raw = prepare_features(X_job_train_raw)
X_job_test_raw = prepare_features(X_job_test_raw)
X_job_train = get_transformed_features(X_job_train_raw, is_train=True)
X_job_test = get_transformed_features(X_job_test_raw, is_train=False)

# Train the model
train_lgb_classifier(X_job_train, y_job_train, X_job_test, y_job_test, "job_type_classifier", job_encoder)
