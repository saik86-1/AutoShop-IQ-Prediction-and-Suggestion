import pandas as pd
import os
import json
from joblib import dump
from sklearn.model_selection import train_test_split

# === CONFIGURATION ===
INPUT_CSV = "C:/Users/mruga/Desktop/New folder/Capstone Project/AutoShopIQ_Repair_chatbot/Data/Invoices_converted_csv/structured_repair_data.csv"
OUTPUT_DIR = "C:/Users/mruga/Desktop/New folder/Capstone Project/AutoShopIQ_Repair_chatbot/Data/Modeling_Prepared"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD DATA ===
df = pd.read_csv(INPUT_CSV)

# === CLEAN INPUT FIELDS ===
df["concern"] = df["concern"].fillna("unknown")
df["make"] = df["make"].fillna("unknown")
df["model"] = df["model"].fillna("unknown")
df["year"] = df["year"].fillna("unknown").astype(str)

# === DEBUG: Check counts ===
print("🔍 Original rows:", len(df))
print("🚨 Missing job_type:", df["job_type"].isna().sum())
print("🚨 Missing part_name:", df["part_name"].isna().sum())

# === ENCODERS ===
def encode_features(df_subset):
    return pd.DataFrame({
        "concern": df_subset["concern"],
        "make": df_subset["make"],
        "model": df_subset["model"],
        "year": df_subset["year"]
    })

# === FILTERED TARGET SETS ===
df_job = df.dropna(subset=["job_type"]).copy()
df_part = df.dropna(subset=["part_name"]).copy()
df_labor_cost = df.dropna(subset=["labor_total_cost"]).copy()
df_labor_hours = df.dropna(subset=["labor_hours"]).copy()
df_part_cost = df.dropna(subset=["part_total_cost"]).copy()

# === ENCODE AND SPLIT ===
def encode_and_split(df_subset, target_column):
    if df_subset.empty:
        print(f"⚠️ WARNING: No rows found for target column '{target_column}'")
        return pd.DataFrame(), pd.DataFrame(), [], [], None

    X = encode_features(df_subset)
    y = df_subset[target_column].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, None

# Train/test splits
X_job_train, X_job_test, y_job_train, y_job_test, _ = encode_and_split(df_job, "job_type")
X_part_train, X_part_test, y_part_train, y_part_test, _ = encode_and_split(df_part, "part_name")
X_labor_cost_train, X_labor_cost_test, y_labor_cost_train, y_labor_cost_test, _ = encode_and_split(df_labor_cost, "labor_total_cost")
X_labor_hours_train, X_labor_hours_test, y_labor_hours_train, y_labor_hours_test, _ = encode_and_split(df_labor_hours, "labor_hours")
X_part_cost_train, X_part_cost_test, y_part_cost_train, y_part_cost_test, _ = encode_and_split(df_part_cost, "part_total_cost")

# Save function
def save_to_csv(name, train, test):
    try:
        if isinstance(train, pd.DataFrame) or isinstance(train, pd.Series):
            train.to_csv(os.path.join(OUTPUT_DIR, f"{name}_train.csv"), index=False)
        if isinstance(test, pd.DataFrame) or isinstance(test, pd.Series):
            test.to_csv(os.path.join(OUTPUT_DIR, f"{name}_test.csv"), index=False)
    except Exception as e:
        print(f"❌ Error saving {name}: {e}")

save_to_csv("X_job", X_job_train, X_job_test)
save_to_csv("y_job", pd.Series(y_job_train), pd.Series(y_job_test))
save_to_csv("X_part", X_part_train, X_part_test)
save_to_csv("y_part", pd.Series(y_part_train), pd.Series(y_part_test))
save_to_csv("X_labor_cost", X_labor_cost_train, X_labor_cost_test)
save_to_csv("y_labor_cost", pd.Series(y_labor_cost_train), pd.Series(y_labor_cost_test))
save_to_csv("X_labor_hours", X_labor_hours_train, X_labor_hours_test)
save_to_csv("y_labor_hours", pd.Series(y_labor_hours_train), pd.Series(y_labor_hours_test))
save_to_csv("X_part_cost", X_part_cost_train, X_part_cost_test)
save_to_csv("y_part_cost", pd.Series(y_part_cost_train), pd.Series(y_part_cost_test))

print("✅ Preprocessing completed and saved.")
