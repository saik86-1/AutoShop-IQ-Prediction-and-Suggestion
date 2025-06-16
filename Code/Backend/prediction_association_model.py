import pandas as pd
import os
import json
from rapidfuzz import fuzz
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from joblib import load
import numpy as np

# === Load XGBoost Models and Transformer ===
MODEL_DIR = r"C:\Users\mruga\Desktop\New folder\Capstone Project\AutoShopIQ_Repair_chatbot\Models"
transformer = load(os.path.join(MODEL_DIR, "feature_transformer.joblib"))
job_model = load(os.path.join(MODEL_DIR, "job_type_classifier.joblib"))
job_encoder = load(os.path.join(MODEL_DIR, "job_type_classifier_encoder.joblib"))

# === Step 1: Load & Normalize Data ===
df = pd.read_csv(r"C:\Users\mruga\Desktop\New folder\Capstone Project\AutoShopIQ_Repair_chatbot\Data\Invoices_converted_csv\structured_repair_data.csv")

for col in ['job_type', 'labor_job_info', 'part_name']:
    df[f'{col}_norm'] = df[col].astype(str).str.strip().str.lower()

# === Step 2: Count Top Normalized Terms ===
labor_info_counts = df['labor_job_info_norm'].value_counts().reset_index()
labor_info_counts.columns = ['labor_job_info_normalized', 'count']
part_name_counts = df['part_name_norm'].value_counts().reset_index()
part_name_counts.columns = ['part_name_normalized', 'count']

def get_similar_terms(base_term, choices, threshold=85, top_n=10):
    matches = [(desc, fuzz.ratio(base_term, desc)) for desc in choices if pd.notna(desc)]
    filtered = sorted([m for m in matches if m[1] >= threshold], key=lambda x: x[1], reverse=True)
    return pd.DataFrame(filtered[:top_n], columns=["matched_term", "similarity_score"])

# === Step 3: Fuzzy Grouping ===
def group_similar_terms(counts_df, col_name, top_n=100):
    top_terms = [term for term in counts_df[col_name].head(top_n).tolist() if pd.notna(term)]
    mapping = {}
    for base in top_terms:
        matches_df = get_similar_terms(base, counts_df[col_name])
        for match in matches_df['matched_term']:
            mapping[match] = base
    return mapping

labor_mapping = group_similar_terms(labor_info_counts, 'labor_job_info_normalized')
part_mapping = group_similar_terms(part_name_counts, 'part_name_normalized')

df['labor_job_info_grouped'] = df['labor_job_info_norm'].map(labor_mapping).fillna(df['labor_job_info_norm'])
df['part_name_grouped'] = df['part_name_norm'].map(part_mapping).fillna(df['part_name_norm'])

# === Step 4: Association Rule Mining ===
grouped_df = df.groupby("ro_number").agg({
    "concern": lambda x: list(set(x.dropna())),
    "labor_job_info_grouped": lambda x: list(set(x.dropna())),
    "part_name_grouped": lambda x: list(set(x.dropna())),
}).reset_index()

grouped_df["transaction"] = grouped_df["concern"] + grouped_df["labor_job_info_grouped"] + grouped_df["part_name_grouped"]
grouped_df["transaction"] = grouped_df["transaction"].apply(lambda x: [i for i in x if pd.notna(i) and str(i).strip().lower() != 'nan'])

te = TransactionEncoder()
te_ary = te.fit(grouped_df["transaction"]).transform(grouped_df["transaction"])
transaction_df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(transaction_df, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# === Step 5: Export Rules ===
output_dir = r"C:\Users\mruga\Desktop\New folder\Capstone Project\AutoShopIQ_Repair_chatbot\Data\association_rule_mining"
os.makedirs(output_dir, exist_ok=True)

concern_keywords = ['check', 'noise', 'diagnostic']
rules_filtered = rules[rules['antecedents'].apply(lambda x: any(k in str(x).lower() for k in concern_keywords))]
rules_filtered.sort_values("confidence", ascending=False).head(20).to_csv(os.path.join(output_dir, "filtered_rules.csv"), index=False)

concern_to_jobs = rules[
    rules['antecedents'].apply(lambda x: any(k in str(x).lower() for k in concern_keywords)) &
    rules['consequents'].apply(lambda x: 'labor' in str(x).lower())
]
concern_to_jobs.to_csv(os.path.join(output_dir, "concern_to_jobs_rules.csv"), index=False)

job_to_parts = rules[
    rules['antecedents'].apply(lambda x: 'labor' in str(x).lower()) &
    rules['consequents'].apply(lambda x: 'part' in str(x).lower())
]
job_to_parts.to_csv(os.path.join(output_dir, "job_to_parts_rules.csv"), index=False)

cost_stats = df.groupby("concern")[["labor_total_cost", "part_total_cost"]].agg(["mean", "median", "count"]).dropna()
cost_stats.to_json(os.path.join(output_dir, "concern_to_cost_stats.json"), orient='index')

# === New: Retrieve Top ML-Based Job Predictions and Rule-Based Parts ===
def get_top_xgb_jobs(user_concern, make, model, year, top_n=3):
    vehicle = f"{make} {model} {year}"
    tier = 'luxury' if make in ["BMW", "Mercedes-Benz", "Audi", "Lexus", "Jaguar", "Infiniti", "Tesla", "Acura", "Porsche", "Cadillac"] \
        else ('economy' if make in ["Toyota", "Honda", "Hyundai", "Kia", "Chevrolet", "Ford", "Nissan", "Mazda", "Volkswagen"] else 'mid')

    input_df = pd.DataFrame([{
        'concern': user_concern,
        'vehicle': vehicle,
        'vehicle_tier': tier,
        'mileage_in': 0
    }])

    X_transformed = transformer.transform(input_df)
    probs = job_model.predict_proba(X_transformed)[0]
    top_indices = np.argsort(probs)[::-1][:top_n]

    solutions = []
    for idx in top_indices:
        job_label = job_encoder.inverse_transform([idx])[0]
        probability = float(round(probs[idx], 2))  # cast to float for JSON serialization

        parts = []
        part_matches = job_to_parts[job_to_parts['antecedents'].apply(lambda x: any(fuzz.partial_ratio(job_label.lower(), str(i).lower()) > 80 for i in x))]
        if part_matches.empty:
            print(f"⚠️ No parts found for job: {job_label}")
            fallback_parts = df['part_name_grouped'].value_counts().head(2).index.tolist()
            for pname in fallback_parts:
                p_cost = df[df['part_name_grouped'] == pname]["part_total_cost"].median()
                l_cost = df[df['labor_job_info_grouped'].str.contains(job_label, case=False, na=False)]["labor_total_cost"].median()
                time = df[df['labor_job_info_grouped'].str.contains(job_label, case=False, na=False)]["labor_hours"].median()
                parts.append({
                    "name": pname,
                    "part_cost": float(round(p_cost, 2)) if not pd.isna(p_cost) else None,
                    "labor_cost": float(round(l_cost, 2)) if not pd.isna(l_cost) else None,
                    "time": float(round(time, 2)) if not pd.isna(time) else None
                })
        part_names = []
        for _, prow in part_matches.iterrows():
            part_names.extend(list(prow['consequents']))
        unique_parts = list(set(part_names))[:2]

        for pname in unique_parts:
            p_cost = df[df['part_name_grouped'].str.contains(pname, case=False, na=False)]["part_total_cost"].median()
            l_cost = df[df['labor_job_info_grouped'].str.contains(job_label, case=False, na=False)]["labor_total_cost"].median()
            time = df[df['labor_job_info_grouped'].str.contains(job_label, case=False, na=False)]["labor_hours"].median()

            parts.append({
                "name": pname,
                "part_cost": float(round(p_cost, 2)) if not pd.isna(p_cost) else None,
                "labor_cost": float(round(l_cost, 2)) if not pd.isna(l_cost) else None,
                "time": float(round(time, 2)) if not pd.isna(time) else None
            })

        solutions.append({
            "job_description": job_label,
            "probability": probability,
            "parts": parts
        })

    return solutions

# === Example Usage with Model ===
example = get_top_xgb_jobs("sound when braking", "Toyota", "Camry", 2018)
print(json.dumps(example, indent=2))

print("✅ XGBoost model integrated with top-3 prediction + parts inference.")
