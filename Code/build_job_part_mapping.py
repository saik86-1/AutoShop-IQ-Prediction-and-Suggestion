import pandas as pd
import joblib
import os

# === INPUT CSV ===
CSV_PATH = "C:/Users/mruga/Desktop/New folder/Capstone Project/AutoShopIQ_Repair_chatbot/Data/Invoices_converted_csv/structured_repair_data.csv"
OUTPUT_PATH = "C:/Users/mruga/Desktop/New folder/Capstone Project/AutoShopIQ_Repair_chatbot/Models/job_to_parts_mapping.joblib"

# === LOAD DATA ===
df = pd.read_csv(CSV_PATH)

# === FILTER OUT EMPTY JOBS/PARTS ===
df = df.dropna(subset=["job_type", "part_name"])

# === BUILD JOB → PARTS MAPPING ===
job_to_parts = df.groupby("job_type")["part_name"].apply(lambda x: list(set(x.dropna()))).to_dict()

# === SAVE TO DISK ===
joblib.dump(job_to_parts, OUTPUT_PATH)

print(f"✅ Mapping saved to: {OUTPUT_PATH}")
print(f"🔧 Sample: {list(job_to_parts.items())[:3]}")
