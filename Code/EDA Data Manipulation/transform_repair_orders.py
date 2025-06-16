import pandas as pd
import ast
import os

# === CONFIG ===
INPUT_CSV = r"C:\Users\mruga\Desktop\New folder\Capstone Project\AutoShopIQ_Repair_chatbot\Data\Invoices_converted_csv\all_repair_orders_millage.csv"
OUTPUT_CSV = r"C:\Users\mruga\Desktop\New folder\Capstone Project\AutoShopIQ_Repair_chatbot\Data\Invoices_converted_csv\structured_repair_data.csv"

# === Load CSV ===
df = pd.read_csv(INPUT_CSV, dtype=str)

# Fill missing dictionary fields
json_cols = ["vehicle", "vehicle_issues", "jobs", "Financials_Overall"]
for col in json_cols:
    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else {})

# === Flatten Function ===
flat_rows = []

for _, row in df.iterrows():
    vehicle = row["vehicle"]
    issues = row["vehicle_issues"]
    jobs = row["jobs"]
    mileage_in = row.get("mileage_in", None)
    mileage_out = row.get("mileage_out", None)

    # Extract vehicle info
    year = vehicle.get("year", "")
    make = vehicle.get("make", "")
    model = vehicle.get("model", "")

    # Loop over concerns
    if isinstance(issues, list):
        for issue in issues:
            concern = issue.get("concern", "")
            finding = issue.get("finding", "")

            # Loop over jobs
            if isinstance(jobs, list):
                for job in jobs:
                    job_type = job.get("job_type", "")

                    # Loop over labor
                    if isinstance(job.get("labor", []), list):
                        for labor in job.get("labor", []):
                            flat_rows.append({
                                "ro_number": row["ro_number"],
                                "make": make,
                                "model": model,
                                "year": year,
                                "mileage_in": mileage_in,
                                "mileage_out": mileage_out,
                                "concern": concern,
                                "job_type": job_type,
                                "labor_job_info": labor.get("job_info", ""),
                                "labor_hours": labor.get("hours", ""),
                                "labor_total_cost": labor.get("total_cost", ""),
                                "source": "labor"
                            })

                    # Loop over parts
                    if isinstance(job.get("parts", []), list):
                        for part in job.get("parts", []):
                            flat_rows.append({
                                "ro_number": row["ro_number"],
                                "make": make,
                                "model": model,
                                "year": year,
                                "mileage_in": mileage_in,
                                "mileage_out": mileage_out,
                                "concern": concern,
                                "job_type": job_type,
                                "part_name": part.get("part_name", ""),
                                "part_total_cost": part.get("total_cost", ""),
                                "source": "part"
                            })

# === Create DataFrame & Save ===
flat_df = pd.DataFrame(flat_rows)
flat_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"✅ Flattened data saved → {OUTPUT_CSV}")
print(f"📊 Total rows: {len(flat_df)}")
