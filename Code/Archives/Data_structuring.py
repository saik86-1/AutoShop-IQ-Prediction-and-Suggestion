import pandas as pd
import ast

# 📌 Define the CSV File Path
CSV_PATH = "C:/Users/mruga/Desktop/New folder/Data Extraction/Manual data Analysis/All Repair vehicle data.csv"

# 📌 Load CSV with all columns as strings
df = pd.read_csv(CSV_PATH, dtype=str)

df.head()

# # 📌 Convert JSON-like columns to actual Python objects
# json_columns = ["vehicle", "vehicle_issues", "jobs", "Financials_Overall"]
# for col in json_columns:
#     df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else None)

# # 📌 Extract vehicle details into separate columns
# df["vehicle_year"] = df["vehicle"].apply(
#     lambda v: v.get("year", None) if isinstance(v, dict) else None
# )
# df["vehicle_make"] = df["vehicle"].apply(
#     lambda v: v.get("make", None) if isinstance(v, dict) else None
# )
# df["vehicle_model"] = df["vehicle"].apply(
#     lambda v: v.get("model", None) if isinstance(v, dict) else None
# )

# # 📌 Remove rows where vehicle make or model is missing
# df_cleaned = df.dropna(subset=["vehicle_make", "vehicle_model"])

# # 📌 Create an empty list to store structured data
# structured_data = []

# # 📌 Iterate over each row in the dataset
# for _, row in df_cleaned.iterrows():
#     ro_id = row["ro_number"]  # Keep RO Number as the common identifier

#     # Extract vehicle details (Only one set per RO)
#     vehicle_data = {
#         "ro_number": ro_id,
#         "vehicle_year": row["vehicle_year"],
#         "vehicle_make": row["vehicle_make"],
#         "vehicle_model": row["vehicle_model"],
#     }

#     # 📌 Expand all issues (each issue will be a new row with the same RO ID)
#     if isinstance(row["vehicle_issues"], list):
#         for issue in row["vehicle_issues"]:
#             issue_data = vehicle_data.copy()
#             issue_data["issue"] = issue.get("concern", None)  # Store issue description
#             structured_data.append(issue_data)

#     # 📌 Expand job details (each job will be a separate row)
#     if isinstance(row["jobs"], list):
#         for job in row["jobs"]:
#             job_data = vehicle_data.copy()
#             job_data["job"] = job.get("job_type", None)  # Store job type

#             # 📌 Extract parts used under each job
#             job_parts = []
#             if isinstance(job, dict) and "parts" in job:
#                 job_parts = [part["name"] for part in job["parts"] if "name" in part]

#             # 📌 Add parts in a structured way
#             job_data["parts"] = ", ".join(job_parts) if job_parts else None
#             structured_data.append(job_data)

# # 📌 Convert structured data into a DataFrame
# df_structured = pd.DataFrame(structured_data)

# # 📌 Save the structured data into a new CSV file
# OUTPUT_PATH = (
#     "C:/Users/mruga/Desktop/New folder/Data Extraction/structured_repair_orders.csv"
# )
# df_structured.to_csv(OUTPUT_PATH, index=False)

# print("\n✅ Data transformation complete!")
# print(f"📂 Structured data saved to: {OUTPUT_PATH}")
