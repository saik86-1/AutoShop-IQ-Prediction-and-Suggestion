import pandas as pd

# === Load the provided CSV files ===
csv_path_main = r"C:\Users\mruga\Desktop\New folder\Capstone Project\AutoShopIQ_Repair_chatbot\Data\Invoices_converted_csv\structured_repair_data_filled.csv"
csv_path_mileage = r"C:/Users/mruga/Desktop/New folder/Capstone Project/AutoShopIQ_Repair_chatbot/Data/Invoices_converted_csv/all_repair_orders_millage.csv"

df_main = pd.read_csv(csv_path_main)
df_mileage = pd.read_csv(csv_path_mileage)

# === Check for missing mileage data in main file ===
missing_mileage_count = df_main["mileage_in"].isna().sum()
print(f"🔍 Missing 'mileage_in' in structured_repair_data.csv: {missing_mileage_count}")

# === Check if any ro_numbers in mileage file are not in the main file ===
extra_ro_numbers = set(df_mileage["ro_number"]) - set(df_main["ro_number"])
print(f"🧾 Repair orders in mileage file not found in structured data: {len(extra_ro_numbers)}")

# (Optional) Show some example ro_numbers that are extra
if extra_ro_numbers:
    print(f"📋 Sample missing ro_numbers: {list(extra_ro_numbers)[:5]}")


# import pandas as pd

# # === File paths ===
# main_path = r"C:\Users\mruga\Desktop\New folder\Capstone Project\AutoShopIQ_Repair_chatbot\Data\Invoices_converted_csv\structured_repair_data.csv"
# mileage_path = r"C:\Users\mruga\Desktop\New folder\Capstone Project\AutoShopIQ_Repair_chatbot\Data\Invoices_converted_csv\all_repair_orders_millage.csv"

# # === Load data ===
# df_main = pd.read_csv(main_path)
# df_mileage = pd.read_csv(mileage_path)

# # === Merge only where mileage is missing in main ===
# missing_mask = df_main["mileage_in"].isna()
# print(f"🔍 Repair orders with missing mileage: {missing_mask.sum()}")

# # Only merge mileage for matching ro_numbers with missing mileage
# df_main.loc[missing_mask, "mileage_in"] = df_main.loc[missing_mask, "ro_number"].map(
#     df_mileage.set_index("ro_number")["mileage_in"]
# )
# df_main.loc[missing_mask, "mileage_out"] = df_main.loc[missing_mask, "ro_number"].map(
#     df_mileage.set_index("ro_number")["mileage_out"]
# )

# # === Recheck how many are still missing ===
# still_missing = df_main["mileage_in"].isna().sum()
# print(f"✅ Missing mileage after merge: {still_missing}")

# # === Save updated file ===
# updated_path = main_path.replace("structured_repair_data.csv", "structured_repair_data_filled.csv")
# df_main.to_csv(updated_path, index=False)
# print(f"📁 Updated CSV saved to → {updated_path}")

