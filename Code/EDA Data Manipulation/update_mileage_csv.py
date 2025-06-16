import os
import pandas as pd
import re
from langchain_community.document_loaders import PyMuPDFLoader

# === CONFIGURATION ===
PDF_FOLDER = r"C:\Users\mruga\Desktop\New folder\Capstone Project\Capstone Invocies\Raw_Invoices\Raw_Invoices"
CSV_PATH = r"C:\Users\mruga\Desktop\New folder\Capstone Project\AutoShopIQ_Repair_chatbot\Data\Invoices_converted_csv\all_repair_orders_millage.csv"
LIMIT = 6148  # Set to None or remove to process all

# === HELPER FUNCTIONS ===

def extract_text_from_pdf(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    return "\n".join([doc.page_content for doc in documents])

def extract_mileage(text):
    # Tries multiple mileage formats
    patterns = [
        r"In: ?([\d,]+)\s*\|\s*Out: ?([\d,]+)",
        r"Mileage In[:\- ]+([\d,]+).*?Mileage Out[:\- ]+([\d,]+)",
        r"Odometer In[:\- ]+([\d,]+).*?Odometer Out[:\- ]+([\d,]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            mileage_in = int(match.group(1).replace(",", ""))
            mileage_out = int(match.group(2).replace(",", ""))
            return mileage_in, mileage_out
    return None, None

# === LOAD CSV ===
df = pd.read_csv(CSV_PATH, dtype=str)

# Add missing mileage columns
if "mileage_in" not in df.columns:
    df["mileage_in"] = None
if "mileage_out" not in df.columns:
    df["mileage_out"] = None

# Track how many and what we missed
processed_count = 0
missing_mileage = []

# === MAIN LOOP ===
for idx, row in df.iterrows():
    if LIMIT and processed_count >= LIMIT:
        break

    pdf_name = row.get("pdf_name")
    
    # ✅ Skip rows that already have mileage
    if pd.notna(row.get("mileage_in")) and pd.notna(row.get("mileage_out")):
        continue
    if pd.notna(pdf_name):
        pdf_path = os.path.join(PDF_FOLDER, pdf_name)
        if os.path.exists(pdf_path):
            try:
                text = extract_text_from_pdf(pdf_path)
                mileage_in, mileage_out = extract_mileage(text)

                df.at[idx, "mileage_in"] = mileage_in
                df.at[idx, "mileage_out"] = mileage_out

                if mileage_in is not None and mileage_out is not None:
                    pass  # ✅ placeholder when no action needed
                    # print(f"✅ {pdf_name} → In: {mileage_in}, Out: {mileage_out}")
                else:
                    print(f"⚠️  Mileage not found in: {pdf_name}")
                    missing_mileage.append(pdf_name)

                processed_count += 1

            except Exception as e:
                print(f"❌ Error reading {pdf_name}: {e}")

# === SAVE UPDATED CSV ===
try:
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
except PermissionError:
    print(f"\n❌ Could not save CSV. Make sure the file is closed: {CSV_PATH}")
# print(f"\n🚀 Done! Mileage updated in {processed_count} files → saved to: {CSV_PATH}")

# === REPORT MISSING MILEAGE ===
if missing_mileage:
    print("\n📝 PDFs where mileage was missing:")
    for name in missing_mileage:
        print(f" - {name}")
else:
    print("\n✅ All files had mileage!")
