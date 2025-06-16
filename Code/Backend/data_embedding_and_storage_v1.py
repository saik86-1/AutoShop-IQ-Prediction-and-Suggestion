from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
import pandas as pd
import ast
import os
from tqdm import tqdm

# === CONFIGURATION ===
CSV_PATH = "C:/Users/mruga/Desktop/New folder/Capstone Project/AutoShopIQ_Repair_chatbot/Data/Invoices_converted_csv/all_repair_orders_millage.csv"
CHROMA_PATH = r"C:/Users/mruga/Desktop/New folder/Capstone Project/AutoShopIQ_Repair_chatbot/chroma_repair_orders"
COLLECTION_NAME = "repair_orders"
EMBEDDING_MODEL = "text-embedding-3-small"

# === STEP 1: Load Data ===
df = pd.read_csv(CSV_PATH, dtype=str)

# === STEP 2: Parse JSON-like Columns ===
json_columns = ["vehicle", "jobs", "vehicle_issues", "Financials_Overall"]
for col in json_columns:
    df[col] = df[col].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) and (x.startswith("{") or x.startswith("[")) else x
    )

# === STEP 3: Initialize LangChain Chroma Store ===
embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
documents = []
metadatas = []

# === STEP 4: Convert Row to Flat Text ===
def prepare_repair_order_text(row):
    vehicle_data = row["vehicle"] if isinstance(row["vehicle"], dict) else ast.literal_eval(row["vehicle"])
    vehicle_issues = row["vehicle_issues"] if isinstance(row["vehicle_issues"], list) else ast.literal_eval(row["vehicle_issues"])
    jobs = row["jobs"] if isinstance(row["jobs"], list) else ast.literal_eval(row["jobs"])
    financial_data_overall = row.get("Financials_Overall", {})
    if isinstance(financial_data_overall, str):
        financial_data_overall = ast.literal_eval(financial_data_overall)

    vehicle_info = f"""
    Vehicle Info: {vehicle_data.get('year', 'Unknown')} {vehicle_data.get('make', 'Unknown')} {vehicle_data.get('model', 'Unknown')} | VIN: {vehicle_data.get('vin', 'Unknown')} | Color: {vehicle_data.get('color', 'Unknown')}
    """

    vehicle_issues_text = "\n".join([
        f"Concern: {issue.get('concern', 'Unknown')}, Finding: {issue.get('finding', 'Not Diagnosed')}"
        for issue in vehicle_issues
    ]) if isinstance(vehicle_issues, list) else "No issues."

    job_text = "\n".join([
        f"Job Type: {job.get('job_type', 'Unknown')}, Parts: {[p.get('part_name') for p in job.get('parts', [])]}, Labor: {[l.get('job_info') for l in job.get('labor', [])]}"
        for job in jobs
    ]) if isinstance(jobs, list) else "No jobs."

    financial_summary = f"""
    Total Labor: ${financial_data_overall.get('labour', 0)}, Parts: ${financial_data_overall.get('part', 0)}, Fees: ${financial_data_overall.get('fees', 0)}
    """ if financial_data_overall else "No financials."

    return f"""
    Repair Order #: {row['ro_number']}
    {vehicle_info}
    Concerns:\n{vehicle_issues_text}
    Jobs:\n{job_text}
    Summary:\n{financial_summary}
    """

# === STEP 5: Build Documents & Metadata ===
for index, row in tqdm(df.iterrows(), total=len(df), desc="🔄 Preparing Documents"):
    try:
        text = prepare_repair_order_text(row)
        vehicle_data = row["vehicle"] if isinstance(row["vehicle"], dict) else ast.literal_eval(row["vehicle"])
        issues = row["vehicle_issues"] if isinstance(row["vehicle_issues"], list) else ast.literal_eval(row["vehicle_issues"])
        concerns = ", ".join([i.get("concern", "Unknown") for i in issues]) if isinstance(issues, list) else ""

        metadata = {
            "ro_number": row.get("ro_number", "").lower(),
            "pdf_name": row.get("pdf_name", "").lower(),
            "vehicle_year": str(vehicle_data.get("year", "")).lower(),
            "vehicle_make": (vehicle_data.get("make") or "").strip().lower(),
            "vehicle_model": (vehicle_data.get("model") or "").strip().lower(),
            "customer_concerns": concerns.lower()
        }

        documents.append(Document(page_content=text, metadata=metadata))
    except Exception as e:
        pass  # Skip problematic rows

# === STEP 6: Store in ChromaDB ===
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_PATH
)

vectorstore.persist()
print("✅ All repair orders embedded and stored successfully.")
