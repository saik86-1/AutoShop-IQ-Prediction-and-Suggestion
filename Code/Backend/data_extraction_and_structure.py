import openai
import json
import os

# from fastapi import FastAPI
# import uvicorn
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.chat_models import ChatOpenAI
import pandas as pd
from langchain.schema import HumanMessage
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain import hub

# from langchain_community.document_loaders import PyPDFLoader
# from langchain.document_loaders import PyMuPDFLoader
# from langchain.chat_models import ChatOpenAI
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain.chains.retrieval import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# Define a model for vehicle info to ensure correct data format


load_dotenv()

# Define paths for input PDFs and output CSVs
pdf_directory = "C:/Users/mruga/Desktop/New folder/Data Extraction/invoices_pdf"


# Define path for the single CSV file
CSV_DIR = "C:/Users/mruga/Desktop/New folder/Data Extraction/invoices_converted_csv/"
CSV_PATH = os.path.join(CSV_DIR, "all_repair_orders.csv")


# Define Pydantic models for structured data validation
class Vehicle(BaseModel):
    year: Optional[str] = None
    make: Optional[str] = None
    model: Optional[str] = None
    engine: Optional[str] = None
    vin: Optional[str] = None
    license_plate: Optional[str] = None
    color: Optional[str] = None
    production_date: Optional[str] = None


class LaborJob(BaseModel):
    job_info: Optional[str] = None
    technician: Optional[str] = None
    hours: Optional[float] = None
    rate_per_hour: Optional[float] = None
    total_cost: Optional[float] = None


class Part(BaseModel):
    part_name: Optional[str] = None
    part_number: Optional[str] = None
    quantity: Optional[float] = None
    cost_per_unit: Optional[float] = None
    retail_price_per_unit: Optional[float] = None
    total_cost: Optional[float] = None
    inventory_status: Optional[str] = None


class Fee(BaseModel):
    fee_type: Optional[str] = None
    calculation_method: Optional[str] = None
    amount: Optional[float] = None


class Financials(BaseModel):
    gross_profit_percentage: Optional[float] = None
    gross_profit_amount: Optional[float] = None
    gross_profit_per_hour: Optional[float] = None
    subtotal: Optional[float] = None
    estimated_fees_tax: Optional[float] = None
    estimated_labor_tax: Optional[float] = None
    estimated_parts_tax: Optional[float] = None
    total_job_cost: Optional[float] = None
    overall_total: Optional[float] = None


class Financials_Overall(BaseModel):
    labour: Optional[float] = None
    part: Optional[float] = None
    sublet: Optional[float] = None
    fees: Optional[float] = None
    discount: Optional[float] = None
    subtotal: Optional[float] = None
    taxes: Optional[float] = None


class Job(BaseModel):
    job_type: str  # Example: "Brake Pad & Rotor Package"
    labor: List[LaborJob] = []
    parts: List[Part] = []
    fees: List[Fee] = []
    financials: Optional[Financials] = None  # Financials related to this job


class VehicleIssue(BaseModel):
    concern: str  # Customer or technician concern
    finding: Optional[str] = None  # Diagnosed issue or solution


class RepairOrder(BaseModel):
    ro_number: str
    customer_name: Optional[str] = None
    vehicle: Vehicle
    vehicle_issues: List[VehicleIssue] = []
    service_writer: Optional[str] = None
    technician: Optional[str] = None
    labor_rate: Optional[float] = None
    customer_time_in: Optional[str] = None
    promised_time_out: Optional[str] = None
    marketing_source: Optional[str] = None
    appointment_option: Optional[str] = None
    jobs: List[Job] = (
        []
    )  # ✅ Now all jobs, labor, parts, fees, and financials are grouped
    Financials_Overall: (
        Financials_Overall  # ✅ This stores overall financials for the entire order
    )


def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF using LangChain's PyMuPDFLoader.
    """
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    extracted_text = "\n".join([doc.page_content for doc in documents])
    return extracted_text


def generate_structured_json(extracted_text):
    """
    Uses GPT-4 (via LangChain) to convert extracted text into structured JSON format.
    """

    schema_json = json.dumps(RepairOrder.model_json_schema(), indent=4)

    prompt = f"""
    Extract structured data from the following repair order text:
    
    {extracted_text}
    
    Convert it into the following JSON format:
    {schema_json}
    
    Ensure:
    - Extract all relevant fields.
    - Maintain data integrity.
    - Preserve numeric values in their proper format.
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI that extracts structured data from repair orders.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        # 🚨 Debugging: Print raw response
        raw_response = response.choices[0].message.content
        print("🔹 RAW OpenAI Response:", raw_response)

        # ✅ Fix: Strip ```json and ``` from the response
        if raw_response.startswith("```json"):
            raw_response = raw_response[7:]  # Remove "```json"
        if raw_response.endswith("```"):
            raw_response = raw_response[:-3]  # Remove ending backticks

        # ✅ Parse cleaned JSON
        structured_data = json.loads(raw_response.strip())
        return structured_data

    except json.JSONDecodeError as e:
        print("🚨 JSON Parsing Error:", e)
        print("❌ Failed to parse response. Raw response:", raw_response)
        return None

    except openai.OpenAIError as e:
        print(f"🚨 OpenAI API Error: {e}")
        return None


def save_to_dataframe(repair_orders, pdf_name):
    """
    Appends extracted repair orders into a single DataFrame and saves it as a CSV file.
    - Uses `model_dump()` for Pydantic v2 compatibility.
    - Adds a `pdf_name` column for tracking the source file.
    - Ensures `RO Number` is the unique identifier to prevent duplicates.
    """

    # ✅ Ensure the directory exists before saving
    os.makedirs(CSV_DIR, exist_ok=True)

    # Convert repair orders to DataFrame using `model_dump()`
    df_new = pd.DataFrame([ro.model_dump() for ro in repair_orders])
    df_new["pdf_name"] = pdf_name  # ✅ Add PDF filename column

    # Handle file permission issues by checking if the file is locked
    try:
        if os.path.exists(CSV_PATH):
            # Load existing data
            df_existing = pd.read_csv(CSV_PATH, dtype=str)

            # ✅ Prevent duplicate RO Numbers
            df_existing = df_existing[
                ~df_existing["ro_number"].isin(df_new["ro_number"])
            ]

            # Append new data
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new  # First-time creation

        # Save to CSV with proper permissions
        df_combined.to_csv(CSV_PATH, index=False, mode="w", encoding="utf-8-sig")

        print(f"✅ Data saved to {CSV_PATH} with {len(df_combined)} total records.")
        return df_combined

    except PermissionError:
        print(
            f"❌ Permission denied: Unable to write to {CSV_PATH}. Close the file if open and try again."
        )
        return None


# Load the existing dataset
if os.path.exists(CSV_PATH):
    existing_df = pd.read_csv(CSV_PATH, dtype=str)
    existing_pdfs = set(
        existing_df["pdf_name"].dropna()
    )  # Convert to set for fast lookup
else:
    existing_pdfs = set()

# Define a list to store skipped PDFs
skipped_files = []


def process_pdfs(pdf_directory):
    """
    Process all PDFs in the given directory, extract text, generate structured data,
    and save to a single CSV. Skips files that are already present in all_repair_orders.csv.
    """

    pdf_files = [
        os.path.join(pdf_directory, file)
        for file in os.listdir(pdf_directory)
        if file.endswith(".pdf")
    ]

    for pdf_file in pdf_files:
        pdf_name = os.path.basename(pdf_file)

        # ✅ Check if PDF is already present in the dataset
        if pdf_name in existing_pdfs:
            print(f"🔹 Skipped {pdf_name} (Already in dataset)")
            skipped_files.append({"pdf_name": pdf_name, "reason": "Already processed"})
            continue  # Skip further processing

        print(f"Processing: {pdf_file}")

        extracted_text = extract_text_from_pdf(pdf_file)
        structured_data = generate_structured_json(extracted_text)

        if structured_data:
            validated_order = RepairOrder(**structured_data)  # Validate using Pydantic

            # ✅ Save data
            save_to_dataframe([validated_order], pdf_name)
            print(f"✅ Processed and saved data from: {pdf_name}")

        else:
            print(f"❌ Skipped {pdf_file} due to missing or invalid structured data.")
            skipped_files.append(
                {"pdf_name": pdf_name, "reason": "Invalid structured data"}
            )

    # ✅ Save skipped files to CSV for tracking
    if skipped_files:
        skipped_df = pd.DataFrame(skipped_files)
        skipped_df.to_csv("skipped_files_log1.csv", index=False)
        print(f"📄 Skipped files log saved to skipped_files_log.csv")


# Run the processing function for all PDFs in the specified directory
process_pdfs(pdf_directory)
